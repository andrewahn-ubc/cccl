from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import random
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
import torch
import yaml


def project_root() -> Path:
    return Path(os.environ.get("RECAP_ROOT", Path(__file__).resolve().parents[1])).resolve()


def canonical(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def stable_run_id(config: dict[str, Any]) -> str:
    payload = json.dumps(canonical(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(canonical(value), indent=2, sort_keys=True) + "\n")


def atomic_yaml(path: Path, value: Any) -> None:
    atomic_text(path, yaml.safe_dump(canonical(value), sort_keys=True))


def atomic_parquet(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(args, cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            return "unknown"

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain") not in ("", "unknown")),
    }


def environment_text(root: Path) -> str:
    packages = {}
    for name in ("torch", "torchvision", "numpy", "pandas", "pyarrow", "scipy", "matplotlib"):
        try:
            module = __import__(name)
            packages[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:  # noqa: BLE001 - capture must never invalidate a scientific run
            packages[name] = f"unavailable: {exc}"
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_gpus": os.environ.get("SLURM_GPUS"),
        "loaded_modules": os.environ.get("LOADEDMODULES"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": packages,
        "git": git_state(root),
    }
    return yaml.safe_dump(payload, sort_keys=True)


class PreemptionRequested(RuntimeError):
    pass


class SignalState:
    requested = False

    def __init__(self) -> None:
        self._old: dict[int, Any] = {}

    def __enter__(self) -> Self:
        def handler(_signum: int, _frame: Any) -> None:
            self.requested = True

        for sig in (signal.SIGUSR1, signal.SIGTERM):
            self._old[sig] = signal.signal(sig, handler)
        return self

    def check(self) -> None:
        if self.requested:
            raise PreemptionRequested("scheduler requested checkpoint/requeue")

    def __exit__(self, *_args: object) -> None:
        for sig, old in self._old.items():
            signal.signal(sig, old)


class RunArtifacts:
    def __init__(self, pilot: str, config: dict[str, Any], root: Path | None = None) -> None:
        self.root = root or project_root()
        self.config = canonical(config)
        self.run_id = str(config["run_id"])
        self.path = self.root / "results" / "pilots" / pilot / self.run_id
        self.path.mkdir(parents=True, exist_ok=True)
        self.events_path = self.path / "events.jsonl"
        self.started = time.time()

    @property
    def done(self) -> bool:
        return (self.path / "DONE").exists()

    def start(self) -> None:
        atomic_yaml(self.path / "config.yaml", self.config)
        atomic_text(self.path / "environment.txt", environment_text(self.root))
        self.status("running")
        self.event("run_started", config=self.config)

    def event(self, kind: str, **values: Any) -> None:
        row = {"time": time.time(), "event": kind, **canonical(values)}
        with self.events_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def status(self, state: str, **values: Any) -> None:
        atomic_json(
            self.path / "status.json",
            {"state": state, "run_id": self.run_id, "elapsed_seconds": time.time() - self.started, **values},
        )

    def complete(self, metrics: list[dict[str, Any]], checkpoint: Any) -> None:
        atomic_parquet(self.path / "metrics.parquet", metrics)
        atomic_torch(self.path / "checkpoint.pt", checkpoint)
        self.event("run_completed", rows=len(metrics))
        self.status("complete", metric_rows=len(metrics))
        atomic_text(self.path / "DONE", "complete\n")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open() as handle:
        return json.load(handle)


@contextlib.contextmanager
def elapsed_timer() -> Iterator[dict[str, float]]:
    box: dict[str, float] = {"start": time.time()}
    try:
        yield box
    finally:
        box["elapsed_seconds"] = time.time() - box["start"]
