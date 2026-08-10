from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .constants import SKEW
from .data import PermutedMNIST
from .manifest import read_manifest
from .methods import evaluate, gate_scores, recycle
from .models import GatedMLP, make_optimizer
from .replay import ReplayBuffer
from .schedules import oracle_transition
from .training import train_offline, train_stream
from .utils import (
    PreemptionRequested,
    RunArtifacts,
    atomic_parquet,
    atomic_text,
    project_root,
    read_json,
    seed_everything,
)


def device_from_environment() -> torch.device:
    request = os.environ.get("RECAP_DEVICE", "cpu")
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("RECAP_DEVICE=cuda but CUDA is unavailable")
    return torch.device(request)


def scientific_prerequisite(root: Path, pilot: str) -> tuple[bool, str]:
    a = read_json(root / "results" / "pilots" / "A" / "A_decision.json", {})
    b = read_json(root / "results" / "pilots" / "B" / "B_decision.json", {})
    d = read_json(root / "results" / "pilots" / "D" / "D_decision.json", {})
    if pilot == "AR" and a.get("decision") != "FAIL_INFEASIBLE":
        return False, "Pilot A width-200 diagnostic was not triggered"
    if pilot == "BR":
        if a.get("decision") not in ("PASS", "MARGINAL"):
            return False, "Pilot A stopped the stronger-skew diagnostic"
        if b.get("decision") == "PASS":
            return False, "Pilot B passed without a stronger-skew retry"
    if pilot == "DR" and d.get("decision") != "MARGINAL":
        return False, "Pilot D gentler-recycle retry was not triggered"
    if pilot in ("P", "C", "D", "DR", "E") and a.get("decision") not in ("PASS", "MARGINAL"):
        return False, "Pilot A did not establish a usable allocation-limited width"
    if pilot in ("P", "C", "D", "DR", "E") and b.get("decision") not in ("PASS", "MARGINAL"):
        return False, "Pilot B did not establish prospective-demand value"
    if pilot == "E" and d.get("decision") not in ("PASS", "MARGINAL"):
        return False, "Pilot D failed the compression gate"
    return True, ""


def skip_run(artifacts: RunArtifacts, reason: str) -> None:
    artifacts.event("run_skipped", reason=reason)
    atomic_parquet(artifacts.path / "metrics.parquet", [{"metric_kind": "skipped", "reason": reason}])
    from .utils import atomic_torch

    atomic_torch(artifacts.path / "checkpoint.pt", {"skipped": True, "reason": reason})
    artifacts.status("skipped", reason=reason)
    atomic_text(artifacts.path / "DONE", "skipped\n")


def prepared_checkpoint(root: Path, seed: int, device: torch.device) -> dict[str, Any]:
    manifest = root / "manifests" / "pilot_P.csv"
    index = next(i for i in range(3) if int(read_manifest(manifest, i)["seed"]) == seed)
    config = read_manifest(manifest, index)
    path = root / "results" / "pilots" / "P" / str(config["run_id"]) / "checkpoint.pt"
    if not path.exists():
        raise FileNotFoundError(f"prepared checkpoint missing: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def run_c(
    config: dict[str, Any], data: PermutedMNIST, artifacts: RunArtifacts, device: torch.device
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state = prepared_checkpoint(artifacts.root, int(config["seed"]), device)
    model = GatedMLP(int(config["width"])).to(device)
    model.load_state_dict(state["model"])
    probes = data.probes([0, 1, 5])
    scores = gate_scores(model, probes)
    baseline = {task: evaluate(model, *batch)["nll"] for task, batch in probes.items()}
    rng = np.random.default_rng(int(config["seed"]) + 110_000)
    rows = []
    for layer_index, gate in enumerate(model.gates):
        permutation = rng.permutation(len(gate))
        positions = np.linspace(0, len(gate) - 1, min(25, len(gate)), dtype=int)
        units = permutation[positions]
        for task, (x, y) in probes.items():
            for unit in units:
                old = float(gate.data[int(unit)].item())
                gate.data[int(unit)] = 0.0
                damage = evaluate(model, x, y)["nll"] - baseline[task]
                gate.data[int(unit)] = old
                rows.append(
                    {
                        "metric_kind": "unit_ablation",
                        "seed": int(config["seed"]),
                        "layer": layer_index,
                        "unit": int(unit),
                        "task": task,
                        "taylor": float(scores["taylor"][task][layer_index][int(unit)]),
                        "signed": float(scores["signed"][task][layer_index][int(unit)]),
                        "fisher": float(scores["fisher"][task][layer_index][int(unit)]),
                        "activation": float(scores["activation"][task][layer_index][int(unit)]),
                        "ablation_damage": float(damage),
                    }
                )
    atomic_parquet(artifacts.path / "score_vs_ablation.parquet", rows)
    return rows, {"model": model.state_dict(), "source_checkpoint": state, "width": int(config["width"])}


def run_d(
    config: dict[str, Any], data: PermutedMNIST, artifacts: RunArtifacts, device: torch.device
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state = prepared_checkpoint(artifacts.root, int(config["seed"]), device)
    model = GatedMLP(int(config["width"])).to(device)
    model.load_state_dict(state["model"])
    optimizer = make_optimizer(model, float(config["lr"]))
    optimizer.load_state_dict(state["optimizer"])
    replay = ReplayBuffer(int(config["buffer_size"]), int(config["seed"]))
    replay.load_state_dict(state["replay"])
    probes = data.probes()
    weights = {task: SKEW[task] for task in probes}
    metric_decision = read_json(artifacts.root / "results" / "pilots" / "C" / "C_decision.json", {})
    metric = str(metric_decision.get("selected_importance_metric", "taylor"))
    method = str(config["method"])
    selection_metric = "random" if method == "random_consolidate" else metric
    steps = 0 if method == "importance_prune" else int(config["sleep_updates"])
    beta = 0.0 if method in ("importance_prune", "importance_ce_only") else float(config["kd_beta"])
    outcome = recycle(
        model,
        optimizer,
        replay,
        probes,
        weights,
        selection_metric,
        float(config["recycle_fraction"]),
        steps,
        beta,
        device,
        np.random.default_rng(int(config["seed"]) + 120_000),
        torch.Generator(device=device).manual_seed(int(config["seed"]) + 130_000),
    )
    if outcome["reset_logit_max_abs"] > 1e-5:
        raise AssertionError(f"reset changed logits by {outcome['reset_logit_max_abs']}")
    mask_damage = outcome["after_mask"]["nll"] - outcome["before"]["nll"]
    recovery = (outcome["after_mask"]["nll"] - outcome["after_sleep"]["nll"]) / max(mask_damage, 1e-8)
    rows = []
    for stage in ("before", "after_mask", "after_sleep", "after_reset"):
        rows.append(
            {
                "metric_kind": "compression_stage",
                "seed": int(config["seed"]),
                "method": method,
                "stage": stage,
                "accuracy": outcome[stage]["accuracy"],
                "nll": outcome[stage]["nll"],
                "recovery_ratio": float(recovery),
                "reset_logit_max_abs": outcome["reset_logit_max_abs"],
                "importance_metric": selection_metric,
            }
        )
    artifacts.event("compression", **outcome, recovery_ratio=recovery, importance_metric=selection_metric)
    return rows, {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "source_checkpoint_seed": int(config["seed"]),
        "outcome": outcome,
    }


def execute(config: dict[str, Any], root: Path) -> None:
    pilot = str(config["pilot"])
    asset_suffix = "_strong" if pilot == "BR" else ""
    asset_path = root / "manifests" / "libraries" / f"seed_{config['seed']}{asset_suffix}.npz"
    if isinstance(config["width"], str):
        raise TypeError("dynamic manifests have not been resolved; run `python -m src.manifest resolve`")
    asset = np.load(asset_path)
    config = {
        **config,
        "task_library": str(asset_path),
        "input_permutations": asset["input_permutations"].tolist(),
        "label_permutations": asset["label_permutations"].tolist(),
        "schedule": asset["schedule"].astype(int).tolist(),
        "generator_probabilities": list(config.get("skew_probabilities", SKEW)),
        "oracle_transition_matrix": oracle_transition(tuple(config.get("skew_probabilities", SKEW))).tolist(),
    }
    artifacts = RunArtifacts(pilot, config, root)
    if artifacts.done:
        print(f"{pilot}/{artifacts.run_id} already complete")
        return
    artifacts.start()
    allowed, reason = scientific_prerequisite(root, pilot)
    if not allowed:
        skip_run(artifacts, reason)
        return
    device = device_from_environment()
    seed_everything(int(config["seed"]))
    data_root = Path(os.environ.get("RECAP_DATA_ROOT", root / "data"))
    data = PermutedMNIST(data_root, asset_path, device)
    try:
        if pilot in ("A", "AR") and config["method"] == "offline_mixture":
            metrics, checkpoint = train_offline(config, data, artifacts, device)
        elif pilot in ("A", "B", "BR", "E"):
            metric = read_json(root / "results" / "pilots" / "C" / "C_decision.json", {}).get(
                "selected_importance_metric", "taylor"
            )
            metrics, checkpoint = train_stream(config, data, artifacts, device, importance_metric=str(metric))
        elif pilot == "P":
            metrics, checkpoint = train_stream(config, data, artifacts, device, stop_after_blocks=15)
        elif pilot == "C":
            metrics, checkpoint = run_c(config, data, artifacts, device)
        elif pilot in ("D", "DR"):
            metrics, checkpoint = run_d(config, data, artifacts, device)
        else:
            raise ValueError(f"unknown pilot {pilot}")
        artifacts.complete(metrics, checkpoint)
    except PreemptionRequested:
        artifacts.event("preempted")
        artifacts.status("preempted", resumable=True)
        raise SystemExit(99)
    except Exception as exc:
        artifacts.event("run_failed", error=repr(exc), traceback=traceback.format_exc())
        artifacts.status("failed", error=repr(exc))
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--root", type=Path, default=project_root())
    args = parser.parse_args()
    execute(read_manifest(args.manifest, args.row), args.root.resolve())


if __name__ == "__main__":
    main()
