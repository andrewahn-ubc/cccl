import json
import os
import subprocess
from pathlib import Path


def test_bootstrap_redirects_caches_and_recovers_partial_venv(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    scratch = tmp_path / "scratch"
    fake_bin = tmp_path / "bin"
    venv = scratch / "recap_venv"
    log = tmp_path / "virtualenv.json"
    fake_bin.mkdir()
    venv.mkdir(parents=True)
    (venv / "partial-file").write_text("interrupted")

    virtualenv = fake_bin / "virtualenv"
    virtualenv.write_text(
        """#!/usr/bin/env python3
import json, os, pathlib, sys
target = pathlib.Path(sys.argv[-1])
(target / "bin").mkdir(parents=True)
python = target / "bin" / "python"
python.write_text("#!/bin/sh\\nexit 0\\n")
python.chmod(0o755)
(target / "bin" / "activate").write_text(f'export PATH="{target}/bin:$PATH"\\n')
pathlib.Path(os.environ["FAKE_VIRTUALENV_LOG"]).write_text(json.dumps({
    "args": sys.argv[1:],
    "app_data": os.environ["VIRTUALENV_OVERRIDE_APP_DATA"],
    "pip_cache": os.environ["PIP_CACHE_DIR"],
    "tmpdir": os.environ["TMPDIR"],
}))
"""
    )
    virtualenv.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SCRATCH": str(scratch),
        "RECAP_ALLOW_INDEX": "1",
        "FAKE_VIRTUALENV_LOG": str(log),
    }
    subprocess.run(
        ["bash", str(root / "scripts" / "bootstrap_env.sh"), str(root), str(venv)],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )

    recorded = json.loads(log.read_text())
    assert recorded["app_data"].startswith(str(scratch))
    assert recorded["pip_cache"].startswith(str(scratch))
    assert recorded["tmpdir"].startswith(str(scratch))
    assert "--no-download" in recorded["args"]
    assert "--app-data" in recorded["args"]
    assert (venv / "bin" / "python").exists()
    assert list(scratch.glob("recap_venv.incomplete.*"))

