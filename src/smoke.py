from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import PermutedMNIST
from .training import train_stream
from .utils import RunArtifacts, project_root, stable_run_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    config = {
        "pilot": "smoke",
        "method": "ER",
        "seed": 0,
        "width": 10,
        "batch_size": 64,
        "blocks": 1,
        "updates_per_block": 3,
        "total_updates": 3,
        "buffer_size": 500,
        "lr": 1e-3,
        "sleep_updates": 0,
        "recycle_fraction": 0.2,
        "kd_beta": 1.0,
        "smoke_version": 2,
    }
    config["run_id"] = stable_run_id(config)
    artifacts = RunArtifacts("smoke", config, args.root)
    if artifacts.done:
        print(artifacts.path)
        return
    artifacts.start()
    data = PermutedMNIST(
        args.data_root,
        args.root / "manifests" / "libraries" / "seed_0.npz",
        torch.device("cpu"),
    )
    metrics, checkpoint = train_stream(config, data, artifacts, torch.device("cpu"))
    artifacts.complete(metrics, checkpoint)
    print(artifacts.path)


if __name__ == "__main__":
    main()
