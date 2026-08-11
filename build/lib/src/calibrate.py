from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import PermutedMNIST
from .models import GatedMLP, make_optimizer
from .utils import atomic_json, project_root, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--root", type=Path, default=project_root())
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU calibration allocation has no visible CUDA device")
    device = torch.device(args.device)
    seed_everything(0)
    data_root = Path(os.environ.get("RECAP_DATA_ROOT", args.root / "data"))
    data = PermutedMNIST(data_root, args.root / "manifests" / "libraries" / "seed_0.npz", device)
    model = GatedMLP(50).to(device)
    optimizer = make_optimizer(model)
    generator = torch.Generator().manual_seed(123)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(args.steps):
        x, y = data.batch(step % 10, 64, generator)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    output = args.root / "results" / "pilots" / "calibration" / f"{args.device}.json"
    atomic_json(
        output,
        {
            "device": args.device,
            "steps": args.steps,
            "elapsed_seconds": elapsed,
            "updates_per_second": args.steps / elapsed,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    )
    print(output)


if __name__ == "__main__":
    main()

