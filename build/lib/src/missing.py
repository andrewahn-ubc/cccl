from __future__ import annotations

import argparse
from pathlib import Path

from .manifest import read_manifest
from .utils import project_root, read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", required=True, choices=("A", "AR", "B", "BR", "P", "C", "D", "DR", "E"))
    parser.add_argument("--root", type=Path, default=project_root())
    args = parser.parse_args()
    manifest = args.root / "manifests" / f"pilot_{args.pilot}.csv"
    with manifest.open() as handle:
        count = sum(1 for _ in handle) - 1
    missing = []
    for index in range(count):
        config = read_manifest(manifest, index)
        status = read_json(args.root / "results" / "pilots" / args.pilot / str(config["run_id"]) / "status.json", {})
        if status.get("state") not in ("complete", "skipped"):
            missing.append(index)
    print(",".join(map(str, missing)))


if __name__ == "__main__":
    main()
