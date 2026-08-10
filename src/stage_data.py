from __future__ import annotations

import argparse
from pathlib import Path

from .data import stage_mnist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)
    stage_mnist(args.data_root)
    print(f"MNIST staged at {args.data_root}")


if __name__ == "__main__":
    main()

