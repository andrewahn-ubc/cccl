from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .constants import (
    BATCH_SIZE,
    BLOCKS,
    BUFFER_SIZE,
    DISCOUNT,
    HORIZON,
    KD_BETA,
    KD_TEMPERATURE,
    RECYCLE_FRACTION,
    SKEW,
    SLEEP_UPDATES,
    STRONG_SKEW,
    TOTAL_UPDATES,
    UPDATES_PER_BLOCK,
)
from .data import make_library_asset
from .utils import project_root, read_json, stable_run_id

COMMON = {
    "batch_size": BATCH_SIZE,
    "blocks": BLOCKS,
    "updates_per_block": UPDATES_PER_BLOCK,
    "total_updates": TOTAL_UPDATES,
    "buffer_size": BUFFER_SIZE,
    "lr": 1e-3,
    "horizon": HORIZON,
    "gamma": DISCOUNT,
    "recycle_fraction": RECYCLE_FRACTION,
    "sleep_updates": SLEEP_UPDATES,
    "kd_beta": KD_BETA,
    "kd_temperature": KD_TEMPERATURE,
    "skew_probabilities": json.dumps(list(SKEW)),
}


def row(pilot: str, method: str, seed: int, width: int | str, **extra: Any) -> dict[str, Any]:
    config = {**COMMON, "pilot": pilot, "method": method, "seed": seed, "width": width, **extra}
    config["run_id"] = "PENDING" if isinstance(width, str) else stable_run_id(config)
    return config


def base_manifests() -> dict[str, list[dict[str, Any]]]:
    return {
        "A": [
            row("A", method, seed, width)
            for width in (10, 25, 50, 100)
            for method in ("offline_mixture", "ER", "periodic_full_reset")
            for seed in (0, 1)
        ],
        "B": [
            row("B", method, seed, 50)
            for method in ("uniform_task", "oracle_frequency_weighted")
            for seed in (0, 1, 2, 3)
        ],
        "AR": [row("AR", "offline_mixture", 0, 200, diagnostic="width_200_feasibility")],
        "BR": [
            row("BR", method, seed, 50, skew_probabilities=json.dumps(list(STRONG_SKEW)), diagnostic="strong_skew")
            for method in ("uniform_task", "oracle_frequency_weighted")
            for seed in (0, 1)
        ],
        "P": [row("P", "ER_checkpoint", seed, "selected") for seed in (0, 1, 2)],
        "C": [row("C", "score_validation", seed, "selected") for seed in (0, 1)],
        "D": [
            row("D", method, seed, "selected")
            for method in (
                "importance_prune",
                "importance_consolidate",
                "random_consolidate",
                "importance_ce_only",
            )
            for seed in (0, 1, 2)
        ],
        "DR": [
            row(
                "DR",
                method,
                seed,
                "selected",
                recycle_fraction=0.10,
                sleep_updates=100,
                diagnostic="gentler_recycle_longer_sleep",
            )
            for method in ("importance_consolidate", "random_consolidate")
            for seed in (0, 1, 2)
        ],
        "E": [
            row("E", method, seed, "selected")
            for method in ("ER", "ER_compute_matched", "random_recycling", "oracle_ReCAP")
            for seed in (0, 1, 2)
        ],
    }


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for item in rows for key in item})
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            writer.writerow(item)
    tmp.replace(path)


def generate(root: Path) -> None:
    manifest_dir = root / "manifests"
    for pilot, rows in base_manifests().items():
        write_manifest(manifest_dir / f"pilot_{pilot}.csv", rows)
    for seed in range(4):
        make_library_asset(manifest_dir / "libraries" / f"seed_{seed}.npz", seed)
    for seed in range(2):
        make_library_asset(manifest_dir / "libraries" / f"seed_{seed}_strong.npz", seed, STRONG_SKEW)


def resolve(root: Path) -> int:
    selection = read_json(root / "results" / "pilots" / "A" / "selected_width.json", {})
    width = int(selection.get("selected_width", 50))
    for pilot in ("P", "C", "D", "DR", "E"):
        rows = base_manifests()[pilot]
        for item in rows:
            item["width"] = width
            item["run_id"] = stable_run_id({key: value for key, value in item.items() if key != "run_id"})
        write_manifest(root / "manifests" / f"pilot_{pilot}.csv", rows)
    return width


def resolve_e(root: Path) -> tuple[int, float, int]:
    selection = read_json(root / "results" / "pilots" / "A" / "selected_width.json", {})
    d_decision = read_json(root / "results" / "pilots" / "D" / "D_decision.json", {})
    width = int(selection.get("selected_width", 50))
    fraction = float(d_decision.get("adopted_recycle_fraction", RECYCLE_FRACTION))
    sleep = int(d_decision.get("adopted_sleep_updates", SLEEP_UPDATES))
    rows = base_manifests()["E"]
    for item in rows:
        item["width"] = width
        item["recycle_fraction"] = fraction
        item["sleep_updates"] = sleep
        item["run_id"] = stable_run_id({key: value for key, value in item.items() if key != "run_id"})
    write_manifest(root / "manifests" / "pilot_E.csv", rows)
    return width, fraction, sleep


def read_manifest(path: Path, index: int) -> dict[str, Any]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if index < 0 or index >= len(rows):
        raise IndexError(f"manifest row {index} outside 0..{len(rows)-1}")
    result: dict[str, Any] = {}
    for key, value in rows[index].items():
        if key in {
            "seed",
            "width",
            "batch_size",
            "blocks",
            "updates_per_block",
            "total_updates",
            "buffer_size",
            "horizon",
            "sleep_updates",
        }:
            result[key] = int(value) if value != "selected" else value
        elif key in {"lr", "gamma", "recycle_fraction", "kd_beta", "kd_temperature"}:
            result[key] = float(value)
        elif key == "skew_probabilities":
            result[key] = json.loads(value)
        else:
            result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "resolve", "resolve-e", "count"))
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.root)
    elif args.command == "resolve":
        print(resolve(args.root))
    elif args.command == "resolve-e":
        print(*resolve_e(args.root))
    else:
        if not args.manifest:
            parser.error("count requires --manifest")
        with args.manifest.open() as handle:
            print(max(0, sum(1 for _ in handle) - 1))


if __name__ == "__main__":
    main()
