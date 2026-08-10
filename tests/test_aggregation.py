from pathlib import Path

import pandas as pd

from src.aggregate import (
    aggregate_a,
    aggregate_b,
    aggregate_c,
    aggregate_d,
    aggregate_e,
)
from src.manifest import generate, read_manifest, resolve
from src.report import generate as generate_report
from src.utils import atomic_json, read_json


def write_run(root: Path, pilot: str, index: int, rows: list[dict]):
    config = read_manifest(root / "manifests" / f"pilot_{pilot}.csv", index)
    path = root / "results" / "pilots" / pilot / config["run_id"]
    path.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(path / "metrics.parquet", index=False)
    atomic_json(path / "status.json", {"state": "complete", "elapsed_seconds": 1})


def test_all_gate_aggregators_and_report(tmp_path: Path):
    generate(tmp_path)
    for index in range(24):
        config = read_manifest(tmp_path / "manifests" / "pilot_A.csv", index)
        accuracy = {"offline_mixture": 0.91, "ER": 0.75, "periodic_full_reset": 0.65}[config["method"]]
        write_run(
            tmp_path,
            "A",
            index,
            [
                {"metric_kind": "preupdate", "step": 0, "accuracy": accuracy - 0.01},
                {"metric_kind": "mixture_eval", "step": 300, "accuracy": accuracy},
            ],
        )
    aggregate_a(tmp_path)
    assert resolve(tmp_path) == 10

    for index in range(8):
        config = read_manifest(tmp_path / "manifests" / "pilot_B.csv", index)
        accuracy = 0.63 if config["method"] == "oracle_frequency_weighted" else 0.60
        write_run(
            tmp_path,
            "B",
            index,
            [{"metric_kind": "preupdate", "step": 0, "block": 0, "task": 0, "accuracy": accuracy}],
        )
    aggregate_b(tmp_path)

    for index in range(2):
        config = read_manifest(tmp_path / "manifests" / "pilot_C.csv", index)
        rows = []
        for unit in range(25):
            damage = unit / 1000
            rows.append(
                {
                    "metric_kind": "unit_ablation",
                    "seed": config["seed"],
                    "layer": 0,
                    "unit": unit,
                    "task": 0,
                    "taylor": damage,
                    "fisher": damage,
                    "activation": damage,
                    "ablation_damage": damage,
                }
            )
        write_run(tmp_path, "C", index, rows)
    aggregate_c(tmp_path)

    for index in range(12):
        config = read_manifest(tmp_path / "manifests" / "pilot_D.csv", index)
        method = config["method"]
        final_accuracy = 0.89 if method == "importance_consolidate" else 0.87
        stages = {
            "before": (0.90, 1.0),
            "after_mask": (0.80, 2.0),
            "after_sleep": (final_accuracy, 1.2),
            "after_reset": (final_accuracy, 1.2),
        }
        write_run(
            tmp_path,
            "D",
            index,
            [
                {
                    "metric_kind": "compression_stage",
                    "stage": stage,
                    "accuracy": values[0],
                    "nll": values[1],
                    "recovery_ratio": 0.8,
                    "reset_logit_max_abs": 0.0,
                }
                for stage, values in stages.items()
            ],
        )
    aggregate_d(tmp_path)

    accuracies = {"ER": 0.75, "ER_compute_matched": 0.77, "random_recycling": 0.77, "oracle_ReCAP": 0.80}
    for index in range(12):
        config = read_manifest(tmp_path / "manifests" / "pilot_E.csv", index)
        rows = [
            {"metric_kind": "preupdate", "step": step, "block": 0, "task": step % 2, "accuracy": accuracies[config["method"]]}
            for step in range(3)
        ]
        if config["method"] == "oracle_ReCAP":
            for task in (0, 1):
                rows.extend(
                    [
                        {"metric_kind": "recycle_task_before", "step": 3, "block": 10, "task": task, "accuracy": 0.90},
                        {"metric_kind": "recycle_task_after_reset", "step": 3, "block": 10, "task": task, "accuracy": 0.85},
                    ]
                )
        write_run(tmp_path, "E", index, rows)
    aggregate_e(tmp_path)
    report = generate_report(tmp_path)
    assert report.exists()
    assert "FULL_GO" in report.read_text()


def test_stronger_skew_retry_can_resolve_pilot_b(tmp_path: Path):
    generate(tmp_path)
    for index in range(8):
        config = read_manifest(tmp_path / "manifests" / "pilot_B.csv", index)
        accuracy = 0.605 if config["method"] == "oracle_frequency_weighted" else 0.60
        write_run(
            tmp_path,
            "B",
            index,
            [{"metric_kind": "preupdate", "step": 0, "block": 0, "task": 0, "accuracy": accuracy}],
        )
    aggregate_b(tmp_path)
    assert read_json(tmp_path / "results" / "pilots" / "B" / "B_decision.json")["decision"] == "MARGINAL"
    for index in range(4):
        config = read_manifest(tmp_path / "manifests" / "pilot_BR.csv", index)
        accuracy = 0.63 if config["method"] == "oracle_frequency_weighted" else 0.60
        write_run(
            tmp_path,
            "BR",
            index,
            [{"metric_kind": "preupdate", "step": 0, "block": 0, "task": 0, "accuracy": accuracy}],
        )
    aggregate_b(tmp_path, finalize=True)
    decision = read_json(tmp_path / "results" / "pilots" / "B" / "B_decision.json")
    assert decision["decision"] == "PASS"
    assert decision["gate_inputs"]["strong_skew_positive_pairs"] == 2
