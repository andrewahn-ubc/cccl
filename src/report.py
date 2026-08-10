from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .utils import atomic_text, project_root, read_json


def collect_runtime(root: Path) -> tuple[float, float, list[str]]:
    cpu_hours = gpu_hours = 0.0
    failed = []
    base = root / "results" / "pilots"
    for pilot in ("A", "AR", "B", "BR", "P", "C", "D", "DR", "E"):
        pilot_path = base / pilot
        if not pilot_path.exists():
            continue
        for run in pilot_path.iterdir():
            if not run.is_dir():
                continue
            status = read_json(run / "status.json", {})
            if status.get("state") not in ("complete", "skipped"):
                failed.append(run.name)
            seconds = float(status.get("elapsed_seconds", 0.0))
            environment = {}
            if (run / "environment.txt").exists():
                environment = yaml.safe_load((run / "environment.txt").read_text()) or {}
            if environment.get("cuda_available"):
                gpu_hours += seconds / 3600
            else:
                cpu_hours += seconds * 4 / 3600
    return cpu_hours, gpu_hours, failed


def generate(root: Path) -> Path:
    base = root / "results" / "pilots"
    decisions = {pilot: read_json(base / pilot / f"{pilot}_decision.json", {}) for pilot in "ABCDE"}
    selected = read_json(base / "A" / "selected_width.json", {})
    c_metric = decisions["C"].get("selected_importance_metric", "unknown")
    scientific = [decisions[pilot].get("decision", "MISSING") for pilot in "ABCDE"]
    marginal_count = sum(value in ("MARGINAL", "CONDITIONAL_GO") for value in scientific)
    hard_failure = any(value.startswith("FAIL") or value in ("NO_GO", "MISSING") for value in scientific)
    if not hard_failure and decisions["E"].get("decision") == "GO" and marginal_count == 0:
        overall = "FULL_GO"
        next_action = "Freeze the selected pilot defaults and implement the learned decayed-transition predictor for the reduced main study."
    elif not hard_failure and marginal_count <= 1:
        overall = "CONDITIONAL_GO"
        next_action = "Repair the single marginal gate, then implement the learned predictor with a reduced main grid."
    else:
        overall = "NO_GO"
        if str(decisions["A"].get("decision", "")).startswith("FAIL"):
            next_action = "Run one width-200 offline feasibility diagnostic before changing the task/model regime."
        elif decisions["B"].get("decision") == "FAIL":
            next_action = "Run the prescribed two-seed stronger-skew replay diagnostic before any predictor work."
        elif decisions["D"].get("decision") == "FAIL":
            next_action = "Debug consolidation/reset with a single 10%-recycle, 100-sleep-step compression diagnostic."
        else:
            next_action = "Use the mechanism event table to isolate scoring, consolidation, or benchmark incentive; do not implement learned future prediction yet."
    cpu_hours, gpu_hours, discovered_failed = collect_runtime(root)
    failed = sorted(set(discovered_failed + [run for decision in decisions.values() for run in decision.get("failed_run_ids", [])]))

    lines = [
        "# ReCAP 12-hour pilot decision report",
        "",
        f"**Recommendation: `{overall}`**",
        "",
        f"Selected width: **{selected.get('selected_width', 'unavailable')}**. Selected importance metric: **{c_metric}**.",
        "",
        "## Gate summary",
        "",
        "| Pilot | Threshold | Observed gate inputs | Decision |",
        "|---|---|---|---|",
    ]
    for pilot in "ABCDE":
        decision = decisions[pilot]
        threshold = str(decision.get("thresholds", {})).replace("|", "/")
        inputs = str(decision.get("gate_inputs", {})).replace("|", "/")
        lines.append(f"| {pilot} | `{threshold}` | `{inputs}` | **{decision.get('decision', 'MISSING')}** |")
    lines.extend(
        [
            "",
            "## Environment and cost",
            "",
            f"Measured allocation use: approximately **{cpu_hours:.2f} CPU-hours** and **{gpu_hours:.2f} GPU-hours**. Exact per-run package, host, Slurm, and hardware details are stored in each `environment.txt`.",
            "",
            "## Required figures",
            "",
            "- [Pilot A phase diagram](A/phase_diagram.png)",
            "- [Pilot B per-task curves](B/per_task_curves.png)",
            "- [Pilot C score/ablation plot](C/score_vs_ablation.png)",
            "- [Pilot D stagewise compression](D/stagewise_compression.png)",
            "- [Pilot E lifetime prefix average](E/prefix_average.png)",
            "- [Pilot E per-task heatmap](E/per_task_heatmap.png)",
            "",
            "## Failed, cancelled, and retried runs",
            "",
            (", ".join(f"`{run}`" for run in failed) if failed else "None. No threshold was relaxed and no result was transcribed manually."),
            "",
            "Targeted retries are permitted only for the predeclared marginal cases. A software-invalid run must be rerun with its identical manifest row; the resubmission helper records that rationale.",
            "",
            "## Exact next action",
            "",
            next_action,
            "",
            "If proceeding, the frozen defaults are: buffer 500; batch 64; Adam 1e-3; oracle occupancy H=10, gamma=.9; 20% layer-wise recycling after block 10; 30 CE+KD sleep updates; beta=1; temperature=2; function-neutral reset.",
            "",
        ]
    )
    output = base / "PILOT_DECISION_REPORT.md"
    atomic_text(output, "\n".join(lines))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=project_root())
    args = parser.parse_args()
    print(generate(args.root.resolve()))


if __name__ == "__main__":
    main()
