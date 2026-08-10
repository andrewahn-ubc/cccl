from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .manifest import read_manifest
from .utils import atomic_json, project_root, read_json


def manifest_rows(root: Path, pilot: str) -> list[dict[str, Any]]:
    path = root / "manifests" / f"pilot_{pilot}.csv"
    with path.open() as handle:
        count = sum(1 for _ in handle) - 1
    return [read_manifest(path, index) for index in range(count)]


def load_runs(root: Path, pilot: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    frames, used, failed = [], [], []
    for config in manifest_rows(root, pilot):
        run_id = str(config["run_id"])
        run_path = root / "results" / "pilots" / pilot / run_id
        status = read_json(run_path / "status.json", {})
        if status.get("state") != "complete":
            failed.append(run_id)
            continue
        frame = pd.read_parquet(run_path / "metrics.parquet")
        for key in ("method", "seed", "width", "run_id"):
            frame[key] = config[key]
        frames.append(frame)
        used.append(run_id)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), used, failed


def load_events(root: Path, pilot: str) -> list[dict[str, Any]]:
    events = []
    for config in manifest_rows(root, pilot):
        path = root / "results" / "pilots" / pilot / str(config["run_id"]) / "events.jsonl"
        if not path.exists():
            continue
        with path.open() as handle:
            for line in handle:
                event = json.loads(line)
                event.update({"run_id": config["run_id"], "method": config["method"], "seed": config["seed"], "width": config["width"]})
                events.append(event)
    return events


def save_decision(path: Path, decision: str, inputs: dict[str, Any], thresholds: dict[str, Any], run_ids: list[str], **extra: Any) -> None:
    atomic_json(
        path,
        {
            "decision": decision,
            "gate_inputs": inputs,
            "thresholds": thresholds,
            "run_ids": run_ids,
            **extra,
        },
    )


def aggregate_a(root: Path, finalize: bool = False) -> None:
    frame, used, failed = load_runs(root, "A")
    output = root / "results" / "pilots" / "A"
    records = []
    for (method, seed, width, run_id), group in frame.groupby(["method", "seed", "width", "run_id"]):
        evaluations = group[group.metric_kind == "mixture_eval"]
        pre = group[group.metric_kind == "preupdate"]
        records.append(
            {
                "method": method,
                "seed": int(seed),
                "width": int(width),
                "run_id": run_id,
                "final_mixture_accuracy": float(evaluations.sort_values("step").iloc[-1].accuracy),
                "lifetime_preupdate_accuracy": float(pre.accuracy.mean()),
                "updates": len(pre),
            }
        )
    summary = pd.DataFrame(records)
    summary.to_csv(output / "phase_diagram.csv", index=False)
    paired = summary.pivot_table(index=["width", "seed"], columns="method", values="final_mixture_accuracy").reset_index()
    by_width = paired.groupby("width").mean(numeric_only=True)
    by_width["gap"] = by_width["offline_mixture"] - by_width["ER"]
    pass_widths = by_width[
        (by_width.offline_mixture >= 0.88) & (by_width.gap >= 0.08) & (by_width.ER >= 0.55)
    ]
    marginal_widths = by_width[(by_width.offline_mixture >= 0.85) & (by_width.gap >= 0.05)]
    selected = None
    if len(pass_widths):
        decision = "PASS"
        selected = int(pass_widths.index.min())
    elif len(marginal_widths):
        decision = "MARGINAL"
        selected = int(marginal_widths.sort_values(["width", "gap"], ascending=[True, False]).index[0])
    elif (by_width.offline_mixture < 0.85).all():
        decision = "FAIL_INFEASIBLE"
    else:
        decision = "FAIL_OVERCAPACITY"
    inputs = {str(int(width)): {key: float(value) for key, value in row.items()} for width, row in by_width.iterrows()}
    if finalize and decision == "FAIL_INFEASIBLE":
        retry, retry_used, retry_failed = load_runs(root, "AR")
        used.extend(retry_used)
        failed.extend(retry_failed)
        if not retry.empty:
            retry_eval = retry[retry.metric_kind == "mixture_eval"].sort_values("step")
            retry_accuracy = float(retry_eval.iloc[-1].accuracy)
            inputs["width_200_diagnostic"] = {"offline_mixture": retry_accuracy}
            decision = "FAIL_INFEASIBLE" if retry_accuracy < 0.85 else "FAIL_GRID_TOO_SMALL"
    thresholds = {"pass": {"offline": 0.88, "offline_minus_er": 0.08, "er": 0.55}, "marginal": {"offline": 0.85, "gap": 0.05}}
    save_decision(output / "A_decision.json", decision, inputs, thresholds, used, failed_run_ids=failed)
    atomic_json(
        output / "selected_width.json",
        {"selected_width": selected if selected is not None else 50, "fallback": selected is None, "A_decision": decision},
    )
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for method, group in summary.groupby("method"):
        values = group.groupby("width").final_mixture_accuracy.agg(["mean", "sem"])
        ax.errorbar(values.index, values["mean"], yerr=values["sem"].fillna(0), marker="o", label=method)
    ax.axhline(0.88, color="grey", linestyle="--", linewidth=1)
    ax.set(xlabel="Hidden width", ylabel="Final mixture accuracy", title="Pilot A: capacity phase diagram")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "phase_diagram.png", dpi=180)
    plt.close(fig)
    learning = frame[frame.metric_kind == "mixture_eval"].groupby(["method", "width", "step"], as_index=False).accuracy.mean()
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for width, ax in zip(sorted(learning.width.unique()), axes):
        for method, group in learning[learning.width == width].groupby("method"):
            ax.plot(group.step, group.accuracy, label=method)
        ax.set_title(f"width {width}")
        ax.set_xlabel("Update")
    axes[0].set_ylabel("Mixture accuracy")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=180)
    plt.close(fig)


def aggregate_b(root: Path, finalize: bool = False) -> None:
    frame, used, failed = load_runs(root, "B")
    output = root / "results" / "pilots" / "B"
    pre = frame[frame.metric_kind == "preupdate"]
    summary = pre.groupby(["method", "seed", "run_id"], as_index=False).accuracy.mean()
    pivot = summary.pivot(index="seed", columns="method", values="accuracy")
    pivot["oracle_minus_uniform"] = pivot["oracle_frequency_weighted"] - pivot["uniform_task"]
    pivot.reset_index().to_csv(output / "replay_policy_pairs.csv", index=False)
    mean_difference = float(pivot.oracle_minus_uniform.mean())
    positive = int((pivot.oracle_minus_uniform > 0).sum())
    decision = "PASS" if mean_difference >= 0.02 and positive >= 3 else "MARGINAL"
    gate_inputs: dict[str, Any] = {
        "base_mean_paired_difference": mean_difference,
        "base_positive_pairs": positive,
        "base_pairs": pivot.oracle_minus_uniform.to_dict(),
    }
    if finalize and decision != "PASS":
        retry, retry_used, retry_failed = load_runs(root, "BR")
        used.extend(retry_used)
        failed.extend(retry_failed)
        retry_pre = retry[retry.metric_kind == "preupdate"] if not retry.empty else pd.DataFrame()
        if len(retry_pre):
            retry_summary = retry_pre.groupby(["method", "seed"], as_index=False).accuracy.mean()
            retry_pivot = retry_summary.pivot(index="seed", columns="method", values="accuracy")
            retry_pivot["oracle_minus_uniform"] = (
                retry_pivot["oracle_frequency_weighted"] - retry_pivot["uniform_task"]
            )
            retry_pivot.reset_index().to_csv(output / "stronger_skew_pairs.csv", index=False)
            retry_mean = float(retry_pivot.oracle_minus_uniform.mean())
            retry_positive = int((retry_pivot.oracle_minus_uniform > 0).sum())
            gate_inputs.update(
                {
                    "strong_skew_mean_paired_difference": retry_mean,
                    "strong_skew_positive_pairs": retry_positive,
                    "strong_skew_pairs": retry_pivot.oracle_minus_uniform.to_dict(),
                }
            )
            decision = "PASS" if retry_mean >= 0.02 and retry_positive == 2 else "MARGINAL" if retry_mean >= 0.01 else "FAIL"
        else:
            decision = "FAIL"
            gate_inputs["strong_skew_error"] = "triggered retry produced no complete rows"
    save_decision(
        output / "B_decision.json",
        decision,
        gate_inputs,
        {"pass_mean_difference": 0.02, "pass_positive_pairs": 3, "marginal_mean_difference": 0.01},
        used,
        failed_run_ids=failed,
    )
    curves = pre.groupby(["method", "task", "block"], as_index=False).accuracy.mean()
    fig, axes = plt.subplots(2, 5, figsize=(12, 5), sharex=True, sharey=True)
    for task, ax in enumerate(axes.flat):
        for method, group in curves[curves.task == task].groupby("method"):
            ax.plot(group.block, group.accuracy, label=method)
        ax.set_title(f"task {task}")
    axes.flat[0].legend(frameon=False, fontsize=7)
    fig.supxlabel("Block")
    fig.supylabel("Pre-update accuracy")
    fig.tight_layout()
    fig.savefig(output / "per_task_curves.png", dpi=180)
    plt.close(fig)
    allocation_rows = []
    for event in load_events(root, "B"):
        if event.get("event") != "buffer_allocation":
            continue
        for task, count in event["allocation"].items():
            allocation_rows.append(
                {
                    "run_id": event["run_id"],
                    "method": event["method"],
                    "seed": event["seed"],
                    "block": event["block"],
                    "task": int(task),
                    "examples": int(count),
                }
            )
    pd.DataFrame(allocation_rows).to_csv(output / "buffer_allocations.csv", index=False)


def bottom_damage_ratio(group: pd.DataFrame, metric: str) -> float:
    count = max(1, int(np.floor(len(group) * 0.2)))
    damage = group.ablation_damage.clip(lower=0)
    selected = damage.loc[group[metric].nsmallest(count).index].mean()
    random_expected = damage.mean()
    if random_expected <= 1e-12:
        return 1.0 if selected <= 1e-12 else float("inf")
    return float(selected / random_expected)


def aggregate_c(root: Path) -> None:
    frame, used, failed = load_runs(root, "C")
    output = root / "results" / "pilots" / "C"
    if frame.empty:
        save_decision(
            output / "C_decision.json",
            "FAIL",
            {"reason": "Pilot C was skipped because an upstream scientific gate failed"},
            {"spearman": 0.40, "positive_both_seeds": True, "bottom_damage_ratio": 0.70},
            used,
            selected_importance_metric="empirical",
            failed_run_ids=failed,
        )
        pd.DataFrame().to_csv(output / "score_summary.csv", index=False)
        return
    metrics = ("taylor", "fisher", "activation")
    rows = []
    for metric in metrics:
        pooled = float(spearmanr(frame[metric], frame.ablation_damage, nan_policy="omit").statistic)
        per_seed = {
            str(int(seed)): float(spearmanr(group[metric], group.ablation_damage, nan_policy="omit").statistic)
            for seed, group in frame.groupby("seed")
        }
        ratio = bottom_damage_ratio(frame, metric)
        rows.append({"metric": metric, "pooled_spearman": pooled, "bottom_quintile_damage_ratio": ratio, **{f"seed_{k}_rho": v for k, v in per_seed.items()}})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "score_summary.csv", index=False)
    correlation_rows = []
    for (layer, task), group in frame.groupby(["layer", "task"]):
        for metric in metrics:
            correlation_rows.append(
                {
                    "layer": int(layer),
                    "task": int(task),
                    "metric": metric,
                    "spearman": float(spearmanr(group[metric], group.ablation_damage, nan_policy="omit").statistic),
                    "n": len(group),
                }
            )
    pd.DataFrame(correlation_rows).to_csv(output / "per_layer_task_correlations.csv", index=False)
    taylor = summary[summary.metric == "taylor"].iloc[0]
    seed_columns = [column for column in summary if column.startswith("seed_")]
    taylor_positive = all(float(taylor[column]) > 0 for column in seed_columns)
    if taylor.pooled_spearman >= 0.40 and taylor_positive and taylor.bottom_quintile_damage_ratio <= 0.70:
        decision, selected = "PASS", "taylor"
    else:
        best = summary.sort_values("pooled_spearman", ascending=False).iloc[0]
        best_positive = all(float(best[column]) > 0 for column in seed_columns)
        if best.pooled_spearman >= 0.40 and best_positive and best.bottom_quintile_damage_ratio <= 0.70:
            decision, selected = "PASS", str(best.metric)
        elif max(summary.pooled_spearman) >= 0.20:
            decision, selected = "MARGINAL", str(best.metric)
        else:
            decision, selected = "FAIL", "empirical"
    inputs = {row["metric"]: {k: v for k, v in row.items() if k != "metric"} for row in rows}
    save_decision(
        output / "C_decision.json",
        decision,
        inputs,
        {"spearman": 0.40, "positive_both_seeds": True, "bottom_damage_ratio": 0.70, "marginal_spearman": 0.20},
        used,
        selected_importance_metric=selected,
        failed_run_ids=failed,
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(frame.taylor, frame.ablation_damage, alpha=0.35, s=14)
    ax.set(xlabel="Taylor gate score", ylabel="Empirical deletion damage (NLL)", title="Pilot C: score validity")
    fig.tight_layout()
    fig.savefig(output / "score_vs_ablation.png", dpi=180)
    plt.close(fig)
    frame.to_parquet(output / "score_vs_ablation.parquet", index=False)


def aggregate_d(root: Path, finalize: bool = False) -> None:
    frame, used, failed = load_runs(root, "D")
    output = root / "results" / "pilots" / "D"
    if frame.empty:
        save_decision(
            output / "D_decision.json",
            "FAIL",
            {"reason": "Pilot D was skipped because an upstream scientific gate failed"},
            {"recovery_ratio": 0.60, "max_accuracy_drop": 0.03, "importance_minus_random": 0.01},
            used,
            failed_run_ids=failed,
        )
        pd.DataFrame().to_csv(output / "recovery_table.csv", index=False)
        return
    frame.to_csv(output / "recovery_table.csv", index=False)
    importance = frame[frame.method == "importance_consolidate"]
    before = importance[importance.stage == "before"].set_index("seed")
    final = importance[importance.stage == "after_reset"].set_index("seed")
    random_final = frame[(frame.method == "random_consolidate") & (frame.stage == "after_reset")].set_index("seed")
    median_recovery = float(importance.drop_duplicates("seed").recovery_ratio.median())
    final_drop = float((before.accuracy - final.accuracy).mean())
    importance_over_random = float((final.accuracy - random_final.accuracy).mean())
    max_drift = float(frame.reset_logit_max_abs.max())
    passed = median_recovery >= 0.60 and final_drop <= 0.03 and importance_over_random >= 0.01 and max_drift <= 1e-5
    marginal = (0.30 <= median_recovery < 0.60) or (0.03 < final_drop <= 0.06)
    decision = "PASS" if passed else "MARGINAL" if marginal and max_drift <= 1e-5 else "FAIL"
    adopted_fraction, adopted_sleep = 0.20, 30
    if finalize and decision == "MARGINAL":
        retry, retry_used, retry_failed = load_runs(root, "DR")
        used.extend(retry_used)
        failed.extend(retry_failed)
        retry.to_csv(output / "retry_recovery_table.csv", index=False)
        if not retry.empty:
            retry_importance = retry[retry.method == "importance_consolidate"]
            retry_before = retry_importance[retry_importance.stage == "before"].set_index("seed")
            retry_final = retry_importance[retry_importance.stage == "after_reset"].set_index("seed")
            retry_random = retry[
                (retry.method == "random_consolidate") & (retry.stage == "after_reset")
            ].set_index("seed")
            median_recovery = float(retry_importance.drop_duplicates("seed").recovery_ratio.median())
            final_drop = float((retry_before.accuracy - retry_final.accuracy).mean())
            importance_over_random = float((retry_final.accuracy - retry_random.accuracy).mean())
            max_drift = float(retry.reset_logit_max_abs.max())
            retry_passed = (
                median_recovery >= 0.60
                and final_drop <= 0.03
                and importance_over_random >= 0.01
                and max_drift <= 1e-5
            )
            retry_marginal = (0.30 <= median_recovery < 0.60) or (0.03 < final_drop <= 0.06)
            decision = "PASS" if retry_passed else "MARGINAL" if retry_marginal and max_drift <= 1e-5 else "FAIL"
            if decision in ("PASS", "MARGINAL"):
                adopted_fraction, adopted_sleep = 0.10, 100
        else:
            decision = "FAIL"
    save_decision(
        output / "D_decision.json",
        decision,
        {
            "median_recovery_ratio": median_recovery,
            "mean_final_accuracy_drop": final_drop,
            "importance_minus_random_final_accuracy": importance_over_random,
            "max_reset_logit_drift": max_drift,
        },
        {"recovery_ratio": 0.60, "max_accuracy_drop": 0.03, "importance_minus_random": 0.01, "max_logit_drift": 1e-5},
        used,
        adopted_recycle_fraction=adopted_fraction,
        adopted_sleep_updates=adopted_sleep,
        failed_run_ids=failed,
    )
    atomic_json(
        output / "logit_invariance.json",
        {"max_abs_logit_drift": max_drift, "tolerance": 1e-5, "pass": max_drift <= 1e-5, "run_ids": used},
    )
    order = ["before", "after_mask", "after_sleep", "after_reset"]
    plot = frame.groupby(["method", "stage"], as_index=False).accuracy.mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, group in plot.groupby("method"):
        values = group.set_index("stage").reindex(order)
        ax.plot(order, values.accuracy, marker="o", label=method)
    ax.set(ylabel="Weighted probe accuracy", title="Pilot D: compression and recovery")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "stagewise_compression.png", dpi=180)
    plt.close(fig)


def aggregate_e(root: Path) -> None:
    frame, used, failed = load_runs(root, "E")
    output = root / "results" / "pilots" / "E"
    if frame.empty or not (frame.metric_kind == "preupdate").any():
        save_decision(
            output / "E_decision.json",
            "NO_GO",
            {"reason": "Pilot E was skipped because an upstream scientific gate failed"},
            {"oracle_minus_er": 0.03, "oracle_minus_random": 0.02, "oracle_minus_compute_matched": 0.02},
            used,
            failed_run_ids=failed,
        )
        pd.DataFrame().to_csv(output / "oracle_pairs.csv", index=False)
        return
    pre = frame[frame.metric_kind == "preupdate"].copy()
    summary = pre.groupby(["method", "seed", "run_id"], as_index=False).accuracy.mean()
    pivot = summary.pivot(index="seed", columns="method", values="accuracy")
    comparisons = {
        "ER": "oracle_minus_er",
        "random_recycling": "oracle_minus_random",
        "ER_compute_matched": "oracle_minus_compute_matched",
    }
    for baseline, label in comparisons.items():
        pivot[label] = pivot.oracle_ReCAP - pivot[baseline]
    pivot.reset_index().to_csv(output / "oracle_pairs.csv", index=False)
    dominant = frame[
        (frame.method == "oracle_ReCAP")
        & (frame.metric_kind.isin(["recycle_task_before", "recycle_task_after_reset"]))
        & (frame.task.isin([0, 1]))
    ]
    if len(dominant):
        collapse_table = dominant.pivot_table(index=["seed", "block", "task"], columns="metric_kind", values="accuracy")
        max_collapse = float((collapse_table.recycle_task_before - collapse_table.recycle_task_after_reset).max())
    else:
        max_collapse = 0.0
    means = {label: float(pivot[label].mean()) for label in comparisons.values()}
    positives = {label: bool((pivot[label] > 0).all()) for label in comparisons.values()}
    go = means["oracle_minus_er"] >= 0.03 and means["oracle_minus_random"] >= 0.02 and means["oracle_minus_compute_matched"] >= 0.02 and all(positives.values()) and max_collapse <= 0.10
    consistent_small_win = min(means.values()) >= 0.01 and all(positives.values())
    d_decision = read_json(root / "results" / "pilots" / "D" / "D_decision.json", {}).get("decision")
    decision = "GO" if go else "CONDITIONAL_GO" if consistent_small_win and d_decision == "PASS" else "NO_GO"
    save_decision(
        output / "E_decision.json",
        decision,
        {"mean_paired_differences": means, "positive_all_seeds": positives, "max_dominant_task_collapse": max_collapse},
        {"oracle_minus_er": 0.03, "oracle_minus_random": 0.02, "oracle_minus_compute_matched": 0.02, "max_dominant_collapse": 0.10},
        used,
        failed_run_ids=failed,
    )
    pre["prefix_accuracy"] = pre.sort_values("step").groupby(["method", "seed"]).accuracy.expanding().mean().reset_index(level=[0, 1], drop=True)
    curves = pre.groupby(["method", "step"], as_index=False).prefix_accuracy.mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, group in curves.groupby("method"):
        ax.plot(group.step, group.prefix_accuracy, label=method)
    ax.set(xlabel="Online update", ylabel="Lifetime prefix accuracy", title="Pilot E: oracle kill test")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "prefix_average.png", dpi=180)
    plt.close(fig)
    heat = pre[pre.method == "oracle_ReCAP"].pivot_table(index="task", columns="block", values="accuracy", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    image = ax.imshow(heat, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set(xlabel="Block", ylabel="Task", title="Oracle ReCAP pre-update accuracy")
    fig.colorbar(image, ax=ax, label="Accuracy")
    fig.tight_layout()
    fig.savefig(output / "per_task_heatmap.png", dpi=180)
    plt.close(fig)
    mechanism_rows = []
    selection_rows = []
    for event in load_events(root, "E"):
        if event.get("event") != "recycle":
            continue
        before, masked, slept, reset = event["before"], event["after_mask"], event["after_sleep"], event["after_reset"]
        mechanism_rows.append(
            {
                "run_id": event["run_id"],
                "method": event["method"],
                "seed": event["seed"],
                "block": event["block"],
                "importance_metric": event["metric"],
                "before_accuracy": before["accuracy"],
                "after_mask_accuracy": masked["accuracy"],
                "after_sleep_accuracy": slept["accuracy"],
                "after_reset_accuracy": reset["accuracy"],
                "mask_nll_damage": masked["nll"] - before["nll"],
                "sleep_nll_recovery": masked["nll"] - slept["nll"],
                "reset_logit_max_abs": event["reset_logit_max_abs"],
                "selected_units": json.dumps(event["selected"], sort_keys=True),
                "selected_unit_ages": json.dumps(event.get("selected_unit_ages", {}), sort_keys=True),
            }
        )
        if event["method"] == "oracle_ReCAP":
            for layer, units in event["selected"].items():
                for unit in units:
                    selection_rows.append(
                        {"block": int(event["block"]), "layer_unit": f"L{layer}:U{unit}", "selected": 1}
                    )
    pd.DataFrame(mechanism_rows).to_csv(output / "mechanism_events.csv", index=False)
    if selection_rows:
        selection = pd.DataFrame(selection_rows).pivot_table(
            index="layer_unit", columns="block", values="selected", aggfunc="sum", fill_value=0
        )
        fig, ax = plt.subplots(figsize=(9, max(4, len(selection) * 0.08)))
        image = ax.imshow(selection, aspect="auto", cmap="magma")
        ax.set(xlabel="Recycle boundary", ylabel="Layer/unit", title="Oracle ReCAP selection frequency")
        ax.set_xticks(range(len(selection.columns)), labels=selection.columns)
        step = max(1, len(selection) // 20)
        ax.set_yticks(range(0, len(selection), step), labels=selection.index[::step], fontsize=6)
        fig.colorbar(image, ax=ax, label="Selections across seeds")
        fig.tight_layout()
        fig.savefig(output / "capacity_selection_heatmap.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", choices=("A", "B", "C", "D", "E"), required=True)
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    aggregate = globals()[f"aggregate_{args.pilot.lower()}"]
    if args.pilot in ("A", "B", "D"):
        aggregate(args.root.resolve(), finalize=args.finalize)
    else:
        aggregate(args.root.resolve())


if __name__ == "__main__":
    main()
