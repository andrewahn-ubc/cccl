from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .constants import BATCH_SIZE, SKEW
from .data import PermutedMNIST
from .methods import evaluate, recycle
from .models import GatedMLP, make_optimizer
from .replay import ReplayBuffer, equal_quotas, largest_remainder_quotas
from .schedules import discounted_occupancy, oracle_transition
from .utils import RunArtifacts, SignalState, atomic_torch


def preupdate(model: GatedMLP, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        return float((logits.argmax(1) == y).float().mean().item()), float(F.cross_entropy(logits, y).item())


def online_update(
    model: GatedMLP,
    optimizer: torch.optim.Optimizer,
    current_x: torch.Tensor,
    current_y: torch.Tensor,
    replay: ReplayBuffer,
    device: torch.device,
    replay_weights: Mapping[int, float] | None = None,
    batch_size: int = BATCH_SIZE,
) -> float:
    replay_size = batch_size // 2 if len(replay) else 0
    sampled = replay.sample(replay_size, device, dict(replay_weights) if replay_weights else None)
    current_size = batch_size - replay_size
    x, y = current_x[:current_size], current_y[:current_size]
    if sampled is not None:
        replay_x, replay_y, _ = sampled
        x, y = torch.cat((x, replay_x)), torch.cat((y, replay_y))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def save_progress(
    path: Path,
    next_block: int,
    model: GatedMLP,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    generator: torch.Generator,
    metrics: list[dict[str, Any]],
    extra_state: dict[str, Any] | None = None,
) -> None:
    atomic_torch(
        path,
        {
            "next_block": next_block,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "replay": replay.state_dict(),
            "data_generator": generator.get_state(),
            "torch_rng": torch.random.get_rng_state(),
            "metrics": metrics,
            "extra_state": extra_state or {},
        },
    )


def load_progress(
    path: Path,
    model: GatedMLP,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    replay.load_state_dict(state["replay"])
    generator.set_state(state["data_generator"])
    torch.random.set_rng_state(state["torch_rng"])
    return int(state["next_block"]), list(state["metrics"]), dict(state.get("extra_state", {}))


def train_stream(
    config: dict[str, Any],
    data: PermutedMNIST,
    artifacts: RunArtifacts,
    device: torch.device,
    stop_after_blocks: int | None = None,
    importance_metric: str = "taylor",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed, width, method = int(config["seed"]), int(config["width"]), str(config["method"])
    probabilities = tuple(float(value) for value in config.get("skew_probabilities", SKEW))
    torch.manual_seed(seed + 30_000)
    model = GatedMLP(width).to(device)
    optimizer = make_optimizer(model, float(config["lr"]))
    replay = ReplayBuffer(int(config["buffer_size"]), seed)
    data_generator = torch.Generator().manual_seed(seed + 50_000)
    reset_generator = torch.Generator(device=device).manual_seed(seed + 60_000)
    event_rng = np.random.default_rng(seed + 70_000)
    metrics: list[dict[str, Any]] = []
    unit_ages = {layer: np.zeros(width, dtype=np.int64) for layer in range(2)}
    start_block = 0
    progress = artifacts.path / "training_state.pt"
    if progress.exists():
        start_block, metrics, extra_state = load_progress(progress, model, optimizer, replay, data_generator, device)
        if "unit_ages" in extra_state:
            unit_ages = {int(layer): np.asarray(values, dtype=np.int64) for layer, values in extra_state["unit_ages"].items()}
        artifacts.event("run_resumed", next_block=start_block)

    schedule = data.schedule
    max_blocks = stop_after_blocks or int(config["blocks"])
    probes = data.probes()
    eval_x, eval_y, _ = data.evaluation_set()
    observed: set[int] = set(replay.observed)
    global_step = start_block * int(config["updates_per_block"])
    high_loss_evaluations = 0
    with SignalState() as signals:
        for block in range(start_block, max_blocks):
            task = int(schedule[block])
            observed.add(task)
            if method == "oracle_frequency_weighted":
                quotas = largest_remainder_quotas(observed, int(config["buffer_size"]), probabilities)
                replay_weights = {t: probabilities[t] for t in observed}
            else:
                # The common protocol reserves exactly 50 slots per task, including
                # tasks not yet observed; this deliberately leaves capacity unused
                # during the forced-introduction prefix.
                quotas = equal_quotas(range(len(SKEW)), int(config["buffer_size"]))
                replay_weights = None
            replay.set_quotas(quotas)
            artifacts.event("block_started", block=block, task=task, quotas=quotas)

            for update in range(int(config["updates_per_block"])):
                signals.check()
                current_x, current_y = data.batch(task, int(config["batch_size"]), data_generator)
                accuracy, nll = preupdate(model, current_x, current_y)
                training_loss = online_update(
                    model,
                    optimizer,
                    current_x,
                    current_y,
                    replay,
                    device,
                    replay_weights=replay_weights,
                    batch_size=int(config["batch_size"]),
                )
                replay.add(task, current_x, current_y)
                metrics.append(
                    {
                        "metric_kind": "preupdate",
                        "step": global_step,
                        "block": block,
                        "update": update,
                        "task": task,
                        "accuracy": accuracy,
                        "nll": nll,
                        "training_loss": training_loss,
                    }
                )
                global_step += 1
                if global_step % 300 == 0:
                    mixture = evaluate(model, eval_x, eval_y)
                    metrics.append(
                        {
                            "metric_kind": "mixture_eval",
                            "step": global_step,
                            "block": block,
                            "update": update,
                            "task": -1,
                            "accuracy": mixture["accuracy"],
                            "nll": mixture["nll"],
                            "training_loss": training_loss,
                        }
                    )
                    if not np.isfinite(mixture["nll"]) or mixture["nll"] > 20:
                        high_loss_evaluations += 1
                    else:
                        high_loss_evaluations = 0
                    if not np.isfinite(mixture["nll"]) or high_loss_evaluations >= 3:
                        raise FloatingPointError(f"diverged at step {global_step}: nll={mixture['nll']}")

            artifacts.event("buffer_allocation", block=block, task=task, allocation=replay.allocation())
            for ages in unit_ages.values():
                ages += 1

            boundary_before_final = block < max_blocks - 1
            if method == "periodic_full_reset" and (block + 1) % 5 == 0 and boundary_before_final:
                torch.manual_seed(seed + 80_000 + block)
                model = GatedMLP(width).to(device)
                optimizer = make_optimizer(model, float(config["lr"]))
                artifacts.event("full_reset", block=block)

            should_extra = method in ("ER_compute_matched", "random_recycling", "oracle_ReCAP")
            recycle_boundary = block >= 9 and boundary_before_final
            if should_extra and recycle_boundary:
                if method == "ER_compute_matched":
                    teacher = copy.deepcopy(model)
                    from .methods import sleep_consolidate

                    losses = sleep_consolidate(
                        model,
                        teacher,
                        optimizer,
                        replay,
                        {},
                        device,
                        {task_index: probabilities[task_index] for task_index in observed},
                        int(config["sleep_updates"]),
                        beta=0.0,
                    )
                    artifacts.event("compute_matched_replay", block=block, losses=losses)
                else:
                    selection_metric = "random" if method == "random_recycling" else importance_metric
                    oracle_weights = discounted_occupancy(
                        oracle_transition(probabilities),
                        task,
                        int(config["horizon"]),
                        float(config["gamma"]),
                    )
                    outcome = recycle(
                        model,
                        optimizer,
                        replay,
                        {key: probes[key] for key in sorted(observed)},
                        {task_index: float(oracle_weights[task_index]) for task_index in observed},
                        selection_metric,
                        float(config["recycle_fraction"]),
                        int(config["sleep_updates"]),
                        float(config["kd_beta"]),
                        device,
                        event_rng,
                        reset_generator,
                    )
                    selected_ages = {
                        str(layer): [int(unit_ages[layer][unit]) for unit in units]
                        for layer, units in outcome["selected"].items()
                    }
                    artifacts.event(
                        "recycle",
                        block=block,
                        metric=selection_metric,
                        selected_unit_ages=selected_ages,
                        **outcome,
                    )
                    for layer, units in outcome["selected"].items():
                        unit_ages[layer][units] = 0
                    for stage in ("before", "after_mask", "after_sleep", "after_reset"):
                        metrics.append(
                            {
                                "metric_kind": f"recycle_{stage}",
                                "step": global_step,
                                "block": block,
                                "update": -1,
                                "task": -1,
                                "accuracy": outcome[stage]["accuracy"],
                                "nll": outcome[stage]["nll"],
                                "training_loss": np.nan,
                            }
                        )
                        for probe_task, measure in outcome["per_task"][stage].items():
                            metrics.append(
                                {
                                    "metric_kind": f"recycle_task_{stage}",
                                    "step": global_step,
                                    "block": block,
                                    "update": -1,
                                    "task": int(probe_task),
                                    "accuracy": measure["accuracy"],
                                    "nll": measure["nll"],
                                    "training_loss": np.nan,
                                }
                            )
                    if outcome["reset_logit_max_abs"] > 1e-5:
                        raise AssertionError(f"function-neutral reset drift {outcome['reset_logit_max_abs']}")

            save_progress(
                progress,
                block + 1,
                model,
                optimizer,
                replay,
                data_generator,
                metrics,
                extra_state={"unit_ages": {str(layer): ages.tolist() for layer, ages in unit_ages.items()}},
            )

    final_eval = evaluate(model, eval_x, eval_y)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "replay": replay.state_dict(),
        "width": width,
        "seed": seed,
        "schedule": schedule,
        "final_mixture": final_eval,
        "blocks_trained": max_blocks,
        "unit_ages": unit_ages,
    }
    return metrics, checkpoint


def train_offline(
    config: dict[str, Any], data: PermutedMNIST, artifacts: RunArtifacts, device: torch.device
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed, width = int(config["seed"]), int(config["width"])
    probabilities = tuple(float(value) for value in config.get("skew_probabilities", SKEW))
    torch.manual_seed(seed + 30_000)
    model = GatedMLP(width).to(device)
    optimizer = make_optimizer(model, float(config["lr"]))
    generator = torch.Generator().manual_seed(seed + 50_000)
    task_rng = np.random.default_rng(seed + 90_000)
    eval_x, eval_y, _ = data.evaluation_set()
    metrics = []
    stale = 0
    best = -1.0
    high_loss_evaluations = 0
    for step in range(int(config["total_updates"])):
        task = int(task_rng.choice(len(probabilities), p=probabilities))
        x, y = data.batch(task, int(config["batch_size"]), generator)
        accuracy, nll = preupdate(model, x, y)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        metrics.append(
            {
                "metric_kind": "preupdate",
                "step": step,
                "block": step // int(config["updates_per_block"]),
                "update": step % int(config["updates_per_block"]),
                "task": task,
                "accuracy": accuracy,
                "nll": nll,
                "training_loss": float(loss.item()),
            }
        )
        if (step + 1) % 300 == 0:
            mixture = evaluate(model, eval_x, eval_y)
            metrics.append(
                {
                    "metric_kind": "mixture_eval",
                    "step": step + 1,
                    "block": step // int(config["updates_per_block"]),
                    "update": step % int(config["updates_per_block"]),
                    "task": -1,
                    "accuracy": mixture["accuracy"],
                    "nll": mixture["nll"],
                    "training_loss": float(loss.item()),
                }
            )
            high_loss_evaluations = high_loss_evaluations + 1 if mixture["nll"] > 20 else 0
            if not np.isfinite(mixture["nll"]) or high_loss_evaluations >= 3:
                raise FloatingPointError(f"offline divergence at step {step + 1}")
            if mixture["accuracy"] - best < 0.002:
                stale += 1
            else:
                stale = 0
                best = mixture["accuracy"]
            if stale >= 3:
                artifacts.event("offline_early_stop", step=step + 1, best_accuracy=best)
                break
    final_eval = evaluate(model, eval_x, eval_y)
    return metrics, {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "width": width,
        "seed": seed,
        "schedule": data.schedule,
        "final_mixture": final_eval,
        "updates_trained": len([row for row in metrics if row["metric_kind"] == "preupdate"]),
    }
