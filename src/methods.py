from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .constants import KD_BETA, KD_TEMPERATURE, RECYCLE_FRACTION
from .models import (
    GatedMLP,
    reset_selected_units,
    restore_selected_parameter_snapshot,
    selected_parameter_snapshot,
)
from .replay import ReplayBuffer


@torch.no_grad()
def evaluate(model: GatedMLP, x: torch.Tensor, y: torch.Tensor, chunk: int = 512) -> dict[str, float]:
    model.eval()
    losses, correct = 0.0, 0
    for start in range(0, len(y), chunk):
        logits = model(x[start : start + chunk])
        target = y[start : start + chunk]
        losses += F.cross_entropy(logits, target, reduction="sum").item()
        correct += int((logits.argmax(1) == target).sum().item())
    return {"nll": losses / len(y), "accuracy": correct / len(y)}


@torch.no_grad()
def evaluate_probes(
    model: GatedMLP,
    probes: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    weights: Mapping[int, float] | None = None,
) -> dict[str, float]:
    weights = weights or {task: 1 / len(probes) for task in probes}
    total_weight = sum(weights.get(task, 0.0) for task in probes)
    result = {"nll": 0.0, "accuracy": 0.0}
    for task, (x, y) in probes.items():
        measure = evaluate(model, x, y)
        weight = weights.get(task, 0.0) / total_weight
        result["nll"] += weight * measure["nll"]
        result["accuracy"] += weight * measure["accuracy"]
    return result


def gate_scores(
    model: GatedMLP,
    probes: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, dict[int, dict[int, torch.Tensor]]]:
    """Return task-conditioned Taylor, signed-gradient, Fisher, and activation scores."""
    model.eval()
    output_norms = [
        model.hidden[1].weight.detach().abs().mean(0),
        model.output.weight.detach().abs().mean(0),
    ]
    all_scores: dict[str, dict[int, dict[int, torch.Tensor]]] = {
        metric: {} for metric in ("taylor", "signed", "fisher", "activation")
    }
    for task, (x, y) in probes.items():
        def single_loss(
            gate0: torch.Tensor, gate1: torch.Tensor, feature: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
            hidden0 = torch.relu(F.linear(feature, model.hidden[0].weight, model.hidden[0].bias)) * gate0
            hidden1 = torch.relu(F.linear(hidden0, model.hidden[1].weight, model.hidden[1].bias)) * gate1
            logits = F.linear(hidden1, model.output.weight, model.output.bias)
            return F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))

        gradient_function = torch.func.grad(single_loss, argnums=(0, 1))
        gradients = torch.func.vmap(gradient_function, in_dims=(None, None, 0, 0))(
            model.gates[0], model.gates[1], x, y
        )
        with torch.no_grad():
            _logits, hidden = model(x, return_hidden=True)
        for layer_index, (gate, per_example_gradient, activation, output_norm) in enumerate(
            zip(model.gates, gradients, hidden, output_norms)
        ):
            gated_gradients = gate.detach().unsqueeze(0) * per_example_gradient.detach()
            signed = gated_gradients.mean(0).cpu()
            all_scores["signed"].setdefault(task, {})[layer_index] = signed
            all_scores["taylor"].setdefault(task, {})[layer_index] = gated_gradients.abs().mean(0).cpu()
            all_scores["fisher"].setdefault(task, {})[layer_index] = per_example_gradient.detach().square().mean(0).cpu()
            all_scores["activation"].setdefault(task, {})[layer_index] = (
                activation.detach().abs().mean(0) * output_norm
            ).cpu()
    return all_scores


def empirical_scores(
    model: GatedMLP,
    probes: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
) -> dict[int, dict[int, torch.Tensor]]:
    baseline = {task: evaluate(model, x, y)["nll"] for task, (x, y) in probes.items()}
    result: dict[int, dict[int, torch.Tensor]] = {}
    for task, (x, y) in probes.items():
        result[task] = {}
        for layer_index, gate in enumerate(model.gates):
            damage = torch.empty_like(gate, device="cpu")
            for unit in range(len(gate)):
                old = gate.data[unit].item()
                gate.data[unit] = 0.0
                damage[unit] = evaluate(model, x, y)["nll"] - baseline[task]
                gate.data[unit] = old
            result[task][layer_index] = damage.clamp_min(0.0)
    return result


def normalized_utility(
    scores: dict[int, dict[int, torch.Tensor]],
    task_weights: Mapping[int, float],
) -> dict[int, torch.Tensor]:
    utilities: dict[int, torch.Tensor] = {}
    layer_indices = sorted(next(iter(scores.values())).keys())
    for layer_index in layer_indices:
        utility = None
        for task, by_layer in scores.items():
            values = by_layer[layer_index].float()
            median = values.median().clamp_min(1e-8)
            normalized = values / median
            term = float(task_weights.get(task, 0.0)) * normalized
            utility = term if utility is None else utility + term
        utilities[layer_index] = utility
    return utilities


def select_units(
    utilities: Mapping[int, torch.Tensor], fraction: float = RECYCLE_FRACTION
) -> dict[int, list[int]]:
    selected = {}
    for layer_index, values in utilities.items():
        count = math.floor(len(values) * fraction)
        if len(values) >= 1 / fraction:
            count = max(1, count)
        selected[layer_index] = torch.argsort(values)[:count].tolist()
    return selected


def choose_units(
    model: GatedMLP,
    probes: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    task_weights: Mapping[int, float],
    metric: str,
    fraction: float,
    rng: np.random.Generator,
) -> tuple[dict[int, list[int]], dict[int, torch.Tensor]]:
    if metric == "random":
        selected = {
            layer: sorted(rng.choice(model.width, size=max(1, int(model.width * fraction)), replace=False).tolist())
            for layer in range(2)
        }
        return selected, {layer: torch.zeros(model.width) for layer in range(2)}
    if metric == "empirical":
        task_scores = empirical_scores(model, probes)
    else:
        score_bundle = gate_scores(model, probes)
        task_scores = score_bundle[metric]
    utilities = normalized_utility(task_scores, task_weights)
    return select_units(utilities, fraction), utilities


def sleep_consolidate(
    model: GatedMLP,
    teacher: GatedMLP,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    selected: dict[int, list[int]],
    device: torch.device,
    task_weights: Mapping[int, float],
    steps: int,
    beta: float = KD_BETA,
    temperature: float = KD_TEMPERATURE,
    batch_size: int = 64,
) -> list[float]:
    teacher.eval()
    losses = []
    frozen = selected_parameter_snapshot(model, selected)
    for _ in range(steps):
        sampled = replay.sample(batch_size, device, dict(task_weights))
        if sampled is None:
            break
        x, y, _tasks = sampled
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        with torch.no_grad():
            teacher_logits = teacher(x)
        ce = F.cross_entropy(logits, y)
        kd = F.kl_div(
            F.log_softmax(logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature**2)
        loss = ce + beta * kd
        loss.backward()
        optimizer.step()
        # Gates zero ordinary gradients, while this restoration also prevents Adam's
        # stored momentum from drifting masked slices during consolidation.
        restore_selected_parameter_snapshot(model, selected, frozen)
        losses.append(float(loss.item()))
    return losses


def recycle(
    model: GatedMLP,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    probes: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    task_weights: Mapping[int, float],
    metric: str,
    fraction: float,
    sleep_steps: int,
    beta: float,
    device: torch.device,
    rng: np.random.Generator,
    reset_generator: torch.Generator,
) -> dict[str, object]:
    teacher = copy.deepcopy(model).eval()
    before = evaluate_probes(model, probes, task_weights)
    per_task_before = {task: evaluate(model, *batch) for task, batch in probes.items()}
    selected, utilities = choose_units(model, probes, task_weights, metric, fraction, rng)
    model.selected_mask(selected)
    after_mask = evaluate_probes(model, probes, task_weights)
    per_task_after_mask = {task: evaluate(model, *batch) for task, batch in probes.items()}
    losses = sleep_consolidate(
        model,
        teacher,
        optimizer,
        replay,
        selected,
        device,
        task_weights,
        sleep_steps,
        beta,
    )
    after_sleep = evaluate_probes(model, probes, task_weights)
    per_task_after_sleep = {task: evaluate(model, *batch) for task, batch in probes.items()}
    probe_x = torch.cat([value[0] for value in probes.values()])
    with torch.no_grad():
        logits_before_reset = model(probe_x).clone()
    reset_selected_units(model, optimizer, selected, reset_generator)
    with torch.no_grad():
        logits_after_reset = model(probe_x)
    drift = float((logits_before_reset - logits_after_reset).abs().max().item())
    after_reset = evaluate_probes(model, probes, task_weights)
    per_task_after_reset = {task: evaluate(model, *batch) for task, batch in probes.items()}
    return {
        "selected": selected,
        "utilities": {str(k): v.tolist() for k, v in utilities.items()},
        "before": before,
        "after_mask": after_mask,
        "after_sleep": after_sleep,
        "after_reset": after_reset,
        "per_task": {
            "before": per_task_before,
            "after_mask": per_task_after_mask,
            "after_sleep": per_task_after_sleep,
            "after_reset": per_task_after_reset,
        },
        "sleep_losses": losses,
        "reset_logit_max_abs": drift,
    }
