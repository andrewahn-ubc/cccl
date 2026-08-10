from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import torch

from .constants import BUFFER_SIZE, N_TASKS, SKEW


def largest_remainder_quotas(
    observed: Iterable[int],
    capacity: int = BUFFER_SIZE,
    probabilities: tuple[float, ...] = SKEW,
    minimum: int = 5,
) -> dict[int, int]:
    observed = sorted({int(t) for t in observed})
    if not observed:
        return {}
    if minimum * len(observed) > capacity:
        raise ValueError("minimum quotas exceed capacity")
    remaining = capacity - minimum * len(observed)
    weights = np.asarray([probabilities[t] for t in observed], dtype=np.float64)
    weights /= weights.sum()
    raw = weights * remaining
    floors = np.floor(raw).astype(int)
    quotas = {task: minimum + int(value) for task, value in zip(observed, floors)}
    left = capacity - sum(quotas.values())
    order = np.argsort(-(raw - floors), kind="stable")
    for index in order[:left]:
        quotas[observed[int(index)]] += 1
    return quotas


def equal_quotas(observed: Iterable[int], capacity: int = BUFFER_SIZE) -> dict[int, int]:
    observed = sorted({int(t) for t in observed})
    if not observed:
        return {}
    base, left = divmod(capacity, len(observed))
    return {task: base + int(i < left) for i, task in enumerate(observed)}


class ReplayBuffer:
    """Task-aware reservoir with deterministic quota changes."""

    def __init__(self, capacity: int = BUFFER_SIZE, seed: int = 0) -> None:
        self.capacity = capacity
        self.rng = np.random.default_rng(seed + 40_000)
        self.x: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.y: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.seen: dict[int, int] = defaultdict(int)
        self.quotas: dict[int, int] = {}

    def __len__(self) -> int:
        return sum(len(items) for items in self.x.values())

    @property
    def observed(self) -> list[int]:
        return sorted(task for task, count in self.seen.items() if count > 0)

    def set_quotas(self, quotas: dict[int, int]) -> None:
        if sum(quotas.values()) > self.capacity or any(value < 0 for value in quotas.values()):
            raise ValueError("invalid replay quotas")
        self.quotas = {int(k): int(v) for k, v in quotas.items()}
        for task in list(self.x):
            quota = self.quotas.get(task, 0)
            if len(self.x[task]) > quota:
                keep = sorted(self.rng.choice(len(self.x[task]), size=quota, replace=False).tolist()) if quota else []
                self.x[task] = [self.x[task][i] for i in keep]
                self.y[task] = [self.y[task][i] for i in keep]

    def add(self, task: int, x: torch.Tensor, y: torch.Tensor) -> None:
        if task not in self.quotas:
            self.set_quotas(equal_quotas(self.observed + [task], self.capacity))
        quota = self.quotas.get(task, 0)
        for feature, target in zip(x.detach().cpu(), y.detach().cpu()):
            self.seen[task] += 1
            if len(self.x[task]) < quota:
                self.x[task].append(feature.clone())
                self.y[task].append(target.clone())
            elif quota:
                index = int(self.rng.integers(self.seen[task]))
                if index < quota:
                    self.x[task][index] = feature.clone()
                    self.y[task][index] = target.clone()
        if len(self) > self.capacity:
            raise AssertionError("replay capacity exceeded")

    def sample(
        self,
        size: int,
        device: torch.device,
        task_weights: dict[int, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        available = [task for task in self.observed if self.x[task]]
        if not available or size <= 0:
            return None
        if task_weights is None:
            weights = np.asarray([self.quotas.get(t, 0) for t in available], dtype=np.float64)
        else:
            weights = np.asarray([task_weights.get(t, 0.0) for t in available], dtype=np.float64)
        if weights.sum() <= 0:
            weights[:] = 1.0
        weights /= weights.sum()
        tasks = self.rng.choice(available, size=size, p=weights)
        xs, ys = [], []
        for task in tasks:
            index = int(self.rng.integers(len(self.x[int(task)])))
            xs.append(self.x[int(task)][index])
            ys.append(self.y[int(task)][index])
        return torch.stack(xs).to(device), torch.stack(ys).long().to(device), torch.as_tensor(tasks, device=device)

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "x": dict(self.x),
            "y": dict(self.y),
            "seen": dict(self.seen),
            "quotas": self.quotas,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.capacity = int(state["capacity"])
        self.x = defaultdict(list, state["x"])
        self.y = defaultdict(list, state["y"])
        self.seen = defaultdict(int, state["seen"])
        self.quotas = {int(k): int(v) for k, v in state["quotas"].items()}
        self.rng.bit_generator.state = state["rng_state"]

    def allocation(self) -> dict[int, int]:
        return {task: len(self.x[task]) for task in range(N_TASKS)}
