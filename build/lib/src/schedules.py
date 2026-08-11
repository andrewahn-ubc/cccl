from __future__ import annotations

import numpy as np

from .constants import BLOCKS, N_TASKS, SKEW


def make_skew_schedule(seed: int, probabilities: tuple[float, ...] = SKEW, blocks: int = BLOCKS) -> list[int]:
    """Force one visit per task, then draw from the fixed skew generator."""
    if len(probabilities) != N_TASKS or not np.isclose(sum(probabilities), 1.0):
        raise ValueError("schedule probabilities must contain ten values summing to one")
    if blocks < N_TASKS:
        raise ValueError("blocks must allow all forced task introductions")
    rng = np.random.default_rng(seed + 20_000)
    forced = rng.permutation(N_TASKS).tolist()
    tail = rng.choice(N_TASKS, size=blocks - N_TASKS, p=probabilities).tolist()
    return [int(x) for x in forced + tail]


def oracle_transition(probabilities: tuple[float, ...] = SKEW) -> np.ndarray:
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    return np.tile(probabilities_array, (len(probabilities), 1))


def discounted_occupancy(transition: np.ndarray, current: int, horizon: int = 10, gamma: float = 0.9) -> np.ndarray:
    transition = np.asarray(transition, dtype=np.float64)
    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError("transition must be square")
    if np.any(transition < 0) or not np.allclose(transition.sum(axis=1), 1.0):
        raise ValueError("transition rows must be nonnegative and sum to one")
    state = np.zeros(transition.shape[0], dtype=np.float64)
    state[current] = 1.0
    occupancy = np.zeros_like(state)
    for step in range(horizon):
        state = state @ transition
        occupancy += (gamma**step) * state
    return occupancy / occupancy.sum()


class DecayedTransitions:
    def __init__(self, n_tasks: int = N_TASKS, alpha: float = 0.5, decay: float = 0.97) -> None:
        self.n_tasks = n_tasks
        self.alpha = alpha
        self.decay = decay
        self.counts = np.zeros((n_tasks, n_tasks), dtype=np.float64)

    def update(self, previous: int, current: int) -> None:
        self.counts *= self.decay
        self.counts[previous, current] += 1.0

    def matrix(self) -> np.ndarray:
        smoothed = self.counts + self.alpha
        return smoothed / smoothed.sum(axis=1, keepdims=True)

    def state_dict(self) -> dict[str, object]:
        return {"counts": self.counts.copy(), "alpha": self.alpha, "decay": self.decay}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.counts = np.asarray(state["counts"], dtype=np.float64).copy()

