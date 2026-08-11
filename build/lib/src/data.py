from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch

from .constants import INPUT_SIZE, N_TASKS, SKEW
from .schedules import make_skew_schedule


def make_library_asset(path: Path, seed: int, probabilities: tuple[float, ...] = SKEW) -> None:
    """Persist task definitions, schedule, and evaluation/probe indices once per seed."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + 10_000)
    input_permutations = np.stack([rng.permutation(28 * 28) for _ in range(N_TASKS)]).astype(np.int32)
    label_permutations = np.stack([rng.permutation(10) for _ in range(N_TASKS)]).astype(np.int16)
    schedule = np.asarray(make_skew_schedule(seed, probabilities), dtype=np.int16)
    counts = np.floor(np.asarray(probabilities) * 2000).astype(int)
    counts[np.argmax(probabilities)] += 2000 - int(counts.sum())
    evaluation_tasks = np.repeat(np.arange(N_TASKS), counts)
    rng.shuffle(evaluation_tasks)
    evaluation_indices = rng.integers(0, 10_000, size=2000, dtype=np.int32)
    probe_indices = np.stack([rng.choice(10_000, size=50, replace=False) for _ in range(N_TASKS)]).astype(np.int32)
    tmp = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(
        tmp,
        input_permutations=input_permutations,
        label_permutations=label_permutations,
        schedule=schedule,
        evaluation_tasks=evaluation_tasks.astype(np.int16),
        evaluation_indices=evaluation_indices,
        probe_indices=probe_indices,
    )
    tmp.replace(path)


def stage_mnist(data_root: Path) -> None:
    from torchvision.datasets import MNIST

    MNIST(str(data_root), train=True, download=True)
    MNIST(str(data_root), train=False, download=True)


class PermutedMNIST:
    def __init__(self, data_root: Path, asset_path: Path, device: torch.device) -> None:
        from torchvision.datasets import MNIST

        asset = np.load(asset_path)
        self.input_permutations = torch.from_numpy(asset["input_permutations"].astype(np.int64))
        self.label_permutations = torch.from_numpy(asset["label_permutations"].astype(np.int64))
        self.schedule = asset["schedule"].astype(int).tolist()
        self.evaluation_tasks = asset["evaluation_tasks"].astype(int)
        self.evaluation_indices = asset["evaluation_indices"].astype(int)
        self.probe_indices = asset["probe_indices"].astype(int)
        train = MNIST(str(data_root), train=True, download=False)
        test = MNIST(str(data_root), train=False, download=False)
        self.train_x = train.data.reshape(-1, 784).float().div_(255.0)
        self.train_y = train.targets.long()
        self.test_x = test.data.reshape(-1, 784).float().div_(255.0)
        self.test_y = test.targets.long()
        self.device = device

    def transform(self, raw_x: torch.Tensor, raw_y: torch.Tensor, task: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = raw_x[:, self.input_permutations[task]]
        one_hot = torch.zeros((x.shape[0], N_TASKS), dtype=x.dtype)
        one_hot[:, task] = 1.0
        y = self.label_permutations[task, raw_y]
        return torch.cat((x, one_hot), dim=1).to(self.device), y.to(self.device)

    def batch(
        self,
        task: int,
        size: int,
        generator: torch.Generator,
        split: Literal["train", "test"] = "train",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_x, source_y = (self.train_x, self.train_y) if split == "train" else (self.test_x, self.test_y)
        indices = torch.randint(len(source_y), (size,), generator=generator)
        return self.transform(source_x[indices], source_y[indices], task)

    def evaluation_set(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for task in range(N_TASKS):
            mask = np.flatnonzero(self.evaluation_tasks == task)
            if not len(mask):
                continue
            indices = torch.from_numpy(self.evaluation_indices[mask].astype(np.int64))
            x, y = self.transform(self.test_x[indices], self.test_y[indices], task)
            xs.append(x)
            ys.append(y)
        # Reconstructing by task groups changes order, which is immaterial for aggregate metrics.
        grouped_tasks = torch.cat(
            [torch.full((int((self.evaluation_tasks == task).sum()),), task, device=self.device) for task in range(N_TASKS)]
        )
        return torch.cat(xs), torch.cat(ys), grouped_tasks

    def probes(self, tasks: list[int] | None = None) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        tasks = tasks if tasks is not None else list(range(N_TASKS))
        result = {}
        for task in tasks:
            indices = torch.from_numpy(self.probe_indices[task].astype(np.int64))
            result[task] = self.transform(self.test_x[indices], self.test_y[indices], task)
        return result


def validate_input(x: torch.Tensor) -> None:
    if x.ndim != 2 or x.shape[1] != INPUT_SIZE:
        raise ValueError(f"expected [batch,{INPUT_SIZE}], received {tuple(x.shape)}")
