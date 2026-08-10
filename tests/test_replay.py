import torch

from src.constants import INPUT_SIZE
from src.replay import ReplayBuffer, equal_quotas, largest_remainder_quotas


def test_replay_capacity_quotas_and_sampling():
    replay = ReplayBuffer(capacity=20, seed=3)
    replay.set_quotas(equal_quotas([0, 1], 20))
    for task in (0, 1):
        x = torch.full((30, INPUT_SIZE), float(task))
        y = torch.full((30,), task)
        replay.add(task, x, y)
    assert len(replay) == 20
    assert replay.allocation()[0] == 10
    assert replay.allocation()[1] == 10
    sampled = replay.sample(12, torch.device("cpu"), {0: 1.0, 1: 0.0})
    assert sampled is not None
    _, _, tasks = sampled
    assert torch.equal(tasks, torch.zeros_like(tasks))


def test_largest_remainder_respects_minimum_and_capacity():
    quotas = largest_remainder_quotas(range(10), capacity=500)
    assert sum(quotas.values()) == 500
    assert min(quotas.values()) >= 5

