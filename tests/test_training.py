from pathlib import Path

import torch

from src.constants import INPUT_SIZE
from src.models import GatedMLP, make_optimizer
from src.replay import ReplayBuffer
from src.training import load_progress, online_update, preupdate, save_progress


def make_batch(generator: torch.Generator):
    return torch.randn(16, INPUT_SIZE, generator=generator), torch.randint(0, 10, (16,), generator=generator)


def test_preupdate_metric_is_taken_before_optimization():
    torch.manual_seed(1)
    model = GatedMLP(5)
    optimizer = make_optimizer(model)
    replay = ReplayBuffer(20, 0)
    x, y = make_batch(torch.Generator().manual_seed(2))
    measured = preupdate(model, x, y)
    with torch.no_grad():
        expected = float((model(x).argmax(1) == y).float().mean())
    online_update(model, optimizer, x, y, replay, torch.device("cpu"), batch_size=16)
    assert measured[0] == expected


def test_resume_reproduces_uninterrupted_next_metric(tmp_path: Path):
    torch.manual_seed(4)
    model = GatedMLP(5)
    optimizer = make_optimizer(model)
    replay = ReplayBuffer(20, 4)
    generator = torch.Generator().manual_seed(8)
    x, y = make_batch(generator)
    online_update(model, optimizer, x, y, replay, torch.device("cpu"), batch_size=16)
    replay.set_quotas({0: 20})
    replay.add(0, x, y)
    state_path = tmp_path / "state.pt"
    save_progress(state_path, 1, model, optimizer, replay, generator, [{"accuracy": 0.1}])

    next_x, next_y = make_batch(generator)
    expected_metric = preupdate(model, next_x, next_y)
    expected_loss = online_update(model, optimizer, next_x, next_y, replay, torch.device("cpu"), batch_size=16)

    resumed_model = GatedMLP(5)
    resumed_optimizer = make_optimizer(resumed_model)
    resumed_replay = ReplayBuffer(20, 99)
    resumed_generator = torch.Generator()
    next_block, metrics, extra = load_progress(
        state_path,
        resumed_model,
        resumed_optimizer,
        resumed_replay,
        resumed_generator,
        torch.device("cpu"),
    )
    actual_x, actual_y = make_batch(resumed_generator)
    actual_metric = preupdate(resumed_model, actual_x, actual_y)
    actual_loss = online_update(
        resumed_model, resumed_optimizer, actual_x, actual_y, resumed_replay, torch.device("cpu"), batch_size=16
    )
    assert next_block == 1 and metrics == [{"accuracy": 0.1}]
    assert extra == {}
    assert torch.equal(next_x, actual_x) and torch.equal(next_y, actual_y)
    assert actual_metric == expected_metric
    assert actual_loss == expected_loss
