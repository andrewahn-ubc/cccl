import copy

import torch
import torch.nn.functional as F

from src.constants import INPUT_SIZE
from src.methods import gate_scores, sleep_consolidate
from src.models import (
    GatedMLP,
    make_optimizer,
    reset_selected_units,
    selected_parameter_snapshot,
)
from src.replay import ReplayBuffer


def test_masking_known_unit_changes_only_expected_activation():
    model = GatedMLP(5)
    x = torch.randn(4, INPUT_SIZE)
    _, before = model(x, return_hidden=True)
    model.gates[0].data[2] = 0
    _, after = model(x, return_hidden=True)
    assert torch.equal(after[0][:, 2], torch.zeros(4))
    assert torch.allclose(before[0][:, [0, 1, 3, 4]], after[0][:, [0, 1, 3, 4]])


def test_taylor_scores_have_one_per_unit_and_do_not_modify_weights():
    model = GatedMLP(7)
    x = torch.randn(8, INPUT_SIZE)
    y = torch.randint(0, 10, (8,))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    scores = gate_scores(model, {0: (x, y)})
    assert scores["taylor"][0][0].shape == (7,)
    assert scores["taylor"][0][1].shape == (7,)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_function_neutral_reset_and_slice_local_optimizer_clear():
    model = GatedMLP(6)
    optimizer = make_optimizer(model)
    x = torch.randn(10, INPUT_SIZE)
    y = torch.randint(0, 10, (10,))
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
    selected = {0: [0], 1: [1]}
    model.selected_mask(selected)
    with torch.no_grad():
        before_logits = model(x).clone()
    untouched_before = optimizer.state[model.hidden[1].weight]["exp_avg"][3, 3].clone()
    reset_selected_units(model, optimizer, selected, torch.Generator().manual_seed(9))
    with torch.no_grad():
        after_logits = model(x)
    assert torch.allclose(before_logits, after_logits, atol=1e-6, rtol=0)
    state = optimizer.state[model.hidden[0].weight]["exp_avg"]
    assert torch.equal(state[0], torch.zeros_like(state[0]))
    assert optimizer.state[model.hidden[1].weight]["exp_avg"][3, 3] == untouched_before


def test_sleep_keeps_masked_parameter_slices_frozen():
    model = GatedMLP(6)
    teacher = copy.deepcopy(model)
    optimizer = make_optimizer(model)
    selected = {0: [0], 1: [1]}
    replay = ReplayBuffer(30, 0)
    replay.set_quotas({0: 30})
    replay.add(0, torch.randn(30, INPUT_SIZE), torch.randint(0, 10, (30,)))
    model.selected_mask(selected)
    before = selected_parameter_snapshot(model, selected)
    sleep_consolidate(
        model,
        teacher,
        optimizer,
        replay,
        selected,
        torch.device("cpu"),
        {0: 1.0},
        steps=2,
    )
    after = selected_parameter_snapshot(model, selected)
    for name in before:
        assert torch.equal(before[name], after[name])
