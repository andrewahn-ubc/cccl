from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn

from .constants import INPUT_SIZE, N_CLASSES


class GatedMLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.hidden = nn.ModuleList([nn.Linear(INPUT_SIZE, width), nn.Linear(width, width)])
        self.output = nn.Linear(width, N_CLASSES)
        self.gates = nn.ParameterList(
            [nn.Parameter(torch.ones(width), requires_grad=True), nn.Parameter(torch.ones(width), requires_grad=True)]
        )
        self.reset_parameters()

    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        # Match nn.Linear's initializer, with an optional generator for deterministic boundary resets.
        for layer in [*self.hidden, self.output]:
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), generator=generator)
            if layer.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in else 0
                nn.init.uniform_(layer.bias, -bound, bound, generator=generator)
        for gate in self.gates:
            gate.data.fill_(1.0)

    def forward(
        self, x: torch.Tensor, return_hidden: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        activations = []
        for layer, gate in zip(self.hidden, self.gates):
            x = torch.relu(layer(x)) * gate
            activations.append(x)
        logits = self.output(x)
        return (logits, activations) if return_hidden else logits

    def optimizer_parameters(self) -> Iterable[nn.Parameter]:
        gate_ids = {id(gate) for gate in self.gates}
        return (parameter for parameter in self.parameters() if id(parameter) not in gate_ids)

    def selected_mask(self, selected: dict[int, list[int]]) -> None:
        for layer_index, indices in selected.items():
            self.gates[layer_index].data[indices] = 0.0

    def unmask_all(self) -> None:
        for gate in self.gates:
            gate.data.fill_(1.0)


def make_optimizer(model: GatedMLP, lr: float = 1e-3) -> torch.optim.Adam:
    return torch.optim.Adam(model.optimizer_parameters(), lr=lr)


def _clear_state_slice(state: dict[str, torch.Tensor | int], indices: tuple[object, ...]) -> None:
    for value in state.values():
        if torch.is_tensor(value) and value.ndim:
            value[indices] = 0


def reset_selected_units(
    model: GatedMLP,
    optimizer: torch.optim.Optimizer,
    selected: dict[int, list[int]],
    generator: torch.Generator | None = None,
) -> None:
    """Function-neutral reset: random incoming, zero outgoing, and slice-local Adam clearing."""
    for layer_index, raw_indices in selected.items():
        indices = torch.as_tensor(sorted(set(raw_indices)), dtype=torch.long, device=model.gates[layer_index].device)
        if not len(indices):
            continue
        incoming = model.hidden[layer_index]
        temporary = torch.empty(
            (len(indices), incoming.weight.shape[1]), device=incoming.weight.device, dtype=incoming.weight.dtype
        )
        nn.init.kaiming_uniform_(temporary, a=math.sqrt(5), generator=generator)
        incoming.weight.data[indices] = temporary
        incoming.bias.data[indices] = 0.0
        outgoing = model.hidden[layer_index + 1] if layer_index + 1 < len(model.hidden) else model.output
        outgoing.weight.data[:, indices] = 0.0

        if incoming.weight in optimizer.state:
            _clear_state_slice(optimizer.state[incoming.weight], (indices, slice(None)))
        if incoming.bias in optimizer.state:
            _clear_state_slice(optimizer.state[incoming.bias], (indices,))
        if outgoing.weight in optimizer.state:
            _clear_state_slice(optimizer.state[outgoing.weight], (slice(None), indices))
    model.unmask_all()


def selected_parameter_snapshot(model: GatedMLP, selected: dict[int, list[int]]) -> dict[str, torch.Tensor]:
    snapshot = {}
    for layer_index, raw_indices in selected.items():
        indices = torch.as_tensor(raw_indices, dtype=torch.long, device=model.gates[layer_index].device)
        incoming = model.hidden[layer_index]
        outgoing = model.hidden[layer_index + 1] if layer_index + 1 < len(model.hidden) else model.output
        snapshot[f"{layer_index}.incoming_weight"] = incoming.weight.data[indices].clone()
        snapshot[f"{layer_index}.incoming_bias"] = incoming.bias.data[indices].clone()
        snapshot[f"{layer_index}.outgoing_weight"] = outgoing.weight.data[:, indices].clone()
    return snapshot


def restore_selected_parameter_snapshot(
    model: GatedMLP, selected: dict[int, list[int]], snapshot: dict[str, torch.Tensor]
) -> None:
    for layer_index, raw_indices in selected.items():
        if not raw_indices:
            continue
        indices = torch.as_tensor(raw_indices, dtype=torch.long, device=model.gates[layer_index].device)
        incoming = model.hidden[layer_index]
        outgoing = model.hidden[layer_index + 1] if layer_index + 1 < len(model.hidden) else model.output
        incoming.weight.data[indices] = snapshot[f"{layer_index}.incoming_weight"]
        incoming.bias.data[indices] = snapshot[f"{layer_index}.incoming_bias"]
        outgoing.weight.data[:, indices] = snapshot[f"{layer_index}.outgoing_weight"]
