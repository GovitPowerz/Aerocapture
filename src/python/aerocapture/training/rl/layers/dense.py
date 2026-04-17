"""Dense (fully-connected) layer matching the Rust DenseLayer variant.

Canonical flat weight order: W (row-major, [out, in]) then b.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

_ACTIVATIONS: dict[str, Callable[[Tensor], Tensor]] = {
    "tanh": torch.tanh,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "asinh": torch.asinh,
    "linear": lambda x: x,
    "swish": lambda x: x * torch.sigmoid(x),
    "mish": lambda x: x * torch.tanh(torch.nn.functional.softplus(x)),
}


class DenseLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int, activation: str) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=True)
        self.activation_name = activation
        self.activation_fn = _ACTIVATIONS[activation]

    def forward(self, x: Tensor, state: None) -> tuple[Tensor, None]:
        """Stateful-compatible signature. State is always None for dense layers."""
        return self.activation_fn(self.linear(x)), None

    def new_state(self, batch_size: int, device: Any) -> None:
        return None

    def extra_repr(self) -> str:
        return f"activation={self.activation_name}"
