"""Window-MLP layer torch mirror.

Zero trainable parameters. Maintains a FIFO ring buffer of the last `n_steps`
inputs and concatenates them into a vector of length `n_steps * input_size`
for the next Dense layer.

Constructible via `build_layer(WindowSpec)` (used by warm-start BPTT and the
cross-language equivalence test). The PPO runtime gate has moved to
`rl/train.py::_derive_hidden_shapes`; PSO still bypasses V2Policy and drives
the Rust runtime via aerocapture_rs.nn_forward.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


class WindowLayer(nn.Module):
    # Registered buffer tracks module dtype/device for new_state.
    # Class-level annotation helps mypy resolve the attribute as Tensor rather
    # than nn.Module | Tensor (nn.Module.__getattr__ default).
    _dtype_anchor: Tensor

    def __init__(self, input_size: int, n_steps: int) -> None:
        super().__init__()
        if input_size <= 0 or n_steps <= 0:
            raise ValueError(f"WindowLayer input_size and n_steps must be positive (got input_size={input_size}, n_steps={n_steps})")
        self.input_size = input_size
        self.n_steps = n_steps
        # Zero-param layers still need a dtype/device anchor for new_state.
        # A non-persistent buffer is the idiomatic torch approach: it doesn't
        # appear in state_dict and it participates in .double() / .to() calls.
        self.register_buffer("_dtype_anchor", torch.zeros(1), persistent=False)

    def forward(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        # x:     (batch, input_size)
        # state: (batch, n_steps, input_size)
        if x.shape[-1] != self.input_size:
            raise AssertionError(f"WindowLayer expected input_size={self.input_size}, got {x.shape[-1]}")
        if state.shape[1:] != (self.n_steps, self.input_size):
            raise AssertionError(f"WindowLayer expected state shape (_, {self.n_steps}, {self.input_size}), got {tuple(state.shape)}")
        new_state = torch.cat([state[:, 1:], x.unsqueeze(1)], dim=1)
        out = new_state.reshape(x.shape[0], -1)
        return out, new_state

    def new_state(self, batch_size: int, device: Any | None = None) -> Tensor:
        # dtype tracks _dtype_anchor so policy.double() / .float() propagates;
        # device override is optional (defaults to the anchor's device so
        # policy.to(device) propagates naturally).
        return torch.zeros(
            batch_size,
            self.n_steps,
            self.input_size,
            dtype=self._dtype_anchor.dtype,
            device=device if device is not None else self._dtype_anchor.device,
        )

    def to_flat(self) -> np.ndarray:
        """Zero trainable parameters; flat representation is empty.

        Matches Rust `LayerWeights for WindowLayer::to_flat` in
        src/rust/src/data/neural.rs.
        """
        return np.array([], dtype=np.float64)

    def from_flat(self, slab: np.ndarray) -> None:
        """Zero trainable parameters; consumes nothing from slab (no-op).

        Mirrors Rust `LayerWeights for WindowLayer::from_flat` which returns 0
        and ignores the input slice. `slab` must be length-0 by construction
        (callers slice by _layer_n_params which returns 0 for Window).
        """
        assert slab.size == 0, f"WindowLayer from_flat: expected empty slab, got size={slab.size}"
