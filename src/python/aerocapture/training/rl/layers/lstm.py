"""LSTM cell matching nn.LSTMCell + Rust LstmLayer bit-for-bit.

Canonical flat weight order (LayerWeights trait + PSO chromosome):
    weight_ih row-major, weight_hh row-major, bias_ih, bias_hh.

Forward equations (PyTorch nn.LSTMCell convention, gate order i, f, g, o):
    i_t = sigmoid(W_ii @ x + b_ii + W_hi @ h + b_hi)
    f_t = sigmoid(W_if @ x + b_if + W_hf @ h + b_hf)
    g_t =    tanh(W_ig @ x + b_ig + W_hg @ h + b_hg)
    o_t = sigmoid(W_io @ x + b_io + W_ho @ h + b_ho)
    c_t = f_t * c_{t-1} + i_t * g_t
    h_t = o_t * tanh(c_t)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


class LstmLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size))
        stdv = hidden_size**-0.5
        for p in self.parameters():
            nn.init.uniform_(p, -stdv, stdv)

    def forward(self, x: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        h_prev, c_prev = state
        H = self.hidden_size
        gates_x = x @ self.weight_ih.t() + self.bias_ih
        gates_h = h_prev @ self.weight_hh.t() + self.bias_hh
        gates = gates_x + gates_h
        i = torch.sigmoid(gates[:, 0 * H : 1 * H])
        f = torch.sigmoid(gates[:, 1 * H : 2 * H])
        g = torch.tanh(gates[:, 2 * H : 3 * H])
        o = torch.sigmoid(gates[:, 3 * H : 4 * H])
        c_new = f * c_prev + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, (h_new, c_new)

    def new_state(self, batch_size: int, device: Any | None = None) -> tuple[Tensor, Tensor]:
        # dtype tracks the parameter dtype so policy.double() / .float() propagates;
        # device defaults to the parameter's device so policy.to(device) propagates
        # naturally (torch.zeros(..., device=None) would silently fall back to CPU).
        target_device = device if device is not None else self.weight_ih.device
        zeros = torch.zeros(batch_size, self.hidden_size, device=target_device, dtype=self.weight_ih.dtype)
        return (zeros, zeros.clone())

    def to_flat(self) -> np.ndarray:
        """Canonical flat order: weight_ih row-major, weight_hh row-major, bias_ih, bias_hh.

        Matches Rust `LayerWeights for LstmLayer::to_flat` in
        src/rust/src/data/neural.rs.
        """
        return np.concatenate(
            [
                self.weight_ih.detach().cpu().numpy().astype(np.float64).ravel(),
                self.weight_hh.detach().cpu().numpy().astype(np.float64).ravel(),
                self.bias_ih.detach().cpu().numpy().astype(np.float64),
                self.bias_hh.detach().cpu().numpy().astype(np.float64),
            ]
        )

    def extra_repr(self) -> str:
        return f"input_size={self.input_size}, hidden_size={self.hidden_size}"
