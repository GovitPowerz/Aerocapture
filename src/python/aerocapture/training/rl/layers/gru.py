"""GRU cell matching nn.GRUCell + Rust GruLayer bit-for-bit.

Canonical flat weight order (LayerWeights trait + PSO chromosome):
    weight_ih row-major, weight_hh row-major, bias_ih, bias_hh.

Forward equations (PyTorch nn.GRUCell convention):
    r_t = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)
    z_t = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)
    n_t = tanh(W_in @ x + b_in + r_t * (W_hn @ h + b_hn))
    h_t = (1 - z_t) * n_t + z_t * h_{t-1}
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class GruLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size))
        stdv = hidden_size**-0.5
        for p in self.parameters():
            nn.init.uniform_(p, -stdv, stdv)

    def forward(self, x: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        H = self.hidden_size
        gates_x = x @ self.weight_ih.t() + self.bias_ih
        gates_h = h @ self.weight_hh.t() + self.bias_hh
        r = torch.sigmoid(gates_x[:, :H] + gates_h[:, :H])
        z = torch.sigmoid(gates_x[:, H : 2 * H] + gates_h[:, H : 2 * H])
        n = torch.tanh(gates_x[:, 2 * H : 3 * H] + r * gates_h[:, 2 * H : 3 * H])
        h_new = (1 - z) * n + z * h
        return h_new, h_new

    def new_state(self, batch_size: int, device: Any) -> Tensor:
        # dtype tracks the parameter dtype so policy.double() / .float() propagates.
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=self.weight_ih.dtype)

    def extra_repr(self) -> str:
        return f"input_size={self.input_size}, hidden_size={self.hidden_size}"
