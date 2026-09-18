"""sLSTM cell torch mirror -- matches Rust SlstmLayer bit-for-bit (unbatched).

Gate order (i, f, z, o) on the 4H axis; single bias; stabilized exponential
gating. State tuple: (h, c, n, m). Preactivation add order matches Rust:
(weight_ih @ x + bias) + weight_hh @ h.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

SlstmState = tuple[Tensor, Tensor, Tensor, Tensor]


class SlstmLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.zeros(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.zeros(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))

    def forward_unbatched(self, x: Tensor, state: SlstmState) -> tuple[Tensor, SlstmState]:
        h, c, n, m = state
        pre = (self.weight_ih @ x + self.bias) + self.weight_hh @ h
        hs = self.hidden_size
        i_pre = pre[:hs]
        f_pre = pre[hs : 2 * hs]
        z = torch.tanh(pre[2 * hs : 3 * hs])
        o = torch.sigmoid(pre[3 * hs : 4 * hs])
        m_new = torch.maximum(f_pre + m, i_pre)
        i_g = torch.exp(i_pre - m_new)
        f_g = torch.exp(f_pre + m - m_new)
        c_new = f_g * c + i_g * z
        n_new = f_g * n + i_g
        h_new = o * (c_new / n_new)
        return h_new, (h_new, c_new, n_new, m_new)

    def new_state(self) -> SlstmState:
        z = lambda: torch.zeros(self.hidden_size, dtype=self.weight_ih.dtype)  # noqa: E731
        return (z(), z(), z(), z())
