"""mLSTM cell torch mirror -- matches Rust MlstmLayer bit-for-bit (unbatched).

Single head, d_qk = d_v = H. State: (C [H,H], n [H], m scalar). Association
notes mirrored from Rust: C update uses ig * (v_r * k_col) == ig * torch.outer(v, k);
k is scaled by 1/sqrt(H) at projection time.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

MlstmState = tuple[Tensor, Tensor, Tensor]


class MlstmLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.w_q = nn.Parameter(torch.zeros(hidden_size, input_size))
        self.b_q = nn.Parameter(torch.zeros(hidden_size))
        self.w_k = nn.Parameter(torch.zeros(hidden_size, input_size))
        self.b_k = nn.Parameter(torch.zeros(hidden_size))
        self.w_v = nn.Parameter(torch.zeros(hidden_size, input_size))
        self.b_v = nn.Parameter(torch.zeros(hidden_size))
        self.w_o = nn.Parameter(torch.zeros(hidden_size, input_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))
        self.w_i = nn.Parameter(torch.zeros(input_size))
        self.b_i = nn.Parameter(torch.zeros(()))
        self.w_f = nn.Parameter(torch.zeros(input_size))
        self.b_f = nn.Parameter(torch.zeros(()))

    def forward_unbatched(self, x: Tensor, state: MlstmState) -> tuple[Tensor, MlstmState]:
        c, n, m = state
        hs = self.hidden_size
        q = self.w_q @ x + self.b_q
        k = (self.w_k @ x + self.b_k) / math.sqrt(hs)
        v = self.w_v @ x + self.b_v
        i_pre = torch.dot(self.w_i, x) + self.b_i
        f_pre = torch.dot(self.w_f, x) + self.b_f
        m_new = torch.maximum(f_pre + m, i_pre)
        i_g = torch.exp(i_pre - m_new)
        f_g = torch.exp(f_pre + m - m_new)
        c_new = f_g * c + i_g * torch.outer(v, k)
        n_new = f_g * n + i_g * k
        denom = torch.clamp(torch.abs(torch.dot(n_new, q)), min=1.0)
        o = torch.sigmoid(self.w_o @ x + self.b_o)
        h_new = o * ((c_new @ q) / denom)
        return h_new, (c_new, n_new, m_new)

    def new_state(self) -> MlstmState:
        dt = self.w_q.dtype
        return (
            torch.zeros(self.hidden_size, self.hidden_size, dtype=dt),
            torch.zeros(self.hidden_size, dtype=dt),
            torch.zeros((), dtype=dt),
        )
