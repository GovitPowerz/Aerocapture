"""CfC cell torch mirror -- matches Rust CfcLayer bit-for-bit (unbatched).

Used ONLY by the cross-language equivalence test; the PPO path rejects CfC
(build_layer raises). Math must track src/rust/src/data/neural/layers/cfc.rs
exactly: lecun_tanh constant order, sigmoid(-t_a * CFC_DT + t_b) gate, and
(1 - g) * ff1 + g * ff2 blend.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

CFC_DT = 1.0


class CfcLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, backbone_units: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.backbone_units = backbone_units
        cat = input_size + hidden_size
        self.w_bb = nn.Parameter(torch.zeros(backbone_units, cat))
        self.b_bb = nn.Parameter(torch.zeros(backbone_units))
        self.w_ff1 = nn.Parameter(torch.zeros(hidden_size, backbone_units))
        self.b_ff1 = nn.Parameter(torch.zeros(hidden_size))
        self.w_ff2 = nn.Parameter(torch.zeros(hidden_size, backbone_units))
        self.b_ff2 = nn.Parameter(torch.zeros(hidden_size))
        self.w_ta = nn.Parameter(torch.zeros(hidden_size, backbone_units))
        self.b_ta = nn.Parameter(torch.zeros(hidden_size))
        self.w_tb = nn.Parameter(torch.zeros(hidden_size, backbone_units))
        self.b_tb = nn.Parameter(torch.zeros(hidden_size))

    def forward_unbatched(self, x: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        cat = torch.cat([x, h])
        xb = 1.7159 * torch.tanh(2.0 * (self.w_bb @ cat + self.b_bb) / 3.0)
        ff1 = torch.tanh(self.w_ff1 @ xb + self.b_ff1)
        ff2 = torch.tanh(self.w_ff2 @ xb + self.b_ff2)
        t_a = self.w_ta @ xb + self.b_ta
        t_b = self.w_tb @ xb + self.b_tb
        g = torch.sigmoid(-t_a * CFC_DT + t_b)
        h_new = (1.0 - g) * ff1 + g * ff2
        return h_new, h_new

    def new_state(self) -> Tensor:
        return torch.zeros(self.hidden_size, dtype=self.w_bb.dtype)
