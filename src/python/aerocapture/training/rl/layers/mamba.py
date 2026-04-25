"""Python torch mirror of the Rust MambaLayer (Phase 4a, PSO-only).

Consumed exclusively by the cross-language equivalence test and (in Phase 4b)
the PPO training path. PSO training bypasses this module entirely -- it goes
through `aerocapture_rs.flat_weights_to_json` + the Rust forward runtime.

The manual `_softplus` / `_expm1_over_x` helpers are 1-for-1 equivalents of the
Rust `pub(crate)` free functions in `src/rust/src/data/neural.rs`. Both sides
must produce bit-identical f64 output (verified by Task 14's equivalence test).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _softplus(x: Tensor) -> Tensor:
    """Numerically stable softplus matching Rust `softplus` bit-for-bit.

    NOT `torch.nn.functional.softplus`, which has a `threshold=20` linear-branch
    fallback that would break equivalence at |x| > 20.
    """
    return x.clamp_min(0.0) + torch.log1p(torch.exp(-x.abs()))


def _expm1_over_x(z: Tensor) -> Tensor:
    """(exp(z) - 1) / z with Taylor fallback for |z| < 1e-8.

    Matches Rust `expm1_over_x` bit-for-bit in the forward pass.

    Implementation note: we compute `taylor + (exact - taylor) * gate`
    rather than `torch.where(gate, taylor, exact)`. The plain-where form
    suffers from the classic double-backward NaN pitfall: PyTorch still
    propagates gradients through the unselected `exact = expm1(z) / z`
    branch, and at z=0 that produces 0/0 in the backward pass. Computing
    the blend as an additive mix preserves gradients cleanly (exact is
    evaluated on a non-zero safe_z, and its contribution is gated to 0
    whenever |z| < 1e-8). Branch point: `|z| < 1e-8` is strict, so at
    exactly |z| == 1e-8 the exact branch wins.
    """
    taylor = 1.0 + 0.5 * z + (z * z) / 6.0
    safe_z = torch.where(z.abs() < 1e-8, torch.ones_like(z), z)
    exact = torch.expm1(z) / safe_z
    gate = (z.abs() >= 1e-8).to(z.dtype)
    return taylor + (exact - taylor) * gate


class MambaLayer(nn.Module):
    """Selective SSM core (Mamba S6) -- PSO-only in Phase 4a.

    Parameters:
        input_size: d_inner; layer fan-in = fan-out.
        d_state:    SSM state dim per channel (N in paper).
        dt_rank:    Bottleneck rank for the Δ projection.

    State contract:
        `new_state()` -> zero-initialized `Tensor` of shape (input_size, d_state),
        dtype tracks parameter dtype (so `policy.double()` propagates).
        `forward(x, h) -> (y, h_new)` where `x: (input_size,)` and `h: (input_size, d_state)`.

    Canonical parameter order (matches Rust `LayerWeights for MambaLayer::to_flat`):
      x_proj_w (dt_rank + 2*d_state, input_size) row-major
      dt_proj_w (input_size, dt_rank)             row-major
      dt_proj_b (input_size,)
      a_log (input_size, d_state)                 row-major
      d_skip (input_size,)
    """

    def __init__(self, input_size: int, d_state: int, dt_rank: int) -> None:
        super().__init__()
        if input_size <= 0 or d_state <= 0 or dt_rank <= 0:
            raise ValueError(f"MambaLayer: all dims must be positive; got input_size={input_size}, d_state={d_state}, dt_rank={dt_rank}")
        if dt_rank > input_size:
            raise ValueError(f"MambaLayer: dt_rank ({dt_rank}) must be <= input_size ({input_size})")
        self.input_size = input_size
        self.d_state = d_state
        self.dt_rank = dt_rank

        self.x_proj_w = nn.Parameter(torch.zeros(dt_rank + 2 * d_state, input_size))
        self.dt_proj_w = nn.Parameter(torch.zeros(input_size, dt_rank))
        self.dt_proj_b = nn.Parameter(torch.zeros(input_size))
        self.a_log = nn.Parameter(torch.zeros(input_size, d_state))
        self.d_skip = nn.Parameter(torch.zeros(input_size))

    def new_state(self) -> Tensor:
        """Return a zero-initialized state tensor with parameter dtype / device."""
        return torch.zeros(
            self.input_size,
            self.d_state,
            dtype=self.x_proj_w.dtype,
            device=self.x_proj_w.device,
        )

    def forward(self, x: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        """Single-step forward.

        Args:
            x: (input_size,) input vector.
            h: (input_size, d_state) current state.

        Returns:
            y: (input_size,) output vector.
            h_new: (input_size, d_state) updated state.
        """
        assert x.shape == (self.input_size,), f"x shape {x.shape} != ({self.input_size},)"
        assert h.shape == (self.input_size, self.d_state), f"h shape {h.shape} != ({self.input_size}, {self.d_state})"

        # 1. Fused x_proj -> split into (dt_pre, B, C)
        proj = self.x_proj_w @ x
        dt_pre = proj[: self.dt_rank]
        b_vec = proj[self.dt_rank : self.dt_rank + self.d_state]
        c_vec = proj[self.dt_rank + self.d_state : self.dt_rank + 2 * self.d_state]

        # 2. dt_proj + softplus -> per-channel positive delta
        dt_lifted = self.dt_proj_w @ dt_pre + self.dt_proj_b
        delta = _softplus(dt_lifted)

        # 3. ZOH discretization + state update (fully vectorized over (d, n))
        a = -torch.exp(self.a_log)  # (input_size, d_state), A < 0
        za = delta.unsqueeze(1) * a  # (input_size, d_state)
        a_bar = torch.exp(za)
        b_bar = delta.unsqueeze(1) * b_vec.unsqueeze(0) * _expm1_over_x(za)
        h_new = a_bar * h + b_bar * x.unsqueeze(1)
        y = h_new @ c_vec + self.d_skip * x
        return y, h_new
