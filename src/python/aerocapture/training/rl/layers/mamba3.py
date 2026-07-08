"""Python torch mirror of the Rust Mamba3Layer (PSO-only spike).

Consumed ONLY by the cross-language equivalence test. PSO training goes through
the Rust runtime (flat_weights_to_json + nn_forward); build_layer raises for the
PPO path. The manual _softplus / _expm1_over_x_* helpers are 1-for-1 equivalents
of the Rust free functions (helpers.rs softplus/expm1_over_x, mamba3.rs
expm1_over_x_complex) -- both sides must produce bit-identical f64 output.
See docs/superpowers/specs/2026-07-07-mamba3-ablation-design.md.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


def _softplus(x: Tensor) -> Tensor:
    return x.clamp_min(0.0) + torch.log1p(torch.exp(-x.abs()))


def _expm1_over_x_real(z: Tensor) -> Tensor:
    taylor = 1.0 + 0.5 * z + (z * z) / 6.0
    safe = torch.where(z.abs() < 1e-8, torch.ones_like(z), z)
    exact = torch.expm1(z) / safe
    gate = (z.abs() >= 1e-8).to(z.dtype)
    return taylor + (exact - taylor) * gate


def _expm1_over_x_complex(zr: Tensor, zi: Tensor) -> tuple[Tensor, Tensor]:
    """Mirror of Rust expm1_over_x_complex: same (exp(z)-1)/z form, NOT torch.expm1.

    Explicit (re, im) arithmetic. Taylor fallback for |z| < 1e-8 (magnitude gate).
    """
    mag = torch.sqrt(zr * zr + zi * zi)
    small = mag < 1e-8
    # Taylor 1 + z/2 + z^2/6 ; z^2 = (zr^2 - zi^2) + i(2 zr zi)
    z2r = zr * zr - zi * zi
    z2i = 2.0 * zr * zi
    t_r = 1.0 + 0.5 * zr + z2r / 6.0
    t_i = 0.5 * zi + z2i / 6.0
    # Exact: exp(z) = e^zr (cos zi + i sin zi); (exp(z)-1) / z
    er = torch.exp(zr)
    ez_r = er * torch.cos(zi)
    ez_i = er * torch.sin(zi)
    num_r = ez_r - 1.0
    num_i = ez_i
    denom = torch.where(small, torch.ones_like(zr), zr * zr + zi * zi)
    e_r = (num_r * zr + num_i * zi) / denom
    e_i = (num_i * zr - num_r * zi) / denom
    g = (~small).to(zr.dtype)
    return t_r + (e_r - t_r) * g, t_i + (e_i - t_i) * g


class Mamba3Layer(nn.Module):
    """Mamba-3 ablation layer mirror. `trapezoidal` / `complex` are orthogonal flags.

    State contract (unbatched, for the equivalence gate):
        `new_state()` -> (h_re, h_im, x_prev, b_prev) tuple of zero tensors.
        `forward_unbatched(x, state) -> (y, state)`.

    Canonical flat order (matches Rust LayerWeights for Mamba3Layer::to_flat):
        x_proj_w (dt_rank + 2*d_state, input_size) row-major
        dt_proj_w (input_size, dt_rank)            row-major
        dt_proj_b (input_size,)
        a_log (input_size, d_state)                row-major
        a_imag (input_size, d_state)   row-major   [iff complex]
        lambda_logit (input_size,)                 [iff trapezoidal]
        d_skip (input_size,)
    """

    def __init__(self, input_size: int, d_state: int, dt_rank: int, trapezoidal: bool, complex: bool) -> None:
        super().__init__()
        self.input_size = input_size
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.trapezoidal = trapezoidal
        self.complex = complex
        self.x_proj_w = nn.Parameter(torch.zeros(dt_rank + 2 * d_state, input_size))
        self.dt_proj_w = nn.Parameter(torch.zeros(input_size, dt_rank))
        self.dt_proj_b = nn.Parameter(torch.zeros(input_size))
        self.a_log = nn.Parameter(torch.zeros(input_size, d_state))
        self.a_imag = nn.Parameter(torch.zeros(input_size, d_state)) if complex else None
        self.lambda_logit = nn.Parameter(torch.zeros(input_size)) if trapezoidal else None
        self.d_skip = nn.Parameter(torch.zeros(input_size))

    def new_state(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        dt = self.x_proj_w.dtype
        return (
            torch.zeros(self.input_size, self.d_state, dtype=dt),
            torch.zeros(self.input_size, self.d_state, dtype=dt),
            torch.zeros(self.input_size, dtype=dt),
            torch.zeros(self.d_state, dtype=dt),
        )

    def forward_unbatched(self, x: Tensor, state: tuple[Tensor, Tensor, Tensor, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:
        h_re, h_im, x_prev, b_prev = state
        proj = self.x_proj_w @ x
        dt_pre = proj[: self.dt_rank]
        b_vec = proj[self.dt_rank : self.dt_rank + self.d_state]
        c_vec = proj[self.dt_rank + self.d_state : self.dt_rank + 2 * self.d_state]
        delta = _softplus(self.dt_proj_w @ dt_pre + self.dt_proj_b)  # (input_size,)
        lam = torch.sigmoid(self.lambda_logit) if self.lambda_logit is not None else torch.ones(self.input_size, dtype=x.dtype)

        ar = -torch.exp(self.a_log)  # (in, N)
        za_r = delta.unsqueeze(1) * ar  # (in, N)
        if self.complex:
            assert self.a_imag is not None
            za_i = delta.unsqueeze(1) * self.a_imag
            r = torch.exp(za_r)
            alpha_r, alpha_i = r * torch.cos(za_i), r * torch.sin(za_i)
            ex_r, ex_i = _expm1_over_x_complex(za_r, za_i)
            bb_r = delta.unsqueeze(1) * b_vec.unsqueeze(0) * ex_r
            bb_i = delta.unsqueeze(1) * b_vec.unsqueeze(0) * ex_i
            nr = alpha_r * h_re - alpha_i * h_im + lam.unsqueeze(1) * bb_r * x.unsqueeze(1)
            ni = alpha_r * h_im + alpha_i * h_re + lam.unsqueeze(1) * bb_i * x.unsqueeze(1)
            if self.trapezoidal:
                cross = (1.0 - lam).unsqueeze(1) * delta.unsqueeze(1) * b_prev.unsqueeze(0) * x_prev.unsqueeze(1)
                nr = nr + alpha_r * cross
                ni = ni + alpha_i * cross
            h_re, h_im = nr, ni
            y = (nr * c_vec.unsqueeze(0)).sum(dim=1) + self.d_skip * x
        else:
            alpha = torch.exp(za_r)
            bb = delta.unsqueeze(1) * b_vec.unsqueeze(0) * _expm1_over_x_real(za_r)
            nr = alpha * h_re + lam.unsqueeze(1) * bb * x.unsqueeze(1)
            if self.trapezoidal:
                nr = nr + (1.0 - lam).unsqueeze(1) * delta.unsqueeze(1) * alpha * b_prev.unsqueeze(0) * x_prev.unsqueeze(1)
            h_re = nr
            y = (nr * c_vec.unsqueeze(0)).sum(dim=1) + self.d_skip * x
        if self.trapezoidal:
            x_prev, b_prev = x.clone(), b_vec.detach().clone()
        return y, (h_re, h_im, x_prev, b_prev)

    def to_flat(self) -> np.ndarray:
        parts = [self.x_proj_w, self.dt_proj_w, self.dt_proj_b, self.a_log]
        if self.complex:
            assert self.a_imag is not None
            parts.append(self.a_imag)
        if self.trapezoidal:
            assert self.lambda_logit is not None
            parts.append(self.lambda_logit)
        parts.append(self.d_skip)
        return np.concatenate([p.detach().cpu().numpy().astype(np.float64).ravel() for p in parts])

    def from_flat(self, slab: np.ndarray) -> None:
        c = 0

        def take(param: nn.Parameter, shape: tuple[int, ...]) -> None:
            nonlocal c
            n = int(np.prod(shape))
            with torch.no_grad():
                param.copy_(torch.from_numpy(np.ascontiguousarray(slab[c : c + n]).reshape(shape)).to(param.dtype))
            c += n

        take(self.x_proj_w, (self.dt_rank + 2 * self.d_state, self.input_size))
        take(self.dt_proj_w, (self.input_size, self.dt_rank))
        take(self.dt_proj_b, (self.input_size,))
        take(self.a_log, (self.input_size, self.d_state))
        if self.complex:
            assert self.a_imag is not None
            take(self.a_imag, (self.input_size, self.d_state))
        if self.trapezoidal:
            assert self.lambda_logit is not None
            take(self.lambda_logit, (self.input_size,))
        take(self.d_skip, (self.input_size,))
