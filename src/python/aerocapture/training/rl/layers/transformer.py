"""TransformerLayer (PyTorch mirror of the Rust implementation).

Cross-language contract (enforced by
tests/test_rust_python_transformer_equivalence.py):

- LayerNorm uses biased (1/N) variance with eps=1e-5 (torch.nn.LayerNorm default).
- GELU is the exact form: 0.5 * x * (1 + erf(x / sqrt(2))), via torch.special.erf.
- Softmax uses max-subtraction over the cache time axis.
- Multi-head split is a contiguous slice along d_model: head h -> [h*d_head .. (h+1)*d_head].
- Positional encoding is relative-to-buffer: newest token at slot cache_len - 1.
- PE offsets for K/V are computed at forward time as (w_k.weight @ pe_table[:cache_len].T).T;
  no bias is included in the PE shift. Matches Rust's precomputed k_pe_offsets
  modulo iteration order (< 1e-10 tolerance, target machine epsilon).

Note: constructible via `build_layer(TransformerSpec)` (used by warm-start BPTT
and the cross-language equivalence test). The PPO runtime gate has moved to
`rl/train.py::_derive_hidden_shapes`; PSO bypasses this module entirely and
drives the Rust runtime via aerocapture_rs.nn_forward.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

_INV_SQRT2: float = 1.0 / math.sqrt(2.0)


def _build_sinusoidal_pe(n_seq: int, d_model: int) -> Tensor:
    """Match Rust build_pe_table iteration order: pos outer, i inner.

    Explicit f64 loop -- no broadcast / arange fusion -- so operand ordering
    matches the Rust sequential implementation.
    """
    pe = torch.zeros(n_seq, d_model, dtype=torch.float64)
    for pos in range(n_seq):
        for i in range(d_model):
            k = i // 2
            div = 10000.0 ** ((2.0 * k) / d_model)
            angle = pos / div
            pe[pos, i] = math.sin(angle) if i % 2 == 0 else math.cos(angle)
    return pe


def _manual_ln(x: Tensor, gamma: Tensor, beta: Tensor, eps: float) -> Tensor:
    # x: (batch, d_model)
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)  # biased (1/N)
    return (x - mean) / torch.sqrt(var + eps) * gamma + beta


def _manual_causal_attention(
    q: Tensor,  # (batch, d_model)
    k_eff: Tensor,  # (batch, cache_len, d_model)
    v_eff: Tensor,  # (batch, cache_len, d_model)
    n_heads: int,
    d_head: int,
) -> Tensor:
    batch, cache_len, d_model = k_eff.shape
    q_h = q.view(batch, n_heads, d_head)  # (batch, n_heads, d_head)
    k_h = k_eff.view(batch, cache_len, n_heads, d_head)
    v_h = v_eff.view(batch, cache_len, n_heads, d_head)
    inv_sqrt_d = 1.0 / math.sqrt(d_head)

    # scores: (batch, n_heads, cache_len)
    scores = torch.einsum("bhd,bihd->bhi", q_h, k_h) * inv_sqrt_d
    max_scores, _ = scores.max(dim=-1, keepdim=True)
    exp_scores = torch.exp(scores - max_scores)
    weights = exp_scores / exp_scores.sum(dim=-1, keepdim=True)

    head_out = torch.einsum("bhi,bihd->bhd", weights, v_h)  # (batch, n_heads, d_head)
    return head_out.reshape(batch, n_heads * d_head)  # (batch, d_model)


class TransformerLayer(nn.Module):
    """Manual 1-layer Transformer block for 1-for-1 Rust equivalence."""

    pe_table: Tensor  # registered buffer; mypy needs explicit annotation

    def __init__(self, d_model: int, n_heads: int, d_ffn: int, n_seq: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_ffn = d_ffn
        self.n_seq = n_seq

        self.w_q = nn.Linear(d_model, d_model, bias=True)
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)
        self.w_o = nn.Linear(d_model, d_model, bias=True)

        self.w_ffn1 = nn.Linear(d_model, d_ffn, bias=True)
        self.w_ffn2 = nn.Linear(d_ffn, d_model, bias=True)

        self.ln1_gamma = nn.Parameter(torch.ones(d_model))
        self.ln1_beta = nn.Parameter(torch.zeros(d_model))
        self.ln2_gamma = nn.Parameter(torch.ones(d_model))
        self.ln2_beta = nn.Parameter(torch.zeros(d_model))

        self.register_buffer(
            "pe_table",
            _build_sinusoidal_pe(n_seq, d_model),
            persistent=False,
        )

    def forward(
        self,
        x: Tensor,  # (batch, d_model)
        state: tuple[Tensor, Tensor],  # (k_cache, v_cache)
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        # 1. LN1
        x_norm1 = _manual_ln(x, self.ln1_gamma, self.ln1_beta, eps=1e-5)
        # 2. QKV
        q = self.w_q(x_norm1)
        k = self.w_k(x_norm1)
        v = self.w_v(x_norm1)
        # 3. Push, evict
        k_cache, v_cache = state
        k_cache = torch.cat([k_cache, k.unsqueeze(1)], dim=1)
        v_cache = torch.cat([v_cache, v.unsqueeze(1)], dim=1)
        if k_cache.shape[1] > self.n_seq:
            k_cache = k_cache[:, 1:]
            v_cache = v_cache[:, 1:]
        cache_len = k_cache.shape[1]
        # 4. PE offsets, relative-to-buffer
        pe_slice = self.pe_table[:cache_len].to(dtype=x.dtype, device=x.device)
        k_pe = (self.w_k.weight @ pe_slice.T).T  # (cache_len, d_model) -- no bias in shift
        v_pe = (self.w_v.weight @ pe_slice.T).T
        k_eff = k_cache + k_pe.unsqueeze(0)
        v_eff = v_cache + v_pe.unsqueeze(0)
        # 5. Attention + residual
        attn_out = _manual_causal_attention(q, k_eff, v_eff, self.n_heads, self.d_head)
        x1 = x + self.w_o(attn_out)
        # 6. LN2 + FFN + residual
        x_norm2 = _manual_ln(x1, self.ln2_gamma, self.ln2_beta, eps=1e-5)
        ffn_hidden = self.w_ffn1(x_norm2)
        ffn_hidden_act = 0.5 * ffn_hidden * (1.0 + torch.special.erf(ffn_hidden * _INV_SQRT2))
        ffn_out = self.w_ffn2(ffn_hidden_act)
        out = x1 + ffn_out
        return out, (k_cache, v_cache)

    def new_state(self, batch_size: int, device: Any | None = None) -> tuple[Tensor, Tensor]:
        target_device = device if device is not None else self.w_q.weight.device
        dtype = self.w_q.weight.dtype
        empty = torch.zeros(batch_size, 0, self.d_model, device=target_device, dtype=dtype)
        return (empty.clone(), empty.clone())

    def to_flat(self) -> np.ndarray:
        """Canonical flat order matching Rust LayerWeights<TransformerLayer>::to_flat:

            w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o,
            w_ffn1, b_ffn1, w_ffn2, b_ffn2,
            ln1_gamma, ln1_beta, ln2_gamma, ln2_beta

        All 2D weights row-major. PyTorch nn.Linear stores weight as
        [out, in] row-major which matches the Rust serialization byte-for-byte.
        """
        parts: list[np.ndarray] = []
        for linear in (self.w_q, self.w_k, self.w_v, self.w_o, self.w_ffn1, self.w_ffn2):
            parts.append(linear.weight.detach().cpu().numpy().astype(np.float64).ravel())
            parts.append(linear.bias.detach().cpu().numpy().astype(np.float64))
        for ln in (self.ln1_gamma, self.ln1_beta, self.ln2_gamma, self.ln2_beta):
            parts.append(ln.detach().cpu().numpy().astype(np.float64))
        return np.concatenate(parts)

    def from_flat(self, slab: np.ndarray) -> None:
        """Load a flat slab in-place, mirroring Rust LayerWeights<TransformerLayer>::from_flat.

        Flat order:
            w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o,
            w_ffn1, b_ffn1, w_ffn2, b_ffn2,
            ln1_gamma, ln1_beta, ln2_gamma, ln2_beta

        PE offsets are recomputed lazily on forward (w_k/w_v are already
        the authoritative values after this load); no explicit rebuild needed.
        """
        d = self.d_model
        f = self.d_ffn
        c = 0

        def _copy_param(param: torch.Tensor, src: np.ndarray) -> None:
            param.copy_(torch.from_numpy(np.ascontiguousarray(src)).to(param.dtype))

        def _copy_linear(linear: torch.nn.Linear, n_out: int, n_in: int) -> None:
            nonlocal c
            n_w = n_out * n_in
            _copy_param(linear.weight, slab[c : c + n_w].reshape(n_out, n_in))
            c += n_w
            _copy_param(linear.bias, slab[c : c + n_out])
            c += n_out

        with torch.no_grad():
            # Q/K/V/O projections: each (d_model x d_model) weight + (d_model,) bias
            _copy_linear(self.w_q, d, d)
            _copy_linear(self.w_k, d, d)
            _copy_linear(self.w_v, d, d)
            _copy_linear(self.w_o, d, d)
            # FFN1: (d_ffn x d_model) + (d_ffn,)
            _copy_linear(self.w_ffn1, f, d)
            # FFN2: (d_model x d_ffn) + (d_model,)
            _copy_linear(self.w_ffn2, d, f)
            # Layer norms: (gamma, beta) x 2, each (d_model,)
            _copy_param(self.ln1_gamma, slab[c : c + d])
            c += d
            _copy_param(self.ln1_beta, slab[c : c + d])
            c += d
            _copy_param(self.ln2_gamma, slab[c : c + d])
            c += d
            _copy_param(self.ln2_beta, slab[c : c + d])
