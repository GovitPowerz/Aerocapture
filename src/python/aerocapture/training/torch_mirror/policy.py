"""Step-wise stateful policy mirroring the Rust `NeuralNetModel` contract.

`V2Policy` iterates the v2 layer list with per-layer state: forward is
`(x_t, state_t-1) -> (y_t, state_t)`; BPTT over sequences is an explicit loop.
Consumed by the supervised warm-start (`forward_seq_means`) and the PPO trainer
(`evaluate`, `sample`). `log_std` is exploration noise only, never exported.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from aerocapture.training.torch_mirror.layers import build_layer
from aerocapture.training.torch_mirror.schemas import LayerSpec


def _zero_state_where_done(state: list[Any], done_mask: Tensor) -> list[Any]:
    """Return a new state list where the entries for `done_mask` envs are zeroed.

    Contract:
      - `None` entries (dense / stateless layers) pass through unchanged.
      - Tensor entries of shape `(B, *)` get `done_mask` rows multiplied by 0.
      - Tuple entries (e.g. LSTM `(h, c)`) recurse into each element.
        Recursion terminates at the Tensor branch.
      - Any other non-None, non-Tensor, non-tuple entry raises TypeError.
        Future multi-tensor state types that aren't expressible as a tuple
        (e.g. Mamba's SSM state dict, Transformer KV cache) must add an
        explicit branch here rather than silently matmul-erroring downstream.
    """
    keep_bool = (~done_mask).unsqueeze(-1)  # (B, 1), bool
    return [_zero_entry(s, keep_bool) for s in state]


def _zero_entry(s: Any, keep_bool: Tensor) -> Any:
    if s is None:
        return None
    if isinstance(s, Tensor):
        if s.ndim < 2:
            raise ValueError(f"_zero_entry: expected Tensor with ndim >= 2 (batch dim + state dims), got ndim={s.ndim}")
        # keep_bool starts as (B, 1); reshape to (B, 1, 1, ..., 1) to broadcast
        # against any trailing dims (Mamba 3D, Window 3D, Transformer KV-cache 3D, ...).
        extra = s.ndim - keep_bool.ndim
        if extra > 0:
            shape = keep_bool.shape + (1,) * extra
            broadcast = keep_bool.view(shape)
        else:
            broadcast = keep_bool
        return s * broadcast.to(dtype=s.dtype, device=s.device)
    if isinstance(s, tuple):
        return tuple(_zero_entry(sub, keep_bool) for sub in s)
    raise TypeError(
        f"_zero_state_where_done: unsupported state entry type {type(s).__name__!r}; "
        "only None, Tensor, or tuple supported. Non-tuple multi-tensor states "
        "(e.g. Mamba SSM, Transformer KV cache) need an explicit extension."
    )


class V2Policy(nn.Module):
    """Step-wise stateful policy matching the Rust NeuralNetModel contract.

    Forward pass: `(x_t, state_t-1) -> (y_t, state_t)`. BPTT over sequences
    is an explicit Python loop in the training code.

    The final layer produces 2 outputs; bank is `atan2(out[0], out[1])`.
    log_std is a separate learnable parameter (not a layer, not exported to JSON)
    used only for PPO/SAC exploration noise.
    """

    def __init__(
        self,
        architecture: Sequence[LayerSpec],
        input_mask: list[int] | None,
        initial_log_std: float = 0.0,
        min_log_std: float = -2.0,
        max_log_std: float = 2.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([build_layer(spec) for spec in architecture])
        self.input_mask = input_mask
        self.log_std = nn.Parameter(torch.full((2,), initial_log_std))
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x: Tensor, state: list[Any]) -> tuple[Tensor, list[Any]]:
        new_state: list[Any] = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            x, new_state[i] = layer(x, state[i])
        return x, new_state

    def new_state(self, batch_size: int, device: object) -> list[Any]:
        # Each layer defines its own new_state() so Phase 1+ layer types
        # (gru/lstm/window/ssm) plug in without touching this method.
        # nn.ModuleList typing is `Module | Tensor`; all our layer modules
        # implement new_state by contract (see layers/ subpackage).
        return [layer.new_state(batch_size, device) for layer in self.layers]  # type: ignore[union-attr,operator]

    def forward_mean_logstd(self, obs: Tensor, state: list[Any]) -> tuple[Tensor, Tensor, list[Any]]:
        """Single-step forward producing policy mean + log_std + new state.

        Mirrors GaussianPolicy.forward_mean_logstd but threads per-layer state.
        The final layer's output becomes `mean`. `log_std` is the state-independent
        parameter clamped at `min_log_std`.
        """
        mean, new_state = self.forward(obs, state)
        log_std = self.log_std.clamp(min=self.min_log_std, max=self.max_log_std)
        return mean, log_std, new_state

    def sample(self, obs: Tensor, state: list[Any]) -> tuple[Tensor, Tensor, Tensor, list[Any]]:
        """Reparameterized sample. Returns (bank, raw, log_prob, new_state).

        `raw` is the unconstrained 2D Gaussian sample, `bank` is
        `atan2(raw[..., 0], raw[..., 1])`, `log_prob` is the Normal density at
        `raw` summed over the action axis.
        """
        mean, log_std, new_state = self.forward_mean_logstd(obs, state)
        std = log_std.exp()
        eps = torch.randn_like(mean)
        raw = mean + std * eps
        bank = torch.atan2(raw[..., 0], raw[..., 1])
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(raw).sum(-1)
        return bank, raw, log_prob, new_state

    def evaluate(
        self,
        obs_seq: Tensor,
        state_0: list[Any],
        dones_seq: Tensor,
        raw_seq: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """BPTT forward over a time chunk.

        Inputs:
            obs_seq    (T, B, obs_dim)
            state_0    list of per-layer state tensors; caller is responsible for
                       `.detach()` before passing in when crossing chunk boundaries.
            dones_seq  (T, B) bool. When True at time t, the per-env state entering
                       step t+1 is zeroed (matches the Rust auto-reset + collect-loop
                       behavior).
            raw_seq    (T, B, action_dim). The raw Gaussian samples whose log_prob
                       we are re-evaluating under the current policy.

        Returns:
            log_probs_seq (T, B)
            entropy_seq   (T, B)
        """
        T = obs_seq.shape[0]
        log_probs = []
        entropies = []
        state = state_0
        log_std = self.log_std.clamp(min=self.min_log_std, max=self.max_log_std)
        std = log_std.exp()
        for t in range(T):
            mean, state = self.forward(obs_seq[t], state)
            dist = torch.distributions.Normal(mean, std)
            lp = dist.log_prob(raw_seq[t]).sum(-1)
            ent = dist.entropy().sum(-1)
            log_probs.append(lp)
            entropies.append(ent)
            # Zero the state per-env when done[t] is True, before next step.
            if t + 1 < T:
                done_mask = dones_seq[t]  # (B,)
                if done_mask.any():
                    state = _zero_state_where_done(state, done_mask)
        return torch.stack(log_probs, dim=0), torch.stack(entropies, dim=0)

    def forward_seq_means(
        self,
        obs_seq: Tensor,
        state_0: list[Any],
        dones_seq: Tensor,
    ) -> Tensor:
        """Supervised-warm-start forward over a time chunk.

        Returns the layer-stack final output (the policy mean) per step. Used by
        the chunked-BPTT supervised pretraining loop in warm_start.py; this is the
        autograd-friendly mirror of `evaluate` minus the Gaussian log-prob math.

        Args:
            obs_seq:   (T, B, obs_dim)
            state_0:   list of per-layer state tensors. Caller is responsible for
                       `.detach()` before passing across chunk boundaries.
            dones_seq: (T, B) bool. When True at time t, the per-env state
                       entering step t+1 is zeroed.

        Returns:
            means: (T, B, out_dim)
        """
        T = obs_seq.shape[0]
        means_list: list[Tensor] = []
        state = state_0
        for t in range(T):
            mean, state = self.forward(obs_seq[t], state)
            means_list.append(mean)
            if t + 1 < T:
                done_mask = dones_seq[t]
                if done_mask.any():
                    state = _zero_state_where_done(state, done_mask)
        return torch.stack(means_list, dim=0)
