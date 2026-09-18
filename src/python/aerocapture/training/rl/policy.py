"""PPO/SAC-side policy modules.

`GaussianPolicy` is the v1 dense MLP (`layer_sizes` / `activations` format,
deterministic bank = atan2(out[0], out[1])), `ValueNetwork` the PPO critic, and
`np_state_to_torch` / `torch_state_to_np` the rollout-buffer state packing. The
stateful v2 policy lives in `aerocapture.training.torch_mirror.policy`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class _Asinh(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.asinh(x)


_ACT: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "linear": nn.Identity,
    "identity": nn.Identity,
    "swish": nn.SiLU,
    "mish": nn.Mish,
    "asinh": _Asinh,
}


def build_mlp(input_dim: int, layer_sizes: Sequence[int], activations: Sequence[str]) -> nn.Sequential:
    if len(layer_sizes) != len(activations):
        raise ValueError(f"len(layer_sizes)={len(layer_sizes)} must equal len(activations)={len(activations)}")
    layers: list[nn.Module] = []
    prev = input_dim
    for size, act in zip(layer_sizes, activations, strict=True):
        layers.append(nn.Linear(prev, size))
        layers.append(_ACT[act]())
        prev = size
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """PPO policy: deterministic MLP + state-independent log_std.

    Output is a pair (out0, out1); deterministic bank = atan2(out0, out1).
    Stochastic sampling is on (out0, out1) in unconstrained space.
    """

    def __init__(
        self,
        input_dim: int,
        layer_sizes: Sequence[int],
        activations: Sequence[str],
        initial_log_std: float = -0.5,
        min_log_std: float = -2.0,
        max_log_std: float = 2.0,
    ) -> None:
        super().__init__()
        if layer_sizes[-1] != 2:
            raise ValueError(f"GaussianPolicy requires out_dim=2 (atan2), got {layer_sizes[-1]}")
        self.trunk = build_mlp(input_dim, layer_sizes, activations)
        self.log_std = nn.Parameter(torch.full((2,), initial_log_std))
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward_mean_logstd(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.trunk(obs)
        return mean, self.log_std.clamp(min=self.min_log_std, max=self.max_log_std)

    def deterministic_bank(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward_mean_logstd(obs)
        return torch.atan2(mean[..., 0], mean[..., 1])

    def load_weights_from_json(self, path: Path) -> None:
        """Load weights from a NeuralNetModel JSON file into the trunk."""
        with path.open() as f:
            doc = json.load(f)
        linear_idx = 0
        for module in self.trunk:
            if isinstance(module, nn.Linear):
                lw = doc["weights"][f"layer_{linear_idx}"]
                module.weight.data = torch.tensor(np.array(lw["w"], dtype=np.float64), dtype=torch.float32)
                module.bias.data = torch.tensor(np.array(lw["b"], dtype=np.float64), dtype=torch.float32)
                linear_idx += 1

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reparameterized sample; return (bank, raw, log_prob).

        The raw 2D latent is the true action space of the Gaussian. The bank
        angle `atan2(raw[0], raw[1])` is what the environment sees. Returning
        `raw` makes SAC's critic and entropy objective consistent with the
        density the policy actually regularizes (target_entropy = -2).
        """
        mean, log_std = self.forward_mean_logstd(obs)
        std = log_std.exp()
        eps = torch.randn_like(mean)
        raw = mean + std * eps
        bank = torch.atan2(raw[..., 0], raw[..., 1])
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(raw).sum(-1)
        return bank, raw, log_prob


def np_state_to_torch(np_state: list[Any]) -> list[Any]:
    """Convert per-layer numpy state list to torch tensors (no grad).

    GRU: s is (B, H) ndarray -> Tensor(B, H).
    LSTM: s is (B, 2, H) ndarray -> tuple(Tensor(B, H), Tensor(B, H)).
    """
    out: list[Any] = []
    for s in np_state:
        if s is None:
            out.append(None)
        elif s.ndim == 3:
            t = torch.from_numpy(s).float()
            out.append((t[:, 0, :], t[:, 1, :]))
        else:
            out.append(torch.from_numpy(s).float())
    return out


def torch_state_to_np(torch_state: list[Any]) -> list[Any]:
    """Convert per-layer torch state list back to numpy (no grad).

    GRU: Tensor(B, H) -> (B, H) ndarray.
    LSTM: tuple(Tensor(B, H), Tensor(B, H)) -> (B, 2, H) ndarray.
    """
    out: list[Any] = []
    for s in torch_state:
        if s is None:
            out.append(None)
        elif isinstance(s, tuple):
            h, c = s
            stacked = torch.stack([h, c], dim=1)
            out.append(stacked.detach().cpu().numpy().astype(np.float32))
        else:
            out.append(s.detach().cpu().numpy().astype(np.float32))
    return out


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], activations: Sequence[str]) -> None:
        super().__init__()
        layer_sizes = list(hidden_sizes) + [1]
        act_list = list(activations[:-1]) + ["linear"]
        self.net = build_mlp(input_dim, layer_sizes, act_list)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(obs).squeeze(-1)
        return out
