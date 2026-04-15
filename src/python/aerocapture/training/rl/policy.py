"""PyTorch policies mirroring the NeuralNetModel JSON format.

Mirrors:
    layer_sizes = [h1, h2, ..., out_dim]
    activations = [act1, act2, ..., act_out]
where activation name maps to nn module (tanh, relu, linear/identity, sigmoid).

Deterministic output mapping to bank angle in [-pi, pi] matches the Rust
runtime's atan2 interpretation when out_dim == 2.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

_ACT: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "linear": nn.Identity,
    "identity": nn.Identity,
    "swish": nn.SiLU,
    "mish": nn.Mish,
}


def _build_mlp(input_dim: int, layer_sizes: Sequence[int], activations: Sequence[str]) -> nn.Sequential:
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
    ) -> None:
        super().__init__()
        if layer_sizes[-1] != 2:
            raise ValueError(f"GaussianPolicy requires out_dim=2 (atan2), got {layer_sizes[-1]}")
        self.trunk = _build_mlp(input_dim, layer_sizes, activations)
        self.log_std = nn.Parameter(torch.full((2,), initial_log_std))
        self.min_log_std = min_log_std

    def forward_mean_logstd(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.trunk(obs)
        return mean, self.log_std.clamp(min=self.min_log_std)

    def deterministic_bank(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward_mean_logstd(obs)
        return torch.atan2(mean[..., 0], mean[..., 1])

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action from Gaussian, return (bank_angle, log_prob)."""
        mean, log_std = self.forward_mean_logstd(obs)
        std = log_std.exp()
        eps = torch.randn_like(mean)
        raw = mean + std * eps
        bank = torch.atan2(raw[..., 0], raw[..., 1])
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(raw).sum(-1)
        return bank, log_prob


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], activations: Sequence[str]) -> None:
        super().__init__()
        layer_sizes = list(hidden_sizes) + [1]
        act_list = list(activations[:-1]) + ["linear"]
        self.net = _build_mlp(input_dim, layer_sizes, act_list)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.Tensor(self.net(obs).squeeze(-1))
