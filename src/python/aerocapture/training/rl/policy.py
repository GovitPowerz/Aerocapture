"""PyTorch policies mirroring the NeuralNetModel JSON format.

Mirrors:
    layer_sizes = [h1, h2, ..., out_dim]
    activations = [act1, act2, ..., act_out]
where activation name maps to nn module (tanh, relu, linear/identity, sigmoid).

Deterministic output mapping to bank angle in [-pi, pi] matches the Rust
runtime's atan2 interpretation when out_dim == 2.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from aerocapture.training.rl.layers import build_layer
from aerocapture.training.rl.schemas import LayerSpec

_ACT: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "linear": nn.Identity,
    "identity": nn.Identity,
    "swish": nn.SiLU,
    "mish": nn.Mish,
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
    ) -> None:
        super().__init__()
        if layer_sizes[-1] != 2:
            raise ValueError(f"GaussianPolicy requires out_dim=2 (atan2), got {layer_sizes[-1]}")
        self.trunk = build_mlp(input_dim, layer_sizes, activations)
        self.log_std = nn.Parameter(torch.full((2,), initial_log_std))
        self.min_log_std = min_log_std

    def forward_mean_logstd(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.trunk(obs)
        return mean, self.log_std.clamp(min=self.min_log_std)

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


class V2Policy(nn.Module):
    """Step-wise stateful policy matching the Rust NeuralNetModel contract.

    Forward pass: `(x_t, state_t-1) -> (y_t, state_t)`. BPTT over sequences
    is an explicit Python loop in the training code.

    The final dense layer produces the pre-interpretation output (2 values for
    atan2, 1 for direct). log_std is a separate learnable parameter (not a layer,
    not exported to JSON) used only for PPO/SAC exploration noise.
    """

    def __init__(
        self,
        architecture: list[LayerSpec],
        output_interpretation: str,
        input_mask: list[int] | None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([build_layer(spec) for spec in architecture])
        self.output_interpretation = output_interpretation
        self.input_mask = input_mask
        action_dim = 2 if output_interpretation == "atan2" else 1
        self.log_std = nn.Parameter(torch.zeros(action_dim))

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


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], activations: Sequence[str]) -> None:
        super().__init__()
        layer_sizes = list(hidden_sizes) + [1]
        act_list = list(activations[:-1]) + ["linear"]
        self.net = build_mlp(input_dim, layer_sizes, act_list)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(obs).squeeze(-1)
        return out
