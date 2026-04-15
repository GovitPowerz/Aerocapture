"""PyTorch → JSON → Python roundtrip: deterministic bank angles must match."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aerocapture.training.rl.export import export_policy_to_json, load_nn_model_json  # noqa: E402
from aerocapture.training.rl.policy import GaussianPolicy  # noqa: E402


def test_pytorch_to_json_roundtrip_deterministic_bank(tmp_path: Path) -> None:
    torch.manual_seed(0)
    policy = GaussianPolicy(input_dim=16, layer_sizes=[32, 32, 2], activations=["tanh", "tanh", "linear"])
    out_json = tmp_path / "best_model.json"
    export_policy_to_json(policy, out_json, input_mask=list(range(16)))

    json_nn = load_nn_model_json(out_json)

    obs_np = np.random.default_rng(0).standard_normal((10, 16)).astype(np.float64)
    obs_torch = torch.from_numpy(obs_np).float()

    torch_bank = policy.deterministic_bank(obs_torch).detach().numpy().astype(np.float64)
    json_bank = np.array([json_nn.forward_bank(row) for row in obs_np])

    assert np.allclose(torch_bank, json_bank, atol=1e-5), f"max diff = {np.abs(torch_bank - json_bank).max()}"
