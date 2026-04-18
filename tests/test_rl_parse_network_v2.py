"""_parse_network_config returns v2 architecture when TOML has [[network.architecture]]."""

from __future__ import annotations

from pathlib import Path

import pytest

# Importing aerocapture.training.rl.train transitively loads aerocapture_rs via
# env.py's top-level import. Skip when the PyO3 bindings aren't installed.
pytest.importorskip("aerocapture_rs")

from aerocapture.training.rl.config import RLConfig  # noqa: E402
from aerocapture.training.rl.schemas import DenseSpec, GruSpec  # noqa: E402


def test_parse_network_config_v2_gru_arch(tmp_path: Path) -> None:
    from aerocapture.training.rl.train import _parse_network_config

    toml = tmp_path / "cfg.toml"
    toml.write_text(
        """
[rl]
n_envs = 2

[network]
input_mask = [0, 1, 2]
output_interpretation = "atan2"

[[network.architecture]]
type = "dense"
input_size = 3
output_size = 4
activation = "tanh"

[[network.architecture]]
type = "gru"
input_size = 4
hidden_size = 4

[[network.architecture]]
type = "dense"
input_size = 4
output_size = 2
activation = "linear"
""".lstrip()
    )
    cfg = RLConfig.from_toml(toml)
    parsed = _parse_network_config(cfg)
    # New contract: returns (input_mask, architecture, input_dim, output_interpretation)
    input_mask, architecture, input_dim, output_interpretation = parsed
    assert input_mask == [0, 1, 2]
    assert input_dim == 3
    assert output_interpretation == "atan2"
    assert len(architecture) == 3
    assert isinstance(architecture[0], DenseSpec)
    assert isinstance(architecture[1], GruSpec)
    assert isinstance(architecture[2], DenseSpec)
    assert architecture[1].hidden_size == 4


def test_parse_network_config_v1_layer_sizes_still_works(tmp_path: Path) -> None:
    """Legacy path: [network] layer_sizes + activations produces equivalent dense-only arch."""
    from aerocapture.training.rl.train import _parse_network_config

    toml = tmp_path / "cfg.toml"
    toml.write_text(
        """
[rl]
n_envs = 2

[network]
input_mask = [0, 1, 2]
layer_sizes = [3, 4, 2]
activations = ["tanh", "linear"]
""".lstrip()
    )
    cfg = RLConfig.from_toml(toml)
    parsed = _parse_network_config(cfg)
    input_mask, architecture, input_dim, output_interpretation = parsed
    assert input_mask == [0, 1, 2]
    assert input_dim == 3
    assert output_interpretation == "atan2"
    assert len(architecture) == 2
    assert all(isinstance(s, DenseSpec) for s in architecture)
    assert architecture[0].input_size == 3
    assert architecture[0].output_size == 4
    assert architecture[1].output_size == 2
