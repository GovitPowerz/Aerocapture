"""The Section 5 RL baseline configs are protocol-matched to the per-scenario population
champions they are compared to (issue #101): same architecture, input mask, per-input
normalization, decoder, co-trained scaffolding, per_draw regime and base MC seed. The
champion side is the committed bundle (articles/paper/data/runs/ou_marginal/<cell>), so a
retrained champion or an edited RL config fails here instead of silently un-matching the
paper's comparison. Pure Python (no bindings)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from aerocapture.training.toml_utils import load_toml_with_bases

REPO = Path(__file__).resolve().parents[1]
RL_CONFIGS = REPO / "configs/training/paper/rl"
BUNDLE = REPO / "articles/paper/data/runs"

# (RL config stem, champion bundle cell)
CELLS = [
    ("dense_p515_ppo_scratch", "ou_marginal/ft_dense_p515"),
    ("dense_p515_ppo_warm", "ou_marginal/ft_dense_p515"),
    ("gru_p1014_ppo_scratch", "ou_marginal/ft_gru_p1014"),
    ("gru_p1014_ppo_warm", "ou_marginal/ft_gru_p1014"),
]
SPEC_KEYS = ("type", "input_size", "output_size", "hidden_size", "activation")


def _spec(layers: list[dict]) -> list[dict]:
    return [{k: layer[k] for k in SPEC_KEYS if k in layer} for layer in layers]


@pytest.mark.parametrize(("stem", "champion"), CELLS)
def test_rl_config_matches_champion(stem: str, champion: str) -> None:
    cfg = load_toml_with_bases(RL_CONFIGS / f"{stem}.toml")
    model = json.loads((BUNDLE / champion / "best_model.json").read_text())
    params = json.loads((BUNDLE / champion / "best_params.json").read_text())

    assert _spec(cfg["network"]["architecture"]) == _spec(model["architecture"])
    assert cfg["network"]["input_mask"] == model["input_mask"]
    assert cfg["network"]["normalization"] == model["normalization"]
    assert cfg["guidance"]["neural_network"]["output_parameterization"] == model["output_param"]
    assert cfg["navigation"]["density_filter_gain"] == params["nav.density_filter_gain"]
    assert cfg["navigation"]["density_gain_max_delta"] == params["nav.density_gain_max_delta"]
    assert cfg["guidance"]["command_shaping"]["max_bank_acceleration"] == params["shaping.max_bank_acceleration"]
    assert cfg["guidance"]["command_shaping"]["enabled"] is True
    assert cfg["monte_carlo"]["noise_seeding"] == "per_draw"
    assert cfg["monte_carlo"]["seed"] == 42  # the 2M final-eval pool is derived from it: pairing needs one seed
    assert cfg["data"]["neural_network"] == f"training_output/paper/rl/{stem}/best_model.json"  # = the trainer's output dir


@pytest.mark.parametrize(("stem", "champion"), CELLS)
def test_bundled_rl_cell_matches_its_config(stem: str, champion: str) -> None:
    """The bundled PPO artifact was trained under the config that claims it."""
    cfg = load_toml_with_bases(RL_CONFIGS / f"{stem}.toml")
    model = json.loads((BUNDLE / "rl" / stem / "best_model.json").read_text())
    assert _spec(model["architecture"]) == _spec(cfg["network"]["architecture"])
    assert model["input_mask"] == cfg["network"]["input_mask"]
