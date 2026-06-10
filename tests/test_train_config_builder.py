"""TOML -> TrainingConfig builder (extracted from train.main for CLI reuse)."""

from __future__ import annotations

from pathlib import Path

import pytest
from aerocapture.training.train import build_training_config_from_toml


def test_builds_eqglide_config() -> None:
    cfg, toml_data = build_training_config_from_toml("configs/training/msr_aller_eqglide_train.toml")
    assert cfg.guidance_type == "equilibrium_glide"
    assert cfg.sim.toml_config == "configs/training/msr_aller_eqglide_train.toml"
    assert cfg.optimizer.validation_n_sims > 0
    assert "monte_carlo" in toml_data


def test_builds_nn_config_with_network_fields() -> None:
    cfg, _ = build_training_config_from_toml("configs/training/msr_aller_gru_pso_train.toml")
    assert cfg.guidance_type == "neural_network"
    assert cfg.network.architecture is not None


def test_missing_guidance_type_raises_system_exit(tmp_path: Path) -> None:
    # A TOML with a valid [optimizer] but no [guidance] type must raise SystemExit.
    # (Without seed_strategy the optimizer parse raises ValueError first, so include it.)
    bad = tmp_path / "bad.toml"
    bad.write_text('[simulation]\nn_sims = 1\n\n[optimizer]\nseed_strategy = "fixed"\n')
    with pytest.raises(SystemExit):
        build_training_config_from_toml(str(bad))
