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


def test_warm_start_block_reaches_config() -> None:
    cfg, _ = build_training_config_from_toml("configs/training/msr_aller_nn_joint_train.toml")
    assert cfg.warm_start.enabled is True


def test_mamba_dt_rank_resolved_without_population_build() -> None:
    """train.py's [network] override chain bypasses NetworkConfig.__post_init__;
    the builder must re-normalize so callers that never build an initial
    population (final_select CLI) get a run_grid-parseable architecture.
    Regression: run_grid "missing field dt_rank" on Mamba cells."""
    cfg, _ = build_training_config_from_toml("configs/training/sweep/mamba_p962.toml")
    assert cfg.network.architecture is not None
    for entry in cfg.network.architecture:
        if entry["type"] == "mamba":
            assert "dt_rank" in entry, entry


def test_rust_validation_rejects_invalid_config(tmp_path: Path) -> None:
    """When the extension is importable the builder runs the Rust no-IO
    validation pass (`aerocapture_rs.validate_config`), so a config Rust would
    reject at the gen-0 run_grid fails here instead."""
    pytest.importorskip("aerocapture_rs")
    base = Path("configs/training/msr_aller_eqglide_train.toml").resolve()
    bad = tmp_path / "bad_nav.toml"
    bad.write_text(f'base = "{base}"\n\n[navigation]\nmode = "EKF"\n')
    with pytest.raises(SystemExit):
        build_training_config_from_toml(str(bad))
