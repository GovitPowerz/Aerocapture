"""Tests for training configuration dataclasses."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
from aerocapture.training.optimizer import OptimizerConfig


def test_optimizer_config_defaults() -> None:
    opt = OptimizerConfig(seed_strategy="adaptive")
    assert opt.algorithm == "ga"
    assert opt.n_pop == 60
    assert opt.n_gen == 2500
    assert opt.seed_pool_interval == 50
    assert opt.training_n_sims == 1
    assert opt.validation_n_sims == 1000
    assert opt.curation_top_k == 5
    assert opt.curation_sample_size == 1000


def test_dv_threshold_parsed_from_toml(tmp_path: Path) -> None:
    """Verify dv_threshold is correctly extracted from TOML cost_function section."""
    toml_content = """\
[cost_function]
dv_threshold = 500.0
g_load_limit = 15.0
heat_flux_limit = 200.0
g_load_weight = 1000.0
heat_flux_weight = 1000.0
"""
    toml_file = tmp_path / "test.toml"
    toml_file.write_text(toml_content)

    with open(toml_file, "rb") as f:
        _toml = tomllib.load(f)

    cost_cfg = _toml.get("cost_function", {})
    cost_kwargs = {
        "dv_threshold": float(cost_cfg.get("dv_threshold", 1000.0)),
        "g_load_limit": float(cost_cfg.get("g_load_limit", 15.0)),
        "heat_flux_limit": float(cost_cfg.get("heat_flux_limit", 200.0)),
        "g_load_weight": float(cost_cfg.get("g_load_weight", 1000.0)),
        "heat_flux_weight": float(cost_cfg.get("heat_flux_weight", 1000.0)),
    }
    assert cost_kwargs["dv_threshold"] == 500.0


def test_dv_threshold_default_when_missing(tmp_path: Path) -> None:
    """When dv_threshold is absent from TOML, default to 1000.0."""
    toml_content = """\
[cost_function]
g_load_limit = 15.0
"""
    toml_file = tmp_path / "test.toml"
    toml_file.write_text(toml_content)

    with open(toml_file, "rb") as f:
        _toml = tomllib.load(f)

    cost_cfg = _toml.get("cost_function", {})
    dv_threshold = float(cost_cfg.get("dv_threshold", 1000.0))
    assert dv_threshold == 1000.0


class TestRunParamSpecs:
    """`run_param_specs` is the one rebuild of a run's chromosome specs (#203): the
    trainer's `build_training_config_from_toml` + `_setup_param_specs`, with the run's
    `warm_start_bounds.json` overlaid. The three offline consumers (animate, the
    final_select CLI, the deploy audit) decode through it, so a decode change cannot
    land in one and not the others.
    """

    ARCH = [
        {"type": "dense", "input_size": 17, "output_size": 4, "activation": "swish"},
        {"type": "dense", "input_size": 4, "output_size": 2, "activation": "asinh"},
    ]
    REPO = Path(__file__).resolve().parents[1]

    @pytest.fixture()
    def toml_path(self, tmp_path: Path) -> Path:
        from tests.fixtures.factories import leaf_toml_with_architecture

        p = tmp_path / "config.toml"
        p.write_text(leaf_toml_with_architecture(self.REPO / "configs/training/msr_aller_nn_atan2_train.toml", self.ARCH))
        return p

    def test_is_the_trainers_own_rebuild_without_a_sidecar(self, tmp_path: Path, toml_path: Path) -> None:
        from aerocapture.training.training_config import _setup_param_specs, build_training_config_from_toml, run_param_specs

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        config, toml_data, specs = run_param_specs(toml_path, run_dir)

        expected_config, expected_toml = build_training_config_from_toml(str(toml_path))
        expected = _setup_param_specs(expected_config, expected_toml, verbose=False)
        assert config.guidance_type == expected_config.guidance_type == "neural_network"
        assert config.network.architecture == expected_config.network.architecture
        assert toml_data == expected_toml
        assert specs == expected
        assert len(specs) == 17 * 4 + 4 + 4 * 2 + 2 + 3  # weights + the three live scaffolding genes

    def test_overlays_the_recorded_weight_bounds(self, tmp_path: Path, toml_path: Path) -> None:
        from aerocapture.training.training_config import run_param_specs

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _, _, rebuilt = run_param_specs(toml_path, run_dir)
        n_weights = len(rebuilt) - 3  # the live scaffolding tail
        (run_dir / "warm_start_bounds.json").write_text(json.dumps([{"name": f"w_{i}", "p_min": -7.0 - i, "p_max": 7.0 + i} for i in range(n_weights)]))
        _, _, specs = run_param_specs(toml_path, run_dir)

        assert len(specs) == len(rebuilt)
        assert [(s.name, s.p_min, s.p_max) for s in specs[:2]] == [("w_0", -7.0, 7.0), ("w_1", -8.0, 8.0)]
        assert specs[n_weights:] == rebuilt[n_weights:]

    @pytest.mark.parametrize("delta", [-1, 1])
    def test_bounds_not_spanning_the_weight_slab_is_an_error(self, tmp_path: Path, toml_path: Path, delta: int) -> None:
        # One short would land the Xavier bound on the last weight, one long a weight bound on a scaffolding gene.
        from aerocapture.training.training_config import run_param_specs

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _, _, rebuilt = run_param_specs(toml_path, run_dir)
        n = len(rebuilt) - 3 + delta
        (run_dir / "warm_start_bounds.json").write_text(json.dumps([{"name": f"w_{i}", "p_min": -1.0, "p_max": 1.0} for i in range(n)]))
        with pytest.raises(SystemExit, match="warm_start_bounds.json has"):
            run_param_specs(toml_path, run_dir)
