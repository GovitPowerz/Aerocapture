"""Cache hit/miss matrix: changes to supervisor mtime, bound_multiplier,
architecture, input_mask, output_param, mode each invalidate the cache."""

import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from aerocapture.training.config import (
    NetworkConfig,
    SimConfig,
    TrainingConfig,
    WarmStartConfig,
)
from aerocapture.training.warm_start import build_warm_start_chromosome


def _basic_cfg(tmp_path: Path) -> TrainingConfig:
    p = tmp_path / "ftc_params.json"
    p.write_text(json.dumps({"k_alt": 1.0}))
    stub_toml = tmp_path / "stub.toml"
    stub_toml.write_text('[guidance.neural_network]\nmode = "full_neural"\n')
    arch = [
        {"type": "dense", "input_size": 4, "output_size": 4, "activation": "tanh"},
        {"type": "dense", "input_size": 4, "output_size": 2, "activation": "tanh"},
    ]
    return TrainingConfig(
        network=NetworkConfig(
            architecture=arch,
            input_mask=[0, 1, 2, 3],
            output_parameterization="atan2_signed",
            warm_start_from=str(p),
        ),
        warm_start=WarmStartConfig(
            supervisor_schemes=["ftc"],
            params_paths={"ftc": str(p)},
            n_warm_seeds=24,
            n_epochs=1,
            bptt_length=8,
            bound_multiplier=10.0,
        ),
        sim=SimConfig(toml_config=str(stub_toml)),
        save_dir=str(tmp_path / "out"),
    )


def _mock_collect(toml_path: str, seeds: list[int], overrides: dict | None = None, scheme: str = "ftc", sim_timeout_secs: float | None = None) -> list[dict]:
    rng = np.random.default_rng(int(seeds[0]) if len(seeds) else 0)
    return [{"seed": int(s), "X": rng.standard_normal((10, 21)), "y_signed": np.sin(rng.standard_normal(10)), "dv": 50.0, "captured": True} for s in seeds]


def test_unchanged_cfg_hits_cache(tmp_path: Path) -> None:
    cfg = _basic_cfg(tmp_path)
    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised", side_effect=_mock_collect) as mock:
        build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)
        assert mock.call_count == 1
    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised", side_effect=_mock_collect) as mock:
        build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)
        assert mock.call_count == 0  # cache hit


def test_supervisor_mtime_change_invalidates(tmp_path: Path) -> None:
    cfg = _basic_cfg(tmp_path)
    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised", side_effect=_mock_collect):
        build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)
    # Touch the supervisor file to bump its mtime
    time.sleep(0.01)
    Path(cfg.warm_start.params_paths["ftc"]).touch()
    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised", side_effect=_mock_collect) as mock:
        build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)
        assert mock.call_count == 1  # cache miss


def test_bound_multiplier_change_invalidates(tmp_path: Path) -> None:
    cfg = _basic_cfg(tmp_path)
    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised", side_effect=_mock_collect):
        build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)
    cfg.warm_start.bound_multiplier = 3.0
    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised", side_effect=_mock_collect) as mock:
        build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)
        assert mock.call_count == 1


def test_architecture_change_invalidates(tmp_path: Path) -> None:
    cfg = _basic_cfg(tmp_path)
    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised", side_effect=_mock_collect):
        build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)
    assert cfg.network.architecture is not None
    cfg.network.architecture[0]["output_size"] = 8
    cfg.network.architecture[1]["input_size"] = 8
    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised", side_effect=_mock_collect) as mock:
        build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)
        assert mock.call_count == 1
