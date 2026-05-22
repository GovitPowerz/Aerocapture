"""End-to-end build_warm_start_chromosome with mocked collect_supervised."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from aerocapture.training.config import (
    NetworkConfig,
    SimConfig,
    TrainingConfig,
    WarmStartConfig,
)
from aerocapture.training.warm_start import build_warm_start_chromosome


@pytest.fixture
def synthetic_supervisor_data():
    rng = np.random.default_rng(0)

    def _collect(toml_path, seeds, overrides, scheme, sim_timeout_secs=None):
        results = []
        for seed in seeds:
            T = 50
            results.append(
                {
                    "seed": int(seed),
                    "X": rng.standard_normal((T, 21)),
                    "y_signed": np.sin(rng.standard_normal(T)),
                    "dv": float(rng.uniform(50, 500)),
                    "captured": True,
                }
            )
        return results

    return _collect


@pytest.fixture
def temp_ftc_params(tmp_path):
    p = tmp_path / "ftc_best_params.json"
    p.write_text(
        json.dumps(
            {
                "k_alt": 1.0,
                "lateral.tau": 5.0,
                "exit.dpdyn_target": 100.0,
                "nav.density_filter_gain": 0.5,
                "thermal.heat_flux_activation": 0.8,
                "shaping.max_bank_acceleration": 30.0,
            }
        )
    )
    return p


@pytest.fixture
def stub_toml(tmp_path):
    p = tmp_path / "stub.toml"
    p.write_text('[guidance.neural_network]\nmode = "full_neural"\n')
    return p


@pytest.fixture
def cfg(tmp_path, temp_ftc_params, stub_toml):
    arch = [
        {"type": "dense", "input_size": 4, "output_size": 4, "activation": "tanh"},
        {"type": "dense", "input_size": 4, "output_size": 1, "activation": "tanh"},
    ]
    return TrainingConfig(
        network=NetworkConfig(
            architecture=arch,
            input_mask=[0, 1, 2, 3],
            output_parameterization="acos_tanh",
            warm_start_from=str(temp_ftc_params),
        ),
        warm_start=WarmStartConfig(
            supervisor_schemes=["ftc"],
            params_paths={"ftc": str(temp_ftc_params)},
            n_warm_seeds=24,  # > min_corpus threshold (max(20, n // 4))
            n_epochs=2,
            bptt_length=16,
            bound_multiplier=10.0,  # generous to avoid clip-rate hard error in this smoke
        ),
        sim=SimConfig(toml_config=str(stub_toml)),
        save_dir=str(tmp_path / "warm_out"),
    )


def test_end_to_end_with_mocked_collect(cfg, synthetic_supervisor_data):
    with patch(
        "aerocapture.training.warm_start._aero_rs.collect_supervised",
        side_effect=synthetic_supervisor_data,
    ):
        chromo = build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)
    assert chromo.dtype == np.float64
    assert chromo.ndim == 1
    assert (chromo >= 0.0).all() and (chromo <= 1.0).all()
    # Loss log written
    assert (Path(cfg.save_dir) / "warm_start_loss.json").exists()
    # Chromosome cached
    assert (Path(cfg.save_dir) / "warm_start_chromosome.npy").exists()
    assert (Path(cfg.save_dir) / "warm_start_cache_key.json").exists()


def test_cache_hit_skips_recomputation(cfg, synthetic_supervisor_data):
    with patch(
        "aerocapture.training.warm_start._aero_rs.collect_supervised",
        side_effect=synthetic_supervisor_data,
    ) as mock:
        chromo1 = build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)
        assert mock.call_count >= 1
    with patch(
        "aerocapture.training.warm_start._aero_rs.collect_supervised",
        side_effect=synthetic_supervisor_data,
    ) as mock2:
        chromo2 = build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)
        assert mock2.call_count == 0  # cache hit, no calls
    assert np.array_equal(chromo1, chromo2)


@pytest.fixture
def magnitude_only_toml(tmp_path):
    p = tmp_path / "magnitude_only.toml"
    p.write_text('[guidance.neural_network]\nmode = "magnitude_only"\n')
    return p


def test_magnitude_only_mode_runs_end_to_end(tmp_path, temp_ftc_params, magnitude_only_toml, synthetic_supervisor_data):
    """Magnitude_only mode (TOML-driven) runs warm-start without error and produces a finite chromosome."""
    arch = [
        {"type": "dense", "input_size": 4, "output_size": 4, "activation": "tanh"},
        {"type": "dense", "input_size": 4, "output_size": 1, "activation": "tanh"},
    ]
    cfg = TrainingConfig(
        network=NetworkConfig(
            architecture=arch,
            input_mask=[0, 1, 2, 3],
            output_parameterization="acos_tanh",
            warm_start_from=str(temp_ftc_params),
        ),
        warm_start=WarmStartConfig(
            supervisor_schemes=["ftc"],
            params_paths={"ftc": str(temp_ftc_params)},
            n_warm_seeds=24,
            n_epochs=2,
            bptt_length=16,
            bound_multiplier=10.0,
        ),
        sim=SimConfig(toml_config=str(magnitude_only_toml)),
        save_dir=str(tmp_path / "warm_out_magonly"),
    )
    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised", side_effect=synthetic_supervisor_data):
        chromo = build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)
    assert chromo.dtype == np.float64
    assert chromo.ndim == 1
    assert (chromo >= 0.0).all() and (chromo <= 1.0).all()


def test_resolve_nn_mode_reads_toml(tmp_path):
    from aerocapture.training.warm_start import _resolve_nn_mode

    toml_path = tmp_path / "test.toml"
    toml_path.write_text('[guidance.neural_network]\nmode = "magnitude_only"\n')
    cfg = TrainingConfig(sim=SimConfig(toml_config=str(toml_path)))
    assert _resolve_nn_mode(cfg) == "magnitude_only"


def test_resolve_nn_mode_defaults_to_full_neural_when_section_absent(tmp_path):
    from aerocapture.training.warm_start import _resolve_nn_mode

    toml_path = tmp_path / "no_section.toml"
    toml_path.write_text("[somewhere_else]\nfoo = 1\n")
    cfg = TrainingConfig(sim=SimConfig(toml_config=str(toml_path)))
    assert _resolve_nn_mode(cfg) == "full_neural"


def test_resolve_nn_mode_raises_on_missing_file(tmp_path):
    from aerocapture.training.warm_start import _resolve_nn_mode

    cfg = TrainingConfig(sim=SimConfig(toml_config=str(tmp_path / "nonexistent.toml")))
    with pytest.raises(FileNotFoundError):
        _resolve_nn_mode(cfg)
