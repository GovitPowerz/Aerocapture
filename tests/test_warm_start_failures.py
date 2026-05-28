"""Failure modes from the spec: missing supervisor params, zero captures,
clip rate > 5%, bptt_length > Transformer n_seq."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("aerocapture_rs")

from aerocapture.training.config import (  # noqa: E402
    NetworkConfig,
    SimConfig,
    TrainingConfig,
    WarmStartConfig,
)
from aerocapture.training.warm_start import (  # noqa: E402
    build_warm_start_chromosome,
)


def _basic_cfg(tmp_path: Path, supervisor_schemes: list[str] | None = None, params_paths: dict[str, str] | None = None) -> TrainingConfig:
    arch = [
        {"type": "dense", "input_size": 4, "output_size": 4, "activation": "tanh"},
        {"type": "dense", "input_size": 4, "output_size": 1, "activation": "tanh"},
    ]
    # Write a stub TOML so _resolve_nn_mode doesn't crash on file-not-found.
    stub_toml = tmp_path / "stub.toml"
    stub_toml.write_text('[guidance.neural_network]\nmode = "full_neural"\n')
    return TrainingConfig(
        network=NetworkConfig(
            architecture=arch,
            input_mask=[0, 1, 2, 3],
            output_parameterization="atan2_signed",
            warm_start_from=str(tmp_path / "ftc_params.json") if params_paths is None else None,
        ),
        warm_start=WarmStartConfig(
            supervisor_schemes=supervisor_schemes or ["ftc"],
            params_paths=params_paths or {},
            n_warm_seeds=24,
            n_epochs=1,
            bptt_length=8,
        ),
        sim=SimConfig(toml_config=str(stub_toml)),
        save_dir=str(tmp_path / "warm_out"),
    )


def test_missing_supervisor_params_raises_filenotfound(tmp_path: Path) -> None:
    cfg = _basic_cfg(tmp_path, supervisor_schemes=["ftc"], params_paths={"ftc": str(tmp_path / "missing.json")})
    cfg.network.warm_start_from = str(tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="ftc"):
        build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)


def test_zero_captures_raises(tmp_path: Path) -> None:
    p = tmp_path / "ftc.json"
    p.write_text("{}")
    cfg = _basic_cfg(tmp_path, params_paths={"ftc": str(p)})
    cfg.network.warm_start_from = str(p)

    def _all_fail(toml_path: str, seeds: list[int], overrides: dict | None = None, scheme: str = "ftc", sim_timeout_secs: float | None = None) -> list[dict]:
        return [{"seed": int(s), "X": np.zeros((5, 21)), "y_signed": np.zeros(5), "dv": 999.0, "captured": False} for s in seeds]

    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised", side_effect=_all_fail), pytest.raises(RuntimeError, match="too small"):
        build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)


def test_clip_rate_above_threshold_raises(tmp_path: Path) -> None:
    """Force clip rate > 5% by training with extreme target values and
    a tiny bound_multiplier so weights blow out of bounds.

    Adaptive bounds are explicitly disabled here so the static Xavier ×
    bound_multiplier guard fires. With adaptive_bounds=True (the default),
    bounds expand to fit the trained values and the guard never triggers
    -- that path is covered by test_adaptive_bounds_no_clipping below."""
    p = tmp_path / "ftc.json"
    p.write_text("{}")
    cfg = _basic_cfg(tmp_path, params_paths={"ftc": str(p)})
    cfg.network.warm_start_from = str(p)
    cfg.warm_start.bound_multiplier = 0.01  # absurdly tight; will clip everything
    cfg.warm_start.adaptive_bounds = False  # disable auto-expand so the static guard fires
    cfg.warm_start.n_epochs = 50  # ensure weights drift

    rng = np.random.default_rng(0)

    def _strong_targets(
        toml_path: str,
        seeds: list[int],
        overrides: dict | None = None,
        scheme: str = "ftc",
        sim_timeout_secs: float | None = None,
    ) -> list[dict]:
        return [
            {
                "seed": int(s),
                "X": rng.standard_normal((20, 21)),
                "y_signed": rng.uniform(-3.0, 3.0, size=20),  # large bank values
                "dv": 50.0,
                "captured": True,
            }
            for s in seeds
        ]

    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised", side_effect=_strong_targets), pytest.raises(RuntimeError, match="clip rate"):
        build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)


def test_adaptive_bounds_no_clipping(tmp_path: Path) -> None:
    """Under the SAME adversarial conditions as test_clip_rate_above_threshold_raises
    (tiny bound_multiplier + many epochs + extreme targets), adaptive_bounds=True
    must produce ZERO clipping by expanding the per-layer-slab bounds to fit the
    trained values. Guarantees the user-friendly path: no runtime error from
    high-drift Adam runs."""
    p = tmp_path / "ftc.json"
    p.write_text("{}")
    cfg = _basic_cfg(tmp_path, params_paths={"ftc": str(p)})
    cfg.network.warm_start_from = str(p)
    cfg.warm_start.bound_multiplier = 0.01  # absurdly tight under static bounds
    cfg.warm_start.adaptive_bounds = True  # explicit (also the default)
    cfg.warm_start.n_epochs = 50

    rng = np.random.default_rng(0)

    def _strong_targets(
        toml_path: str,
        seeds: list[int],
        overrides: dict | None = None,
        scheme: str = "ftc",
        sim_timeout_secs: float | None = None,
    ) -> list[dict]:
        return [
            {
                "seed": int(s),
                "X": rng.standard_normal((20, 21)),
                "y_signed": rng.uniform(-3.0, 3.0, size=20),
                "dv": 50.0,
                "captured": True,
            }
            for s in seeds
        ]

    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised", side_effect=_strong_targets):
        chromo, weight_specs = build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)
    # Encoding succeeded; every weight is in [0, 1] (no clipping at the boundary either).
    assert chromo.dtype == np.float64
    assert (chromo > 0.0).all() and (chromo < 1.0).all(), (
        "adaptive_bounds should expand bounds with 2x safety margin so every encoded value is STRICTLY inside (0, 1), "
        f"got min={chromo.min()}, max={chromo.max()}"
    )
    # Bounds widened beyond the absurdly tight 0.01 × Xavier configured value.
    for s in weight_specs:
        assert s.p_max > 0.01, f"adaptive bounds should be wider than the floor for {s.name}; got p_max={s.p_max}"


def test_bptt_length_greater_than_n_seq_raises(tmp_path: Path) -> None:
    p = tmp_path / "ftc.json"
    p.write_text("{}")
    cfg = _basic_cfg(tmp_path, params_paths={"ftc": str(p)})
    cfg.network.warm_start_from = str(p)
    cfg.network.architecture = [
        {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
        {"type": "transformer", "d_model": 8, "n_heads": 2, "d_ffn": 16, "n_seq": 4},
        {"type": "dense", "input_size": 8, "output_size": 1, "activation": "tanh"},
    ]
    cfg.warm_start.bptt_length = 16  # > n_seq=4

    rng = np.random.default_rng(0)

    def _ok(toml_path: str, seeds: list[int], overrides: dict | None = None, scheme: str = "ftc", sim_timeout_secs: float | None = None) -> list[dict]:
        return [{"seed": int(s), "X": rng.standard_normal((40, 21)), "y_signed": np.zeros(40), "dv": 50.0, "captured": True} for s in seeds]

    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised", side_effect=_ok), pytest.raises(ValueError, match="bptt_length.*n_seq"):
        build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)
