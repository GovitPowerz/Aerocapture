"""Per-architecture warm-start smoke: each of 6 layer types completes
end-to-end on a tiny config and produces a valid chromosome."""

import json
from unittest.mock import patch

import aerocapture_rs as r
import numpy as np
import pytest
from aerocapture.training.config import (
    NetworkConfig,
    SimConfig,
    TrainingConfig,
    WarmStartConfig,
)
from aerocapture.training.encoding import nn_param_specs_from_v2
from aerocapture.training.rl.schemas import LayerSpec
from aerocapture.training.warm_start import build_warm_start_chromosome
from pydantic import TypeAdapter


def _ftc_params(tmp_path):
    p = tmp_path / "ftc_params.json"
    p.write_text(json.dumps({"k_alt": 1.0}))
    return p


def _stub_toml(tmp_path):
    p = tmp_path / "stub.toml"
    p.write_text('[guidance.neural_network]\nmode = "full_neural"\n')
    return p


def _mock_collect_factory(traj_T=40, input_dim=21):
    rng = np.random.default_rng(0)

    def _inner(toml_path, seeds, overrides, scheme, sim_timeout_secs=None):
        return [
            {"seed": int(s), "X": rng.standard_normal((traj_T, input_dim)), "y_signed": np.sin(rng.standard_normal(traj_T)), "dv": 50.0, "captured": True}
            for s in seeds
        ]

    return _inner


@pytest.mark.slow
@pytest.mark.parametrize(
    "arch_name, arch",
    [
        (
            "dense",
            [
                {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
                {"type": "dense", "input_size": 8, "output_size": 2, "activation": "linear"},
            ],
        ),
        (
            "window",
            [
                {"type": "window", "input_size": 4, "n_steps": 3},
                {"type": "dense", "input_size": 12, "output_size": 8, "activation": "tanh"},
                {"type": "dense", "input_size": 8, "output_size": 2, "activation": "linear"},
            ],
        ),
        (
            "gru",
            [
                {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
                {"type": "gru", "input_size": 8, "hidden_size": 8},
                {"type": "dense", "input_size": 8, "output_size": 2, "activation": "linear"},
            ],
        ),
        (
            "lstm",
            [
                {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
                {"type": "lstm", "input_size": 8, "hidden_size": 8},
                {"type": "dense", "input_size": 8, "output_size": 2, "activation": "linear"},
            ],
        ),
        (
            "transformer",
            [
                {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
                {"type": "transformer", "d_model": 8, "n_heads": 2, "d_ffn": 16, "n_seq": 16},
                {"type": "dense", "input_size": 8, "output_size": 2, "activation": "linear"},
            ],
        ),
        (
            "mamba",
            [
                {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
                {"type": "mamba", "input_size": 8, "d_state": 4, "dt_rank": 2},
                {"type": "dense", "input_size": 8, "output_size": 2, "activation": "linear"},
            ],
        ),
    ],
)
def test_warm_start_per_arch_smoke(arch_name, arch, tmp_path):
    p = _ftc_params(tmp_path)
    stub_toml = _stub_toml(tmp_path)
    cfg = TrainingConfig(
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
            bound_multiplier=10.0,  # generous; smoke focuses on plumbing, not clipping
        ),
        sim=SimConfig(toml_config=str(stub_toml)),
        save_dir=str(tmp_path / f"warm_out_{arch_name}"),
    )

    with patch("aerocapture.training.warm_start._aero_rs.collect_supervised", side_effect=_mock_collect_factory()):
        chromo = build_warm_start_chromosome(cfg=cfg, base_mc_seed=42)

    # Width matches param specs
    validated = TypeAdapter(list[LayerSpec]).validate_python(arch)
    expected_width = len(nn_param_specs_from_v2(validated, bound_multiplier=10.0))
    assert chromo.shape == (expected_width,)
    assert (chromo >= 0.0).all() and (chromo <= 1.0).all()

    # Quick decode + Rust forward: build a JSON via flat_weights_to_json from the un-normalized weights
    # and assert nn_forward returns a finite output.
    weight_specs = nn_param_specs_from_v2(validated, bound_multiplier=10.0)
    flat = np.array([s.p_min + chromo[i] * (s.p_max - s.p_min) for i, s in enumerate(weight_specs)])
    json_path = tmp_path / f"model_{arch_name}.json"
    # Real flat_weights_to_json signature: (flat, architecture_json, path, input_mask, output_param)
    r.flat_weights_to_json(flat.tolist(), json.dumps(arch), str(json_path), None, None)
    out = r.nn_forward(str(json_path), [0.1, 0.2, 0.3, 0.4])
    assert all(np.isfinite(out)), f"arch={arch_name} nn_forward output {out} not finite"
