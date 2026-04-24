"""Tests for Mamba PSO ParamSpec generation and config arm dispatch."""

from __future__ import annotations

import math

from aerocapture.training.config import _layer_n_params, _layer_output_size
from aerocapture.training.encoding import _layer_param_specs, nn_param_specs_from_v2
from aerocapture.training.rl.schemas import DenseSpec, MambaSpec


def test_mamba_param_specs_total_count_matches_formula():
    spec = MambaSpec(type="mamba", input_size=32, d_state=16, dt_rank=2)
    specs = _layer_param_specs(spec, bound_multiplier=1.0)
    # Formula: input_size * (3*d_state + 2*dt_rank + 2) = 32 * (48 + 4 + 2) = 32 * 54 = 1728
    assert len(specs) == 1728


def test_mamba_param_specs_layout_matches_canonical_order():
    spec = MambaSpec(type="mamba", input_size=4, d_state=2, dt_rank=1)
    specs = _layer_param_specs(spec, bound_multiplier=1.0)
    # Canonical order:
    #   1. x_proj_w: (dt_rank + 2*d_state, input_size) = (5, 4) = 20
    #   2. dt_proj_w: (input_size, dt_rank) = (4, 1) = 4
    #   3. dt_proj_b: (input_size,) = 4
    #   4. a_log: (input_size, d_state) = (4, 2) = 8
    #   5. d_skip: (input_size,) = 4
    # Total = 40
    assert len(specs) == 40

    names = [s.name for s in specs]
    assert all(n.startswith("x_proj_w") for n in names[:20])
    assert all(n.startswith("dt_proj_w") for n in names[20:24])
    assert all(n.startswith("dt_proj_b") for n in names[24:28])
    assert all(n.startswith("a_log") for n in names[28:36])
    assert all(n.startswith("d_skip") for n in names[36:40])


def test_mamba_param_specs_hippo_centers():
    spec = MambaSpec(type="mamba", input_size=2, d_state=3, dt_rank=1)
    specs = _layer_param_specs(spec, bound_multiplier=1.0)
    x_proj_n = (1 + 2 * 3) * 2  # 14
    dt_proj_w_n = 2 * 1  # 2
    dt_proj_b_n = 2  # 2
    a_log_start = x_proj_n + dt_proj_w_n + dt_proj_b_n
    expected_centers = []
    for _d in range(2):
        for n in range(3):
            expected_centers.append(math.log(n + 1))
    for i, expected in enumerate(expected_centers):
        assert abs(specs[a_log_start + i].default - expected) < 1e-15, f"a_log spec {i}: got {specs[a_log_start + i].default}, expected {expected}"


def test_mamba_param_specs_d_skip_centers_are_one():
    spec = MambaSpec(type="mamba", input_size=4, d_state=2, dt_rank=1)
    specs = _layer_param_specs(spec, bound_multiplier=1.0)
    d_skip_specs = [s for s in specs if s.name.startswith("d_skip")]
    assert len(d_skip_specs) == 4
    for s in d_skip_specs:
        assert s.default == 1.0
        assert s.p_min == 0.0  # 1.0 - 1.0
        assert s.p_max == 2.0  # 1.0 + 1.0


def test_layer_n_params_mamba():
    spec = MambaSpec(type="mamba", input_size=32, d_state=16, dt_rank=2)
    assert _layer_n_params(spec) == 1728


def test_layer_output_size_mamba_equals_input_size():
    spec = MambaSpec(type="mamba", input_size=32, d_state=16, dt_rank=2)
    assert _layer_output_size(spec) == 32


def test_nn_param_specs_from_v2_handles_mamba():
    arch = [
        DenseSpec(type="dense", input_size=23, output_size=8, activation="tanh"),
        MambaSpec(type="mamba", input_size=8, d_state=4, dt_rank=2),
        DenseSpec(type="dense", input_size=8, output_size=2, activation="linear"),
    ]
    specs = nn_param_specs_from_v2(arch, bound_multiplier=1.0)
    # Dense(23->8): 23*8+8=192; Mamba(8,4,2): 8*(12+4+2)=144; Dense(8->2): 18
    assert len(specs) == 192 + 144 + 18
