"""Tests for init_v2_population Mamba arm diversity and center-agreement invariants."""

from __future__ import annotations

import math

import numpy as np
from aerocapture.training.encoding import _layer_param_specs
from aerocapture.training.initialization_v2 import init_v2_population
from aerocapture.training.rl.schemas import MambaSpec


def test_mamba_init_produces_correct_param_count() -> None:
    arch = [MambaSpec(type="mamba", input_size=4, d_state=2, dt_rank=1)]
    rng = np.random.default_rng(0)
    pop = init_v2_population(arch, n_pop=8, bound_multiplier=1.0, rng=rng)
    # d_inner=4: x_proj_w=(1+2*2)*4=20, dt_proj_w=4*1=4, dt_proj_b=4, a_log=4*2=8, d_skip=4 => 40
    assert pop.shape == (8, 40)


def test_mamba_init_all_individuals_differ_on_every_slice() -> None:
    """Regression test: dt_proj_b / a_log / d_skip must NOT be identical across
    the PSO population (would kill exploration).
    """
    arch = [MambaSpec(type="mamba", input_size=4, d_state=2, dt_rank=1)]
    rng = np.random.default_rng(123)
    pop = init_v2_population(arch, n_pop=16, bound_multiplier=1.0, rng=rng)

    x_proj_n = 5 * 4  # 20
    dt_proj_w_n = 4 * 1  # 4
    dt_proj_b_n = 4  # 4
    a_log_n = 4 * 2  # 8
    d_skip_n = 4  # 4

    slices = {
        "x_proj_w": slice(0, x_proj_n),
        "dt_proj_w": slice(x_proj_n, x_proj_n + dt_proj_w_n),
        "dt_proj_b": slice(x_proj_n + dt_proj_w_n, x_proj_n + dt_proj_w_n + dt_proj_b_n),
        "a_log": slice(x_proj_n + dt_proj_w_n + dt_proj_b_n, x_proj_n + dt_proj_w_n + dt_proj_b_n + a_log_n),
        "d_skip": slice(x_proj_n + dt_proj_w_n + dt_proj_b_n + a_log_n, x_proj_n + dt_proj_w_n + dt_proj_b_n + a_log_n + d_skip_n),
    }
    for name, sl in slices.items():
        std_across_pop = pop[:, sl].std(axis=0)
        assert (std_across_pop > 1e-9).all(), f"slice {name} has zero-variance columns: {std_across_pop}"


def test_mamba_init_values_within_paramspec_bounds() -> None:
    """Load-bearing: each init value must fall inside [p_min, p_max] from _layer_param_specs."""
    spec = MambaSpec(type="mamba", input_size=4, d_state=2, dt_rank=1)
    arch = [spec]
    rng = np.random.default_rng(42)
    pop = init_v2_population(arch, n_pop=32, bound_multiplier=1.0, rng=rng)

    ps = _layer_param_specs(spec, bound_multiplier=1.0)
    assert pop.shape[1] == len(ps)

    for i, param_spec in enumerate(ps):
        col = pop[:, i]
        # Allow a tiny tolerance for jitter tail draws + float precision
        assert (col >= param_spec.p_min - 1e-9).all(), f"param {i} ({param_spec.name}): {col.min()} below p_min {param_spec.p_min}"
        assert (col <= param_spec.p_max + 1e-9).all(), f"param {i} ({param_spec.name}): {col.max()} above p_max {param_spec.p_max}"


def test_mamba_init_a_log_mean_is_hippo() -> None:
    spec = MambaSpec(type="mamba", input_size=2, d_state=4, dt_rank=1)
    arch = [spec]
    rng = np.random.default_rng(7)
    pop = init_v2_population(arch, n_pop=10000, bound_multiplier=1.0, rng=rng)

    x_proj_n = (1 + 2 * 4) * 2
    dt_proj_w_n = 2 * 1
    dt_proj_b_n = 2
    a_log_start = x_proj_n + dt_proj_w_n + dt_proj_b_n

    a_log_mean = pop[:, a_log_start : a_log_start + 2 * 4].mean(axis=0)
    # HiPPO: for each d in [0, 2), n in [0, 4): center = log(n+1); row-major (d outer, n inner)
    expected = np.array([math.log(n + 1) for _d in range(2) for n in range(4)])
    assert np.allclose(a_log_mean, expected, atol=0.01)  # jitter_std=0.01, 10000 samples


def test_mamba_init_d_skip_mean_is_one() -> None:
    spec = MambaSpec(type="mamba", input_size=4, d_state=2, dt_rank=1)
    arch = [spec]
    rng = np.random.default_rng(99)
    pop = init_v2_population(arch, n_pop=10000, bound_multiplier=1.0, rng=rng)

    d_skip_start = 5 * 4 + 4 * 1 + 4 + 4 * 2  # = 36
    d_skip = pop[:, d_skip_start : d_skip_start + 4]
    assert np.allclose(d_skip.mean(axis=0), 1.0, atol=0.01)
