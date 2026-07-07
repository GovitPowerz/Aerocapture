"""Mamba3 PSO ParamSpec count must match n_params across all flag combos."""

from __future__ import annotations

from aerocapture.training.config import _layer_n_params
from aerocapture.training.encoding import _mamba3_specs
from aerocapture.training.rl.schemas import Mamba3Spec


def _n_expected(input_size: int, d_state: int, dt_rank: int, trap: bool, cplx: bool) -> int:
    base = input_size * (3 * d_state + 2 * dt_rank + 2)
    return base + (input_size * d_state if cplx else 0) + (input_size if trap else 0)


def test_mamba3_specs_length_matches_n_params() -> None:
    for disc, trap in (("euler", False), ("trapezoidal", True)):
        for sm, cplx in (("real", False), ("complex", True)):
            spec = Mamba3Spec(type="mamba3", input_size=16, d_state=8, dt_rank=1, discretization=disc, state_mode=sm)
            specs = _mamba3_specs(spec, 0, 2.0)
            expected = _n_expected(16, 8, 1, trap, cplx)
            assert len(specs) == expected, (disc, sm)
            # config._layer_n_params must agree with the spec count.
            entry = {"type": "mamba3", "input_size": 16, "d_state": 8, "dt_rank": 1, "discretization": disc, "state_mode": sm}
            assert _layer_n_params(entry) == expected, (disc, sm)


def test_mamba3_specs_lambda_center_is_near_euler() -> None:
    spec = Mamba3Spec(type="mamba3", input_size=4, d_state=2, dt_rank=1, discretization="trapezoidal", state_mode="real")
    specs = _mamba3_specs(spec, 0, 2.0)
    lam = [s for s in specs if s.name.startswith("lambda_logit")]
    assert len(lam) == 4
    assert all(s.default == 4.0 for s in lam)  # sigmoid(4) ~ 0.98 == near-euler start
