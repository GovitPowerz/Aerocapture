"""build_scaffolding_initial_slab seeds FTC's best_params + jitter into the chromosome."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from aerocapture.training.param_spaces import _NN_SCAFFOLDING_PARAMS
from aerocapture.training.train import build_scaffolding_initial_slab


def test_seed_centers_at_ftc_optimum(tmp_path: Path) -> None:
    ftc_params = {s.name: s.default for s in _NN_SCAFFOLDING_PARAMS}
    ftc_path = tmp_path / "best_params.json"
    ftc_path.write_text(json.dumps(ftc_params))

    rng = np.random.default_rng(0)
    n_pop = 32
    slab = build_scaffolding_initial_slab(ftc_path, _NN_SCAFFOLDING_PARAMS, n_pop, rng, jitter=0.0)

    assert slab.shape == (n_pop, 17)
    from aerocapture.training.encoding import encode_to_normalized

    expected_row = encode_to_normalized(ftc_params, list(_NN_SCAFFOLDING_PARAMS))
    np.testing.assert_allclose(slab[0], expected_row, atol=1e-15)
    np.testing.assert_allclose(slab[-1], expected_row, atol=1e-15)


def test_seed_jitter_keeps_values_in_unit_box(tmp_path: Path) -> None:
    ftc_params = {s.name: s.default for s in _NN_SCAFFOLDING_PARAMS}
    ftc_path = tmp_path / "best_params.json"
    ftc_path.write_text(json.dumps(ftc_params))

    rng = np.random.default_rng(0)
    slab = build_scaffolding_initial_slab(ftc_path, _NN_SCAFFOLDING_PARAMS, 100, rng, jitter=0.02)

    assert slab.shape == (100, 17)
    assert (slab >= 0.0).all() and (slab <= 1.0).all()


def test_missing_ftc_params_fails_loud(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    missing = tmp_path / "absent.json"
    try:
        build_scaffolding_initial_slab(missing, _NN_SCAFFOLDING_PARAMS, 4, rng, jitter=0.02)
    except FileNotFoundError as e:
        assert "absent.json" in str(e)
        return
    raise AssertionError("expected FileNotFoundError for missing FTC params")
