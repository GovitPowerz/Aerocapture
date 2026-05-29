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


def test_build_default_scaffolding_slab_no_file(tmp_path: Path) -> None:
    """live seeding builds the slab from ParamSpec defaults, touching no file."""
    from aerocapture.training.encoding import encode_to_normalized
    from aerocapture.training.param_spaces import _NN_LIVE_PARAMS
    from aerocapture.training.train import build_default_scaffolding_slab

    rng = np.random.default_rng(0)
    slab = build_default_scaffolding_slab(_NN_LIVE_PARAMS, n_pop=8, rng=rng, jitter=0.0)

    assert slab.shape == (8, 3)
    expected_row = encode_to_normalized({s.name: s.default for s in _NN_LIVE_PARAMS}, list(_NN_LIVE_PARAMS))
    for row in slab:
        np.testing.assert_allclose(row, expected_row)
    assert slab.min() >= 0.0 and slab.max() <= 1.0


def test_build_default_scaffolding_slab_jitter_bounds() -> None:
    from aerocapture.training.param_spaces import _NN_LIVE_PARAMS
    from aerocapture.training.train import build_default_scaffolding_slab

    rng = np.random.default_rng(1)
    slab = build_default_scaffolding_slab(_NN_LIVE_PARAMS, n_pop=100, rng=rng, jitter=0.02)
    assert slab.shape == (100, 3)
    assert slab.min() >= 0.0 and slab.max() <= 1.0
