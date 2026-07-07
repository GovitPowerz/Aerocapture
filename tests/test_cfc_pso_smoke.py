"""PSO plumbing smoke for the CfC probe layer.

Arch: Dense(23 -> 8, tanh) -> Cfc(8, 6, 5) -> Dense(6 -> 2, linear).
Param count: 192 + (5*14 + 5 + 4*(6*5 + 6)) + 14 = 192 + 219 + 14 = 425.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("aerocapture_rs")
import aerocapture_rs  # type: ignore[import-not-found]  # noqa: E402


@pytest.mark.slow
def test_cfc_pso_smoke(tmp_path: Path) -> None:
    from aerocapture.training.encoding import nn_param_specs_from_v2
    from aerocapture.training.initialization_v2 import init_v2_population
    from aerocapture.training.rl.schemas import CfcSpec, DenseSpec

    architecture_specs: list[DenseSpec | CfcSpec] = [
        DenseSpec(type="dense", input_size=23, output_size=8, activation="tanh"),
        CfcSpec(type="cfc", input_size=8, hidden_size=6, backbone_units=5),
        DenseSpec(type="dense", input_size=6, output_size=2, activation="linear"),
    ]
    architecture_dicts = [s.model_dump() for s in architecture_specs]

    param_specs = nn_param_specs_from_v2(architecture_specs, bound_multiplier=2.0)
    assert len(param_specs) == 425, f"Expected 425 params, got {len(param_specs)}"

    rng = np.random.default_rng(42)
    pop = init_v2_population(architecture_dicts, n_pop=4, bound_multiplier=2.0, rng=rng)
    assert pop.shape == (4, 425)
    assert np.all(np.isfinite(pop))

    json_path = tmp_path / "cfc_pso_best.json"
    aerocapture_rs.flat_weights_to_json(pop[0].astype(np.float64).tolist(), json.dumps(architecture_dicts), str(json_path), None)

    loaded = json.loads(json_path.read_text())
    assert loaded["format_version"] == 2
    assert [e["type"] for e in loaded["architecture"]] == ["dense", "cfc", "dense"]
    layer_1 = loaded["weights"]["layer_1"]
    for key in ("w_bb", "b_bb", "w_ff1", "b_ff1", "w_ff2", "b_ff2", "w_ta", "b_ta", "w_tb", "b_tb"):
        assert key in layer_1, f"missing cfc weight key: {key!r}"

    out = np.asarray(aerocapture_rs.nn_forward(str(json_path), np.zeros(23, dtype=np.float64).tolist()), dtype=np.float64)
    assert out.shape == (2,)
    assert all(math.isfinite(v) for v in out), f"non-finite output: {out}"
