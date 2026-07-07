"""PSO plumbing smoke for the Mamba-3 ablation layer (both flags on).

Arch: Dense(23 -> 8, tanh) -> Mamba3(8, 4, 1, trapezoidal, complex) -> Dense(8 -> 2, linear).
Param count:
  Dense(23 -> 8):  23*8 + 8 = 192
  Mamba3(8, 4, 1): 8*(3*4 + 2*1 + 2) = 128, +complex 8*4=32, +trapz 8 = 168
  Dense(8 -> 2):   8*2 + 2 = 18
  Total:           378

Verifies the PSO serialization path (nn_param_specs_from_v2 -> init_v2_population
-> flat_weights_to_json -> nn_forward) works end to end for mamba3, and that the
deployed JSON carries the conditional a_imag / lambda_logit weight blocks.
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
def test_mamba3_pso_smoke(tmp_path: Path) -> None:
    from aerocapture.training.encoding import nn_param_specs_from_v2
    from aerocapture.training.initialization_v2 import init_v2_population
    from aerocapture.training.rl.schemas import DenseSpec, Mamba3Spec

    architecture_specs: list[DenseSpec | Mamba3Spec] = [
        DenseSpec(type="dense", input_size=23, output_size=8, activation="tanh"),
        Mamba3Spec(type="mamba3", input_size=8, d_state=4, dt_rank=1, discretization="trapezoidal", state_mode="complex"),
        DenseSpec(type="dense", input_size=8, output_size=2, activation="linear"),
    ]
    architecture_dicts = [s.model_dump() for s in architecture_specs]

    param_specs = nn_param_specs_from_v2(architecture_specs, bound_multiplier=2.0)
    assert len(param_specs) == 378, f"Expected 378 params, got {len(param_specs)}"

    rng = np.random.default_rng(42)
    pop = init_v2_population(architecture_dicts, n_pop=4, bound_multiplier=2.0, rng=rng)
    assert pop.shape == (4, 378)
    assert np.all(np.isfinite(pop))

    json_path = tmp_path / "mamba3_pso_best.json"
    aerocapture_rs.flat_weights_to_json(pop[0].astype(np.float64).tolist(), json.dumps(architecture_dicts), str(json_path), None)

    loaded = json.loads(json_path.read_text())
    assert loaded["format_version"] == 2
    assert [e["type"] for e in loaded["architecture"]] == ["dense", "mamba3", "dense"]
    m3_spec = loaded["architecture"][1]
    assert m3_spec["discretization"] == "trapezoidal"
    assert m3_spec["state_mode"] == "complex"

    layer_1 = loaded["weights"]["layer_1"]
    for key in ("x_proj_w", "dt_proj_w", "dt_proj_b", "a_log", "a_imag", "lambda_logit", "d_skip"):
        assert key in layer_1, f"missing mamba3 weight key: {key!r}"

    out = np.asarray(aerocapture_rs.nn_forward(str(json_path), np.zeros(23, dtype=np.float64).tolist()), dtype=np.float64)
    assert out.shape == (2,)
    assert all(math.isfinite(v) for v in out), f"non-finite output: {out}"
