"""Phase 4a PSO smoke test for Mamba SSM.

Runs on a reduced Dense(23 -> 8, tanh) -> Mamba(8, d_state=4, dt_rank=1) ->
Dense(8 -> 2, linear) arch (338 trainable params).

Param count:
  Dense(23 -> 8):         23*8 + 8 = 192
  Mamba(8, 4, 1):         8*(3*4 + 2*1 + 2) = 8*16 = 128
  Dense(8 -> 2):          8*2 + 2 = 18
  Total:                  338

Verifies:
  - nn_param_specs_from_v2 produces 338 ParamSpecs for this architecture.
  - init_v2_population produces a (n_pop, 338) array with finite values.
  - aerocapture_rs.flat_weights_to_json serializes the best individual to a
    valid JSON v2 file with ["dense", "mamba", "dense"] architecture.
  - JSON has the expected Mamba weight keys (x_proj_w, dt_proj_w, dt_proj_b,
    a_log, d_skip).
  - aerocapture_rs.nn_forward returns a finite (2,) tuple on a 23-element input.
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
def test_mamba_pso_smoke(tmp_path: Path) -> None:
    from aerocapture.training.encoding import nn_param_specs_from_v2
    from aerocapture.training.initialization_v2 import init_v2_population
    from aerocapture.training.rl.schemas import DenseSpec, MambaSpec

    architecture_specs: list[DenseSpec | MambaSpec] = [
        DenseSpec(type="dense", input_size=23, output_size=8, activation="tanh"),
        MambaSpec(type="mamba", input_size=8, d_state=4, dt_rank=1),
        DenseSpec(type="dense", input_size=8, output_size=2, activation="linear"),
    ]
    architecture_dicts = [s.model_dump() for s in architecture_specs]

    # Verify total param count.
    param_specs = nn_param_specs_from_v2(architecture_specs, bound_multiplier=2.0)
    assert len(param_specs) == 338, f"Expected 338 params, got {len(param_specs)}"

    # Initial population.
    rng = np.random.default_rng(42)
    pop_physical = init_v2_population(architecture_dicts, n_pop=4, bound_multiplier=2.0, rng=rng)
    assert pop_physical.shape == (4, 338)
    assert np.all(np.isfinite(pop_physical)), "non-finite values in initial population"

    # Serialize first individual to JSON v2 via the PyO3 helper.
    best_flat = pop_physical[0].astype(np.float64)
    json_path = tmp_path / "mamba_pso_best.json"
    aerocapture_rs.flat_weights_to_json(
        best_flat.tolist(),
        json.dumps(architecture_dicts),
        str(json_path),
        None,  # input_mask
    )

    # Verify JSON schema.
    loaded = json.loads(json_path.read_text())
    assert loaded["format_version"] == 2
    assert [e["type"] for e in loaded["architecture"]] == ["dense", "mamba", "dense"]

    # Mamba weight keys must all be present.
    layer_1 = loaded["weights"]["layer_1"]
    for key in ("x_proj_w", "dt_proj_w", "dt_proj_b", "a_log", "d_skip"):
        assert key in layer_1, f"missing mamba weight key: {key!r}"

    # Dense layers must also have weights.
    assert "layer_0" in loaded["weights"]
    assert "layer_2" in loaded["weights"]

    # Run a single Rust forward to confirm the model is valid.
    obs = np.zeros(23, dtype=np.float64)
    out = aerocapture_rs.nn_forward(str(json_path), obs.tolist())
    out_arr = np.asarray(out, dtype=np.float64)
    assert out_arr.shape == (2,)
    assert all(math.isfinite(v) for v in out_arr), f"non-finite output: {out_arr}"
