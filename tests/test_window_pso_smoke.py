"""Phase 2b PSO training smoke test for Window-MLP.

Runs 2 PSO generations on a reduced Window(4, 4) -> Dense(16, 4, swish) ->
Dense(4, 2, linear) arch (~78 trainable params in the downstream Dense
layers; Window contributes zero). Verifies:

- init_v2_population produces a (n_pop, 78) chromosome array with finite values.
- aerocapture_rs.flat_weights_to_json serializes the best individual to a
  valid JSON v2 file with ["window", "dense", "dense"] architecture.
- aerocapture_rs.nn_forward returns a finite 2-tuple on a zero input.

This does NOT run a real Rayon PSO loop end-to-end -- that happens in the
full training pipeline. The smoke test covers the Phase 2b-specific code
paths (Window param specs, init, JSON serialization) and confirms the Rust
runtime accepts the Window-containing model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("aerocapture_rs")
import aerocapture_rs


@pytest.mark.slow
def test_window_pso_two_gens_smoke(tmp_path: Path) -> None:
    from aerocapture.training.encoding import nn_param_specs_from_v2
    from aerocapture.training.initialization_v2 import init_v2_population
    from aerocapture.training.rl.schemas import DenseSpec, WindowSpec

    # Reduced arch: Window(4, 4) -> Dense(16, 4, swish) -> Dense(4, 2, linear).
    architecture_specs: list[DenseSpec | WindowSpec] = [
        WindowSpec(type="window", input_size=4, n_steps=4),
        DenseSpec(type="dense", input_size=16, output_size=4, activation="swish"),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    architecture_dicts = [s.model_dump() for s in architecture_specs]

    # n_params = 0 (window) + (16*4 + 4) + (4*2 + 2) = 0 + 68 + 10 = 78.
    param_specs = nn_param_specs_from_v2(architecture_specs, bound_multiplier=2.0)
    assert len(param_specs) == 78

    # Initial population.
    rng = np.random.default_rng(1234)
    pop_physical = init_v2_population(architecture_dicts, n_pop=4, bound_multiplier=2.0, rng=rng)
    assert pop_physical.shape == (4, 78)
    assert np.all(np.isfinite(pop_physical))

    # Serialize first individual to JSON v2 via the PyO3 helper.
    best_flat = pop_physical[0].astype(np.float64)
    json_path = tmp_path / "window_pso_best.json"
    aerocapture_rs.flat_weights_to_json(
        best_flat.tolist(),
        json.dumps(architecture_dicts),
        str(json_path),
        "atan2",
        None,
    )

    # Verify JSON schema.
    loaded = json.loads(json_path.read_text())
    assert loaded["format_version"] == 2
    assert [e["type"] for e in loaded["architecture"]] == ["window", "dense", "dense"]
    # Window entry is spec-only; Dense entries have weights.
    assert "layer_0" not in loaded.get("weights", {})
    assert "layer_1" in loaded["weights"]
    assert "layer_2" in loaded["weights"]

    # Run a single Rust forward to confirm the model is valid.
    obs = np.zeros(4, dtype=np.float64)
    out = aerocapture_rs.nn_forward(str(json_path), obs.tolist())
    out_arr = np.asarray(out, dtype=np.float64)
    assert out_arr.shape == (2,)
    assert np.all(np.isfinite(out_arr))
