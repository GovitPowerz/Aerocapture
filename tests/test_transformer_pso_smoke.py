"""Phase 3a PSO smoke test for Transformer.

Runs on a reduced Dense(8 -> 4, linear) -> Transformer(d_model=4, n_heads=2,
d_ffn=8, n_seq=3) -> Dense(4 -> 2, linear) arch (218 trainable params).
Verifies:

- init_v2_population produces a (n_pop, 218) chromosome array with finite values.
- aerocapture_rs.flat_weights_to_json serializes the best individual to a
  valid JSON v2 file with ["dense", "transformer", "dense"] architecture.
- JSON has the expected Transformer weight keys (w_q, b_q, ..., ln2_beta).
- aerocapture_rs.nn_forward returns a finite 2-tuple on a small input.

This does NOT run a real Rayon PSO loop end-to-end -- that is covered by the
full training pipeline. The smoke test covers the Phase 3a-specific code paths
(Transformer param specs, init, JSON serialization) and confirms the Rust
runtime accepts the Transformer-containing model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("aerocapture_rs")
import aerocapture_rs  # type: ignore[import-not-found]


@pytest.mark.slow
def test_transformer_pso_two_gens_smoke(tmp_path: Path) -> None:
    from aerocapture.training.encoding import nn_param_specs_from_v2
    from aerocapture.training.initialization_v2 import init_v2_population
    from aerocapture.training.rl.schemas import DenseSpec, TransformerSpec

    # Reduced arch: Dense(8 -> 4, linear) -> Transformer(4, 2, 8, 3) -> Dense(4 -> 2, linear).
    architecture_specs: list[DenseSpec | TransformerSpec] = [
        DenseSpec(type="dense", input_size=8, output_size=4, activation="linear"),
        TransformerSpec(type="transformer", d_model=4, n_heads=2, d_ffn=8, n_seq=3),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    architecture_dicts = [s.model_dump() for s in architecture_specs]

    # n_params:
    #   Dense 0: 8*4 + 4 = 36
    #   Transformer: 4*(4^2) + 2*(8*4) + 8 + 9*4 = 64 + 64 + 8 + 36 = 172
    #   Dense 2: 4*2 + 2 = 10
    #   Total: 218
    param_specs = nn_param_specs_from_v2(architecture_specs, bound_multiplier=2.0)
    assert len(param_specs) == 218

    # Initial population.
    rng = np.random.default_rng(42)
    pop_physical = init_v2_population(architecture_dicts, n_pop=4, bound_multiplier=2.0, rng=rng)
    assert pop_physical.shape == (4, 218)
    assert np.all(np.isfinite(pop_physical))

    # Serialize first individual to JSON v2 via the PyO3 helper.
    best_flat = pop_physical[0].astype(np.float64)
    json_path = tmp_path / "transformer_pso_best.json"
    aerocapture_rs.flat_weights_to_json(
        best_flat.tolist(),
        json.dumps(architecture_dicts),
        str(json_path),
        None,
    )

    # Verify JSON schema.
    loaded = json.loads(json_path.read_text())
    assert loaded["format_version"] == 2
    assert [e["type"] for e in loaded["architecture"]] == ["dense", "transformer", "dense"]

    # Transformer weight keys must all be present.
    layer_1 = loaded["weights"]["layer_1"]
    expected_keys = [
        "w_q",
        "b_q",
        "w_k",
        "b_k",
        "w_v",
        "b_v",
        "w_o",
        "b_o",
        "w_ffn1",
        "b_ffn1",
        "w_ffn2",
        "b_ffn2",
        "ln1_gamma",
        "ln1_beta",
        "ln2_gamma",
        "ln2_beta",
    ]
    for key in expected_keys:
        assert key in layer_1, f"missing transformer weight key: {key}"

    # Dense layers must also have weights.
    assert "layer_0" in loaded["weights"]
    assert "layer_2" in loaded["weights"]

    # Run a single Rust forward to confirm the model is valid.
    obs = np.zeros(8, dtype=np.float64)
    out = aerocapture_rs.nn_forward(str(json_path), obs.tolist())
    out_arr = np.asarray(out, dtype=np.float64)
    assert out_arr.shape == (2,)
    assert np.all(np.isfinite(out_arr))
