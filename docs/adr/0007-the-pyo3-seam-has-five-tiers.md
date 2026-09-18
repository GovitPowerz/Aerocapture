# ADR-0007: The PyO3 seam has five tiers; a new entry names its tier and its gate

**Status:** accepted · **Date:** 2026-09-18 (issue #103)

## Context

`aerocapture_rs` is the one seam between the Rust simulator and the Python training and
evaluation code. Before this decision its interface was sixteen `#[pyfunction]`s, three
classes and five constants registered as one flat list under a one-line module doc. Two of
those entries carry every number in the paper (`run_grid` for training, ADR-0004; `run_batch`
for every deploy-side evaluation); three had no production caller at all (`run`,
`load_config`, `candidate_inputs`), and the CLI bit-identity gate compared the CLI against a
path (`run`) that nothing in production used. A reader could not tell which entries mattered
without grepping the callers.

## Decision

The seam is documented and registered in five tiers, each naming the test that proves it:

- **evaluate**: `run_grid` (gate `tests/test_run_grid.py`), `run_batch` (gates
  `tests/test_pyo3.py::TestBitIdenticalRegression` against the CLI, `tests/test_noise_seeding.py`
  for the noise regime), `run_mc`, `run_with_draws`.
- **config**: `validate_config`, `load_config`. `load_config` keeps one job: the Rust-side
  base-resolution oracle for `toml_utils.load_toml_with_bases`, asserted over every committed
  config (`tests/test_pyo3.py::TestLoadConfig::test_base_resolution_parity`).
- **contract**: `candidate_inputs` (the ONE candidate-input schema; the retired
  `NN_INPUT_NAMES` constant and `default_normalization()` were its projections),
  `final_record_indices`, `layer_schema`, the width constants (drift oracles; gates
  `tests/test_record_index_drift.py`, `tests/test_layer_schema_drift.py`).
- **nn**: `flat_weights_to_json`, `collect_supervised` / `collect_nn_inputs`, and the gate-only
  `nn_forward` / `nn_forward_sequence` (the Rust side of the per-layer equivalence tests; no
  training path calls them).
- **env**: `BatchedSimulation` (gate `tests/test_env_pyo3.py`).

`run` and its `SimResult` class are retired: one run is `run_batch(toml, [{}])` row 0, so the
CLI gate now compares the CLI against the path every deploy-side number takes. Before the
deletion, `run(t).final_record` was checked bit-equal to both `run_batch(t, [{}]).final_records[0]`
and `run_mc(t).final_records[0]` on the golden config.

A new entry names its tier in the `lib.rs` module doc and the test that gates it, and is
registered in that tier's block. An entry with no production caller and no gate does not
belong on the seam.

## Consequences

- Numbers do not move: the retired entries were adapters over the same `run_for_api`, and
  goldens, `test_pyo3_matches_subprocess`, `test_run_grid.py` and `test_noise_seeding.py` are
  unchanged.
- `README.md`'s bindings example and CLAUDE.md's "Key API" lead with `run_batch`.
- Python derives the candidate-input names and the default normalization from
  `candidate_inputs()`; the pure-Python fallback tuple in `config.py` stays, asserted equal
  element-wise by the drift test.
