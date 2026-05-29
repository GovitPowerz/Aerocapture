# NN input behavior report

Date: 2026-05-29
Status: design approved, pre-implementation
Branch: `feature/nn-input-report` (off `feature/nn-bank-decoders`)

## Motivation

NN guidance can fly itself into input regions the network was never well-conditioned
for -- inputs that saturate past the expected `[-1, 1]` normalized range, spike, drift,
or differ systematically between good and bad runs. The new `scaled_pi` / `delta`
decoders make this especially worth watching. We need a diagnostic that plots the NN's
**own closed-loop inputs** (not a teacher's) across a Monte Carlo ensemble, colored by
final DV correction, so problematic inputs are visible and quantifiable.

Existing tooling does not expose this:

- `run_mc(include_trajectories=True)` returns a fixed 17-column per-tick trajectory; it
  does not include the 31-element NN candidate-input vector.
- `collect_supervised` returns per-tick `X (T,31)` but **forces a non-NN teacher scheme**
  (default FTC) and skips the NN model -- so its inputs are the teacher's, not the NN's.

## Goal

A standalone report that, for a trained NN scheme:

1. Runs an MC ensemble with the **deployed NN** and captures the per-tick 31-input
   candidate vector + final DV + capture flag.
2. Splits trajectories blue (low final DV) / red (high final DV) by a DV threshold.
3. Renders one spaghetti + per-class envelope panel per candidate input, against both
   time and energy, with `+-1` guide lines.
4. Emits a per-input summary table flagging saturation and good-vs-bad separation.

## Component design

### 1. Data capture: `aerocapture_rs.collect_nn_inputs` (Rust/PyO3)

A sibling to `collect_supervised` in `src/rust/aerocapture-py/src/lib.rs`. Differences
from `collect_supervised`:

- **Runs the configured guidance** (the deployed NN) -- NO teacher-scheme override.
  Requires `guidance.type = "neural_network"` and a loadable `[data] neural_network`;
  errors clearly otherwise.
- Reuses the existing per-tick candidate-vector trace (`build_nn_input` via `FULL_MASK`,
  already 31-wide) by enabling the supervised-collect flag while the NN scheme runs.
- Additionally stashes, per tick, `sim_time` and the **truth energy** in MJ/kg (same
  `energy_mj_kg` convention the corridor charts use, computed from the truth state via
  `total_energy`), so panels can be drawn against either axis.

Returns a `list[dict]` (one per seed):
`{X: ndarray(T,31), time: ndarray(T,), energy: ndarray(T,), dv: float, captured: bool}`.

Shared scaffolding with `collect_supervised` (per-seed loop, `py.detach`, trace
extraction, override merging) is factored into a common helper rather than copy-pasted.

**Implementation note (resolve in plan):** the per-tick trace tuple is currently
`(Vec<f64> nn_input, f64 y_signed, f64 prev_realized)` (in `runner.rs` /
`tick.rs`). Either widen it with `(time, energy)` or attach a parallel time/energy
trace. Widening the existing tuple is simplest but touches `collect_supervised`'s
extraction too -- keep that function's output unchanged by reading only the fields it
already uses.

### 2. Analysis module: `aerocapture.training.nn_input_report` (standalone CLI)

Mirrors `ablation.py` / `sensitivity.py` structure:

- Resolves the NN model + config (reuses `_resolve_nn_path` / `load_toml_with_bases`
  patterns from `ablation.py`).
- Draws `--n-sims` reserved seeds (disjoint from training/validation/final-eval pools;
  reuse `make_reserved_seeds` with a dedicated offset, or accept that this is a read-only
  diagnostic and use a fixed base -- decided in plan, default to a dedicated offset for
  cleanliness).
- Calls `collect_nn_inputs`, assembles the per-trajectory list.
- **DV classification:** blue if `dv < dv_threshold`, red otherwise. `dv_threshold`
  defaults to the TOML `cost_function.dv_threshold` (via the same kwargs reader
  `ablation.py` uses); `--dv-threshold` overrides.
- Computes the summary table (S4), renders charts (S3), writes outputs (S5).

### 3. Charts: `aerocapture.training.charts_nn_inputs` (new file)

Reuses `charts.py` conventions (seaborn theme, `_spaghetti_alpha`, percentile-band
style). One panel **per candidate input x per axis** = 31 x {time, energy} = 62 panels,
arranged in a grid (paged in the PDF). Each panel:

- Blue spaghetti (low-DV trajectories) and red spaghetti (high-DV), alpha via
  `_spaghetti_alpha(n)`.
- Per-class **p5-p95 envelope band** (blue band over blue trajectories, red band over
  red), computed by **binning samples on the x-axis** (time or energy) into a shared grid
  and taking per-bin percentiles over each class -- a percentile extension of the
  existing `_compute_envelope` (which bins by energy and takes min/max for the corridor).
  Binning handles ragged trajectory lengths and non-monotonic energy uniformly, so no
  per-trajectory interpolation is needed.
- Horizontal **guide lines at `+1` and `-1`** (the expected normalized input range) so
  out-of-range excursions are obvious at a glance.
- Title = `NN_INPUT_NAMES[i]`; rendered **greyed / annotated "(unused)"** when index `i`
  is not in the model's `input_mask`.

Output: individual SVGs, named `nn_input_{NN}_{name}_{time|energy}.svg`.

### 4. Per-input summary table (`summary.json` + PDF table)

Per candidate input, over all (trajectory x timestep) samples:

- `p1`, `p50`, `p99` of the value.
- `frac_out_of_range`: fraction of samples with `|value| > 1` (saturation / scaling flag).
- `separation`: `|mean_red - mean_blue| / pooled_std` -- how strongly the input
  discriminates high-DV from low-DV runs (a large value means the input is implicated in
  bad outcomes; near-zero means it carries no good/bad signal).
- `in_mask`: whether the input reaches the deployed network.

Sorted to surface the worst offenders (highest `frac_out_of_range`, then `separation`).
Written as JSON and rendered as a table in the PDF.

### 5. CLI + outputs

```
python -m aerocapture.training.nn_input_report <training_dir> --toml <config.toml> \
    [--n-sims 500] [--dv-threshold F] [--output-dir DIR]
```

Outputs under `<training_dir>/nn_input_report/` (or `--output-dir`):

- 62 panel SVGs.
- `summary.json`.
- `nn_input_report.pdf` (Typst, only if `typst` is installed; degrade gracefully to SVGs
  + JSON otherwise -- mirrors the existing report's Typst-optional behavior).

## Testing

Rust (`src/rust/aerocapture-py` + core):

- `collect_nn_inputs` runs the NN scheme (not a teacher): returned `X` width 31,
  `time`/`energy` lengths == `X` rows, `dv`/`captured` populated, time monotonic.
- Rejects a config with `guidance.type != neural_network` (clear error).
- `collect_supervised` output is unchanged by the trace-tuple change (regression).

Python:

- DV classification threshold logic (blue/red split at the boundary, TOML default vs
  `--dv-threshold` override).
- Summary stats: `frac_out_of_range`, `separation` on synthetic traces with known values;
  `in_mask` correctness.
- Ragged-trajectory resampling onto the common grid (envelope band shape).
- Chart SVG generation on a small synthetic ensemble (both axes, greyed unused panel).
- CLI smoke (small `--n-sims`).

## Out of scope (deferred)

- Animating input evolution over training generations (the report is a single
  post-hoc snapshot of one deployed model).
- Per-input remediation suggestions (the summary surfaces offenders; acting on them
  is the user's call).
- Integrating into the auto-generated end-of-training `report.py` PDF (standalone only).

## Risks / call-outs

- **Trace-tuple change touches `collect_supervised`.** The plan must keep
  `collect_supervised`'s returned dict byte-identical (a regression test guards this) so
  warm-start is unaffected.
- **Ragged trajectory lengths + non-monotonic energy.** Handled by the binning approach
  (S3): samples are pooled into x-axis bins per class, so trajectories of different
  lengths and non-monotonic energy contribute to whichever bins their samples fall in --
  no per-trajectory interpolation, no monotonicity assumption. Spaghetti lines are drawn
  raw (matplotlib handles non-monotonic x). Bins with too few samples in a class are
  dropped from that class's band.
