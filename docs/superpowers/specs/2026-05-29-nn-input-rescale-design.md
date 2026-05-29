# NN input rescaling (asinh signed-log) + periapsis altitude

Date: 2026-05-29
Status: design approved, pre-implementation
Branch: `feature/nn-input-rescale` (off `feature/nn-input-report`)

## Motivation

The NN input behavior report (`nn_input_report`) showed several candidate inputs spend a
large fraction of every flight *outside* the network's expected `[-1, 1]` normalized
range -- effectively rail-pinned / poorly conditioned. Measured saturation
(`frac_out_of_range`) on real ensembles:

| input | current scaling | delta | scaled_pi |
|---|---|---|---|
| `sma_error` (13) | `/5e5` | 43% | 54% |
| `apoapsis_alt` (14) | `clamp(±10e6)/1e6 − 1` | 39% | 46% |
| `radial_velocity` (2) | `2(vr/1e3+1.2)/1.5 − 1` | 1% | 46% |
| `orbital_energy` (3) | `−μ/2a / 6e6` | 10% | 26% |
| `hdot_nominal` (18) | `/500` | 15% | 22% |

These are wide-dynamic-range, sign-bearing quantities; a fixed linear scale (or a clamp,
as on `apoapsis_alt`) cannot keep them in range across capture / skip-out / crash regimes.

Ablation cannot currently rank input *usefulness* (the deployed models learned
near-"hold" policies that barely read their inputs -- zeroing any single input moves DV by
~0), so **discarding** inputs is deferred until an input-sensitive trained model exists.
This pass does the data-justified work: **rescale** the saturating inputs and **add
periapsis altitude** (a strong capture-targeting signal the user requested).

## Goal

1. Replace clamps + ad-hoc linear scalings on the five saturating inputs with a single
   principled **`asinh(x / s)`** signed-log transform.
2. Add `periapsis_alt` as a new candidate input (asinh).
3. Scale factors `s` chosen data-driven so the operating range fills `[-1, 1]`.
4. Verify via the input report that saturation drops below ~5% per rescaled input.

## Design

### 1. Transform: `asinh(x / s)`

Signed-log: linear near 0 (`asinh(z) ≈ z` for small `z`), log-like in the tails
(`asinh(z) ≈ sign(z)·ln(2|z|)` for large `|z|`), sign-preserving (handles negative
apoapsis/periapsis on hyperbolic / crashing orbits, where `log` is undefined). Rust:
`f64::asinh`. No clamp.

### 2. Scale factors -- data-driven, no new instrumentation

Each `s_i` is set so `asinh(p99_raw_i / s_i) ≈ 1.3` -> `s_i = p99_raw_i / sinh(1.3)`
(`sinh(1.3) ≈ 1.698`). The `1.3` target leaves modest headroom (p99 maps to ~1.3, a hair
outside `[-1, 1]`, true tails compress beyond).

`p99_raw_i` is measured in pure Python from `aerocapture_rs.run_mc(..., include_trajectories=True)`
(no Rust instrumentation needed): per-tick `r`, `v`, `fpa`, `lat` from the 17-column
trajectory give absolute energy, `a`, `e` -> apoapsis/periapsis altitudes;
`radial_velocity = v·sin(fpa)`; `orbital_energy` directly; `hdot_nominal` via the ref-traj
interpolation; `sma_error = a − a_target`. The measurement is a one-off analysis pass that
prints the six constants; they are then baked into `build_nn_input` as named consts.

The constants are produced by the plan's first task and are NOT placeholders -- the
procedure (measure p99 -> divide by sinh(1.3)) deterministically yields them.

### 3. `build_nn_input` (`src/rust/src/gnc/guidance/neural.rs`)

Named scale constants near the top of the module, e.g.:

```rust
const S_RADIAL_VELOCITY: f64 = /* p99/sinh(1.3) */;
const S_ORBITAL_ENERGY: f64 = /* ... */;   // applied to the raw -mu/(2a) energy (J/kg)
const S_SMA_ERROR: f64 = /* ... */;
const S_APOAPSIS_ALT: f64 = /* ... */;     // meters
const S_HDOT_NOMINAL: f64 = /* ... */;
const S_PERIAPSIS_ALT: f64 = /* ... */;    // meters
```

Rescale in place (indices unchanged):
- `full_input[2]  = (velocity_radial / S_RADIAL_VELOCITY).asinh();`
- `full_input[3]  = (raw_orbital_energy / S_ORBITAL_ENERGY).asinh();`  (raw = `-mu/(2a)`)
- `full_input[13] = (nav.orbital_errors[0] / S_SMA_ERROR).asinh();`
- `full_input[14] = (orbit.apoapsis_alt / S_APOAPSIS_ALT).asinh();`   (unclamped)
- `full_input[18] = (hdot_nominal / S_HDOT_NOMINAL).asinh();`

Add new input at index 31:
- `full_input[31] = (orbit.periapsis_alt / S_PERIAPSIS_ALT).asinh();`

`NN_FULL_INPUT_SIZE`: 31 -> 32. The `(sin,cos)` bank-history pairs at 25-30 are untouched.
The module-doc input table is updated (new index 31, rescaled note on 2/3/13/14/18).

### 4. Plumbing (the 31 -> 32 + new-index ripples)

- `src/rust/src/simulation/tick.rs`: `FULL_MASK` -> `[0..32]` (`[usize; 32]`).
- `src/python/aerocapture/training/ablation.py`: `NN_INPUT_NAMES` -> 32 entries
  (append `"periapsis_alt"`).
- `src/python/aerocapture/training/warm_start.py`: `_CANDIDATE_INPUT_WIDTH` 31 -> 32.
- `src/python/aerocapture/training/config.py`: `_RUNTIME_CANDIDATE_WIDTH` 31 -> 32.
- `src/rust/aerocapture-py/src/lib.rs`: the `NN_INPUT_WIDTH` const in `collect_supervised`
  and `collect_nn_inputs` 31 -> 32 (X arrays become `(T, 32)`).
- `input_mask` validation upper bound 31 -> 32 (Rust `data/mod.rs` /
  `validate_output_parameterization` site and any Python mirror).

The default mask (`None` -> `[0..16]`) is unchanged in *membership*, but indices 2/3/13/14
are in it, so default-mask models' input *values* change (see Consequences).

### 5. Training configs

`configs/training/msr_aller_nn_delta_train.toml` and
`configs/training/msr_aller_nn_scaledpi_train.toml`: append index `31` to `input_mask`
(29 -> 30 indices) and bump the first `[[network.architecture]]` `input_size` 29 -> 30.
Other schemes' configs are left untouched (they keep their masks; the rescaled-index
*values* change, which is a retrain, not a config edit).

### 6. Consequences

- **`neural` guidance golden regenerates.** Rescaling indices 2/3/13/14/18 changes
  `build_nn_input` output, invalidating `tests/reference_data/rust_golden/` for the
  `neural` scheme only (the other 5 goldens don't touch the NN). Regenerate by running the
  updated binary on the neural test config and replacing the CSV; eyeball the diff is
  confined to NN-driven columns.
- **All existing NN models become invalid** (weights trained on the old scalings) --
  expected; they get retrained. The deployed delta/scaledpi/gru `best_model.json` files
  will produce different (meaningless) output until retrained.
- **Cross-language equivalence unaffected** -- it tests `nn_forward` (the network), not
  `build_nn_input`.

### 7. Acceptance

Re-run `nn_input_report` on a freshly-(even briefly-)trained or smoke model and confirm
each rescaled input (`radial_velocity`, `orbital_energy`, `sma_error`, `apoapsis_alt`,
`hdot_nominal`) shows `frac_out_of_range` < ~5%, and `periapsis_alt` is present and
in-range. (Saturation is a property of the *scaling* + state distribution, not the model,
so even an untrained / zero-weight model gives a valid saturation read.)

## Testing

Rust (`gnc/guidance/neural.rs`):
- `NN_FULL_INPUT_SIZE == 32`.
- Each rescaled index equals `asinh(raw / s_i)` for a known fixture state.
- `periapsis_alt` at index 31, finite, equals `asinh(orbit.periapsis_alt / s)`.
- A huge raw value (e.g. apoapsis 1e9 m) yields a bounded, small-magnitude asinh output
  (saturation sanity -- the whole point).
- Default-mask path still returns length 16, all finite.

Python:
- `NN_INPUT_NAMES` length 32, unique, contains `periapsis_alt`.
- `tests/test_ablation.py` length assertion 31 -> 32.
- Updated delta/scaledpi configs load with `len(input_mask) == architecture[0].input_size == 30`.

Golden:
- Regenerate the `neural` golden; the other 5 stay bit-identical.

## Out of scope (deferred)

- **Discarding** inputs -- needs an input-sensitive trained model to ablate (current ones
  ignore inputs). Revisit after a real PSO/warm-start run; `sma_error` is the leading
  drop candidate once `apo`+`peri` are present (`sma = (apo+peri)/2 + R`), but no ablation
  evidence backs it yet.
- Rescaling other schemes' config masks (gru/lstm/etc.) -- they retrain on the new
  scalings without config edits; adding periapsis to their masks is a per-scheme choice.
- Tuning the `1.3` asinh target per input.
