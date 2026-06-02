# NN Input Vector v2: Renormalize + pdyn_error + Live Correction-DV

**Date:** 2026-06-01
**Branch:** `feature/nn-input-rescale`
**Status:** Design approved, pending implementation plan

## Motivation

After iterative pruning we landed on a shared 13-input mask that performs well
(atan2 132/158, scaled_pi 139/181, delta 143/192 m/s DV p50/p95) but with two
known gaps:

1. **delta's p95 tail** (162 -> 192) traced to the dropped corridor-tracking
   reference. The control-theoretic fix is `pdyn_error` (current - nominal
   dynamic pressure), the direct corridor feedback signal -- not `pdyn_nominal`
   (only the reference) and not `pdyn_current` (largely redundant with
   `drag_accel`, which is already in the mask).
2. **Crude linear normalization** on several inputs -- notably `drag_accel`
   (`raw/50 - 1`) and `lift_accel` (`raw/10`) -- that do not fill the `[-1, 1]`
   operating range, unlike the six data-driven `asinh` inputs (~1% saturation).

Separately, we want to test **cost-aligned inputs**: the orbital correction
delta-V decomposed into its three signed components. The terminal `dv1/dv2/dv3`
in the final record are not causal (computed once at capture, they are the
objective), but `compute_deltav` evaluated **per tick on the current osculating
orbit** is causal and arguably the most directly cost-aligned input available.

## Decisions (locked during brainstorming)

| Decision | Choice |
| --- | --- |
| Which input is "13" | `pdyn_error` (index 19), added to all three decoders |
| Renormalization scope | Re-derive **all** input scales, data-driven |
| Correction-DV shape | 3 signed components, `asinh`, hyperbolic-guarded; add all 3 to every mask |

## Design

### 1. Candidate vector 32 -> 35 (append-only)

Three new inputs appended at the end so masks/models on other branches do not
shift:

| index | name | source |
| --- | --- | --- |
| 32 | `predicted_dv1` | `compute_deltav(...).dv1` on current osculating orbit |
| 33 | `predicted_dv2` | `.dv2` |
| 34 | `predicted_dv3` | `.dv3` |

`NN_FULL_INPUT_SIZE = 35` (`data/neural.rs`). The append-only choice means the
candidate vector is no longer grouped by theme; accepted trade-off.

### 2. Live correction-DV computation + hyperbolic guard

In `build_nn_input`, after `orbit` is available:

```rust
let (dv1, dv2, dv3) = if orbit.eccentricity < 1.0 {
    let dv = n::compute_deltav(&orbit, &data.target_orbit, &data.parking_orbit, planet);
    if dv.dv1.is_finite() && dv.dv2.is_finite() && dv.dv3.is_finite() {
        (dv.dv1, dv.dv2, dv.dv3)
    } else {
        (DV_SENTINEL, DV_SENTINEL, DV_SENTINEL)
    }
} else {
    // hyperbolic / open orbit: apoapsis undefined, compute_deltav ill-defined
    (DV_SENTINEL, DV_SENTINEL, DV_SENTINEL)
};
full_input[32] = (dv1 / S_DV).asinh();
full_input[33] = (dv2 / S_DV).asinh();
full_input[34] = (dv3 / S_DV).asinh();
```

`DV_SENTINEL` is chosen so `asinh(DV_SENTINEL / S_DV) = 1.5` exactly
(`DV_SENTINEL = S_DV * sinh(1.5)`), an out-of-band-but-bounded value the network
reads as "still capturing / far from target." Sign is meaningless pre-capture,
so all three saturate to the same `+1.5`.

Post-capture the components carry the true signed per-axis distance-to-target in
DV units. `compute_deltav` already has every argument it needs in
`build_nn_input` (`orbit`, `data.target_orbit`, `data.parking_orbit`, `planet`).

### 3. Data-driven renormalization (all inputs)

A one-time calibration utility (`src/python/aerocapture/training/calibrate_inputs.py`):

1. Runs `aerocapture_rs.collect_nn_inputs` over a reserved seed pool to collect
   the **normalized** `(T, 35)` candidate trace for every seed.
2. Inverts each input's **known current transform** to recover the raw
   distribution (asinh: `raw = s * sinh(norm)`; affine: `raw = (norm - b) / a`).
3. Per input, computes p1/p50/p99 of the raw values and classifies:
   - **Heavy-tailed / spiky** (accelerations `drag_accel`, `lift_accel`,
     `accel_magnitude`; the DV components; existing wide-range inputs):
     `asinh(raw / s)` with `s = max(|p1|, |p99|) / sinh(1.0)` so p99 lands at
     ~1.0, no clamp.
   - **Bounded / symmetric** (`*2 - 1` fractions, `ecc_excess`,
     `inclination_error`, reference `cos_bank`): affine
     `(raw - center) / halfwidth` with `center = (p1 + p99) / 2`,
     `halfwidth = (p99 - p1) / 2`.
4. Emits a `const S_*` block (and affine center/halfwidth literals) to paste into
   `neural.rs`, each annotated with the p1/p99 it was derived from.

Classification rule (encoded in the script): use `asinh` when the tail ratio
`max(|p1|, |p99|) / median(|raw|)` exceeds a threshold (heavy tail) **or** the
input is an acceleration / DV; otherwise affine. Borderline cases are listed in
the script output for manual confirmation.

The six existing asinh inputs (radial_velocity, orbital_energy, sma_error,
apoapsis_alt, hdot_nominal, periapsis_alt) are re-confirmed by the same pass and
expected to barely move (already ~1% saturation). The real fixes land on
`drag_accel` / `lift_accel` / `accel_magnitude`.

Scale constants remain named `const S_*` in `neural.rs` (established pattern,
e.g. `S_RADIAL_VELOCITY: f64 = 8.802043e+02`). Python never re-derives scales --
`build_nn_input` runs only in Rust; `collect_nn_inputs` / `collect_supervised`
call it through PyO3, and the cross-language equivalence tests feed explicit
vectors to `nn_forward` (so they are unaffected by `build_nn_input` changes).

### 4. Masks -> 17-input (all three decoders)

Shared mask `[0, 2, 3, 5, 6, 7, 11, 12, 18, 19, 27, 28, 29, 30, 32, 33, 34]`
(current 13 + `pdyn_error`(19) + `predicted_dv` 32/33/34), first-layer
`input_size` -> 17 in:

- `configs/training/msr_aller_nn_train_consolidated.toml` (atan2)
- `configs/training/msr_aller_nn_scaledpi_train.toml` (scaled_pi)
- `configs/training/msr_aller_nn_delta_train.toml` (delta)

### 5. Touch list

**Rust**
- `gnc/guidance/neural.rs`: new/updated scale consts (`S_DV` + re-derived `S_*`
  + affine literals), `build_nn_input` renormalization of the affine/linear
  inputs, the 3 live DV inputs + hyperbolic guard, updated input-name doc
  comment (35 entries).
- `data/neural.rs`: `NN_FULL_INPUT_SIZE 32 -> 35`. `validate_mask` already
  checks `idx >= NN_FULL_INPUT_SIZE` and `ablated_input` against the same const,
  so updating the const is sufficient -- no validation-logic change.
- `config.rs`: no change (mask parsing is size-agnostic; validation lives in
  `data/neural.rs`).

**Python**
- `training/ablation.py`: `NN_INPUT_NAMES` 32 -> 35 (append `predicted_dv1`,
  `predicted_dv2`, `predicted_dv3`).
- `training/nn_input_report.py`: name list / any literal 32.
- `training/calibrate_inputs.py`: new calibration script (section 3).

**Configs**: 3 masks + `input_size` (section 4).

**Tests**
- Regenerate the `neural` Rust golden (input vector changed; non-NN goldens
  unaffected).
- Update the `NN_INPUT_NAMES` length test (32 -> 35) + DV-name canary.
- Update any Rust unit test asserting the 32-element candidate size.
- Cross-language equivalence tests (`test_v2_rust_python_equivalence.py`):
  unaffected (explicit vectors), but confirm they do not assert
  `NN_FULL_INPUT_SIZE`.

### 6. Calibration workflow (one-time, reproducible)

1. Add the 3 DV inputs with a provisional `S_DV` (domain guess ~128 from
   ~150 m/s typical component); rebuild PyO3.
2. Run `calibrate_inputs.py` -> raw distributions for all 35 -> emit `const S_*`
   block.
3. Paste consts into `neural.rs`, rebuild, re-run calibration to verify
   p99 ~= 1.0 across inputs (one feedback loop).

## Risks / open notes

- **`predicted_dv` partial redundancy** with `apoapsis_alt` / `sma_error` /
  `eccentricity_excess`: same underlying distance-to-target, but in DV units and
  nonlinearly combined -- a genuinely different projection. Post-retrain
  ablation decides whether it earns its place.
- **Retrain required**: changing scales + adding inputs invalidates all existing
  13-input NN models. All three decoders retrain from scratch (chromosome width
  17 != 13 -> auto-resume will refuse; use `--from-scratch` or fresh dirs).
- **Sentinel discontinuity** at `e = 1.0`: the DV inputs jump from `+1.5` to the
  live value as the orbit closes. This is a real feature edge but bounded and
  monotone-ish (live value is large near capture, shrinking as it settles); the
  network sees a smooth-ish "far -> near" transition.

## Validation plan

After implementation + retrain of all three decoders:
1. Final-eval MC (1000 sims) vs the 13-input baseline -- expect delta tail to
   tighten (pdyn_error) and overall DV to hold or improve.
2. Ablation on all three 17-input models -- confirm the new inputs
   (`pdyn_error`, `predicted_dv*`) carry weight and nothing collapsed.
3. Input report -- confirm renormalized inputs (esp. `drag_accel`, `lift_accel`)
   now fill `[-1, 1]` with ~1% saturation.
