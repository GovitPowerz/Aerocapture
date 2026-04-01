# NN Expanded Inputs & Full-Envelope Guidance

**Date:** 2026-04-02
**Status:** Draft

## Goal

Expand the neural network guidance scheme from 8 to 16 inputs, drawing on all
useful data available from navigation/sensors. Enable the NN to operate across
both capture and exit phases as a single phase-blind controller (Approach A),
replacing the need for the shared exit-phase pdyn-feedback controller.

## Current State

The NN (`neural.rs`) takes 8 inputs (eccentricity excess, inclination error,
radial velocity, orbital energy, velocity, acceleration magnitude, heat flux
fraction, heat load fraction), produces a signed bank angle via atan2, and
bypasses lateral guidance, thermal limiter, and exit guidance. In the current
dispatch logic it runs its capture-phase match arm regardless of `guidance_phase`
-- it already handles all phases, but with only capture-relevant inputs.
Default architecture: 8-12-2.

## Design

### Input Vector (8 -> 16)

Existing inputs retain their indices and normalization. New inputs are appended.

| #  | Input              | Source                          | Normalization               | Rationale                        |
|----|--------------------|---------------------------------|-----------------------------|----------------------------------|
| 0  | Eccentricity excess| `e - 1.0`                      | raw                         | (existing)                       |
| 1  | Inclination error  | `(i - i_target)` deg            | `* 3/5`                     | (existing)                       |
| 2  | Radial velocity    | `V * sin(gamma)`                | affine to ~[-1,1]           | (existing)                       |
| 3  | Orbital energy     | `-mu / (2a)`                    | `/ 6e6`                     | (existing)                       |
| 4  | Velocity           | `V`                             | affine to ~[-1,1]           | (existing)                       |
| 5  | Accel magnitude    | `sqrt(drag^2 + lift^2)`         | `/ 20 - 1`                  | (existing)                       |
| 6  | Heat flux fraction | `[0,1]`                         | `* 2 - 1`                   | (existing)                       |
| 7  | Heat load fraction | `[0,1]`                         | `* 2 - 1`                   | (existing)                       |
| 8  | Altitude           | `r - R_eq` (m -> km)            | `(alt_km - 65) / 65`        | Depth in atmosphere              |
| 9  | Flight path angle  | `gamma` (rad)                   | `/ 0.3`                     | Vertical vs horizontal flight    |
| 10 | Latitude           | `lat` (rad)                     | `/ (pi/2)`                  | Gravity/wind variation           |
| 11 | Drag acceleration  | `accel_estimated[0]` (m/s^2)    | `/ 50 - 1`                  | Separate from lift               |
| 12 | Lift acceleration  | `accel_estimated[1]` (m/s^2)    | `/ 10`                      | Separate from drag               |
| 13 | SMA error          | `orbital_errors[0]` (m)         | `/ 5e5`                     | Direct orbital targeting         |
| 14 | Apoapsis altitude  | `orbit.apoapsis_alt` (m)        | `/ 1e6 - 1`                 | Key exit phase observable (negative when hyperbolic) |
| 15 | Bounce flag        | `nav.bounce_flag` (0 or 1)      | `* 2 - 1` -> {-1, 1}        | Phase discriminator              |

Normalization ranges chosen so typical MSR values map to roughly [-1, 1].

### Phase Dispatch (No Change)

The NN already bypasses exit guidance, thermal limiter, and lateral guidance in
`dispatch.rs` (renamed from `ftc.rs`). When `guidance_phase == 2`, the NN
continues running its capture-phase match arm -- the bounce flag input (#15)
lets it learn phase-dependent behavior internally. No dispatch logic changes.

### Guidance Dispatcher Refactor

**Rename `ftc.rs` -> `dispatch.rs`:**
- `guidance_step()` stays in `dispatch.rs`
- `FtcState` -> `GuidanceState`
- `FtcOutput` -> `GuidanceOutput`

**Extract FTC into its own file (`ftc.rs`):**
- `capture_guidance()` and `compute_gains()` move to new `ftc.rs`
- Public entry point: `ftc_bank_angle()` matching the pattern of other schemes
- Tests for FTC-specific logic move with the code

**`guidance/mod.rs`:** Update module declarations.

**Files affected by rename:**
- `runner.rs`, `simulation/init.rs` -- `FtcState`/`FtcOutput` references
- All guidance test files referencing these types

### Default Network Architecture

Default changes from `[8, 12, 2]` to `[16, 24, 2]`. TOML `layer_sizes` override
continues to work as before. Single hidden layer with proportionally scaled width.

### Python Training Side

- **`initialization.py`**: Uses `layer_sizes` from config; works automatically
  with the new default.
- **`evaluate.py`**: No changes (passes weights to Rust, doesn't construct inputs).
- **`param_spaces.py`**: No changes (NN is weight-based).
- **Training TOML** (`msr_aller_nn_train_consolidated.toml`): Update default
  `layer_sizes` to `[16, 24, 2]` or remove to pick up new default.
- **NN JSON format**: Schema unchanged; first layer weight matrix grows from
  8 columns to 16.

### Test Updates

**`neural.rs` tests:**
- All hardcoded weight matrices widen from 8 to 16 columns
- `zero_weight_nn`, `fixed_small_nn`, the 8->3->2 network all need updating
- Proptest structure unchanged, just wider weights
- `test_nav()` fixture already has all NavigationOutput fields

**`dispatch.rs` tests (moved from `ftc.rs`):**
- Phase dispatch tests use `GuidanceType::Ftc`, unaffected by NN input changes
- `FtcState`/`FtcOutput` renamed to `GuidanceState`/`GuidanceOutput`

**New `ftc.rs` tests:**
- `capture_guidance` and `compute_gains` tests relocate from old `ftc.rs`
- No logic changes

**Integration/E2E tests:**
- Any test config referencing NN with hardcoded weight files needs regenerated weights

**Python tests:**
- Tests constructing NN chromosomes or weight matrices need 16-input layer sizes

### Breaking Changes

- Existing NN weight files (JSON) are incompatible -- retraining required
- `FtcState`/`FtcOutput` renamed -- downstream code must update imports

### Not Changed

- Phase transition logic in `estimator.rs`
- `NavigationOutput` struct (already carries all needed fields)
- Other guidance schemes (EqGlide, EnergyController, PredGuid, FNPAG,
  PiecewiseConstant, FTC)
- Thermal limiter, lateral guidance
- Exit guidance module (`exit.rs`) -- still used by unsigned-magnitude schemes
