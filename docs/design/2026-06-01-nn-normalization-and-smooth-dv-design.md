# NN Input Pipeline v3: Unified Normalization Schema + Smooth Always-Defined DV

**Date:** 2026-06-01
**Branch:** `feature/nn-input-rescale` (or a fresh branch off it)
**Status:** Design approved, pending implementation plan
**Predecessor:** `docs/superpowers/specs/2026-06-01-nn-input-vector-v2-design.md` (added the DV inputs + data-driven renorm; this spec generalizes the normalization and fixes the DV sentinel).

## Motivation

Two coupled cleanups, validated by the v2 work + the post-deploy review:

1. **Normalization is fragmented and code-baked.** Each input's transform is a bespoke inline expression in `build_nn_input` (asinh for some, affine for others, `*2-1` for flags, `tanh(.../30)` for telemetry), and the scale constants live in `neural.rs` AND are mirrored by hand in `calibrate_inputs.py::CURRENT_TRANSFORMS` (guarded only by a parity test). Recalibration requires editing Rust and rebuilding.

2. **The DV sentinel is a wart.** `predicted_dv1/2/3` emit a `+1.5` normalized sentinel pre-capture (hyperbolic orbit). It is not truly out-of-band (`asinh` is unbounded, so a live `|dv| ≈ 2.13·S` collides with it), it is a dead constant for ~21% of the trajectory (the capture phase, where the DV inputs are otherwise the top-ranked signals), and it conflates "undefined" with a magnitude.

## Decisions (locked during brainstorming)

| Decision | Choice |
| --- | --- |
| Normalization schema | Uniform `{transform, scale, center}` per input, **divisor form** `norm = transform((raw − center) / scale)`, `transform ∈ {none, asinh, tanh}` |
| Where it lives | Embedded in the model JSON (self-describing, like `output_param`); optional TOML `[network.normalization]` override; absent → baked defaults (backward-compatible) |
| DV sentinel | **Removed.** All three DV components redefined to be defined + smooth across `e=1`; they become plain `asinh` inputs in the schema |
| `predicted_dv1` | Energy-closing burn at current periapsis (vis-viva) — "Δv to shed excess energy and close the orbit" |
| Input removal | **Not done.** The 35-candidate library + mask stays (removing/renumbering would invalidate trained models for ~zero runtime gain). |

## Part A — Unified normalization schema

### A.1 The form

For every numeric candidate input, after its input-specific *extraction* produces a raw scalar:

```
norm = transform((raw - center) / scale)
```

with `transform ∈ {none, asinh, tanh}`. The affine `(raw - center)/scale` is always present (degenerate = `center 0, scale 1`); the nonlinearity is optional on top. `scale` is the characteristic magnitude (the data-driven divisor), `center` the value subtracted before scaling.

This subsumes every current transform:

| current expression | `{transform, scale, center}` |
| --- | --- |
| `asinh(raw / S)` | `{asinh, S, 0}` |
| `(raw − center) / half` | `{none, half, center}` |
| `raw` (sin/cos, cos_bank) | `{none, 1, 0}` |
| `tanh(t / 30)` (telemetry 23/24) | `{tanh, 30, 0}` |
| `flag·2 − 1` (bounce_flag) | `{none, 0.5, 0.5}` |

`asinh` = soft/unbounded (graceful tails); `tanh` = hard-bounded to (−1,1); `none` = pure affine.

### A.2 Where it lives — model JSON + TOML override

- **Model JSON (`NnModelFile`):** add `#[serde(default)] normalization: Option<Vec<NormSpec>>` — a per-candidate-input list of length `NN_FULL_INPUT_SIZE` (indices align with the candidate vector, NOT the mask). `NormSpec { transform: NormTransform, scale: f64, center: f64 }` with `NormTransform` a `#[serde(rename_all="snake_case")]` enum `{ None, Asinh, Tanh }`. Absent (`None`) → fall back to the baked default table (backward-compat for existing/legacy models).
- **TOML override:** optional `[network.normalization]` (or `[[network.normalization]]` list) parsed in `config.rs`, overlaid onto the loaded model's block (same precedence pattern as `input_mask`/`output_parameterization` overrides).
- **Single source of truth:** with the block embedded, `calibrate_inputs.py::CURRENT_TRANSFORMS` and the `neural.rs` const block both go away; the parity test (`tests/test_nn_scale_parity.py`) is retired.

### A.3 build_nn_input refactor — extract, then normalize

Split `build_nn_input` into two phases:

1. **Extraction** (input-specific, unchanged logic): compute each candidate's *raw* scalar into a `[f64; NN_FULL_INPUT_SIZE]` — orbit elements, nav fields, ref-traj interpolations, `compute_deltav` (see Part B), and the angle→(sin,cos) pairs. The sin/cos extraction stays special (one angle → two raw outputs, each already in [−1,1]).
2. **Normalization** (uniform loop): `for i in 0..N { norm[i] = apply_norm(raw[i], &norm_spec[i]); }` where `apply_norm` matches on `transform`. `norm_spec` comes from the model (or the baked default table when absent). Then mask-select as today.

`apply_norm` is a small pure function: `match transform { None => v, Asinh => v.asinh(), Tanh => v.tanh() }` applied to `(raw - center)/scale`.

### A.4 Baked default table (backward-compat)

A `const DEFAULT_NORMALIZATION: [NormSpec; NN_FULL_INPUT_SIZE]` in `neural.rs` holding today's calibrated values (the post-Part-B values for the DV inputs). Used when a model carries no `normalization` block. This is the ONE place defaults live; calibration overwrites per-model.

## Part B — Smooth, always-defined DV (sentinel removed)

All three DV components are redefined to be finite and continuous across the `e=1` (parabolic) boundary. Periapsis radius `rp = a(1−e)` is positive for every conic (only apoapsis blows up for hyperbolics), so periapsis-referenced burns are continuous through capture.

Let `mu = planet.mu`, `req = planet.equatorial_radius`, `a = orbit.semi_major_axis` (negative for hyperbolic), `rp = req + orbit.periapsis_alt`, `ra_t = req + parking.apoapsis` (target apoapsis), `rp_t = req + parking.periapsis` (target periapsis).

**`predicted_dv1` — energy-closing burn (NEW, the user's "loss of energy to close"):** Δv at the current periapsis to bring apoapsis (hence energy) to the target apoapsis.
```
v_cur = sqrt(mu * (2/rp - 1/a))                  // vis-viva at current periapsis; a<0 -> higher speed
a_t1  = (rp + ra_t) / 2                           // SMA of orbit with periapsis rp, apoapsis ra_t
v_tgt = sqrt(mu * (2/rp - 1/a_t1))
dv1   = v_cur - v_tgt
```
Always defined (`rp>0`, `a` finite≠0; at `e=1`, `1/a→0` so `v_cur → sqrt(2mu/rp)` = escape speed — continuous). Pre-capture: large retro Δv = energy to shed to close. Post-capture: the apoapsis/energy correction.

**`predicted_dv2` — periapsis correction (apoapsis-referenced):** keep the current `compute_deltav` `dv1` formula (burn at apoapsis to set periapsis to target), but **define `dv2 = 0` when the orbit is hyperbolic / apoapsis undefined** (`orbit.eccentricity >= 1.0` or `rapoge` non-finite/≤0). This is the continuous limit: the current formula's Δv → 0 as `rapoge → ∞` (the `1/(rapoge·(rapoge+r))` terms vanish), so extending with 0 introduces no discontinuity. Physically honest: periapsis trim is deferred until captured.

**`predicted_dv3` — inclination plane change:** unchanged — `compute_deltav`'s `dv3` (node velocity from target orbit × `sin(Δi/2)`) is already defined for all orbits.

### B.1 Implementation note

This needs a dedicated NN-input DV function (e.g. `maneuver::predicted_dv_for_nn(orbit, parking, target, planet) -> [f64; 3]`) rather than reusing `compute_deltav` verbatim, because (a) `dv1` is redefined (energy-closing, periapsis-referenced) and (b) `dv2` gets the hyperbolic→0 guard. The terminal-cost `compute_deltav` (used by `runner.rs` for the final maneuver plan) is UNCHANGED — only the NN-input DV is redefined. Keep them separate functions.

### B.2 Sentinel removal cascade

Removing the sentinel deletes from `build_nn_input` / `neural.rs`: `DV_SENTINEL_NORM`, the `e<1` gate + `is_finite` backstop branch for the DV block, and the per-component sentinel handling. The DV inputs become three ordinary `asinh` entries in the normalization schema (`{asinh, S_DVi, 0}`). `calibrate_inputs.py`'s `drop_sentinel` + `_DV_INDICES` special-casing is removed (no sentinel ticks to exclude — DV is meaningful throughout, so the full distribution calibrates honestly).

## Touch list

**Rust**
- `data/neural.rs`: `NormSpec` + `NormTransform` types; `normalization: Option<Vec<NormSpec>>` on `NnModelFile` (serde default); `DEFAULT_NORMALIZATION` const; thread the resolved per-input specs into `build_nn_input`.
- `gnc/guidance/neural.rs`: refactor `build_nn_input` into extract-then-normalize; `apply_norm` helper; remove sentinel + per-index inline transforms (replaced by the uniform loop + default table); call the new DV function.
- `orbit/maneuver.rs`: new `predicted_dv_for_nn(...)` (dv1 energy-closing, dv2 periapsis with hyperbolic→0, dv3 inclination). `compute_deltav` untouched.
- `config.rs`: parse `[network.normalization]` TOML override; overlay onto the model block.

**Python**
- `training/calibrate_inputs.py`: emit a `normalization` JSON block (write into the model / a sidecar) instead of Rust consts; drop `CURRENT_TRANSFORMS`, `_ASINH_CONST_NAME`, `_AFFINE_CONST_NAME`, `drop_sentinel`, `_DV_INDICES`. The inversion still needs to know the *current* per-input transform to recover raw — read it from the model's `normalization` block (or the default table exported once), closing the dual-maintenance loop.
- `training/model_io.py` / `rl/export.py`: round-trip the `normalization` block (export/load).
- `tests/test_nn_scale_parity.py`: **retire** (single source of truth makes it moot) — or repoint it to assert the model block matches the default table only when absent.

**Tests**
- Rust: `apply_norm` unit tests (each transform, divisor+center); DV continuity test across `e=1` (sweep eccentricity through 1.0, assert `predicted_dv*` continuous, no jump > tolerance); `dv2 == 0` for hyperbolic; `dv1` = energy-closing sign/magnitude sanity; model with explicit `normalization` block overrides the default; model without it uses the default (bit-identical to current output for the legacy 16-input golden).
- `neural` golden: it uses a 16-input model (`input_mask=None` → indices 0-15), so the DV redefinition (indices 32-34) does NOT touch it. The normalization refactor must keep 0-15 **bit-identical** (the divisor form `(raw-center)/scale` must equal the old `raw/S` and `(raw-center)/half` expressions exactly — they do algebraically). So the expectation is the golden stays unchanged; regenerate ONLY if a float reassociation shifts it (investigate first — an unexpected golden change signals a refactor bug, not an intended change).
- Python: `calibrate_inputs` emits a valid normalization block; round-trip export/load.

**Configs / models**
- Existing deployed models have no `normalization` block → use defaults (backward-compat), BUT their DV inputs now use the redefined DV → **retrain required** for any model using DV inputs (all three current decoders). Document in the plan.

## Consequences

- **Retrain required** for the three decoders (DV semantics changed). Expected — the user is iterating; the redefined `dv1` should be a *stronger* input throughout (the old `dv1` was the weakest at 0.32 precisely because it was the dead/undefined one).
- **Dual-maintenance eliminated:** one source of truth for normalization (the model block + a single default table). Parity test retired.
- **Recalibration without Rust rebuild:** `calibrate_inputs.py` writes the block directly.
- **`build_nn_input` simplifies:** uniform normalization loop; the only remaining special-case is angle→(sin,cos) extraction.
- The candidate-vector width (35) and the 17-input shared mask are UNCHANGED.

## Validation plan

1. **Continuity:** Rust test sweeping a trajectory state through `e = 1 ± ε`, asserting `predicted_dv1/2/3` are continuous (no jump) and finite.
2. **Backward-compat:** a model with no `normalization` block reproduces today's normalized values bit-for-bit (the default table == current baked transforms, modulo the DV redefinition at 32-34); all 6 goldens (5 non-NN + the 16-input `neural`) stay bit-identical, since the `neural` golden uses only indices 0-15.
3. **Retrain + compare:** retrain the three decoders; final-eval DV vs the current 124/143 (atan2), 129/146 (delta), 132/159 (scaled_pi). Expect hold-or-improve, especially with a live pre-capture DV gradient.
4. **Ablation:** confirm `predicted_dv1` (redefined) is now a strong input across decoders (was the weak one), and the report shows the DV inputs no longer have the ~21% sentinel saturation (they fill `[-1,1]` like any asinh input).
