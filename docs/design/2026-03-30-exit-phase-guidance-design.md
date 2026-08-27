# Exit Phase Guidance Design

**Date:** 2026-03-30
**Status:** Approved
**Ref:** IMPROVEMENTS.md §4.2, TODO.md line 6

## Problem

The navigation layer tracks `guidance_phase` (capture=1, exit=2, emergency=3) with complete bounce detection, phase transition logic, and crash detection — but both `navigate()` and `navigate_ekf()` unconditionally override `guidance_phase = 1` at the end of every step. No exit-phase guidance algorithm exists. The `SimPhase` config enum (`Full`, `CaptureOnly`, `ExitOnly`, `Preprogrammed`) is parsed but never consumed.

Five exit-phase parameters are loaded from TOML and stored in `GuidanceParams` but never read by guidance:
- `exit_velocity_threshold` (4400.0 m/s) — velocity at which phase 1 → 2 fires
- `exit_pdyn_margin` (1.75) — dynamic pressure scaling for exit target
- `exit_altitude_threshold` (60.0 km → 60000.0 m) — altitude for exit density lookup
- `exit_radial_vel_gain` (10.0 Pa/(m/s)) — damping gain on radial velocity error
- `exit_apoapsis_threshold` (100.0 m) — apoapsis comparison threshold

## Scope

- **In scope:** FTC + the four unsigned-magnitude schemes (EqGlide, EnergyController, PredGuid, FNPAG). These five schemes share a common exit controller after the phase transition.
- **Out of scope:** NN and PiecewiseConstant (they produce signed bank angles for the full trajectory, including the exit leg). GA-optimizability of exit params for non-FTC schemes (follow-up enhancement).

## Design

### 1. Navigation — Remove phase override, activate `SimPhase`

**Files:** `src/rust/src/gnc/navigation/estimator.rs`

Remove the hardcoded `nav_state.guidance_phase = 1` override in both `navigate()` (line 242) and `navigate_ekf()` (~line 566). Replace with `SimPhase`-conditional logic:

- `SimPhase::Full` — phase logic runs as-is (bounce detection → velocity threshold → phase 2 transition → crash detection → phase 3)
- `SimPhase::CaptureOnly` — force `guidance_phase = 1` always (equivalent to today's behavior; regression baseline)
- `SimPhase::ExitOnly` — force `guidance_phase = 2` always (useful for isolated testing of exit guidance)
- `SimPhase::Preprogrammed` — same as `Full` for now

The `capture_time` accumulator is corrected: it only increments when `guidance_phase == 1`, which happens naturally once the override is removed.

**Threading `SimPhase`:** Add `sim_phase` to `SimData` (populated from `SimInput.sim_phase` during config loading). The `navigate()` and `navigate_ekf()` functions already receive `&SimData`, so no signature changes needed.

### 2. New module — `gnc/guidance/exit.rs`

**New file:** `src/rust/src/gnc/guidance/exit.rs`

A shared exit-phase longitudinal controller. Stateless function:

```rust
pub fn exit_guidance(
    nav: &NavigationOutput,
    data: &SimData,
    planet: &PlanetConfig,
    reference_velocity: f64,  // radial velocity latched at phase transition
) -> f64  // bank angle magnitude (rad)
```

**Algorithm (dynamic pressure feedback with apoapsis correction):**

1. **Target dynamic pressure:** `pdyn_target = density_exit * V^2 * exit_pdyn_margin` — where `density_exit` = atmosphere density at `exit_altitude_threshold` (already computed every nav step, available in `nav.density_exit`), and `exit_pdyn_margin` = 1.75 (scaling factor for target drag on the ascending leg).

2. **Current dynamic pressure:** `pdyn_current = 0.5 * density_guidance * V^2` (from estimated density and velocity).

3. **Dynamic pressure correction:** `pdyn_correction = (pdyn_current - pdyn_target) / pdyn_current_safe` — normalized error between current and target dynamic pressure. When current pdyn exceeds target, this drives bank angle toward lift-down (more drag), decelerating the vehicle to lower the apoapsis.

4. **Radial velocity damping:** `radial_vel_correction = exit_radial_vel_gain * (velocity_radial - reference_velocity) / pdyn_current_safe` — damps altitude rate oscillations relative to the radial velocity latched at the moment of phase transition. Normalized by pdyn for gain consistency.

5. **Predictor-corrector:** `cos_bank = pdyn_correction + radial_vel_correction`. The pdyn term drives toward target energy dissipation; the radial velocity term provides damping. Apoapsis targeting emerges from the pdyn feedback: controlling drag on the ascending leg directly controls how much energy is removed before atmosphere exit, which determines the final apoapsis.

6. **Clamp and convert:** Clamp `cos_bank` to [-1, 1], return `acos(cos_bank)`.

The function is stateless. The only persistent value it needs (`reference_velocity`) is latched at the phase transition and carried via `FtcState.reference_velocity`.

### 3. Guidance dispatch — Phase branch in `ftc.rs`

**File:** `src/rust/src/gnc/guidance/ftc.rs`

Add a phase-aware dispatch in `guidance_step()` before the scheme-specific `match guidance_type`:

```
let uses_exit_guidance = !matches!(
    guidance_type,
    GuidanceType::PiecewiseConstant | GuidanceType::NeuralNetwork
);

if nav.guidance_phase == 2 && uses_exit_guidance && longitudinal_active == 1 {
    bank_angle_longitudinal = exit::exit_guidance(nav, data, planet, state.reference_velocity);
} else {
    bank_angle_longitudinal = match guidance_type { ... };  // existing dispatch
}
```

The `uses_exit_guidance` check is computed from `guidance_type` directly (not from `skip_lateral`, which is computed later in the current code for the lateral guidance decision).

Key behaviors:
- **NN and PiecewiseConstant excluded:** These schemes produce signed bank angles for the full trajectory, so exit guidance is bypassed.
- **Phase 3 (emergency/crash):** `longitudinal_active` will be 0 (energy outside activation window as the vehicle descends), so guidance defaults to `reference_bank_angle.abs()` (full lift-up). No special handling needed.
- **Lateral guidance remains active during exit phase:** The existing energy window gate (`lateral_activation` / `lateral_inhibition`) in `lateral.rs` already controls when lateral runs. If energy is still within the lateral window during exit, inclination correction continues. No changes needed in `lateral.rs`.

**Runner wiring** (`src/rust/src/simulation/runner.rs`): After the navigation step, if `nav_out.phase_transition_flag == 1`, copy `nav_out.reference_velocity` into `ftc_state.reference_velocity`. One-line addition.

### 4. Config plumbing — Exit params for non-FTC schemes

Currently the five `exit_*` params only appear in FTC-specific TOML configs (e.g., `configs/nominal/msr_aller_ftc_consolidated.toml`). Since exit guidance is now shared across five schemes, these params need to be available to all.

**Approach:** Move the exit params into the mission-level TOML configs (`configs/missions/mars.toml`, `configs/missions/earth.toml`) in a `[guidance.ftc]` section. All scheme-specific configs inherit from mission configs via `base`, so exit params become universally available through inheritance.

The Rust parser already handles missing `exit_*` fields gracefully (they default to 0.0 via `GuidanceParams::default()`). If exit params are 0.0, exit guidance produces `cos_bank = 0` → 90-degree bank angle — a safe neutral default.

GA-optimizability of exit params for non-FTC schemes is a follow-up enhancement. The FTC-tuned defaults (4400, 1.75, 60, 10, 100) are a reasonable starting point.

### 5. Output changes

**File:** `src/rust/src/simulation/runner.rs`

The `photo.csv` output currently computes its own independent phase value (lines 906-910) using a parallel heuristic based on altitude and bounce state. Replace this with `nav_out.guidance_phase` (cast to f64) so the output reflects the actual GNC phase. One-line change.

### 6. Testing strategy

**Rust unit tests (`exit.rs`):**
- Exit guidance returns finite bank angle for typical ascending-leg states
- Bank angle magnitude stays in [0, pi]
- Apoapsis error drives bank in the correct direction (too-high apoapsis → more drag → lower bank angle)
- `reference_velocity` damping term reduces commanded bank rate
- Proptest: any valid post-bounce state produces finite, bounded output

**Rust integration tests (`ftc.rs`):**
- Phase dispatch: `guidance_phase=1` routes to capture guidance, `guidance_phase=2` routes to exit guidance
- NN and PiecewiseConstant ignore phase (still produce their own bank angles when phase=2)

**Rust integration tests (`estimator.rs`):**
- `SimPhase::Full`: phase transitions from 1 → 2 after bounce + velocity threshold
- `SimPhase::CaptureOnly`: phase stays 1 regardless of state
- `SimPhase::ExitOnly`: phase stays 2 regardless of state
- Phase transition latches `reference_velocity` and sets `phase_transition_flag`

**E2E tests:**
- Run a full FTC simulation with `phase = "full"` and verify capture
- Run with `phase = "capture_only"` and verify identical output to current behavior (regression)
- Compare DV between `capture_only` and `full` — exit guidance should improve orbit insertion accuracy

**Python tests:**
- PyO3: verify `phase_transition_flag` appears in trajectory data when `include_trajectories=True`

## Files Modified

| File | Change |
|------|--------|
| `src/rust/src/gnc/navigation/estimator.rs` | Remove phase override in `navigate()` and `navigate_ekf()`, add `SimPhase` gating |
| `src/rust/src/gnc/guidance/exit.rs` | **New file** — shared exit-phase longitudinal controller |
| `src/rust/src/gnc/guidance/mod.rs` | Add `pub mod exit;` |
| `src/rust/src/gnc/guidance/ftc.rs` | Phase-aware dispatch before scheme match, import exit module |
| `src/rust/src/simulation/runner.rs` | Wire `reference_velocity` on phase transition, replace photo phase heuristic |
| `src/rust/src/data/mod.rs` | Add `sim_phase` field to `SimData` |
| `configs/missions/mars.toml` | Add exit params to `[guidance.ftc]` section |
| `configs/missions/earth.toml` | Add exit params to `[guidance.ftc]` section |
| Various FTC-specific configs | Remove exit params (now inherited from mission base) |

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Which schemes | FTC + EqGlide + EnergyCtrl + PredGuid + FNPAG | NN and PiecewiseConstant produce signed full-trajectory bank angles |
| Exit targets | Apoapsis (longitudinal) + inclination (lateral) | Bank magnitude controls apoapsis via drag; lateral guidance already handles inclination |
| Algorithm | Dynamic pressure feedback with apoapsis correction | Matches the existing exit_* param scaffolding; uses pre-tuned values |
| Architecture | New `exit.rs` module | Clean separation from capture guidance; follows per-algorithm-per-file pattern |
| SimPhase | Activated as mode selector | Free regression path (CaptureOnly), testing aid (ExitOnly), nearly zero cost |
| Lateral during exit | No changes — stays active via energy window | Already gated correctly; inclination correction is valuable on the ascending leg |
