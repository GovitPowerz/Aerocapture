# Rust Variable Rename Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all French/Fortran legacy variable names with explicit English across the Rust codebase, strip all Fortran-origin comments.

**Architecture:** Bottom-up rename: leaf modules first (no dependents), then mid-level guidance modules, then core GNC structs (highest fan-out), then consumers (runner, init, output), then integration tests. Each tier compiles and passes tests before moving on.

**Tech Stack:** Rust (Edition 2024), cargo test, cargo clippy

**Spec:** `docs/superpowers/specs/2026-03-10-rust-variable-rename-design.md`

---

## Chunk 1: Tier 1 — Leaf Modules

### Task 1: Rename RK4 internals and strip Fortran comments

**Files:**
- Modify: `src/rust/src/integration/rk4.rs`

This file has 3 legacy local variable names and Fortran-origin comments/doc references. The function signature parameters `ix` and `qk` are public API.

- [ ] **Step 1: Rename `ix` parameter → `gill_toggle` in function signature and doc comment**

In `src/rust/src/integration/rk4.rs`, rename the `ix` parameter of `rk4_increment()` (line 23) and all uses within the function body (lines 28, 38, 46). Also rename in the doc comment (lines 7, 15).

```
Old: pub fn rk4_increment(... ix: &mut i32, qk: &mut [f64], ...)
New: pub fn rk4_increment(... gill_toggle: &mut i32, accumulator: &mut [f64], ...)
```

Rename `ix_f` (line 28) to `gill_toggle_f`.

- [ ] **Step 2: Rename `qk` parameter → `accumulator` in function signature and doc comment**

In the same function, rename `qk` (line 24) to `accumulator` everywhere in the body (lines 16, 36, 43, 44, 51).

- [ ] **Step 3: Rename `xk` local → `step_increment` in function body**

Rename the `let xk = ...` bindings on lines 34, 42, 50 to `let step_increment = ...` and update all uses on the same lines.

- [ ] **Step 4: Strip Fortran-origin comments**

Remove or rewrite these comments:
- Line 2: `//! Matches Fortran rkutta.f exactly.` → delete
- Line 7: `//! The \`qk\` and \`ix\` variables persist across the 4 calls.` → update to use new names: `//! The \`accumulator\` and \`gill_toggle\` variables persist across the 4 calls.`
- Line 15: `/// - \`ix\`: internal variable (modified in place, initially 0)` → `/// - \`gill_toggle\`: Gill's variant toggle (-1 or +1, modified in place, initially 0)`
- Line 16: `/// - \`qk\`: internal storage (modified in place)` → `/// - \`accumulator\`: internal RK4 storage (modified in place)`
- Line 30: `// xk = dt * derivs` → delete (redundant with code)

- [ ] **Step 5: Update test code in the same file**

In the `#[cfg(test)]` module (lines 58-182), rename all uses of `qk` → `accumulator` and `ix` → `gill_toggle` in the test helper `rk4_step()` (lines 68-69) and in each test function (lines 93-94, 115-116, 141-142, 167-168).

- [ ] **Step 6: Update callers of `rk4_increment` — `simulation/runner.rs`**

In `src/rust/src/simulation/runner.rs`:
- `SimState` struct fields (lines 40-41): `qk` → `accumulator`, `ix` → `gill_toggle`
- `run_single()` initialization (lines 318-319): `qk: [0.0; 8]` → `accumulator: [0.0; 8]`, `ix: 0` → `gill_toggle: 0`
- `integrate_step()` function (lines 693, 698): `sim.ix` → `sim.gill_toggle`, `sim.qk` → `sim.accumulator`

- [ ] **Step 7: Build and test**

Run: `cd src/rust && cargo test 2>&1 | tail -5`
Expected: all tests pass, no warnings

- [ ] **Step 8: Commit**

```bash
git add src/rust/src/integration/rk4.rs src/rust/src/simulation/runner.rs
git commit -m "refactor: rename RK4 legacy variables (xk/qk/ix → step_increment/accumulator/gill_toggle)"
```

### Task 2: Rename coordinates.rs legacy locals and strip Fortran comments

**Files:**
- Modify: `src/rust/src/gnc/navigation/coordinates.rs`
- Modify: `src/rust/src/orbit/elements.rs` (uses same variable names as locals)

These are all local variables (not struct fields), so changes are file-scoped.

- [ ] **Step 1: Rename locals in `to_absolute_cartesian()` (coordinates.rs lines 200-225)**

```
posita → position_abs
vitesl → velocity_local
plocal → local_to_geocentric (already a good name in comment)
vitesr → velocity_geocentric
vitese → velocity_entrainment
vitesa → velocity_abs
```

- [ ] **Step 2: Rename locals in `total_energy()` (coordinates.rs lines 241-244)**

```
posita → position_abs  (from to_absolute_cartesian return)
vitesa → velocity_abs  (from to_absolute_cartesian return)
vitabs → speed_abs
rayvec → radius
```

- [ ] **Step 3: Strip Fortran comments in coordinates.rs**

Remove these:
- Line 215: `// Fortran: xomega = [0, 0, omega]` → delete
- Line 230: `/// Matches Fortran enrtot.f.` → delete
- Any other `Matches Fortran` or `Fortran:` comments in the file

- [ ] **Step 4: Rename locals in `from_spherical()` (elements.rs lines 28-108)**

```
posita → position_abs
vitesa → velocity_abs
xmocin → angular_momentum
rayvec → radius
vitabs → speed_abs
xcinet → angular_momentum_magnitude
enrorb_raw → energy_raw
sigenr → energy_sign
enrorb → energy
demiax → semi_major_axis_raw
parexc → eccentricity_param
excent → eccentricity_raw
cosinc → cos_inclination
sininc → sin_inclination
xincli → inclination_raw
gomega → raan_raw
posvit → pos_dot_vel
pomega → arg_periapsis_raw
```

- [ ] **Step 5: Strip Fortran comments in elements.rs**

Remove:
- Line 3: `//! Matches Fortran orbito.f exactly.` → delete
- Line 11: `/// Matches Fortran orbito.f.` → delete
- Line 25: `let enrmin = 1e-6; // Fortran: satorb common, small threshold` → `let enrmin = 1e-6; // small threshold to avoid parabolic singularity`

- [ ] **Step 6: Rename locals in runner.rs that shadow these names**

In `src/rust/src/simulation/runner.rs`:
- `run_single()` lines 537-548: `vitesa` → `velocity_abs`, `vitabs` → `speed_abs`, `xenerg` → `energy`, `vitrad` → `velocity_radial`
- `build_photo_values()` lines 638-649: `vitesa` → `velocity_abs`, `vitabs` → `speed_abs`, `enerjr` → `energy`, `vitrad` → `velocity_radial`

- [ ] **Step 7: Build and test**

Run: `cd src/rust && cargo test 2>&1 | tail -5`
Expected: all tests pass

- [ ] **Step 8: Commit**

```bash
git add src/rust/src/gnc/navigation/coordinates.rs src/rust/src/orbit/elements.rs src/rust/src/simulation/runner.rs
git commit -m "refactor: rename French variable names in coordinates.rs and elements.rs"
```

### Task 3: Strip Fortran comments from physics modules

**Files:**
- Modify: `src/rust/src/physics/gravity.rs`
- Modify: `src/rust/src/physics/aerodynamics.rs`
- Modify: `src/rust/src/physics/atmosphere.rs`
- Modify: `src/rust/src/physics/winds.rs`

These files have no legacy variable names, only Fortran-origin comments to strip.

- [ ] **Step 1: Strip all `Matches Fortran` and `Fortran:` comments from all 4 physics files**

Search for and remove/rewrite any `Fortran`, `guilon`, `conphy`, `frayon` references in comments.

- [ ] **Step 2: Build and test**

Run: `cd src/rust && cargo test 2>&1 | tail -5`
Expected: all tests pass

- [ ] **Step 3: Commit**

```bash
git add src/rust/src/physics/
git commit -m "refactor: strip Fortran-origin comments from physics modules"
```

### Task 4: Tier 1 verification

- [ ] **Step 1: Run full test suite**

Run: `cd src/rust && cargo test 2>&1 | tail -10`
Expected: all ~172 tests pass

---

## Chunk 2: Tier 2 — Mid-Level Modules

### Task 5: Strip Fortran comments and rename `PdynTableEntry` in guidance_params.rs

**Files:**
- Modify: `src/rust/src/data/guidance_params.rs`

Fields are already English. Need to strip Fortran mapping comments and rename `PdynTableEntry` → `DynamicPressureTableEntry`.

- [ ] **Step 1: Rename `PdynTableEntry` → `DynamicPressureTableEntry`**

Rename the struct on line 11 and update all references across the codebase:
- `src/rust/src/data/guidance_params.rs` (struct def + uses)
- `src/rust/src/config.rs` (TOML parsing)
- `src/rust/src/gnc/guidance/ftc.rs` (tbgain function)

Search: `rg PdynTableEntry src/rust/src/` to find all references.

- [ ] **Step 2: Strip any Fortran-origin comments in guidance_params.rs**

Remove any `Fortran`, `amorft`, `pulsft`, `margmu`, `gaindh`, `coridx`, `pdacti`, `pdinib`, `enrlat` references from comments. Also update the doc comment that references legacy column names (around line 175 if present): `pdyneq`, `vitrad`, `xinccr`, `gitref` → use English equivalents.

- [ ] **Step 3: Build and test**

Run: `cd src/rust && cargo test 2>&1 | tail -5`

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/data/guidance_params.rs src/rust/src/config.rs src/rust/src/gnc/guidance/ftc.rs
git commit -m "refactor: rename PdynTableEntry, strip Fortran comments in guidance_params.rs"
```

### Task 6: Strip Fortran comments from remaining Tier 2 files

**Files:**
- Modify: `src/rust/src/gnc/control/pilot.rs`
- Modify: `src/rust/src/gnc/guidance/equilibrium_glide.rs`
- Modify: `src/rust/src/gnc/guidance/energy_controller.rs`
- Modify: `src/rust/src/gnc/guidance/predguid.rs`
- Modify: `src/rust/src/gnc/guidance/fnpag.rs`
- Modify: `src/rust/src/gnc/guidance/neural.rs`
- Modify: `src/rust/src/gnc/mod.rs`
- Modify: `src/rust/src/data/dispersions.rs`
- Modify: `src/rust/src/data/mod.rs`
- Modify: `src/rust/src/data/aerodynamics.rs`
- Modify: `src/rust/src/data/atmosphere.rs`
- Modify: `src/rust/src/integration/sequencer.rs`
- Modify: `src/rust/src/orbit/maneuver.rs`
- Modify: `src/rust/src/simulation/init.rs`
- Modify: `src/rust/src/config.rs`

These files have Fortran-origin comments (~144 occurrences across 26 files total) but no variable renames needed. The approach: search for `Fortran`, subroutine names (`guilon`, `guicap`, etc.), and remove/rewrite every such comment.

- [ ] **Step 1: Grep for all Fortran references and strip them**

Use: `rg -n "Fortran|guilon|guicap|tbgain|guilat|naviga|photra|orbito|enrtot|xvabsl|vigite|guialf|realit|simmsr|finmsr|conphy|rkutta|lectci|frayon|geodes|cartes|faeros|pilote|entree|etafin" src/rust/src/`

For each match, either delete the comment line or rewrite it to describe the behavior without referencing Fortran.

- [ ] **Step 2: Build and test**

Run: `cd src/rust && cargo test 2>&1 | tail -5`

- [ ] **Step 3: Commit**

```bash
git add src/rust/src/
git commit -m "refactor: strip all remaining Fortran-origin comments from Rust codebase"
```

### Task 7: Tier 2 verification

- [ ] **Step 1: Run full test suite + clippy**

Run: `cd src/rust && cargo test && cargo clippy 2>&1 | tail -10`
Expected: all tests pass, no clippy warnings

---

## Chunk 3: Tier 3 — Core GNC Structs

### Task 8: Rename FtcState fields

**Files:**
- Modify: `src/rust/src/gnc/guidance/ftc.rs`

FtcState has 14 legacy field names. This struct is used in `runner.rs`, `init.rs`, and integration tests.

- [ ] **Step 1: Rename all FtcState fields**

In the struct definition (lines 15-45) and `FtcState::new()` (lines 48-69):

```
gitcom → bank_angle_commanded
gitpre → bank_angle_previous
gpilpr → pilot_bank_angle_previous
alfcom → aoa_commanded
sgngit → roll_sign
somgit → cumulative_bank_change
nbroll → n_reversals
indrvr → reversal_active
rolway → roll_way           (keep as-is, already readable)
trevrs → reversal_duration
iprepr → securization_counters
iguida → guidance_active
vitref → reference_velocity
```

Note: `vitgit` field does not exist on FtcState (it's computed locally as `bank_rate`). `n_secur` and `n_active` are already English.

- [ ] **Step 2: Update all FtcState field references in ftc.rs functions**

Update `guidance_step()`, `guicap()`, and `guilat()` — everywhere `state.gitcom`, `state.gitpre`, etc. appear.

- [ ] **Step 3: Rename FtcOutput fields**

In the struct definition (lines 73-81):

```
gitcom → bank_angle_commanded
alfcom → aoa_commanded
vitgit → bank_rate
ilongi → longitudinal_active
isatur → rate_saturated
indrol → roll_reversal_active
```

Update all assignments: `out.gitcom`, `out.alfcom`, `out.vitgit`, `out.ilongi`, `out.isatur`, `out.indrol`.

- [ ] **Step 4: Rename FTC local variables in guidance_step()**

```
sgnpre → previous_roll_sign
enrjlt → energy
gitlon → bank_angle_longitudinal
ilongi → longitudinal_active (local, not just struct field)
ilater → lateral_active
vgitmx → max_bank_rate
tguida → guidance_period
vitgit → bank_rate (local computation)
isatur → rate_saturated (local)
indrol → roll_reversal_active (local)
```

- [ ] **Step 5: Rename FTC local variables in guicap()**

```
vitrel → velocity_relative
vitrad → velocity_radial
pdyneq → dynamic_pressure_equilibrium
pdyneq_safe → dynamic_pressure_equilibrium_safe
cmunom → cos_bank_nominal
prenom → dynamic_pressure_nominal
hdtnom → altitude_rate_nominal
gaindh → gain_altitude_rate (from tbgain return)
gainpd → gain_dynamic_pressure (from tbgain return)
cosmuc → cos_bank_commanded
isecur → is_securized
gitlon → bank_angle_longitudinal (return value, already matches Step 4)
```

- [ ] **Step 6: Rename FTC local variables in tbgain()**

```
inumer → table_index (already Option<usize>, but found var is the unwrapped value)
coefpd_a → pressure_coeff
amorft → damping_capture
pulsft → frequency_capture
srefer → reference_area
xmasse → mass
gaindh → gain_altitude_rate
gainpd → gain_dynamic_pressure
```

- [ ] **Step 7: Rename FTC local variables in guilat()**

```
sgnpre → previous_roll_sign
xinccr → inclination_error
vitrel → velocity_relative
coridx → corridor_slope
coridy → corridor_intercept
xinmax → inclination_max
dgitcm → bank_angle_change
vgitmx → max_bank_rate
tguida → guidance_period
```

- [ ] **Step 8: Rename function parameter `gitpil` → `pilot_bank_angle` in guidance_step()**

Line 89: `gitpil: f64` → `pilot_bank_angle: f64`. Also rename `gitref` parameter (line 91) → `reference_bank_angle`.

- [ ] **Step 9: Rename function `guicap` → `capture_guidance` and `guilat` → `lateral_guidance`**

These are private functions but their names are pure French.

- [ ] **Step 10: Update ftc.rs test code**

In the `#[cfg(test)]` module (lines 427-733):
- Update all `NavigationOutput` field accesses: `positn`, `vitesn`, `acceln`, `coefan`, `roguid`, `roexit`, `pdynan`, `energn` → new names (these will be renamed in Task 10, so for now just update FtcState/FtcOutput field accesses)
- Update test assertions: `out.gitcom` → `out.bank_angle_commanded`, `out.alfcom` → `out.aoa_commanded`, `out.vitgit` → `out.bank_rate`
- Rename local test vars: `gitref` → `reference_bank_angle`

- [ ] **Step 11: Update all FtcState/FtcOutput field accesses in runner.rs**

Search for `ftc_state.` and `ftc_out.` and update:
- `ftc_state.alfcom` → `ftc_state.aoa_commanded` (line 392)
- `ftc_state.nbroll` → `ftc_state.n_reversals` (line 603)
- `ftc_out.gitcom` → `ftc_out.bank_angle_commanded` (line 425)
- `ftc_out.alfcom` → `ftc_out.aoa_commanded` (line 438)
- `ftc_out.ilongi` → `ftc_out.longitudinal_active` (line 449)

**Note:** Do NOT expect `cargo test` to pass yet — NavigationOutput fields (used in ftc.rs inline tests and all guidance module tests) are still using old names. Those get renamed in Task 10. The build will compile `runner.rs` and `ftc.rs` production code, but inline `#[cfg(test)]` modules will still break on old NavigationOutput field names.

- [ ] **Step 12: Commit (ftc.rs + runner.rs consumer fixup together)**

```bash
git add src/rust/src/gnc/guidance/ftc.rs src/rust/src/simulation/runner.rs
git commit -m "refactor: rename FtcState/FtcOutput fields from French to English"
```

### Task 10: Rename NavigationState and NavigationOutput fields

**Files:**
- Modify: `src/rust/src/gnc/navigation/estimator.rs`

- [ ] **Step 1: Rename NavigationState fields**

In struct definition (lines 24-30) and `NavigationState::new()` (lines 39-47):

```
coefro → density_gain
vitpre → previous_radial_velocity
ibounc → bounce_flag
iphase → guidance_phase
tcaptr → capture_time
```

- [ ] **Step 2: Rename NavigationOutput fields**

In struct definition (lines 52-72):

```
positn → position_estimated       ([f64; 3])
vitesn → velocity_estimated       ([f64; 3])
acceln → acceleration_estimated   ([f64; 2])
coefan → aero_coefficients        ([f64; 2])
roguid → density_guidance         (f64)
roexit → density_exit             (f64)
pdynan → dynamic_pressure_estimated (f64)
energn → energy_estimated         (f64)
ecartn → orbital_errors           ([f64; 4])
ibounc → bounce_flag              (i32)
iphase → guidance_phase           (i32)
icrash → crash_flag               (i32)
indext → phase_transition_flag    (i32)
vitref → reference_velocity       (f64)
tcaptr → capture_time             (f64)
```

- [ ] **Step 3: Rename local variables in navigate()**

```
vitrel → velocity_relative    (line 110)
acdram → drag_acceleration_measured (line 122)
roesti → density_estimated    (line 134)
coefar → aero_factor          (line 158)
vitrad → velocity_radial      (line 199)
dvitrd → delta_radial_velocity (line 219)
```

- [ ] **Step 4: Update all field references in navigate() body**

Update every `out.positn`, `nav_state.coefro`, etc. to use new names. Also update the `navigate()` parameter comments (`positr`, `vitesr`, `alfcom`, `temsim` — these are parameter names, keep them or rename to `true_position`, `true_velocity`, `commanded_aoa`, `sim_time`).

- [ ] **Step 5: Strip Fortran comments in estimator.rs**

Remove:
- Line 3: `//! Matches Fortran naviga.f.`
- Line 14: `/// Matches Fortran common /pernav/ dispos(3), disvit(3), disdra.`
- Line 76: `/// Matches Fortran naviga.f.`
- Line 102: `// Matches naviga.f lines 140-143`
- Line 145: `// coefro = (1-λ)*coefro + λ*(roesti/rorefr)` → update var names
- Line 201: `// Phase management (matches naviga.f lines 256-299)`
- Line 232: `// Fortran has "iphase=1" hardcoded at line 301 (override)`
- All other `naviga.f`, `Fortran`, `imodel=0`, `imodel=1` references

- [ ] **Step 6: Update estimator.rs test code**

In the `#[cfg(test)]` module (lines 245-680):
- Update all `NavigationOutput` field accesses in assertions and fixture builders
- `out.positn` → `out.position_estimated`, `out.vitesn` → `out.velocity_estimated`, etc.
- `nav_state.coefro` → `nav_state.density_gain`, `nav_state.ibounc` → `nav_state.bounce_flag`
- Rename test vars: `coefro_values` → `density_gain_values`

- [ ] **Step 7: Update all consumers of NavigationOutput across the codebase**

Update both function body field accesses AND `NavigationOutput { ... }` struct literal constructors (especially in inline `#[cfg(test)]` blocks) in these files:
- `src/rust/src/gnc/guidance/ftc.rs` — `nav.positn`, `nav.vitesn`, `nav.roguid`, `nav.coefan` (in capture_guidance, lateral_guidance, and `#[cfg(test)]` module's `test_nav()` builder + proptest `NavigationOutput { ... }` literals)
- `src/rust/src/gnc/guidance/equilibrium_glide.rs` — `nav.positn`, `nav.vitesn`, `nav.roguid`, `nav.acceln`, `nav.coefan` (function body + `#[cfg(test)]` NavigationOutput constructors)
- `src/rust/src/gnc/guidance/energy_controller.rs` — similar (function body + `#[cfg(test)]`)
- `src/rust/src/gnc/guidance/predguid.rs` — similar (function body + `#[cfg(test)]`)
- `src/rust/src/gnc/guidance/fnpag.rs` — similar (function body + `#[cfg(test)]`)
- `src/rust/src/gnc/guidance/neural.rs` — similar (function body + `#[cfg(test)]`)
- `src/rust/src/simulation/runner.rs` — `nav_out.pdynan`, `nav_out.roguid`
- `src/rust/tests/common/fixtures.rs` — `NavigationOutput { positn: ..., vitesn: ..., ... }`

**Note:** `src/rust/src/simulation/output.rs` — no renames needed (no Fortran comments, no legacy field accesses).

- [ ] **Step 8: Build and test**

Run: `cd src/rust && cargo test 2>&1 | tail -10`
Expected: all tests pass

- [ ] **Step 9: Commit**

```bash
git add src/rust/src/gnc/ src/rust/src/simulation/ src/rust/tests/
git commit -m "refactor: rename NavigationState/NavigationOutput fields from French to English"
```

### Task 11: Tier 3 verification

- [ ] **Step 1: Run full test suite**

Run: `cd src/rust && cargo test 2>&1 | tail -10`
Expected: all ~172 tests pass

---

## Chunk 4: Tier 4 — Wiring, Consumers & Final Cleanup

### Task 12: Rename remaining runner.rs legacy locals

**Files:**
- Modify: `src/rust/src/simulation/runner.rs`

- [ ] **Step 1: Rename runner.rs local variables**

```
degrad (line 301)           → DEG_TO_RAD (const)
gitref (line 336)           → reference_bank_angle
somgit_deg (line 366)       → cumulative_bank_change_deg
pdynan_for_photo (line 367) → dynamic_pressure_for_photo
romver_for_photo (line 368) → density_estimate_for_photo
xsauve (line 564)           → final_record
```

- [ ] **Step 2: Rename build_photo_values() parameters and locals**

```
degrad parameter    → deg_to_rad
pdynan parameter    → dynamic_pressure
romver parameter    → density_estimate
isimul parameter    → sim_index
somgit parameter    → cumulative_bank_change
altitr (line 624)   → altitude
xlatit (line 624)   → latitude
enerjr (line 648)   → energy
vitrad (line 649)   → velocity_radial
iphase (line 651)   → phase (already a local, just rename)
```

- [ ] **Step 3: Strip remaining Fortran comments in runner.rs**

Remove:
- Line 2: `//! Matches Fortran simmsr.f + realit.f + finmsr.f.`
- Line 612: `/// Build a photo snapshot line matching Fortran photra.f format.`
- Any other `Fortran` references

- [ ] **Step 4: Build and test**

Run: `cd src/rust && cargo test 2>&1 | tail -5`

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/simulation/runner.rs
git commit -m "refactor: rename remaining French locals in runner.rs"
```

### Task 13: Update integration tests

**Files:**
- Modify: `src/rust/tests/common/fixtures.rs`
- Modify: `src/rust/tests/guidance_regression.rs`
- Modify: `src/rust/tests/error_paths.rs`
- Modify: `src/rust/tests/e2e.rs`
- Modify: `src/rust/tests/edge_cases.rs`
- Modify: `src/rust/tests/config_loading.rs`

- [ ] **Step 1: Update fixtures.rs**

Update `nav_from_state()` to use new NavigationOutput field names:
```
positn → position_estimated
vitesn → velocity_estimated
acceln → acceleration_estimated
coefan → aero_coefficients
roguid → density_guidance
roexit → density_exit
pdynan → dynamic_pressure_estimated
energn → energy_estimated
```

- [ ] **Step 2: Update guidance_regression.rs**

Search for any old field names and update. These tests access FtcOutput and NavigationOutput fields.

- [ ] **Step 3: Update error_paths.rs, e2e.rs, edge_cases.rs**

Search for any old field names and update.

- [ ] **Step 4: Strip Fortran-reference comments in all test files**

- [ ] **Step 5: Build and test**

Run: `cd src/rust && cargo test 2>&1 | tail -10`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add src/rust/tests/
git commit -m "refactor: update integration tests for renamed fields"
```

### Task 14: Final verification

- [ ] **Step 1: Run full check suite**

Run: `cd /Users/govit/Git/Govit/Aerocapture && ./check_all.sh 2>&1 | tail -20`
Expected: all Rust tests pass, fmt check passes, clippy passes

- [ ] **Step 2: Verify no legacy names remain**

Run: `rg "gitcom|alfcom|vitgit|ilongi|isatur|indrol|positn|vitesn|acceln|coefan|roguid|roexit|pdynan|energn|ecartn|icrash|indext|coefro|vitpre|ibounc|iphase|tcaptr|gitpre|gpilpr|sgngit|somgit|nbroll|indrvr|trevrs|iprepr|iguida|romver|xsauve|posita|vitesl|vitese|vitesa|vitabs|rayvec|xmocin|enrjlt|gitlon|vitrel|vitrad|cmunom|prenom|hdtnom|cosmuc|amorft|pulsft|srefer|xmasse|xinccr|xinmax|coridx|coridy|dgitcm|vgitmx|tguida|roesti|acdram|dvitrd|enerjr|altitr|xlatit|degrad|coefar|pdyneq|isimul|coefpd" src/rust/src/ src/rust/tests/`

Expected: no matches (or only in string literals / output column names, not variable names)

- [ ] **Step 3: Verify no Fortran comments remain**

Run: `rg -i "fortran|guilon|guicap|tbgain|guilat|naviga\.f|photra|orbito\.f|enrtot|xvabsl|vigite|guialf|realit\.f|simmsr|finmsr|conphy|rkutta|lectci|frayon|geodes|cartes|faeros|pilote\.f|entree\.f|etafin" src/rust/src/ src/rust/tests/`

Expected: no matches

- [ ] **Step 4: Final commit (if any stragglers found)**

```bash
git add src/rust/
git commit -m "refactor: clean up any remaining legacy references"
```

### Task 15: Update spec and plan as complete

- [ ] **Step 1: Mark spec status as Complete**

Edit `docs/superpowers/specs/2026-03-10-rust-variable-rename-design.md`, change `**Status**: Approved` → `**Status**: Complete`

- [ ] **Step 2: Commit**

```bash
git add docs/
git commit -m "docs: mark variable rename spec as complete"
```
