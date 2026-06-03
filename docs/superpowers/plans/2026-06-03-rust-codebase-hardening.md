# Rust Codebase Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix every finding from the 2026-06-03 Rust codebase review (Critical → Low): silent-config-drop, navigation divergence, density-inversion sign, cross-subsystem duplication, FFI panic ergonomics, dead code, numerical nits, and split the god-files into module trees — without regressing the bit-validated golden trajectories.

**Architecture:** Phase by risk. Establish config-validation correctness first, strip dead surface, then do behavior-preserving deduplication (goldens MUST stay green), then apply trajectory-altering bug fixes as isolated commits each gated by a golden-diff checkpoint, then FFI ergonomics, numerical robustness, and finally the high-churn module-split restructure. Verify continuously against `tests/reference_data/rust_golden/`.

**Tech Stack:** Rust 2024 (nalgebra, libm, serde, rayon), PyO3 0.28 + numpy + maturin, cargo test / clippy / fmt.

---

## Execution Context (read once before starting)

- **Worktree:** `/Users/govit/Git/Govit/Aerocapture/.claude/worktrees/feature+rust-codebase-hardening`, branch `worktree-feature+rust-codebase-hardening`. All paths below are relative to `src/rust/` unless noted. The user's uncommitted `configs/training/common.toml` lives in the MAIN tree, not here — this worktree is clean.
- **Baseline:** `cargo test -p aerocapture` → **572 passing, 0 failing** at start.
- **Golden snapshot:** `/tmp/golden_baseline_hashes.txt` holds SHA-1 of all 12 golden CSVs. The 6 guidance goldens are compared with **approximate** rel-tol (`compare_csv_approx`), the ref/guided/high_bank goldens are separate E2E cases.
- **Golden regime (load-bearing):** every golden config is **bias-mode nav** (no `[navigation]` section) + **fixed Gill RK4** (no `[integration]` section) + **nominal (no MC dispersion draws on the single golden sim)**. Consequence: EKF-path fixes and adaptive-path fixes cannot move goldens; bias-path density fixes only move goldens if the nominal MSR trajectory actually hits the changed branch.
- **Golden checkpoint protocol (behavior-change tasks only):** after the fix, run `cargo test -p aerocapture --test guidance_regression --test e2e` and re-hash the goldens. If any hash differs from baseline: **STOP, do not regenerate, surface the diff + a one-paragraph physical justification to the user, and wait for approval.** Only after approval, regenerate via the documented command and re-commit.
- **Git hygiene:** stage only the files a task touches with explicit paths. **Never `git add -A` / `git add .`** (avoids sweeping unrelated artifacts). Commit messages end with the Co-Authored-By trailer per repo convention.
- **PyO3 rebuild (Phase 5 only):** from the REPO ROOT, `uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml`. Subcrate-local builds go stale — always use the manifest-path form from root.
- **Per-task loop:** write failing test → run it (confirm fail) → implement → run (confirm pass) → run the relevant golden/regression gate → commit. Do not batch unrelated changes into one commit.
- **No early stopping / no scope-trimming:** every task in every phase ships. If a task turns out to be a no-op (e.g. a "dead" fn has a caller), record that in the commit message and move on — do not silently skip.

---

## Phase 1 — Config-layer correctness (silent-drop class)

Valid configs are unaffected (only error behavior for invalid input changes), so **goldens stay green throughout Phase 1**. Run `cargo test -p aerocapture` after each task.

### Task 1.1: Propagate unknown dispersion-level errors (C1)

**Files:**
- Modify: `src/data/mod.rs` (the 10 `DispersionLevel::from_str(&d.level).unwrap_or(DispersionLevel::Medium)` sites in `build_dispersion_config`, starting ~line 839)
- Test: `src/data/mod.rs` `#[cfg(test)]` module (add)

- [ ] **Step 1: Write the failing test** — add to the `data::tests` module:

```rust
#[test]
fn unknown_dispersion_level_errors_not_silent_medium() {
    // A typo in `level` must be rejected, NOT silently treated as Medium.
    let toml_str = r#"
[monte_carlo.initial_state]
level = "of"
"#;
    let mc: TomlMonteCarlo = toml::from_str(toml_str).unwrap();
    let err = build_dispersion_config(&mc).unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("of") || msg.to_lowercase().contains("level"),
        "error should name the bad level, got: {msg}");
}
```

- [ ] **Step 2: Run, confirm it fails** — `cargo test -p aerocapture unknown_dispersion_level_errors -- --nocapture`. Expected: PASS-as-Medium currently means `unwrap_err()` panics ("called unwrap_err on Ok") → test FAILS.

- [ ] **Step 3: Implement** — replace all 10 occurrences. Because the level is read inside `.and_then(|d| {...})` closures returning `Option`, refactor `build_dispersion_config` so each domain resolves the level via `?`. Introduce a helper at the top of the function:

```rust
// Resolve a domain's level, propagating typos as a hard error instead of
// silently defaulting to Medium (which inverts intent for `level = "off"`).
fn resolve_level(raw: &str) -> Result<DispersionLevel, DataError> {
    DispersionLevel::from_str(raw)
        .map_err(|_| DataError::Config(format!("unknown dispersion level: {raw:?}")))
}
```

Then convert each domain block from `mc.initial_state.as_ref().and_then(|d| { let level = ...unwrap_or(Medium); ... Some(s) })` to a `match`-on-`Option` form that can return `Err`:

```rust
let initial_state = match mc.initial_state.as_ref() {
    None => None,
    Some(d) => {
        let level = resolve_level(&d.level)?;
        if level == DispersionLevel::Off { None } else {
            let mut s = InitialStateSigmas::from_level(level);
            if level == DispersionLevel::Custom { /* existing custom reads, unchanged for now */ }
            Some(s)
        }
    }
};
```

Apply the identical transform to all 10 domains (initial_state, atmosphere, aerodynamics, navigation, mass, vehicle, pilot, nav_filter, wind, and any 10th). Use `DataError::Config` if that variant exists; otherwise use the variant the function already returns (check the `-> Result<_, DataError>` signature and the existing `?` at the `SamplingMethod::from_str(s)?` site ~line 1045 for the right constructor).

- [ ] **Step 4: Run, confirm pass** — `cargo test -p aerocapture` (full suite; the 9 other domains must still parse valid levels). Expected: 573 passing (572 + new test).

- [ ] **Step 5: Commit**

```bash
git add src/data/mod.rs
git commit -m "fix(config): error on unknown dispersion level instead of silent Medium

A typo in [monte_carlo.<domain>] level (e.g. \"of\" for \"off\") silently ran
Medium dispersions, inverting intent. Propagate from_str errors via ? across
all 10 domains. Valid configs unchanged; goldens unaffected.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 1.2: Reject unknown custom-dispersion keys (C2)

**Files:**
- Modify: `src/data/mod.rs` (the per-domain `if level == Custom { d.custom.get("...") }` blocks)
- Test: `src/data/mod.rs` tests

**Design decision:** validate the `custom` map keys against a per-domain allowed-set and hard-fail on any unconsumed key when `level = "custom"`. (We do NOT add global `#[serde(deny_unknown_fields)]` — base-inheritance merges can legitimately carry extra top-level keys, and a global deny would break existing layered configs.)

- [ ] **Step 1: Write the failing test**:

```rust
#[test]
fn unknown_custom_dispersion_key_errors() {
    let toml_str = r#"
[monte_carlo.initial_state]
level = "custom"
flight_path = 0.5
"#; // typo: should be flight_path_angle
    let mc: TomlMonteCarlo = toml::from_str(toml_str).unwrap();
    let err = build_dispersion_config(&mc).unwrap_err();
    assert!(format!("{err:?}").contains("flight_path"),
        "must name the unknown custom key");
}

#[test]
fn known_custom_dispersion_key_accepted() {
    let toml_str = r#"
[monte_carlo.initial_state]
level = "custom"
flight_path_angle = 0.5
"#;
    let mc: TomlMonteCarlo = toml::from_str(toml_str).unwrap();
    assert!(build_dispersion_config(&mc).is_ok());
}
```

- [ ] **Step 2: Run, confirm fail** — the typo test fails (currently `Ok`).

- [ ] **Step 3: Implement** — add a checked-consume helper and an allowed-set per domain. Replace the silent `d.custom.get("x")` pattern with explicit consumption that tracks which keys were used, then errors on leftovers:

```rust
// Consume known custom keys, error on any leftover (typo guard).
fn take_custom(custom: &HashMap<String, f64>, allowed: &[&str]) -> Result<(), DataError> {
    for k in custom.keys() {
        if !allowed.contains(&k.as_str()) {
            return Err(DataError::Config(format!(
                "unknown custom dispersion key: {k:?} (allowed: {allowed:?})"
            )));
        }
    }
    Ok(())
}
```

For each domain's Custom branch, declare the allowed list (matching the existing `.get("...")` calls — initial_state: `["altitude","longitude","latitude","velocity","flight_path_angle","azimuth"]`, atmosphere: `["density"]`, etc.), call `take_custom(&d.custom, ALLOWED)?` before reading, then keep the existing `if let Some(&v) = d.custom.get(...)` reads. (The single-helper consolidation comes in Task 7.x; here we only add the guard so the fix lands behind tests immediately.)

- [ ] **Step 4: Run, confirm pass** — `cargo test -p aerocapture`. Expected: 575 passing.

- [ ] **Step 5: Commit** (`git add src/data/mod.rs`; message: `fix(config): reject unknown custom-dispersion keys instead of silently dropping`).

### Task 1.3: Error on unknown integration mode (L4)

**Files:** Modify `src/config.rs` `IntegrationMode::from_toml` (~line 121); Test `src/config.rs` tests.

- [ ] **Step 1: Failing test** — assert `mode = "adaptiv"` (typo) produces an error rather than silently `FixedGill`. (If `from_toml` currently returns `Self`, change its signature to `Result<Self, _>` and update the call site; check callers first with `rg 'IntegrationMode::from_toml'`.)
- [ ] **Step 2: Confirm fail.**
- [ ] **Step 3: Implement** — replace the `_ => Self::FixedGill` arm with an error for any non-empty unrecognized string; an ABSENT `[integration]` section must still default to `FixedGill` (preserve that path — only an explicitly-wrong string errors).
- [ ] **Step 4: Run full suite** — confirm all golden configs (which omit `[integration]`) still load as FixedGill. Expected: 576 passing.
- [ ] **Step 5: Commit** (`git add src/config.rs`).

### Task 1.4: Make `run_batch` n_sims>1 contract explicit (M9)

**Files:** Modify `aerocapture-py/src/lib.rs` (`run_batch`) / `aerocapture-py/src/batch.rs` (~line 73).

- [ ] **Step 1:** Decide: error vs documented-discard. Choose **error** — `run_batch` keeping only result 0 per override while silently dropping 19/20 is a footgun. Return `PyValueError` when any override resolves to `n_sims > 1`, directing callers to `run_mc`.
- [ ] **Step 2:** Update the `eprintln!`-warn path to a `return Err(PyValueError::new_err(...))`. Update the pyfunction docstring to state the one-result-per-override contract.
- [ ] **Step 3:** Test in `tests/test_pyo3.py` (Phase 5 rebuild required to run) — add a `test_run_batch_rejects_multi_sim`. Mark it to run in Phase 5.
- [ ] **Step 4: Commit** (`git add aerocapture-py/src/lib.rs aerocapture-py/src/batch.rs`). Note: Python-side test executes in Phase 5 after maturin rebuild.

---

## Phase 2 — Dead code, no-ops, doc accuracy (behavior-preserving)

Every task here must keep `cargo test -p aerocapture` at full green and goldens bit-identical. For each deletion: `rg '<symbol>' src/ tests/ aerocapture-py/` first to PROVE zero non-definition/non-test callers; if a caller exists, downgrade to "document why retained" and note it.

### Task 2.1: Remove dead `attitude.rs`

- [ ] `rg 'rate_limited_bank|mod attitude|control::attitude' src/ aerocapture-py/` — confirm only the definition + its own tests reference it (shaping is done by `CommandShaper`/`pilot`).
- [ ] Delete `src/gnc/control/attitude.rs`; remove `pub mod attitude;` from `src/gnc/control/mod.rs`.
- [ ] `cargo test -p aerocapture` (green). Commit (`git add src/gnc/control/attitude.rs src/gnc/control/mod.rs`).

### Task 2.2: Remove dead `aero_forces` / divergent `AeroForces`

- [ ] `rg 'aero_forces|AeroForces' src/ aerocapture-py/ tests/` — confirm no non-test caller (the live force path is `physics::aerodynamics::compute_*`). Note the divergent heat-flux formula in the commit message (it's the reason to delete, not keep).
- [ ] Delete the dead fn + struct in `src/physics/aerodynamics.rs` (and the parallel `#[allow(dead_code)]` in `src/data/aerodynamics.rs` if its `AeroForces` is also unused — verify separately).
- [ ] Test green. Commit.

### Task 2.3: Remove dead orbit/maneuver/guidance surface

- [ ] `compute_deltav_optimal` (`src/orbit/maneuver.rs:140`): confirm no caller, delete (third copy of vis-viva, replaced by Task 3.1 helper anyway).
- [ ] `geodetic_to_cartesian` (`src/gnc/navigation/coordinates.rs:76`): confirm no caller, delete.
- [ ] `_httnom` discarded interpolation (`src/gnc/guidance/ftc.rs:49`): delete the line; if `ref_traj.altitude_rate` has no other consumer, leave the column (data file shape) but remove the dead read.
- [ ] Unused `Guidance` trait + `ReferenceGuidance` (`src/gnc/guidance/mod.rs:26`, `src/gnc/guidance/reference.rs`): confirm dispatch uses the `is_reference` flag, not the trait; delete the trait + `reference.rs` + its `pub mod reference;`. (If `reference.rs` exports anything live, keep that and delete only the dead trait.)
- [ ] Test green after EACH deletion (separate commits per symbol for clean revertability).

### Task 2.4: Remove no-op operations

- [ ] `ftc.rs:75` `cos_bank_commanded.acos().abs()` → drop `.abs()` (acos ∈ [0,π] ≥ 0). **Golden-sensitive** (FTC golden): run `guidance_regression` — must stay bit-identical (the `.abs()` is provably a no-op, so it will). Commit only if green.
- [ ] `equilibrium_glide.rs:98` redundant `bank.clamp(15°,120°)` after the cos-clamp: the 15° floor is dead (cos-clamp yields ≥18.2°). Remove the redundant clamp AND fix the misleading "never below 15°" comment to state the real 18.2° floor. Run eqglide golden — bit-identical expected. Commit.
- [ ] `data/neural.rs:965` `WindowLayer::from_flat` dead `assert!(… || { true })`: replace with the plain no-op body (consume 0, ignore tail) + a one-line comment. Test green. Commit.

### Task 2.5: `exit_apoapsis_threshold` — wire or remove (dead tunable)

- [ ] `rg 'exit_apoapsis_threshold' src/ aerocapture-py/ ../../configs/` — confirm parsed + stored but never read by non-test code; exit.rs header admits it's reserved.
- [ ] **Decision: remove** (YAGNI — re-add when the feature lands). Delete the field from `GuidanceParams`/`TomlFtcParams`, its parse/convert site in `config.rs`/`data/mod.rs`, and any `configs/` keys. Run full suite + load every `configs/test/*golden*.toml` (the E2E + guidance_regression do this). Green. Commit.

### Task 2.6: Doc accuracy + gravity oracle test (L8, M6, M7, L3)

- [ ] `src/data/nn_state.rs` + `src/data/neural.rs`: drop stale "Phase N+ adds …" roadmap comments (all phases shipped); fix the `DEFAULT_NORMALIZATION` header comment that contradicts its per-entry DV comments. Doc-only. Commit.
- [ ] GRU/LSTM "bit-identical" claim (M6): soften the in-file doc comments on `dot_plus_bias` and the GRU/LSTM headers to "agrees with the torch mirror to machine epsilon (different reduction order than torch addmm)". **Do NOT reorder `dot_plus_bias`** — reordering would change Rust float output and could move the neural golden. Doc-only. Commit.
- [ ] EKF `predict()` open-loop (M7): expand the existing `TODO` at `src/gnc/navigation/ekf.rs:101` into a clear note that F is time-invariant and ignores specific force (density-estimator + position-error scaffold, not full strapdown INS). Doc-only. Commit.
- [ ] `elements.rs:53` eccentricity parabolic guard (L3): the `|ecc_param − 1| < 1e-20` branch tests near-CIRCULAR, not parabolic, and the real protection is `.abs()`. **Fix the comment only** (state that `.abs()` guards the sqrt and the 1e-20 branch zeroes near-circular e); do not touch the math (removing the branch risks a 0.0-vs-sqrt(tiny) bit shift in goldens). Doc-only. Commit.
- [ ] Gravity J3/J4 (M8-doc): add a source citation comment (Vallado §) at `src/physics/gravity.rs:35`, AND add ONE new unit test asserting a J3 and J4 acceleration component against an independently hand-computed value (the existing tests only sanity-bound J3/J4 < 5%). Test must pass. Commit (`git add src/physics/gravity.rs`).

---

## Phase 3 — Behavior-preserving deduplication (goldens MUST stay bit-identical)

This is the highest-value, highest-care phase. After EVERY task: `cargo test -p aerocapture` full green AND re-hash goldens vs `/tmp/golden_baseline_hashes.txt` — **any drift here is a bug in the extraction, not an intended change; fix the extraction.**

### Task 3.1: Extract shared vis-viva / inclination ΔV helpers (M3 dedup half)

**Files:** Modify `src/orbit/maneuver.rs`; Test `src/orbit/maneuver.rs` tests.

- [ ] **Step 1: Characterization test first** — capture current `compute_deltav` + `predicted_dv_for_nn` outputs on 3 representative inputs (one bound, one near-target, one hyperbolic) as exact `assert!((got - EXPECT).abs() < 1e-12)` literals (compute EXPECT by running the current code once). This pins behavior across the refactor.
- [ ] **Step 2:** Extract `fn hohmann_leg_dv(mu, r_apsis, r_from, r_to) -> f64` and `fn inclination_dv(mu, r, di) -> f64` (the latter WITHOUT the `rayneu>0` guard initially, to preserve `compute_deltav`'s exact current behavior). Rewrite the three call sites to use them. **Do not change guard semantics yet** — that is Task 4.5.
- [ ] **Step 3:** Run characterization test + full suite — bit-identical. Commit.

### Task 3.2: Unify the three `final_record`/termination/ΔV assemblers (H3a)

**Files:** Modify `src/simulation/runner.rs` (`run_single` ~1100-1198, `build_final_record` ~1335-1411), `src/simulation/tick.rs` (`ifinal` arm ~457). Consumer: `aerocapture-py/src/env.rs:204`.

- [ ] **Step 1:** Add `fn ifinal_for(term: TermReason) -> i32` (single source of truth; `None => unreachable!("ifinal requested for a non-terminated state")` — genuinely unreachable, all 3 call sites are post-termination). Replace the three inline matches (incl. the divergent `0 // should not happen`).
- [ ] **Step 2:** Extract `fn assemble_final_record(state: &SimState, data: &SimData, planet: &PlanetConfig, captured: bool, energy: f64, ecc: f64) -> [f64; 52]` from the duplicated block; have BOTH `run_single` and `build_final_record` call it. Keep `build_final_record`'s public signature (it's the env.rs consumer).
- [ ] **Step 3:** `tests/env_equivalence.rs` is the cross-path guard — run it explicitly: `cargo test -p aerocapture --test env_equivalence`. Then full suite + goldens. Bit-identical. Commit.

### Task 3.3: Collapse the two `SimState` constructors (H3b)

**Files:** Modify `src/simulation/runner.rs` (`run_single` init ~846-1005 vs `build_sim_state` ~231-375).

- [ ] **Step 1:** Have `run_single` call `build_sim_state(...)` to construct the base state, then patch only the CLI-specific fields (`is_single`/`write_photo`/the `sim_idx * 10_000` seed offset). Identify the exact delta set by diffing the two literals.
- [ ] **Step 2:** Run full suite + goldens. Bit-identical (same seed math → same RNG → same trajectory). Commit.

### Task 3.4: Hoist the trajectory-projection closure + RunOutput assembly (H3c)

**Files:** Modify `src/simulation/runner.rs` (`run_for_api` ~583-620, `run_for_api_with_draws` ~681-718).

- [ ] Extract `fn project_trajectory(photo_lines: &[[f64; 30]]) -> Vec<[f64; 17]>` and a `fn to_run_output(...)` shared by both entry points. The magic photo-column indices live in ONE place now.
- [ ] `tests/pyo3.rs` / `tests/e2e.rs` cover the API path — run them. Goldens bit-identical. Commit.

### Task 3.5: Navigation shared `finalize` helper (H1a — behavior-preserving extraction)

**Files:** Modify `src/gnc/navigation/estimator.rs` (`navigate` 81-287 vs `navigate_ekf` 373-627).

- [ ] **Step 1:** Extract the shared post-density tail (energy block, orbital-elements block, bounce detection, crash, SimPhase gating, capture_time accumulation, `out.*` population) into `fn finalize_navigation_output(out: &mut NavigationOutput, ns: &mut NavigationState, data: &SimData, ...)`. **Preserve each path's CURRENT behavior exactly** — including the EKF path's missing `exit_phase_locked` (that divergence is fixed separately in Task 4.1, isolated). The phase-management block differs between the two, so for now factor only the genuinely-identical tail (energy/orbital/crash/SimPhase/out-population) and leave the phase block inline in each.
- [ ] **Step 2:** Full suite + goldens (bias-mode goldens exercise `navigate`, not `navigate_ekf`, but both must compile + pass their unit tests). Bit-identical. Commit.

### Task 3.6: Guidance `securize_cos_bank` helper + single signed-bank predicate (H4)

**Files:** Modify `src/gnc/guidance/dispatch.rs` (~162-303), `equilibrium_glide.rs`, `energy_controller.rs`, `predguid.rs`, `ftc.rs`.

- [ ] **Step 1:** In `dispatch.rs`, compute `let is_signed_bank_scheme = matches!(scheme, PiecewiseConstant) || nn_full_neural;` ONCE and derive all four gates (`uses_exit_guidance`, `uses_thermal_limiter`, the pre-lateral magnitude gate, `skip_lateral`) from it. Run guidance_regression (all 6) — bit-identical (pure refactor of a boolean). Commit.
- [ ] **Step 2:** Thread the already-computed `energy` (dispatch.rs:162) and `altitude` (geodetic) into the capture-scheme signatures so ftc/energy_ctrl/predguid/piecewise/eqglide stop recomputing `total_energy`/`geodetic_from_spherical`. The recompute is the identical function on identical inputs → bit-identical. Run all 6 goldens. Commit.
- [ ] **Step 3:** Extract `fn securize_cos_bank(cos_ref: f64, feedback: &[(f64, f64, f64)]) -> f64` = `acos(clamp(cos_ref + Σ gain·err/denom, -1, 1))` and route eqglide/energy_ctrl/predguid/ftc through it. **Critical:** preserve the exact float operation ORDER per scheme (sum the feedback terms in the same sequence the original code did) or goldens shift. Run all 6 goldens after each scheme conversion (one commit per scheme for bisectability). Bit-identical required.
- [ ] **Step 4:** Add a shared `const DEFAULT_FALLBACK_BANK_RAD: f64` for the triplicated `60.0_f64.to_radians()` no-ref fallback. Commit.

### Task 3.7: Neural `LayerSpec::io()` accessor (H5)

**Files:** Modify `src/data/neural.rs` (the triple chain-match in `from_v2_json` ~1631-1673; construction arms in `from_v2_json` + `from_flat_weights_v2`).

- [ ] **Step 1:** Add `impl LayerSpec { fn io(&self) -> (usize /*in*/, usize /*out*/, &'static str /*label*/) }` (one match, the single home for the `output_size`/`hidden_size`/`d_model`/`n_steps*input_size`/`input_size` output rule). Replace the three consecutive matches in `from_v2_json`'s chain validator with calls to `io()`.
- [ ] **Step 2:** The cross-language equivalence tests (`tests/` JSON roundtrip) + `from_v2_json`/`from_flat_weights_v2` unit tests are the guard — run `cargo test -p aerocapture neural` and the full suite. Output bit-identical (same arithmetic, fewer match sites). Commit.
- [ ] **Step 3 (optional, same task):** add the missing positivity validation (`d_model/d_ffn/n_seq != 0`) to the `from_flat_weights_v2` Transformer arm to match `from_v2_json` (low-sev asymmetry noted in review). Add a unit test that `from_flat_weights_v2` rejects `d_model=0`. Commit.

---

## Phase 4 — Behavior-changing fixes (ISOLATED commits, golden checkpoint each)

Each task: implement → `cargo test -p aerocapture --test guidance_regression --test e2e` → re-hash goldens. **If a golden moved: STOP, present diff + justification, await approval before regenerating.** Expectation per the golden regime (bias + fixed + nominal): tasks 4.1, 4.6, 4.7 are golden-neutral by construction; 4.2/4.3/4.4/4.5 are golden-neutral unless the nominal MSR trajectory hits the changed branch (verify empirically).

### Task 4.1: EKF exit-phase lock (H1b)

- [ ] **Failing test:** in `estimator.rs` tests, drive `navigate_ekf` past the exit transition then feed a step with `velocity_relative >= vphase && velocity_radial < 0`; assert `guidance_phase` stays 2 (currently reverts to 1).
- [ ] Confirm fail.
- [ ] **Implement:** add the `!legacy.exit_phase_locked` guard to the EKF phase block (`estimator.rs:578`) and set `legacy.exit_phase_locked = true` on transition — mirroring `navigate` (236-245). The `exit_phase_locked` field already exists on `NavigationState`.
- [ ] Test pass; full suite; **golden checkpoint** (bias goldens → no move expected). Commit (`fix(nav): EKF mode honors exit-phase irreversibility, matching bias mode`).

### Task 4.2: Density-inversion sign guard (H2)

- [ ] **Failing test:** construct a nav state where `denom = Cx·cosα + Cz·sinα < 0`; assert the bias-mode density estimate is REJECTED (gain held, not driven to the 0.1 floor) rather than negative.
- [ ] Confirm fail.
- [ ] **Implement:** at `estimator.rs:154`, change the guard from `denom.abs() > 1e-10` to `denom > 1e-10` (positive denominator is the only physical regime) on BOTH bias and EKF paths; a non-positive denom yields `density_estimated = 0.0` (held).
- [ ] Test pass; full suite; **golden checkpoint** (nominal MSR keeps denom>0 → no move expected; if it moves, the nominal trajectory was relying on a non-physical branch — surface it). Commit.

### Task 4.3: Bias density-filter trigger consistency (M4)

- [ ] **Failing test:** a guard-tripped step (`density_estimated = 0`) must NOT drag `density_gain` toward the 0.1 floor in bias mode (match EKF's `> 0` gate).
- [ ] Implement: add `&& density_estimated > 0.0` to the bias filter trigger (`estimator.rs:169`).
- [ ] Test pass; full suite; **golden checkpoint**. Commit.

### Task 4.4: Unconditional density-gain clamp (M5)

- [ ] **Failing test:** with `rho_model` underflowed (~0) below 100 km, an out-of-range pre-existing `density_gain` must still be re-clamped to [0.1, 10.0].
- [ ] Implement: hoist `nav_state.density_gain = nav_state.density_gain.clamp(0.1, 10.0);` OUT of the `if rho_model.abs() > 1e-30` block (`estimator.rs:178`) so it runs every tick before `out.density_guidance` is formed.
- [ ] Test pass; full suite; **golden checkpoint**. Commit.

### Task 4.5: Maneuver inclination guard (M3 behavior half)

- [ ] **Failing test:** `compute_deltav` with a pathological target (`rayneu <= 0`) must return finite, not NaN.
- [ ] Implement: switch `compute_deltav`'s inclination call to the GUARDED `inclination_dv` (add the `r > 0` guard inside the Task 3.1 helper; `predicted_dv_for_nn` already wants it). Single guarded helper now serves both.
- [ ] Test pass; full suite; **golden checkpoint** (nominal captures have rayneu>0 → no move). Commit.

### Task 4.6: Unify `bounce_alt` convention (M8)

- [ ] **Failing/characterization test:** assert the adaptive-path `bounce_alt` equals the geodetic altitude (not `state[0] - equatorial_radius`) for a known bounce state.
- [ ] **Implement:** make the adaptive path (`tick.rs:350`) record `bounce_alt` via `geodetic_from_spherical(...)` — the SAME convention the fixed path (`tick.rs:399`) already uses. This unifies toward the fixed/geodetic convention, so **fixed-mode goldens are unchanged by construction**; only adaptive runs change (and become consistent + oblateness-correct).
- [ ] Test pass; full suite; **golden checkpoint** (fixed goldens → no move expected). Commit.

### Task 4.7: ifinal None-arm consistency follow-up

- [ ] Already unified in Task 3.2 via `ifinal_for`. Add a regression test asserting the RL/env path and CLI path agree on `ifinal` for each `TermReason`. (`tests/env_equivalence.rs` likely already covers this — extend if not.) Commit.

---

## Phase 5 — PyO3 / FFI ergonomics (rebuild required)

Rebuild once at the start: from REPO ROOT, `uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml`. Run `pytest tests/test_pyo3.py tests/test_v2_rust_python_equivalence.py` after each task.

### Task 5.1: env.rs panic → clean PyResult / defensive fallback (M1)

**Files:** `aerocapture-py/src/env.rs:66, 210, 346`.

- [ ] **Step 1:** Constructor (`new`, line 66): replace `.expect("neural_net model required")` with a checked `Err(PyValueError::new_err("RL env requires [data] neural_network"))` — fail-fast with a clean Python error instead of a `PanicException`. Store a guaranteed-present handle (or keep the Arc and rely on the construction-time check).
- [ ] **Step 2:** `build_obs_for_env` (line 346): since the model is guaranteed by construction, replace `.expect()` with the same handle (no second fallible read) — removes the latent panic in the `py.detach()` + Rayon hot loop.
- [ ] **Step 3:** `unreachable!()` (line 210, inside the parallel step closure): replace with a defensive `0` mapping (or `debug_assert!` + safe fallback) so a future refactor of the `term != None` guard cannot poison a Rayon worker.
- [ ] **Step 4:** Rebuild; `pytest tests/test_pyo3.py` + add a `test_rl_env_without_nn_raises_valueerror`. Green. Commit.

### Task 5.2: results.rs unwrap → `?` and typed-empty arrays (M2)

**Files:** `aerocapture-py/src/results.rs` (6 `from_vec2().unwrap()` + 2 empty-shape sites).

- [ ] Replace `PyArray2::from_vec2(...).unwrap()` with `?` (propagate as `PyErr`) on the `#[getter]` paths.
- [ ] Empty-trajectory: return `PyArray2::<f64>::zeros(py, [0, 17], false)` (correct column count) instead of `from_vec2(py, &[])` (which yields (0,0)). Mirror the `[0, NN_INPUT_WIDTH]` pattern `collect_supervised` already uses.
- [ ] Rebuild; add `test_empty_trajectory_has_17_columns`. Green. Commit.

### Task 5.3: lib.rs collect_* dedup + clippy nits (H4-FFI, L12)

**Files:** `aerocapture-py/src/lib.rs` (`collect_supervised` ~447-583 vs `collect_nn_inputs` ~595-695), env.rs/lib.rs clippy.

- [ ] Extract the shared per-seed run + trace-accumulation into `fn collect_trace(...)` taking a guidance-type-override option and a dict-assembly closure; both pyfunctions delegate.
- [ ] Clippy: add `type` aliases for the complex `env.rs` return tuples (lines 117, 157) and the `lib.rs` `per_seed` Vec (477, 607); fix `needless_range_loop` at lib.rs:225 (`for (j, x) in row.iter_mut().enumerate()`); bundle `flat_weights_to_json`'s 8 args into a small `struct` to clear `too_many_arguments`.
- [ ] Rebuild; `pytest tests/test_pyo3.py tests/test_v2_rust_python_equivalence.py`; `cargo clippy --workspace --all-targets` → zero warnings. Commit.

---

## Phase 6 — Numerical robustness + hot-loop nits (behavior-preserving)

### Task 6.1: `brent` → non-panicking (L6)

**Files:** `src/integration/events.rs:19-26`, caller `check_events_and_locate:275`.

- [ ] **Step 1:** Characterization test: the existing brent tests + event-detection E2E pin behavior for valid brackets.
- [ ] **Step 2:** Change `brent` to return `f64` still but replace the `assert!(fa*fb <= 0.0)` with: if signs agree, return the endpoint with smaller `|f|` (clamp fallback) instead of panicking. (The sole caller pre-checks the sign and dense endpoints are bit-exact, so this fallback is currently unreachable — it's defense against future misuse, not a behavior change.) Keep the valid-bracket path byte-identical.
- [ ] Full suite + event-detection + goldens bit-identical. Commit.

### Task 6.2: `error_norm` scale floor (L7)

**Files:** `src/integration/dopri45.rs:129`.

- [ ] Floor the per-component scale: `let scale = (atol[i] + rtol * y[i].abs()).max(1e-300);`. With the current all-positive `DOPRI45_ATOL` this is a no-op (scale already > 0), so adaptive E2E + dense-output tests stay bit-identical — it removes a latent div-by-zero the moment `atol` is ever TOML-exposed.
- [ ] Full suite + dopri45 tests bit-identical. Commit.

### Task 6.3: Named constants for magic numbers (L5)

**Files:** `src/gnc/guidance/fnpag.rs:271`, `src/simulation/tick.rs` + `runner.rs`, `src/gnc/navigation/estimator.rs`/`ekf.rs`.

- [ ] Promote fnpag's `50e3` altitude breakpoint to a named `const` (or a `GuidanceParams` field — choose `const` to avoid widening the GA chromosome). Name the bounce sentinel `1e34` → `BOUNCE_ALT_UNSET`, the `20e3` apoapsis-crash gate → `MIN_BOUNCE_ALT_FOR_CRASH`, and the density-factor bounds → `const DENSITY_FACTOR_MIN: f64 = 0.1; MAX = 10.0;` (derive EKF's `-0.9/9.0` as `MIN-1.0`/`MAX-1.0`). Same values → bit-identical. Commit per file-group.

### Task 6.4: Hot-loop hygiene (L10, L11)

- [ ] `tick.rs` per-tick `eprintln!` debug (lines ~243-256) + the extra `geodetic_from_spherical` it triggers: gate behind a `cfg!(debug_assertions)` or remove. Behavior-preserving for release (only removes stderr spew). Commit.
- [ ] `RunState` (`init.rs:17`): verify all nested types (`EntryConditions`, `NavigationBiases`, `PilotBiases`) are `Copy` (most data structs already derive it). If so, `#[derive(Copy)]` on `RunState` and replace the per-tick `state.run_state.clone()` sites (`tick.rs:303/307/339`, `runner.rs:857`) with copies. If any nested type is NOT Copy, leave as-is and note why. Full suite green. Commit.

---

## Phase 7 — Module-split restructure (high-churn, behavior-preserving, LAST)

These are pure code moves. After EACH split: `cargo build` + `cargo test -p aerocapture` full green + goldens bit-identical + `cargo fmt` + `cargo clippy`. One module-extraction per commit for revertability. Use `git mv` where a whole file moves so history follows.

### Task 7.1: Split `data/neural.rs` (4876 → module tree)

- [ ] Create `src/data/neural/` with `mod.rs` (NeuralNetModel, LayerSpec, save_json/from_v2_json/from_flat_weights_v2, normalization, OutputParam), and per-layer submodules `layers/{dense,gru,lstm,window,transformer,mamba}.rs` (each `XxxLayer` struct + its `LayerWeights` impl + forward helpers). Move the `#[cfg(test)]` block (~half the file) into `src/data/neural/tests.rs` (`#[cfg(test)] mod tests;`).
- [ ] Keep all paths re-exported from `data::neural::*` so no call site outside the module changes. Run full suite + cross-language equivalence + goldens. Bit-identical. Commit.

### Task 7.2: Split `simulation/runner.rs` (2301 → modules)

- [ ] Extract `simulation/run_init.rs` (`build_sim_state`, the unified SimState constructor from Task 3.3), `simulation/finalize.rs` (`assemble_final_record`, `ifinal_for`, `is_pending_crash`, virtual-DV), keeping `runner.rs` as the thin `run`/`run_core`/`run_for_api*` orchestration + the integration dispatch. Move runner's `#[cfg(test)]` to `simulation/runner_tests.rs`.
- [ ] Full suite + env_equivalence + e2e + goldens. Bit-identical. Commit.

### Task 7.3: Restructure `config.rs` defaults + split (M10, L9)

- [ ] Replace the ~25 value-named `default_X()` free fns with per-struct `impl Default` + `#[serde(default)]` on the struct (follow the existing `TomlPeriods`/`TomlPilot` pattern). The shared `default_0_3` reused for two unrelated fields becomes two explicit struct-default values. **Serde-default behavior must be identical** — add a test deserializing an empty `[guidance.ftc]`/etc. and asserting every field equals its prior default. Bit-identical config → bit-identical goldens. Commit.
- [ ] Replace the blanket `#[allow(dead_code)]` on `SimInput`/`SimData` with targeted per-field allows; wire or remove `stats_only`/`save_results` (verify with `rg`). Commit.
- [ ] (Optional, if config.rs still unwieldy) split the `Toml*` structs into `config/toml_types.rs` and keep resolution logic (`deep_merge`, `resolve_toml_bases`, `from_toml`) in `config/mod.rs`. Bit-identical. Commit.

### Task 7.4: Collapse the 9× dispersion custom-override block (H-dup, finishes C2)

**Files:** `src/data/mod.rs:838-1016`.

- [ ] Now that Task 1.2 added per-domain guards, replace the 9 copy-pasted Custom blocks with a single table-driven helper: for each domain, a `&[(&str, fn(&mut S, f64))]` mapping (or a small per-struct `apply_custom(level, map) -> Result<Self>` trait). The allowed-set check (Task 1.2) lives once inside it. ~180 lines → one helper + 9 tables.
- [ ] The Task 1.1/1.2 tests + full suite guard this. Bit-identical for valid configs. Commit.

### Task 7.5: Move oversized test modules to siblings (estimator, others)

- [ ] `estimator.rs` (~960 test lines of 1588): move `#[cfg(test)] mod tests` to `src/gnc/navigation/estimator_tests.rs`. Same for any remaining file >1300 lines that is mostly tests (`dispersions.rs`, `config.rs` if not split above). `git mv`-style; behavior-preserving. Full suite green. Commit per file.

---

## Phase 8 — Final verification + commit

### Task 8.1: Full gates

- [ ] `cargo test -p aerocapture` — all green (≥ baseline 572 + new tests).
- [ ] `cargo fmt --all --check` — clean.
- [ ] `cargo clippy --workspace --all-targets` — zero warnings (the 6 baseline warnings resolved in Task 5.3).
- [ ] Rebuild PyO3 from repo root (`--manifest-path` form); `pytest tests/test_pyo3.py tests/test_v2_rust_python_equivalence.py tests/test_gru_pso_smoke.py` — green.
- [ ] Re-hash all 12 goldens vs `/tmp/golden_baseline_hashes.txt`. Any file that changed MUST have an approved Phase-4 checkpoint behind it; list them with their justification in the final summary.
- [ ] Run `./check_all.sh` from repo root (Rust test + fmt --check + clippy + release build) — green.

### Task 8.2: smart-commit

- [ ] Invoke the `smart-commit` skill, instructing it to take the WHOLE `worktree-feature+rust-codebase-hardening` branch into account (sync CLAUDE.md / README for the removed dead config keys — `exit_apoapsis_threshold`, `stats_only`/`save_results` if removed — the new module structure under `data/neural/` and `simulation/`, the config-validation behavior change, and the navigation/density/bounce_alt fixes). Stage only the branch's own changes; the user's `common.toml` edit is in the main tree and must not be touched.

---

## Self-Review (completed during authoring)

- **Spec coverage:** every review finding maps to a task — C1→1.1, C2→1.2/7.4, H1→3.5+4.1, H2→4.2, H3→3.2/3.3/3.4, H4→3.6/5.3, H5→3.7, M1→5.1, M2→5.2, M3→3.1/4.5, M4→4.3, M5→4.4, M6→2.6, M7→2.6, M8→4.6, M9→1.4, M10→7.3, L1→2.1/2.2/2.3/2.5, L2→2.4, L3→2.6, L4→1.3, L5→6.3, L6→6.1, L7→6.2, L8→2.6, L9→7.3, L10/L11→6.4, L12→5.3, restructure→7.1-7.5.
- **Behavior-change isolation:** all trajectory-altering fixes are in Phase 4, one per commit, each with a golden checkpoint; behavior-preserving dedup (Phase 3) and restructure (Phase 7) keep goldens bit-identical.
- **Type consistency:** helper names are used consistently — `ifinal_for`, `assemble_final_record`, `build_sim_state`, `finalize_navigation_output`, `securize_cos_bank`, `hohmann_leg_dv`, `inclination_dv`, `LayerSpec::io`, `resolve_level`, `take_custom`, `collect_trace`, `project_trajectory`.
- **No placeholders:** new logic ships with real code/tests; mechanical deletions/moves ship with exact symbols + grep-verify + gate commands.
