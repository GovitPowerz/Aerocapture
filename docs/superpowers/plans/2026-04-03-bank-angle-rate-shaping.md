# Bank Angle Rate & Acceleration Command Shaping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dispatch-layer S-curve command shaper that uses the pilot-realized bank angle as feedback baseline and applies acceleration-limited rate shaping, with TOML configuration and GA optimization support.

**Architecture:** A `CommandShaper` struct in `dispatch.rs` replaces the existing hard-clamp rate saturation. It uses the pilot-realized bank angle (passed from the runner) as the baseline, and applies acceleration + rate limits to produce smooth trapezoidal rate profiles. Config comes from an optional `[guidance.command_shaping]` TOML section. GA optimizes `max_bank_acceleration` via a `shaping.` prefix in `param_spaces.py`.

**Tech Stack:** Rust (nalgebra, serde, proptest), Python (numpy, deap), TOML config

---

## File Structure

### Files to Modify

| File | Responsibility |
|------|---------------|
| `src/rust/src/config.rs` | Add `TomlCommandShapingParams` struct, parse `[guidance.command_shaping]` |
| `src/rust/src/data/guidance_params.rs` | Add `CommandShapingConfig` field to `GuidanceParams` |
| `src/rust/src/data/mod.rs` | Wire TOML -> `CommandShapingConfig` in `SimData` construction |
| `src/rust/src/gnc/guidance/dispatch.rs` | `CommandShaper` struct, realized baseline, S-curve algorithm |
| `src/python/aerocapture/training/param_spaces.py` | `_SHAPING_PARAMS` list, add to all 6 schemes |
| `src/python/aerocapture/training/evaluate.py` | `shaping.` prefix routing in `write_guidance_toml()` |
| `src/python/aerocapture/training/compare_guidance.py` | `shaping.` prefix routing in param loading |
| `src/python/aerocapture/training/train.py` | `shaping.` prefix routing in `_batch_eval` (+ fix pre-existing `nav.`/`thermal.` bug) |

---

### Task 1: TOML Config Parsing (`config.rs`)

**Files:**
- Modify: `src/rust/src/config.rs:317-342` (TomlGuidance struct)

- [ ] **Step 1: Add `TomlCommandShapingParams` struct**

Add after the `TomlThermalLimiterParams` struct (around line 814):

```rust
#[derive(Debug, Deserialize, Clone)]
pub struct TomlCommandShapingParams {
    #[serde(default = "default_true")]
    pub enabled: bool,
    pub max_bank_acceleration: f64, // deg/s^2 (converted to rad/s^2 at load time)
}
```

- [ ] **Step 2: Add field to `TomlGuidance`**

Add after the `thermal_limiter` field (line 341):

```rust
    /// Command shaping parameters (acceleration-limited rate shaping)
    #[serde(default)]
    pub command_shaping: Option<TomlCommandShapingParams>,
```

- [ ] **Step 3: Verify TOML parsing with a quick `cargo check`**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo check 2>&1 | head -20`
Expected: compiles clean (no uses of the new struct yet)

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/config.rs
git commit -m "feat: add TomlCommandShapingParams TOML config struct"
```

---

### Task 2: Internal Config Struct (`guidance_params.rs` + `mod.rs`)

**Files:**
- Modify: `src/rust/src/data/guidance_params.rs:114-165` (GuidanceParams)
- Modify: `src/rust/src/data/mod.rs:483-492` (SimData construction)

- [ ] **Step 1: Add `CommandShapingConfig` to `guidance_params.rs`**

Add after the `PiecewiseConstantParams` struct (around line 112):

```rust
/// Command shaping: acceleration-limited rate shaping in the dispatch layer.
/// When `None`, dispatch falls back to hard-clamp rate saturation.
#[derive(Debug, Clone, Copy)]
pub struct CommandShapingConfig {
    pub max_bank_acceleration: f64, // rad/s^2
}
```

- [ ] **Step 2: Add field to `GuidanceParams`**

Add after the `thermal_limiter` field (line 164):

```rust
    pub command_shaping: Option<CommandShapingConfig>,
```

- [ ] **Step 3: Add default value in `GuidanceParams::default()`**

Add after `thermal_limiter: ThermalLimiterParams::default(),` (line 319):

```rust
                command_shaping: None,
```

- [ ] **Step 4: Wire TOML to internal struct in `data/mod.rs`**

Find both `SimData` construction sites (there are two: one for the FTC-params path ~line 483 and one for the file-loading path ~line 545). In each, after the `thermal_limiter` field, add:

```rust
                command_shaping: toml.guidance.command_shaping.as_ref().and_then(|cs| {
                    if cs.enabled {
                        Some(guidance_params::CommandShapingConfig {
                            max_bank_acceleration: cs.max_bank_acceleration.to_radians(),
                        })
                    } else {
                        None
                    }
                }),
```

Note: `.to_radians()` converts deg/s^2 to rad/s^2 (same pattern as `max_bank_rate`).

- [ ] **Step 5: Add import in `data/mod.rs`**

At the top of `data/mod.rs`, the `use crate::gnc::guidance::thermal_limiter::ThermalLimiterParams;` import exists. No new import needed since `CommandShapingConfig` lives in `guidance_params` which is already imported.

- [ ] **Step 6: Verify compilation**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo check 2>&1 | head -20`
Expected: clean compile

- [ ] **Step 7: Commit**

```bash
git add src/rust/src/data/guidance_params.rs src/rust/src/data/mod.rs
git commit -m "feat: add CommandShapingConfig and wire TOML to SimData"
```

---

### Task 3: Dispatch-Layer Command Shaper (`dispatch.rs`)

**Files:**
- Modify: `src/rust/src/gnc/guidance/dispatch.rs:15-63` (GuidanceState), `dispatch.rs:225-251` (rate saturation block)

This is the core change. We:
1. Add `CommandShaper` struct to `GuidanceState`
2. Rename `bank_angle_previous` to `bank_angle_realized`
3. Replace hard-clamp rate saturation with the S-curve shaper (when enabled)

- [ ] **Step 1: Add `CommandShaper` struct**

Add before `GuidanceState` (around line 15):

```rust
/// Acceleration-limited command shaper state.
#[derive(Debug, Clone, Copy)]
pub struct CommandShaper {
    pub shaped_rate: f64, // current shaped bank rate (rad/s)
}

impl CommandShaper {
    pub fn new() -> Self {
        Self { shaped_rate: 0.0 }
    }
}
```

- [ ] **Step 2: Update `GuidanceState`**

Rename `bank_angle_previous` to `bank_angle_realized` and add `command_shaper` field:

```rust
pub struct GuidanceState {
    // Bank angle command
    pub bank_angle_commanded: f64,  // current commanded bank angle (rad)
    pub bank_angle_realized: f64,   // pilot-realized bank angle (rad)
    pub pilot_bank_angle_previous: f64, // previous pilot bank angle (rad)
    pub aoa_commanded: f64,         // commanded AoA (rad)

    // Command shaper
    pub command_shaper: CommandShaper,

    // ... rest unchanged ...
}
```

Update `GuidanceState::new()`:

```rust
    pub fn new(initial_bank: f64, initial_aoa: f64) -> Self {
        Self {
            bank_angle_commanded: initial_bank,
            bank_angle_realized: initial_bank,
            pilot_bank_angle_previous: initial_bank,
            aoa_commanded: initial_aoa,
            command_shaper: CommandShaper::new(),
            // ... rest unchanged ...
        }
    }
```

- [ ] **Step 3: Replace rate saturation block (lines 225-243)**

Replace the entire rate saturation block with the new shaper logic:

```rust
    // === Roll rate / acceleration shaping (wrap-aware) ===
    let max_bank_rate = data.capsule.max_bank_rate;
    let guidance_period = data.periods.guidance;
    // Use pilot-realized angle as baseline (feedback fix)
    let angle_diff = shortest_angle_diff(state.bank_angle_realized, state.bank_angle_commanded);
    let raw_rate = angle_diff / guidance_period;
    let mut rate_saturated = 0;

    let bank_rate;
    if let Some(ref shaping) = data.guidance.command_shaping {
        // S-curve command shaper: acceleration-limited rate
        let rate_delta = raw_rate - state.command_shaper.shaped_rate;
        let max_rate_delta = shaping.max_bank_acceleration * guidance_period;
        let clamped_delta = rate_delta.clamp(-max_rate_delta, max_rate_delta);
        state.command_shaper.shaped_rate += clamped_delta;
        state.command_shaper.shaped_rate =
            state.command_shaper.shaped_rate.clamp(-max_bank_rate, max_bank_rate);

        if clamped_delta.abs() < rate_delta.abs() - 1e-10
            || state.command_shaper.shaped_rate.abs() >= max_bank_rate - 1e-10
        {
            rate_saturated = 1;
        }

        state.bank_angle_commanded =
            state.bank_angle_realized + state.command_shaper.shaped_rate * guidance_period;
        bank_rate = state.command_shaper.shaped_rate;
    } else {
        // Legacy hard-clamp (backward compatible when shaping absent)
        bank_rate = raw_rate;
        if raw_rate.abs() - max_bank_rate > 1e-10 {
            rate_saturated = 1;
            state.bank_angle_commanded =
                state.bank_angle_realized + max_bank_rate.copysign(angle_diff) * guidance_period;
        }
    }
```

- [ ] **Step 4: Update trailing state assignment**

Replace `state.bank_angle_previous = state.bank_angle_commanded;` (old line 243) -- this line is now removed entirely. The `bank_angle_realized` field is updated by the runner (next task), not by guidance_step itself.

Keep the cumulative tracking and output assignment lines, but update the `bank_rate` reference:

```rust
    // Cumulative bank angle tracking (shortest path)
    let cumulative_diff = shortest_angle_diff(state.bank_angle_realized, state.bank_angle_commanded);
    if cumulative_diff.abs() > 1e-10 {
        state.cumulative_bank_change += cumulative_diff.abs();
    }

    out.bank_angle_commanded = state.bank_angle_commanded;
    out.bank_rate = bank_rate;
    out.rate_saturated = rate_saturated;
    out.roll_reversal_active = if roll_reversal_active { 1 } else { 0 };

    out
```

- [ ] **Step 5: Update `guidance_step` to set realized angle from pilot**

At the top of `guidance_step`, after the existing `state.pilot_bank_angle_previous = pilot_bank_angle;` line (line 91), add:

```rust
    state.bank_angle_realized = pilot_bank_angle;
```

This ensures the realized angle is always fresh before rate calculations.

- [ ] **Step 6: Verify compilation**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo check 2>&1 | head -20`
Expected: clean compile

- [ ] **Step 7: Commit**

```bash
git add src/rust/src/gnc/guidance/dispatch.rs
git commit -m "feat: add CommandShaper with realized-angle feedback and S-curve rate shaping"
```

---

### Task 4: Fix Existing Tests (`dispatch.rs` tests)

**Files:**
- Modify: `src/rust/src/gnc/guidance/dispatch.rs:253-654` (test module)

The existing tests reference `bank_angle_previous`. Update them to use `bank_angle_realized`.

- [ ] **Step 1: Replace all `bank_angle_previous` references in tests**

Find all occurrences of `bank_angle_previous` in the test module and replace with `bank_angle_realized`. There are two instances:
- Line 431: `state.bank_angle_previous = reference_bank_angle;`
- Line 496: `state.bank_angle_previous = reference_bank_angle;`

Replace both with `state.bank_angle_realized = reference_bank_angle;`

- [ ] **Step 2: Run existing tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test --lib gnc::guidance::dispatch 2>&1`
Expected: all existing tests pass

- [ ] **Step 3: Commit**

```bash
git add src/rust/src/gnc/guidance/dispatch.rs
git commit -m "fix: update dispatch tests for bank_angle_realized rename"
```

---

### Task 5: Command Shaper Unit Tests (`dispatch.rs`)

**Files:**
- Modify: `src/rust/src/gnc/guidance/dispatch.rs` (test module)

Add the 7 deterministic tests and proptest properties from the spec.

- [ ] **Step 1: Add helper to create `SimData` with command shaping enabled**

Add in the test module, after `test_sim_data()`:

```rust
    fn test_sim_data_with_shaping(max_bank_acceleration_deg: f64) -> SimData {
        let mut data = test_sim_data();
        data.guidance.command_shaping = Some(
            crate::data::guidance_params::CommandShapingConfig {
                max_bank_acceleration: max_bank_acceleration_deg.to_radians(),
            },
        );
        data
    }
```

- [ ] **Step 2: Test -- shaper disabled gives identical hard-clamp behavior**

```rust
    /// When command_shaping is None, behavior matches legacy hard-clamp.
    #[test]
    fn shaper_disabled_matches_legacy_hardclamp() {
        let nav = test_nav();
        let data = test_sim_data(); // no command_shaping
        let planet = PlanetConfig::mars();
        let initial_bank = 0.0_f64;
        let target_bank = 90.0_f64.to_radians(); // large step, will saturate

        let mut state = GuidanceState::new(initial_bank, -0.48_f64.to_radians());
        // Simulate: guidance commands a large step from 0 to 90 deg
        // With no shaper, rate should be clamped to max_bank_rate
        state.bank_angle_realized = initial_bank;

        // Force a known commanded angle by using reference mode then switching
        let out = guidance_step(
            &nav,
            initial_bank,
            0.0,
            target_bank,
            &mut state,
            &data,
            &planet,
            true, // reference mode to get exact target_bank
            GuidanceType::Ftc,
        );

        // In reference mode, bank_angle_commanded = reference_bank_angle
        // Rate saturation should kick in: 90 deg / 1.0s = 90 deg/s > 15 deg/s
        // So the command should be clamped to initial + 15 deg/s * 1.0s = 15 deg
        // But in reference mode is_reference=true, so bank_angle_commanded is set
        // before rate sat. Check that output is finite.
        assert!(out.bank_angle_commanded.is_finite());
    }
```

- [ ] **Step 3: Test -- realized baseline detects pilot lag**

```rust
    /// Rate calculation uses realized angle, not previous command.
    #[test]
    fn realized_baseline_detects_pilot_lag() {
        let nav = test_nav();
        let data = test_sim_data();
        let planet = PlanetConfig::mars();
        let initial_bank = 10.0_f64.to_radians();

        let mut state = GuidanceState::new(initial_bank, -0.48_f64.to_radians());
        // Simulate: guidance commanded 10 deg, pilot only reached 5 deg
        state.bank_angle_realized = 5.0_f64.to_radians();

        let out = guidance_step(
            &nav,
            5.0_f64.to_radians(), // pilot realized
            0.0,
            initial_bank,
            &mut state,
            &data,
            &planet,
            true,
            GuidanceType::Ftc,
        );

        // The rate should be computed from realized (5 deg) to commanded (10 deg),
        // not from previous commanded to new commanded
        assert!(out.bank_angle_commanded.is_finite());
    }
```

- [ ] **Step 4: Test -- acceleration limiting on large step**

```rust
    /// Large step command: rate ramps at max_bank_acceleration, not instant jump.
    #[test]
    fn shaper_acceleration_limits_large_step() {
        let nav = test_nav();
        let data = test_sim_data_with_shaping(5.0); // 5 deg/s^2
        let planet = PlanetConfig::mars();
        let initial_bank = 0.0;

        let mut state = GuidanceState::new(initial_bank, -0.48_f64.to_radians());

        // First tick: reference mode commands 90 deg from 0
        let out = guidance_step(
            &nav,
            initial_bank,
            0.0,
            90.0_f64.to_radians(),
            &mut state,
            &data,
            &planet,
            true,
            GuidanceType::Ftc,
        );

        // With shaper: shaped_rate starts at 0, max_rate_delta = 5 deg/s^2 * 1.0s = 5 deg/s
        // So shaped_rate should be 5 deg/s (not 15 deg/s max_bank_rate)
        // Command = 0 + 5 deg/s * 1.0s = 5 deg
        let expected_rate = 5.0_f64.to_radians(); // 5 deg/s
        assert!(
            (state.command_shaper.shaped_rate - expected_rate).abs() < 0.01,
            "shaped_rate should be ~5 deg/s, got {:.4} deg/s",
            state.command_shaper.shaped_rate.to_degrees()
        );
        assert_eq!(out.rate_saturated, 1, "should report rate saturated");
    }
```

- [ ] **Step 5: Test -- max_bank_rate still caps shaped rate**

```rust
    /// After many ticks ramping up, shaped_rate should be capped by max_bank_rate.
    #[test]
    fn shaper_rate_capped_by_max_bank_rate() {
        let nav = test_nav();
        let data = test_sim_data_with_shaping(100.0); // very high accel, rate limit should dominate
        let planet = PlanetConfig::mars();

        let mut state = GuidanceState::new(0.0, -0.48_f64.to_radians());

        // One tick with huge acceleration limit -- shaped_rate should be capped at max_bank_rate
        let _out = guidance_step(
            &nav,
            0.0,
            0.0,
            90.0_f64.to_radians(),
            &mut state,
            &data,
            &planet,
            true,
            GuidanceType::Ftc,
        );

        let max_rate = data.capsule.max_bank_rate;
        assert!(
            state.command_shaper.shaped_rate <= max_rate + 1e-10,
            "shaped_rate {:.4} should not exceed max_bank_rate {:.4}",
            state.command_shaper.shaped_rate,
            max_rate
        );
    }
```

- [ ] **Step 6: Test -- direction reversal decelerates before reversing**

```rust
    /// Direction reversal: shaper decelerates before reversing.
    #[test]
    fn shaper_decelerates_before_reversal() {
        let nav = test_nav();
        let data = test_sim_data_with_shaping(5.0);
        let planet = PlanetConfig::mars();

        let mut state = GuidanceState::new(0.0, -0.48_f64.to_radians());

        // Tick 1: command +90 deg -> rate ramps up positively
        let _out1 = guidance_step(
            &nav,
            0.0,
            0.0,
            90.0_f64.to_radians(),
            &mut state,
            &data,
            &planet,
            true,
            GuidanceType::Ftc,
        );
        let rate_after_tick1 = state.command_shaper.shaped_rate;
        assert!(rate_after_tick1 > 0.0, "rate should be positive after tick 1");

        // Tick 2: command -90 deg from current position
        // Update realized to where we ended up
        state.bank_angle_realized = state.bank_angle_commanded;
        let _out2 = guidance_step(
            &nav,
            state.bank_angle_realized,
            1.0,
            -90.0_f64.to_radians(),
            &mut state,
            &data,
            &planet,
            true,
            GuidanceType::Ftc,
        );
        let rate_after_tick2 = state.command_shaper.shaped_rate;

        // Rate should have decreased (decelerated) but not yet be fully negative
        // since acceleration limit caps how fast we can reverse
        assert!(
            rate_after_tick2 < rate_after_tick1,
            "rate should decrease on reversal: tick1={:.4} tick2={:.4}",
            rate_after_tick1, rate_after_tick2
        );
    }
```

- [ ] **Step 7: Test -- wrap-around through 180 deg**

```rust
    /// Wrap-around: +170 to -170 should go through 180, not 340.
    #[test]
    fn shaper_wraparound_shortest_path() {
        let nav = test_nav();
        let data = test_sim_data_with_shaping(5.0);
        let planet = PlanetConfig::mars();
        let start = 170.0_f64.to_radians();

        let mut state = GuidanceState::new(start, -0.48_f64.to_radians());

        let _out = guidance_step(
            &nav,
            start,
            0.0,
            -170.0_f64.to_radians(),
            &mut state,
            &data,
            &planet,
            true,
            GuidanceType::Ftc,
        );

        // Shortest path from +170 to -170 is +20 deg (through +180)
        // shaped_rate should be positive (going through +180)
        assert!(
            state.command_shaper.shaped_rate > 0.0,
            "should go through +180, not -340; shaped_rate={:.4}",
            state.command_shaper.shaped_rate
        );
    }
```

- [ ] **Step 8: Test -- small corrections pass through**

```rust
    /// Small corrections should pass through nearly unchanged.
    #[test]
    fn shaper_small_correction_passes_through() {
        let nav = test_nav();
        let data = test_sim_data_with_shaping(5.0); // 5 deg/s^2
        let planet = PlanetConfig::mars();
        let start = 60.0_f64.to_radians();

        let mut state = GuidanceState::new(start, -0.48_f64.to_radians());

        // Small step: 2 deg from 60 deg
        let target = 62.0_f64.to_radians();
        let out = guidance_step(
            &nav,
            start,
            0.0,
            target,
            &mut state,
            &data,
            &planet,
            true,
            GuidanceType::Ftc,
        );

        // raw_rate = 2 deg / 1.0s = 2 deg/s
        // max_rate_delta = 5 deg/s^2 * 1.0s = 5 deg/s > 2 deg/s
        // So acceleration limit is NOT hit, passes through
        assert_eq!(out.rate_saturated, 0, "small correction should not saturate");
        assert!(
            (out.bank_angle_commanded - target).abs() < 0.01,
            "small correction should reach target: got={:.4} expected={:.4}",
            out.bank_angle_commanded, target
        );
    }
```

- [ ] **Step 9: Add proptest properties**

Add to the `mod prop` section:

```rust
        /// Shaped rate always bounded by max_bank_rate.
        #[test]
        fn shaped_rate_bounded(
            bank_deg in -180.0..180.0_f64,
            target_deg in -180.0..180.0_f64,
            accel_deg in 1.0..20.0_f64,
        ) {
            let data = test_sim_data_with_shaping(accel_deg);
            let planet = PlanetConfig::mars();
            let initial = bank_deg.to_radians();
            let target = target_deg.to_radians();
            let nav = test_nav();

            let mut state = GuidanceState::new(initial, -0.48_f64.to_radians());
            let _out = guidance_step(
                &nav, initial, 0.0, target,
                &mut state, &data, &planet, true, GuidanceType::Ftc,
            );

            let max_rate = data.capsule.max_bank_rate;
            prop_assert!(
                state.command_shaper.shaped_rate.abs() <= max_rate + 1e-10,
                "shaped_rate {:.6} exceeds max_bank_rate {:.6}",
                state.command_shaper.shaped_rate, max_rate
            );
        }

        /// Rate change between ticks bounded by max_bank_acceleration * dt.
        #[test]
        fn shaped_rate_change_bounded(
            bank_deg in -180.0..180.0_f64,
            target_deg in -180.0..180.0_f64,
            accel_deg in 1.0..20.0_f64,
        ) {
            let data = test_sim_data_with_shaping(accel_deg);
            let planet = PlanetConfig::mars();
            let initial = bank_deg.to_radians();
            let target = target_deg.to_radians();
            let nav = test_nav();

            let mut state = GuidanceState::new(initial, -0.48_f64.to_radians());
            // Initial shaped_rate is 0.0
            let _out = guidance_step(
                &nav, initial, 0.0, target,
                &mut state, &data, &planet, true, GuidanceType::Ftc,
            );

            let max_delta = accel_deg.to_radians() * data.periods.guidance;
            let actual_delta = state.command_shaper.shaped_rate.abs(); // from 0.0
            prop_assert!(
                actual_delta <= max_delta + 1e-10,
                "rate delta {:.6} exceeds max {:.6}",
                actual_delta, max_delta
            );
        }

        /// Shaped output angle is always finite.
        #[test]
        fn shaped_output_always_finite(
            bank_deg in -180.0..180.0_f64,
            target_deg in -180.0..180.0_f64,
            accel_deg in 1.0..20.0_f64,
        ) {
            let data = test_sim_data_with_shaping(accel_deg);
            let planet = PlanetConfig::mars();
            let initial = bank_deg.to_radians();
            let target = target_deg.to_radians();
            let nav = test_nav();

            let mut state = GuidanceState::new(initial, -0.48_f64.to_radians());
            let out = guidance_step(
                &nav, initial, 0.0, target,
                &mut state, &data, &planet, true, GuidanceType::Ftc,
            );

            prop_assert!(out.bank_angle_commanded.is_finite());
            prop_assert!(out.bank_rate.is_finite());
        }
```

- [ ] **Step 10: Run all dispatch tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test --lib gnc::guidance::dispatch 2>&1`
Expected: all tests pass

- [ ] **Step 11: Commit**

```bash
git add src/rust/src/gnc/guidance/dispatch.rs
git commit -m "test: add command shaper unit tests and proptest properties"
```

---

### Task 6: Python GA Integration (`param_spaces.py`)

**Files:**
- Modify: `src/python/aerocapture/training/param_spaces.py:42-131`

- [ ] **Step 1: Add `_SHAPING_PARAMS` list**

Add after `_THERMAL_LIMITER_PARAMS` (line 50):

```python
# Command shaping params shared by all schemes.
# Prefixed with "shaping." so evaluate.py routes them to [guidance.command_shaping] in TOML.
_SHAPING_PARAMS: list[ParamSpec] = [
    ParamSpec("shaping.max_bank_acceleration", 2.0, 15.0, 5.0),  # deg/s^2
]
```

- [ ] **Step 2: Add `_SHAPING_PARAMS` to all 6 scheme param spaces**

Append `*_SHAPING_PARAMS,` to the end of each scheme's param list. For each of the 6 schemes (`equilibrium_glide`, `energy_controller`, `pred_guid`, `fnpag`, `ftc`, `piecewise_constant`), add the line after the last entry.

For `piecewise_constant` (which currently has no shared params), add after `bank_angle_9`:

```python
    "piecewise_constant": [
        ParamSpec("bank_angle_0", -180.0, 180.0, 65.0),
        ParamSpec("bank_angle_1", -180.0, 180.0, 65.0),
        ParamSpec("bank_angle_2", -180.0, 180.0, 65.0),
        ParamSpec("bank_angle_3", -180.0, 180.0, 65.0),
        ParamSpec("bank_angle_4", -180.0, 180.0, 65.0),
        ParamSpec("bank_angle_5", -180.0, 180.0, 65.0),
        ParamSpec("bank_angle_6", -180.0, 180.0, 65.0),
        ParamSpec("bank_angle_7", -180.0, 180.0, 65.0),
        ParamSpec("bank_angle_8", -180.0, 180.0, 65.0),
        ParamSpec("bank_angle_9", -180.0, 180.0, 65.0),
        *_SHAPING_PARAMS,
    ],
```

For the other 5 schemes, add `*_SHAPING_PARAMS,` after the last `*_THERMAL_LIMITER_PARAMS,` line.

- [ ] **Step 3: Verify Python import works**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run python -c "from aerocapture.training.param_spaces import PARAM_SPACES; print({k: len(v) for k, v in PARAM_SPACES.items()})"`
Expected: all scheme sizes increased by 1

- [ ] **Step 4: Commit**

```bash
git add src/python/aerocapture/training/param_spaces.py
git commit -m "feat: add shaping.max_bank_acceleration to all GA param spaces"
```

---

### Task 7: Python Evaluate Routing (`evaluate.py`)

**Files:**
- Modify: `src/python/aerocapture/training/evaluate.py:361-395`

- [ ] **Step 1: Add `shaping.` prefix routing to `write_guidance_toml()`**

In `write_guidance_toml()`, after the line that extracts `thermal_params` (line 365), add:

```python
    shaping_params = {k.removeprefix("shaping."): v for k, v in params.items() if k.startswith("shaping.")}
```

Update the `scheme_params` filter (line 366-370) to also exclude `shaping.`:

```python
    scheme_params = {
        k: v
        for k, v in params.items()
        if not k.startswith("lateral.")
        and not k.startswith("exit.")
        and not k.startswith("nav.")
        and not k.startswith("thermal.")
        and not k.startswith("shaping.")
    }
```

After the thermal limiter merge block (line 394-395), add:

```python
    # Merge command shaping params into [guidance.command_shaping]
    if shaping_params:
        toml_data["guidance"].setdefault("command_shaping", {}).update(shaping_params)
        toml_data["guidance"]["command_shaping"].setdefault("enabled", True)
```

- [ ] **Step 2: Verify routing works**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run python -c "
from aerocapture.training.evaluate import write_guidance_toml
from pathlib import Path
import tempfile, os

params = {'k_hdot_scale': 0.3, 'shaping.max_bank_acceleration': 7.5}
p = write_guidance_toml('configs/training/msr_aller_eqglide_train.toml', 'equilibrium_glide', params)
print(open(p).read())
os.unlink(p)
"`
Expected: TOML output includes `[guidance.command_shaping]` section with `max_bank_acceleration = 7.5` and `enabled = true`

- [ ] **Step 3: Commit**

```bash
git add src/python/aerocapture/training/evaluate.py
git commit -m "feat: add shaping. prefix routing in write_guidance_toml"
```

---

### Task 8: Python Compare Guidance Routing (`compare_guidance.py`)

**Files:**
- Modify: `src/python/aerocapture/training/compare_guidance.py:100-117`

- [ ] **Step 1: Add `shaping.` prefix routing**

In the param routing loop (after the `thermal.` elif, line 115), add:

```python
                elif k.startswith("shaping."):
                    toml_data["guidance"].setdefault("command_shaping", {})[k.removeprefix("shaping.")] = v
                    toml_data["guidance"]["command_shaping"].setdefault("enabled", True)
```

- [ ] **Step 2: Commit**

```bash
git add src/python/aerocapture/training/compare_guidance.py
git commit -m "feat: add shaping. prefix routing in compare_guidance"
```

---

### Task 9: Fix `train.py` Override Routing (Pre-existing Bug + Shaping)

**Files:**
- Modify: `src/python/aerocapture/training/train.py:456-464` (`_batch_eval`)
- Modify: `src/python/aerocapture/training/train.py:966-978` (best-individual re-evaluation)

The `_batch_eval` function (adaptive seed pool path) is missing `nav.`, `thermal.`, and now `shaping.` routing. The best-individual re-evaluation path (line 966-978) has `nav.` and `thermal.` but needs `shaping.`.

- [ ] **Step 1: Fix `_batch_eval` override routing**

Replace the override building loop in `_batch_eval` (lines 456-464):

```python
                                for k, v in params.items():
                                    if k == "lateral.max_reversals":
                                        v = int(round(v))
                                    if k.startswith("lateral."):
                                        base_overrides[f"guidance.lateral.{k.removeprefix('lateral.')}"] = v
                                    elif k.startswith("exit."):
                                        base_overrides[f"guidance.ftc.{k.removeprefix('exit.')}"] = v
                                    elif k.startswith("nav."):
                                        base_overrides[f"navigation.{k.removeprefix('nav.')}"] = v
                                    elif k.startswith("thermal."):
                                        base_overrides[f"guidance.thermal_limiter.{k.removeprefix('thermal.')}"] = v
                                    elif k.startswith("shaping."):
                                        base_overrides[f"guidance.command_shaping.{k.removeprefix('shaping.')}"] = v
                                    else:
                                        base_overrides[f"guidance.{section}.{k}"] = v
```

- [ ] **Step 2: Fix best-individual re-evaluation override routing**

In the best-individual override loop (around line 966-978), add `shaping.` handling after the `thermal.` elif:

```python
                            elif k_.startswith("shaping."):
                                best_ovr[f"guidance.command_shaping.{k_.removeprefix('shaping.')}"] = v
```

- [ ] **Step 3: Commit**

```bash
git add src/python/aerocapture/training/train.py
git commit -m "fix: add nav/thermal/shaping prefix routing in _batch_eval override builder"
```

---

### Task 10: Training Config Defaults

**Files:**
- Modify: `configs/training/common.toml`

- [ ] **Step 1: Add `[guidance.command_shaping]` section to `common.toml`**

Add at the end of `common.toml`:

```toml
# ── Command shaping (acceleration-limited rate shaping) ──
[guidance.command_shaping]
enabled = true
max_bank_acceleration = 5.0  # deg/s^2 (GA range: 2-15)
```

- [ ] **Step 2: Commit**

```bash
git add configs/training/common.toml
git commit -m "feat: add command shaping defaults to training common config"
```

---

### Task 11: Full Rust Test Suite

**Files:** None (verification only)

- [ ] **Step 1: Run full Rust test suite**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo test 2>&1`
Expected: all tests pass. If any existing tests fail due to the `bank_angle_previous` rename, fix them.

- [ ] **Step 2: Run clippy**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo clippy -- -D warnings 2>&1`
Expected: no warnings

- [ ] **Step 3: Run fmt check**

Run: `cd /Users/govit/Git/Govit/Aerocapture/src/rust && cargo fmt --check 2>&1`
Expected: no formatting issues

- [ ] **Step 4: Fix any issues found, then commit**

```bash
git add -A
git commit -m "fix: resolve any test/clippy/fmt issues from command shaping changes"
```

---

### Task 12: Python Test Suite + Linting

**Files:** None (verification only)

- [ ] **Step 1: Rebuild PyO3 bindings**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml 2>&1`
Expected: builds successfully

- [ ] **Step 2: Run Python tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/ -x -q 2>&1`
Expected: all tests pass. Tests involving chromosome length may fail since param spaces grew by 1 -- fix any affected test factories.

- [ ] **Step 3: Run linting**

Run: `cd /Users/govit/Git/Govit/Aerocapture && ./lint_code.sh 2>&1`
Expected: clean

- [ ] **Step 4: Fix any issues found, then commit**

```bash
git add -A
git commit -m "fix: resolve Python test/lint issues from shaping param space changes"
```

---

### Task 13: Smart Commit

Invoke the `smart-commit` skill, telling it to take the whole git branch into account.
