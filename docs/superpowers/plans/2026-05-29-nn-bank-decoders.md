# NN bank decoders (`scaled_pi` + `delta`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two `full_neural` NN bank-angle decoders — `scaled_pi` (`n·π·tanh`, knob `n`) and `delta` (bounded increment on previous realized bank, wrapped at the guidance boundary) — and reencode bank-angle history inputs as seam-free `(sin, cos)` pairs.

**Architecture:** `OutputParam` gains `ScaledPi` and `Delta` variants; the decoder lives in `nn_bank_angle` (post-`forward`), reusing the existing `angle_utils::wrap_to_pi`. A new `GuidanceState::prev_realized_bank_for_nn` telemetry field (mirroring `prev_bank_for_nn`) feeds both the `delta` base and the prev-realized `(sin,cos)` input. Config carries `scaled_pi_n` / `delta_max`; warm-start gains two target-encoding branches. PSO-first: no `V2Policy`/PPO changes.

**Tech Stack:** Rust (edition 2024, nalgebra), PyO3 (`aerocapture_rs`), Python 3.14 (pymoo PSO), pytest, cargo test.

**Spec:** `docs/superpowers/specs/2026-05-29-nn-bank-decoders-design.md`

**Branch:** `feature/nn-bank-decoders` (already created; spec already committed).

---

## File map

- `src/rust/src/data/neural.rs` — `OutputParam` variants; `NeuralNetModel`/`NnJsonFileV2` knob fields; validators; `build_nn_input` (`(sin,cos)` slots, `NN_FULL_INPUT_SIZE` 25→31, `prev_realized_bank` param); `nn_bank_angle` decoder arms + `prev_realized_bank` param.
- `src/rust/src/gnc/guidance/dispatch.rs` — `GuidanceState::prev_realized_bank_for_nn` field + init; snapshot + pass to `nn_bank_angle`.
- `src/rust/src/simulation/tick.rs` — `FULL_MASK` 25→31; pass `prev_realized` to `build_nn_input`; update `prev_realized_bank_for_nn` post-guidance; push realized into `supervised_trace`.
- `src/rust/src/simulation/runner.rs` — `supervised_trace` tuple widened to carry realized bank (2 type sites).
- `src/rust/src/config.rs` — `TomlNeuralNetworkParams` gains `scaled_pi_n` / `delta_max`.
- `src/rust/src/data/mod.rs` — `validate_output_parameterization` extension; cross-check match arms; `full_neural` guard; TOML knob override onto loaded model.
- `src/rust/aerocapture-py/src/lib.rs` — `collect_supervised` emits `prev_realized` (T,).
- `src/python/aerocapture/training/config.py` — `NetworkConfig` gains `scaled_pi_n` / `delta_max`.
- `src/python/aerocapture/training/warm_start.py` — `scaled_pi` / `delta` loss-target branches.
- `configs/training/msr_aller_nn_scaledpi_train.toml`, `configs/training/msr_aller_nn_delta_train.toml` — new PSO leaf configs.
- `src/python/aerocapture/training/compare_guidance.py`, `train_all.sh` — scheme registration.

---

## Task 1: `OutputParam` variants + model knob fields + output validators

**Files:**
- Modify: `src/rust/src/data/neural.rs` (enum ~109, `NnJsonFileV2` ~1120, `NeuralNetModel` ~1135, `validate_output_size` ~1193, `validate_output_activation` ~1218, constructors `from_v2_json`/`from_flat_weights_v2`)
- Test: inline `#[cfg(test)]` in `src/rust/src/data/neural.rs`

- [ ] **Step 1: Write failing validator tests**

Add to the `tests` module in `neural.rs`:

```rust
#[test]
fn scaled_pi_requires_output_size_1() {
    assert!(NeuralNetModel::validate_output_size(1, OutputParam::ScaledPi, "<t>").is_ok());
    assert!(NeuralNetModel::validate_output_size(2, OutputParam::ScaledPi, "<t>").is_err());
}

#[test]
fn delta_requires_output_size_1() {
    assert!(NeuralNetModel::validate_output_size(1, OutputParam::Delta, "<t>").is_ok());
    assert!(NeuralNetModel::validate_output_size(2, OutputParam::Delta, "<t>").is_err());
}

#[test]
fn scaled_pi_and_delta_require_tanh_last_activation() {
    for p in [OutputParam::ScaledPi, OutputParam::Delta] {
        assert!(NeuralNetModel::validate_output_activation(Activation::Tanh, p, "<t>").is_ok());
        assert!(NeuralNetModel::validate_output_activation(Activation::Linear, p, "<t>").is_err());
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/rust && cargo test -p aerocapture data::neural 2>&1 | tail -20`
Expected: compile error — `OutputParam` has no variant `ScaledPi`/`Delta`.

- [ ] **Step 3: Add the enum variants**

In `neural.rs`, extend the enum (keep it a plain `Copy + Eq` unit enum — knobs live on the model, not in the variant):

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputParam {
    #[default]
    Atan2Signed,
    AcosTanh,
    ScaledPi,
    Delta,
}
```

- [ ] **Step 4: Add knob fields to model + JSON file structs with serde defaults**

Add default fns near the top of `neural.rs` (after imports):

```rust
fn default_scaled_pi_n() -> f64 { 1.0 }
fn default_delta_max() -> f64 { 0.35 }
```

In `NnJsonFileV2`:

```rust
    #[serde(default = "default_scaled_pi_n")]
    scaled_pi_n: f64,
    #[serde(default = "default_delta_max")]
    delta_max: f64,
```

In `NeuralNetModel` (after `output_param`):

```rust
    /// Half-range multiplier for `ScaledPi`: `bank = scaled_pi_n * π * out[0]`.
    pub scaled_pi_n: f64,
    /// Per-step increment bound for `Delta`: `bank = prev_realized + delta_max * out[0]`.
    pub delta_max: f64,
```

- [ ] **Step 5: Extend the two validators**

`validate_output_size`:

```rust
        let expected = match output_param {
            OutputParam::Atan2Signed => 2,
            OutputParam::AcosTanh | OutputParam::ScaledPi | OutputParam::Delta => 1,
        };
```

`validate_output_activation` — broaden the guard to all three tanh-headed decoders:

```rust
        let needs_tanh = matches!(
            output_param,
            OutputParam::AcosTanh | OutputParam::ScaledPi | OutputParam::Delta
        );
        if needs_tanh && last_activation != Activation::Tanh {
            return Err(DataError(format!(
                "output_param={:?} requires last-layer activation=Tanh, got {:?} in {}. \
                 Without tanh, out[0] is unbounded.",
                output_param, last_activation, path
            )));
        }
        Ok(())
```

- [ ] **Step 6: Thread knobs through constructors + fix all literal construction sites**

In `from_v2_json`, set `scaled_pi_n: file.scaled_pi_n, delta_max: file.delta_max,`. In `from_flat_weights_v2` and any other `NeuralNetModel { .. }` literal, add `scaled_pi_n: default_scaled_pi_n(), delta_max: default_delta_max(),`. Add the two keys to the `NnJsonFileV2` constructed in `save_json`. Let the compiler enumerate every missing site:

Run: `cd src/rust && cargo build -p aerocapture 2>&1 | rg "missing field" | head -40`
Fix each reported site (test fixtures use the defaults).

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd src/rust && cargo test -p aerocapture data::neural 2>&1 | tail -20`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/rust/src/data/neural.rs
git commit -m "feat(nn): add ScaledPi + Delta OutputParam variants with knob fields"
```

---

## Task 2: `GuidanceState::prev_realized_bank_for_nn` telemetry field

**Files:**
- Modify: `src/rust/src/gnc/guidance/dispatch.rs` (`GuidanceState` struct + `new()` ~81)
- Modify: `src/rust/src/simulation/tick.rs` (post-guidance update ~206-213)
- Test: inline `#[cfg(test)]` in `dispatch.rs`

- [ ] **Step 1: Write failing test for field init**

Add to `dispatch.rs` tests:

```rust
#[test]
fn guidance_state_inits_prev_realized_bank() {
    let s = GuidanceState::new(0.5, 0.1, None);
    assert_eq!(s.prev_realized_bank_for_nn, 0.5);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd src/rust && cargo test -p aerocapture guidance_state_inits_prev_realized 2>&1 | tail -15`
Expected: compile error — no field `prev_realized_bank_for_nn`.

- [ ] **Step 3: Add the field + init**

In the `GuidanceState` struct (near `prev_bank_for_nn`):

```rust
    /// Previous-tick pilot-realized bank (rad). Backs the `delta` decoder base
    /// and the prev-realized (sin,cos) NN input. Updated post-guidance in tick.rs.
    pub prev_realized_bank_for_nn: f64,
```

In `GuidanceState::new`, after `prev_bank_for_nn: initial_bank,`:

```rust
            prev_realized_bank_for_nn: initial_bank,
```

- [ ] **Step 4: Update it post-guidance in tick.rs**

In `tick.rs`, in the "Update NN-input telemetry for the NEXT tick" block (right after `state.guidance_state.prev_bank_for_nn = new_bank;`):

```rust
        state.guidance_state.prev_realized_bank_for_nn = guidance_out.bank_angle_realized;
```

- [ ] **Step 5: Run to verify pass + full build**

Run: `cd src/rust && cargo test -p aerocapture guidance_state_inits_prev_realized 2>&1 | tail -15`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/gnc/guidance/dispatch.rs src/rust/src/simulation/tick.rs
git commit -m "feat(nn): track prev_realized_bank_for_nn telemetry (delta base + input)"
```

---

## Task 3: `(sin,cos)` angle inputs + `NN_FULL_INPUT_SIZE` 25→31

**Files:**
- Modify: `src/rust/src/data/neural.rs` (`NN_FULL_INPUT_SIZE` ~1133, `build_nn_input` ~57 signature + body)
- Modify: `src/rust/src/gnc/guidance/dispatch.rs` (internal `build_nn_input` call — via Task 4), `src/rust/src/simulation/tick.rs` (`FULL_MASK` ~167, `build_nn_input` call ~175)
- Test: inline `#[cfg(test)]` in `neural.rs`

- [ ] **Step 1: Write failing tests**

```rust
#[test]
fn full_input_has_31_slots_with_sincos_pairs() {
    let (nav, data, planet) = make_nn_test_fixture(); // existing helper used by other neural tests
    let prev_realized = 0.7_f64;
    let full = build_nn_input(
        &nav, None, None, &data, &planet,
        0.0, 0.0, Some(0.0), 0.3, 0.0, 0.0, prev_realized,
    );
    // None mask => default [..16], length 16 (backward compat)
    assert_eq!(full.len(), 16);
}

#[test]
fn prev_realized_sincos_roundtrips() {
    let (nav, data, planet) = make_nn_test_fixture();
    let prev_realized = 2.5_f64; // > π/2, exercises both sin & cos signs
    let mask: Vec<usize> = (0..31).collect();
    let v = build_nn_input(
        &nav, Some(&mask), None, &data, &planet,
        0.0, 0.0, Some(0.0), 0.3, 0.0, 0.0, prev_realized,
    );
    let recovered = v[29].atan2(v[30]); // sin at 29, cos at 30
    assert_relative_eq!(recovered, prev_realized, epsilon = 1e-12);
}
```

> If `make_nn_test_fixture` does not exist, reuse whatever fixture the existing `neural.rs` tests build (look at the top of the test module — several tests already construct `NavigationOutput` + `SimData` + `PlanetConfig`). Extract a small local helper if needed.

- [ ] **Step 2: Run to verify fail**

Run: `cd src/rust && cargo test -p aerocapture full_input_has_31 prev_realized_sincos 2>&1 | tail -20`
Expected: compile error — `build_nn_input` takes 11 args, not 12.

- [ ] **Step 3: Bump the constant + add the param + slots**

`NN_FULL_INPUT_SIZE`:

```rust
/// 16 baseline + 4 ref-traj + 1 exit-bank teacher + 4 lateral telemetry
/// + 6 (sin,cos) bank-history pairs (exit teacher / prev commanded / prev realized).
pub const NN_FULL_INPUT_SIZE: usize = 31;
```

Add `prev_realized_bank: f64` as the final parameter of `build_nn_input` (after `inclination_error_integral`). After the existing index-24 line and before the ablation block, append:

```rust
    // -- (sin, cos) bank-history pairs (indices 25-30), seam-free cyclic encoding --
    full_input[25] = exit_bank.sin();
    full_input[26] = exit_bank.cos();
    full_input[27] = prev_bank_signed.sin();
    full_input[28] = prev_bank_signed.cos();
    full_input[29] = prev_realized_bank.sin();
    full_input[30] = prev_realized_bank.cos();
```

(`exit_bank` is already in scope from index-20 computation; `prev_bank_signed` is already a param.)

- [ ] **Step 4: Fix the two non-dispatch callers**

`tick.rs` — widen `FULL_MASK` and pass the new arg:

```rust
            const FULL_MASK: [usize; 31] = [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            ];
```

In the `build_nn_input(...)` call in `tick.rs`, add as the last argument:

```rust
                state.guidance_state.prev_realized_bank_for_nn,
```

(The `dispatch.rs` caller is updated in Task 4, where `nn_bank_angle` gains the param.)

- [ ] **Step 5: Run to verify pass**

Run: `cd src/rust && cargo test -p aerocapture full_input_has_31 prev_realized_sincos 2>&1 | tail -20`
Expected: PASS. (Build will still fail until Task 4 fixes `nn_bank_angle`; if so, temporarily run just these two tests after stubbing — but it is cleaner to do Task 4 next and build once.)

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/data/neural.rs src/rust/src/simulation/tick.rs
git commit -m "feat(nn): expand candidate inputs to 31 with (sin,cos) bank-history pairs"
```

---

## Task 4: Decoder arms (`ScaledPi`, `Delta`) + `prev_realized_bank` in `nn_bank_angle`

**Files:**
- Modify: `src/rust/src/gnc/guidance/neural.rs` (`nn_bank_angle` ~187 signature, internal `build_nn_input` call, decode `match` ~215)
- Modify: `src/rust/src/gnc/guidance/dispatch.rs` (call site ~223)
- Test: inline `#[cfg(test)]` in `neural.rs`

- [ ] **Step 1: Write failing decoder tests**

```rust
#[test]
fn scaled_pi_scales_and_wraps() {
    // n=2, tanh head; force out[0] via a bias-only 1-output tanh net.
    let nn = make_single_output_tanh_net(/*bias*/ 0.0, OutputParam::ScaledPi, /*n*/ 2.0, /*dmax*/ 0.0);
    let mut st = NnState::for_model(&nn);
    let (nav, data, planet) = make_nn_test_fixture();
    // tanh(0) = 0 => bank = 2*π*0 = 0
    let b = nn_bank_angle(&nav, &nn, &mut st, &data, &planet, 0.0, 0.0, Some(0.0), 0.0, 0.0, 0.0, 0.0);
    assert_relative_eq!(b, 0.0, epsilon = 1e-12);
}

#[test]
fn delta_integrates_on_prev_realized_bounded_and_wrapped() {
    // delta_max = 0.2, tanh head, bias chosen so tanh(bias) ~ 1 (saturates) => +0.2 step.
    let nn = make_single_output_tanh_net(/*bias*/ 5.0, OutputParam::Delta, /*n*/ 1.0, /*dmax*/ 0.2);
    let mut st = NnState::for_model(&nn);
    let (nav, data, planet) = make_nn_test_fixture();
    let prev_realized = 1.0_f64;
    let b = nn_bank_angle(&nav, &nn, &mut st, &data, &planet, 0.0, 0.0, Some(0.0), 0.0, 0.0, 0.0, prev_realized);
    // bank ≈ prev_realized + 0.2 * tanh(5) ≈ 1.0 + ~0.2, well within (-π, π]
    assert!((b - (prev_realized + 0.2 * 5.0_f64.tanh())).abs() < 1e-9);
    assert!(b > -std::f64::consts::PI && b <= std::f64::consts::PI);
}
```

> `make_single_output_tanh_net(bias, param, n, dmax)` is a small local test helper: build a `NeuralNetModel` whose only layer is `Dense` with `w = [[0.0; in]]`, `b = [bias]`, `activation = Tanh`, `output_param = param`, `scaled_pi_n = n`, `delta_max = dmax`, `input_mask = Some((0..31).collect())`. Model the in-size on the mask length (31). Mirror the existing `acos_tanh_parameterization_emits_acos_of_output` test's construction.

- [ ] **Step 2: Run to verify fail**

Run: `cd src/rust && cargo test -p aerocapture scaled_pi_scales delta_integrates 2>&1 | tail -20`
Expected: compile error — `nn_bank_angle` arity / missing match arms.

- [ ] **Step 3: Add the param + decoder arms**

Add `use crate::gnc::control::angle_utils::wrap_to_pi;` to `neural.rs` imports. Add `prev_realized_bank: f64` as the final parameter of `nn_bank_angle`, and pass it as the final arg of the internal `build_nn_input(...)` call. Replace the decode `match`:

```rust
    use crate::data::neural::OutputParam;
    use std::f64::consts::PI;
    let output = nn.forward(nn_state, &masked);
    match nn.output_param {
        OutputParam::Atan2Signed => output[0].atan2(output[1]),
        OutputParam::AcosTanh => output[0].acos(),
        OutputParam::ScaledPi => wrap_to_pi(nn.scaled_pi_n * PI * output[0]),
        OutputParam::Delta => wrap_to_pi(prev_realized_bank + nn.delta_max * output[0]),
    }
```

- [ ] **Step 4: Update the dispatch.rs call site**

In `dispatch.rs`, snapshot the field before the `nn_state` mut borrow (next to `let integral = ...;`):

```rust
                let prev_realized = state.prev_realized_bank_for_nn;
```

and pass it as the final arg of `neural::nn_bank_angle(...)`:

```rust
                    integral,
                    prev_realized,
                );
```

- [ ] **Step 5: Run to verify pass + whole-crate build**

Run: `cd src/rust && cargo test -p aerocapture scaled_pi_scales delta_integrates 2>&1 | tail -20 && cargo build -p aerocapture 2>&1 | tail -5`
Expected: tests PASS, build clean.

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/gnc/guidance/neural.rs src/rust/src/gnc/guidance/dispatch.rs
git commit -m "feat(nn): decode ScaledPi (n·π·tanh) + Delta (bounded increment on realized), wrapped"
```

---

## Task 5: Config knobs + load-time validation (mode-compat hard errors)

**Files:**
- Modify: `src/rust/src/config.rs` (`TomlNeuralNetworkParams` ~1047)
- Modify: `src/rust/src/data/mod.rs` (`validate_output_parameterization` ~1010, cross-check ~686, full_neural guard ~715, knob override after model load ~671)
- Test: inline `#[cfg(test)]` in `data/mod.rs`

- [ ] **Step 1: Write failing validation tests**

Add to the `output_parameterization validation tests` module in `data/mod.rs`:

```rust
#[test]
fn scaled_pi_with_magnitude_only_rejects() {
    let err = validate_output_parameterization(
        Some("scaled_pi"),
        guidance_params::NeuralNetMode::MagnitudeOnly,
        Some(&one_output_tanh_arch()), // helper used by acos_tanh tests
    ).unwrap_err();
    assert!(err.0.contains("scaled_pi") && err.0.contains("full_neural"));
}

#[test]
fn delta_with_full_neural_and_tanh_accepts() {
    assert!(validate_output_parameterization(
        Some("delta"),
        guidance_params::NeuralNetMode::FullNeural,
        Some(&one_output_tanh_arch()),
    ).is_ok());
}

#[test]
fn scaled_pi_with_linear_last_activation_rejects() {
    let err = validate_output_parameterization(
        Some("scaled_pi"),
        guidance_params::NeuralNetMode::FullNeural,
        Some(&one_output_linear_arch()), // helper: last dense activation=linear
    ).unwrap_err();
    assert!(err.0.contains("tanh") || err.0.contains("activation"));
}
```

> Reuse the architecture-building helpers the existing `acos_tanh_*` tests use (`acos_tanh_with_valid_config_accepts` builds a 1-output tanh arch). If they are inline, factor `one_output_tanh_arch()` / `one_output_linear_arch()` from them.

- [ ] **Step 2: Run to verify fail**

Run: `cd src/rust && cargo test -p aerocapture output_parameterization 2>&1 | tail -25`
Expected: FAIL — `scaled_pi`/`delta` currently fall through the early `return Ok(())`.

- [ ] **Step 3: Extend `validate_output_parameterization`**

Current body early-returns `Ok` for anything that is not `Some("acos_tanh")`. Restructure so signed tanh decoders are also checked. Replace the top guard:

```rust
fn validate_output_parameterization(
    output_param: Option<&str>,
    neural_mode: guidance_params::NeuralNetMode,
    architecture: Option<&[neural::LayerSpec]>,
) -> Result<(), DataError> {
    use guidance_params::NeuralNetMode::*;
    let param = match output_param {
        Some(p) => p,
        None => return Ok(()),
    };

    // Required mode per decoder.
    let required_full_neural = matches!(param, "atan2_signed" | "scaled_pi" | "delta");
    let required_magnitude_only = matches!(param, "acos_tanh");

    if required_magnitude_only && neural_mode != MagnitudeOnly {
        return Err(DataError(format!(
            "output_parameterization='{}' is only legal with mode='magnitude_only' \
             (it cannot emit signed bank)",
            param
        )));
    }
    if required_full_neural && param != "atan2_signed" && neural_mode != FullNeural {
        return Err(DataError(format!(
            "output_parameterization='{}' is only legal with mode='full_neural' \
             (it emits a signed bank; magnitude_only expects an unsigned magnitude)",
            param
        )));
    }

    // Single-output tanh-head decoders share the acos_tanh architecture constraints.
    let needs_single_tanh = matches!(param, "acos_tanh" | "scaled_pi" | "delta");
    if !needs_single_tanh {
        return Ok(());
    }
    // ... keep the existing acos_tanh architecture checks below, but make their
    // error strings reference `param` instead of the hardcoded "acos_tanh", and
    // require last layer dense, output_size == 1, activation == tanh.
```

Keep the existing arch-walking block (v2 required, last-layer dense, `output_size == 1`, `activation == tanh`), swapping hardcoded `'acos_tanh'` in the messages for `{param}`.

- [ ] **Step 4: Extend the cross-check + full_neural guard in `from_toml`**

In the cross-check match (`data/mod.rs` ~686):

```rust
            let toml_enum = match toml_param.as_str() {
                "atan2_signed" => neural::OutputParam::Atan2Signed,
                "acos_tanh" => neural::OutputParam::AcosTanh,
                "scaled_pi" => neural::OutputParam::ScaledPi,
                "delta" => neural::OutputParam::Delta,
                other => { /* unchanged error, add the two names to the expected list */ }
            };
```

Generalize the defense-in-depth guard (~715) so a signed-only model under `magnitude_only` is also rejected (currently only guards `acos_tanh` under non-`magnitude_only`):

```rust
        if let Some(nn) = &neural_net {
            match nn.output_param {
                neural::OutputParam::AcosTanh if neural_mode != guidance_params::NeuralNetMode::MagnitudeOnly => {
                    return Err(DataError("loaded model output_param='acos_tanh' requires mode='magnitude_only'".into()));
                }
                neural::OutputParam::ScaledPi | neural::OutputParam::Delta
                    if neural_mode != guidance_params::NeuralNetMode::FullNeural =>
                {
                    return Err(DataError(format!(
                        "loaded model output_param={:?} emits a signed bank and requires mode='full_neural'",
                        nn.output_param
                    )));
                }
                _ => {}
            }
        }
```

- [ ] **Step 5: Add TOML knob fields + override onto the loaded model**

In `config.rs` `TomlNeuralNetworkParams`:

```rust
    /// Half-range multiplier for output_parameterization="scaled_pi".
    #[serde(default)]
    pub scaled_pi_n: Option<f64>,
    /// Per-step increment bound (rad) for output_parameterization="delta".
    #[serde(default)]
    pub delta_max: Option<f64>,
```

In `data/mod.rs`, after the model is loaded and validated (after the mask/ablation validation ~671, before or after the cross-check), apply TOML overrides so the runtime honors the training-time knob even if the JSON predates the field:

```rust
        if let Some(nn) = neural_net.as_mut()
            && let Some(tnn) = &toml.guidance.neural_network
        {
            if let Some(n) = tnn.scaled_pi_n { nn.scaled_pi_n = n; }
            if let Some(d) = tnn.delta_max { nn.delta_max = d; }
        }
```

(Ensure `neural_net` is a `mut` binding; adjust the `let` if needed.)

- [ ] **Step 6: Run to verify pass + whole-crate build + clippy**

Run: `cd src/rust && cargo test -p aerocapture output_parameterization 2>&1 | tail -25`
Expected: PASS.
Run: `cd src/rust && cargo build 2>&1 | tail -5 && cargo clippy -p aerocapture 2>&1 | tail -10`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/rust/src/config.rs src/rust/src/data/mod.rs
git commit -m "feat(nn): config knobs + hard mode-compat validation for scaled_pi/delta"
```

---

## Task 6: `supervised_trace` carries realized bank + `collect_supervised` emits it

**Files:**
- Modify: `src/rust/src/simulation/runner.rs` (`supervised_trace` type, 2 sites: ~155, ~448)
- Modify: `src/rust/src/simulation/tick.rs` (push tuple ~196)
- Modify: `src/rust/aerocapture-py/src/lib.rs` (`collect_supervised` emit ~506-529)
- Test: `tests/test_pyo3.py` (Python) — new assertion on the dict key

- [ ] **Step 1: Write failing Python test**

Add to `tests/test_pyo3.py` (or the nearest collect_supervised test; if none, add a small one):

```python
def test_collect_supervised_emits_prev_realized(ftc_train_toml):
    import aerocapture_rs
    out = aerocapture_rs.collect_supervised(str(ftc_train_toml), [12345])
    assert len(out) >= 1
    rec = out[0]
    assert "prev_realized" in rec
    assert rec["prev_realized"].shape == rec["y_signed"].shape  # (T,)
    assert rec["X"].shape[1] == 31  # full candidate vector
```

> Use whatever fixture points at an FTC training TOML; mirror the existing `collect_supervised` test setup if present.

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_pyo3.py -k collect_supervised_emits_prev_realized -x 2>&1 | tail -20`
Expected: FAIL — KeyError `prev_realized` (and `X` width 25 ≠ 31 until bindings rebuilt).

- [ ] **Step 3: Widen the trace tuple**

In `runner.rs`, change both declarations:

```rust
    pub(crate) supervised_trace: Vec<(Vec<f64>, f64, f64)>,
```
```rust
    supervised_trace: Vec<(Vec<f64>, f64, f64)>,
```

(`Vec::new()` inits and the `std::mem::take` move sites need no change.)

- [ ] **Step 4: Push realized into the trace**

In `tick.rs`, change the push to include the prev-realized base used to build this tick's input:

```rust
            state
                .supervised_trace
                .push((nn_input, guidance_out.pre_shaper_signed, state.guidance_state.prev_realized_bank_for_nn));
```

> Note: `prev_realized_bank_for_nn` here still holds the PREVIOUS tick's realized bank (it is updated lower in the block, after this push), matching the value `build_nn_input` consumed. Verify the push precedes the telemetry-update lines.

- [ ] **Step 5: Emit `prev_realized` from `collect_supervised`**

In `lib.rs`, where `y_signed` is accumulated and the dict is built, add a parallel `prev_realized` vec:

```rust
        let mut y_signed: Vec<f64> = Vec::with_capacity(n_steps);
        let mut prev_realized: Vec<f64> = Vec::with_capacity(n_steps);
        for (_x, bank, realized) in &supervised_trace {
            y_signed.push(*bank);
            prev_realized.push(*realized);
        }
        // ... existing X build (now 31-wide) ...
        let pr_array = numpy::PyArray1::from_vec(py, prev_realized);
        dict.set_item("prev_realized", pr_array)?;
```

(Adjust the existing destructuring loop that builds `y_signed`/`X` to the 3-tuple.)

- [ ] **Step 6: Rebuild bindings + run to verify pass**

Run (from repo root, per PyO3 rebuild rule):
```bash
uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml 2>&1 | tail -5
uv run pytest tests/test_pyo3.py -k collect_supervised_emits_prev_realized -x 2>&1 | tail -20
```
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/rust/src/simulation/runner.rs src/rust/src/simulation/tick.rs src/rust/aerocapture-py/src/lib.rs tests/test_pyo3.py
git commit -m "feat(nn): collect_supervised emits per-step prev_realized for delta warm-start"
```

---

## Task 7: Python `NetworkConfig` knobs + warm-start target branches

**Files:**
- Modify: `src/python/aerocapture/training/config.py` (`NetworkConfig` ~19-35)
- Modify: `src/python/aerocapture/training/warm_start.py` (target dispatch ~479-493; thread `prev_realized` + knobs into `_chunked_bptt_train`)
- Test: `tests/` — a focused warm-start target unit test

- [ ] **Step 1: Write failing target-encoding test**

Add `tests/test_warm_start_targets.py`:

```python
import math
import numpy as np
import torch
from aerocapture.training.warm_start import encode_supervised_target  # new pure helper

def test_scaled_pi_target_is_y_over_n_pi_clamped():
    y = torch.tensor([0.0, math.pi / 2, math.pi])
    out = encode_supervised_target("scaled_pi", y, prev_realized=None, scaled_pi_n=2.0, delta_max=0.0)
    expected = torch.clamp(y / (2.0 * math.pi), -1.0, 1.0)
    assert torch.allclose(out, expected, atol=1e-12)

def test_delta_target_is_wrapped_diff_over_max_clamped():
    y = torch.tensor([1.0, 1.0])
    prev = torch.tensor([0.9, 1.5])
    out = encode_supervised_target("delta", y, prev_realized=prev, scaled_pi_n=0.0, delta_max=0.2)
    # shortest_angle_diff(prev, y) / 0.2, clamped to [-1, 1]
    diff = torch.tensor([0.1, -0.5])
    expected = torch.clamp(diff / 0.2, -1.0, 1.0)
    assert torch.allclose(out, expected, atol=1e-9)
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_warm_start_targets.py -x 2>&1 | tail -20`
Expected: FAIL — `encode_supervised_target` does not exist.

- [ ] **Step 3: Add the knob fields to `NetworkConfig`**

In `config.py`:

```python
    scaled_pi_n: float = 1.0
    delta_max: float = 0.35
```

Ensure the TOML `[network]`/`[guidance.neural_network]` reader that populates `NetworkConfig` copies these two keys (mirror how `output_parameterization` is read).

- [ ] **Step 4: Implement `encode_supervised_target` + wire into `_chunked_bptt_train`**

In `warm_start.py`, add a pure helper (with a torch-side shortest-angle-diff):

```python
def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + math.pi, 2 * math.pi) - math.pi

def encode_supervised_target(output_param, y, prev_realized, scaled_pi_n, delta_max):
    """Per-decoder supervised target read directly from the tanh head (means[...,0])."""
    if output_param == "scaled_pi":
        return torch.clamp(y / (scaled_pi_n * math.pi), -1.0, 1.0)
    if output_param == "delta":
        diff = _wrap_to_pi(y - prev_realized)
        return torch.clamp(diff / delta_max, -1.0, 1.0)
    raise ValueError(f"encode_supervised_target: {output_param!r} is not delta/scaled_pi")
```

Extend the dispatch in `_chunked_bptt_train` (the `if output_param == "acos_tanh": ... elif "atan2_signed": ... else: raise`):

```python
            elif output_param in ("scaled_pi", "delta"):
                pred = means[..., 0]  # tanh-activated head, already in [-1, 1]
                target = encode_supervised_target(
                    output_param, y_t,
                    prev_realized=pr_t if output_param == "delta" else None,
                    scaled_pi_n=network.scaled_pi_n, delta_max=network.delta_max,
                )
                loss = nn.functional.mse_loss(pred, target)
```

Thread `prev_realized` from the collected dict through the chunking the same way `y_signed` is threaded: where the corpus is assembled from `collect_supervised` records, carry `rec["prev_realized"]` alongside `rec["y_signed"]`, slice it into the same BPTT chunks (`pr_t`), and pass it to the loss branch. (For `atan2_signed`/`acos_tanh` it is unused.)

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_warm_start_targets.py -x 2>&1 | tail -20`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/config.py src/python/aerocapture/training/warm_start.py tests/test_warm_start_targets.py
git commit -m "feat(nn): warm-start target encoding for scaled_pi + delta decoders"
```

---

## Task 8: PSO training configs + scheme registration

**Files:**
- Create: `configs/training/msr_aller_nn_scaledpi_train.toml`
- Create: `configs/training/msr_aller_nn_delta_train.toml`
- Modify: `src/python/aerocapture/training/compare_guidance.py` (`SCHEMES`, `_NN_DEPLOY_SCHEMES`)
- Modify: `train_all.sh` (aliases)

- [ ] **Step 1: Author the `scaled_pi` config**

Inherit `nn_common.toml`, `full_neural`, 1-output tanh head, `input_mask` including the new pairs. `configs/training/msr_aller_nn_scaledpi_train.toml`:

```toml
base = ["nn_common.toml"]
results_suffix = "scaledpi"

[guidance.neural_network]
mode = "full_neural"
output_parameterization = "scaled_pi"
scaled_pi_n = 2.0

[[network.architecture]]
type = "dense"
input_size = 22
output_size = 32
activation = "tanh"

[[network.architecture]]
type = "dense"
input_size = 32
output_size = 1
activation = "tanh"

# 16 baseline + ref-traj 16-19 + prev-realized (sin,cos) 29,30
input_mask = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,29,30]
```

> `input_size` of layer 0 MUST equal `input_mask` length (22 here). Pick the mask to taste; the example includes the prev-realized pair so the seam-free history reaches the net. Confirm `nn_common.toml` does not already pin a conflicting `[[network.architecture]]` (arrays REPLACE under deep-merge, so the leaf fully specifies it).

- [ ] **Step 2: Author the `delta` config**

`configs/training/msr_aller_nn_delta_train.toml`: identical shape, swap the decoder block:

```toml
base = ["nn_common.toml"]
results_suffix = "delta"

[guidance.neural_network]
mode = "full_neural"
output_parameterization = "delta"
delta_max = 0.35

[[network.architecture]]
type = "dense"
input_size = 22
output_size = 32
activation = "tanh"

[[network.architecture]]
type = "dense"
input_size = 32
output_size = 1
activation = "tanh"

input_mask = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,29,30]
```

- [ ] **Step 3: Smoke-load both configs via Rust**

Run:
```bash
cargo build --release --manifest-path src/rust/Cargo.toml 2>&1 | tail -3
./src/rust/target/release/aerocapture configs/training/msr_aller_nn_scaledpi_train.toml 2>&1 | tail -5 || true
```
Expected: no config-load `DataError` (a run may need `best_model.json`; the goal here is that parsing + validation pass. If it errors only on a missing model file, that is fine — validation of the TOML block has passed).

- [ ] **Step 4: Register the schemes**

In `compare_guidance.py`, add `neural_network_scaledpi_pso` and `neural_network_delta_pso` to `SCHEMES` and `_NN_DEPLOY_SCHEMES`, each pointing at its training TOML (mirror the existing `neural_network_gru_pso` entry shape).

In `train_all.sh`, add aliases (mirror existing): `scaledpi` / `nn_scaledpi` → scaled_pi config; `delta` / `nn_delta` → delta config.

- [ ] **Step 5: Quick PSO smoke (1 gen, tiny pop) on the delta config**

Run:
```bash
uv run python -m aerocapture.training.train configs/training/msr_aller_nn_delta_train.toml \
    --n-gen 1 --n-pop 4 --no-tui --skip-report 2>&1 | tail -15
```
Expected: completes a generation, writes `training_output/.../best_model.json` (v2, `output_param="delta"`). If warm-start is configured in `nn_common.toml`, it should collect + pretrain without error.

- [ ] **Step 6: Commit**

```bash
git add configs/training/msr_aller_nn_scaledpi_train.toml configs/training/msr_aller_nn_delta_train.toml src/python/aerocapture/training/compare_guidance.py train_all.sh
git commit -m "feat(nn): PSO training configs + scheme registration for scaled_pi/delta"
```

---

## Task 9: Golden regression + full verification + docs sync

**Files:**
- Verify only: `tests/reference_data/rust_golden/*`
- Modify (docs): handled by `smart-commit`

- [ ] **Step 1: Add a default-mask invariance test (golden safety net)**

In `neural.rs` tests, assert the `None`-mask path is unaffected by the 31-wide vector:

```rust
#[test]
fn default_mask_path_unchanged_by_new_inputs() {
    let (nav, data, planet) = make_nn_test_fixture();
    let v = build_nn_input(&nav, None, None, &data, &planet,
        0.0, 0.0, Some(0.0), 0.3, 0.0, 0.0, /*prev_realized*/ 9.9);
    assert_eq!(v.len(), 16);
    // prev_realized (a wild value) must NOT leak into the default-mask vector.
    assert!(v.iter().all(|x| x.is_finite()));
}
```

- [ ] **Step 2: Run the full Rust suite + golden regressions**

Run: `cd src/rust && cargo test 2>&1 | tail -25`
Expected: all pass; the 6 guidance golden regressions remain bit-identical (new decoders are opt-in; `NN_FULL_INPUT_SIZE` growth does not touch default-mask models).

- [ ] **Step 3: Run `check_all.sh` + Python lint/tests**

Run:
```bash
./check_all.sh 2>&1 | tail -20
./lint_code.sh 2>&1 | tail -20
uv run pytest tests -q 2>&1 | tail -25
```
Expected: Rust fmt/clippy/test clean; ruff/mypy clean; pytest green.

- [ ] **Step 4: Commit any test additions**

```bash
git add src/rust/src/data/neural.rs
git commit -m "test(nn): default-mask invariance guard for 31-wide candidate vector"
```

- [ ] **Step 5: Sync docs + final branch commit (smart-commit)**

Invoke the `smart-commit` skill, instructing it to take the **whole `feature/nn-bank-decoders` branch** into account: update `CLAUDE.md` (NN decoder list: add `scaled_pi`/`delta`, the `(sin,cos)` inputs, `NN_FULL_INPUT_SIZE` 31, the new training configs + `train_all.sh` aliases, the extended validation) and `README.md` as needed, then commit.

---

## Self-review notes

- **Spec coverage:** config surface (T1,T5), input vector 25→31 (T3), decoders + boundary wrap (T1,T4), state plumbing (T2), warm-start encoding (T6,T7), PSO configs (T8), testing + golden bit-identity (T1-T9), out-of-scope PPO untouched. All spec sections map to a task.
- **Layout decision resolved:** append-only, `NN_FULL_INPUT_SIZE = 31`, indices 25-30 are the three `(sin,cos)` pairs; existing 0-24 meanings preserved; default mask `[0..16]` untouched (golden bit-identity). Indices 20/22 keep their seamed singles in the full array but new configs select the `(sin,cos)` pairs instead.
- **`delta` warm-start source:** explicit `prev_realized` field from `collect_supervised` (T6), robust to the model's `input_mask` (does not decode from input columns).
- **Type consistency:** `prev_realized_bank` is the final param of both `build_nn_input` and `nn_bank_angle`; `supervised_trace` is `Vec<(Vec<f64>, f64, f64)>` everywhere; `OutputParam` stays `Copy + Eq` (knobs on the model); `scaled_pi_n`/`delta_max` named identically across Rust model, TOML, and Python `NetworkConfig`.
