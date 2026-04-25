# Phase 4a Mamba SSM MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land a 1-layer selective SSM (Mamba S6 core) as the sixth stateful layer type on the Phase 0/1/1.5/2a/2b/3a stack. Trained on PSO only; PPO paths fail loudly at `build_layer` / `load_policy_from_json` with a clear pointer to the spec.

**Architecture:** Rust `Layer` gains a `Mamba(Box<MambaLayer>)` variant with 5 parameter fields (`x_proj_w`, `dt_proj_w`, `dt_proj_b`, `a_log`, `d_skip`). Per-tick forward: fused `x_proj` emits `[Δ_pre | B | C]`, `dt_proj + softplus` lifts Δ to per-channel positive step size, ZOH discretizes `A = -exp(A_log)` per channel-state pair, recurrence `h_{new} = Ā*h + B̄*x` updates the 2D state `(input_size, d_state)`, output `y = h_{new} @ C + D*x`. Python mirror lives in `rl/layers/mamba.py` with manual softplus / expm1_over_x for bit-equivalence; `build_layer(MambaSpec)` and `load_policy_from_json` raise `NotImplementedError` to gate the PPO path.

**Tech Stack:** Rust 2024 edition (nalgebra for linear algebra, `f64::exp_m1` / `f64::ln_1p` for numerical stability), PyO3 for Python bindings, Python 3.14, PyTorch (`torch.expm1`, `torch.where` for the Taylor crossover), Pydantic v2 discriminated unions, pymoo PSO, pytest.

**Spec:** `docs/superpowers/specs/2026-04-24-phase-4a-mamba-ssm-mvp-design.md`

**Branch:** `feature/mamba-ssm-mvp` (spec already committed as `7d74ea7`)

---

## Task 0: TODO.md marker

**Files:**
- Modify: `TODO.md`

- [ ] **Step 1: Mark Phase 4 in progress with 4a split**

Open `TODO.md`, find the Phase 4 block, and replace it with:

```markdown
### Phase 4a -- Mamba Selective SSM MVP (PSO only) [DOING 2026-04-24 on feature/mamba-ssm-mvp]
- [ ] Rust `MambaLayer` + `Layer::Mamba(Box<MambaLayer>)` + `LayerSpec::Mamba { input_size, d_state, dt_rank }` + `LayerState::Mamba { h }` + `TomlLayerSpec::Mamba`
- [ ] `LayerWeights for MambaLayer` canonical flat order: x_proj_w, dt_proj_w, dt_proj_b, a_log, d_skip
- [ ] `softplus` + `expm1_over_x` pub(crate) helpers with Taylor crossover at |z| < 1e-8
- [ ] Python `MambaLayer` torch module (manual softplus / expm1_over_x / ZOH) + `MambaSpec` pydantic + `build_layer` PPO-rejection guard
- [ ] `_mamba_specs` (Xavier on x_proj + dt_proj with dt_rank^{-0.5} scaling, HiPPO log(n+1) on a_log, inv_softplus(U(1e-3, 1e-1)) on dt_proj_b, 1.0 on d_skip) + per-individual jitter in `_init_mamba_layer`
- [ ] Training config `msr_aller_mamba_pso_train.toml` (Dense(23 -> 32, swish) -> Mamba(32, 16) x2 -> Dense(32 -> 2, asinh), 4290 params) + `compare_guidance` + `train_all.sh` registration
- [ ] Cross-language equivalence + warm-up + PSO smoke + PPO-rejection tests (CI wiring)

Spec: `docs/superpowers/specs/2026-04-24-phase-4a-mamba-ssm-mvp-design.md`.
Plan: `docs/superpowers/plans/2026-04-24-phase-4a-mamba-ssm-mvp-plan.md`.

### Phase 4b -- Mamba PPO-BPTT (follow-up)
- [ ] Deferred from 4a. Requires `_zero_state_where_done` 2D-tensor branch, `hidden_shapes` arm returning `(d_inner, d_state)`, ndim==4 rollout-buffer dispatch in `ppo_update_bptt`, obs-norm bake-in for Mamba-as-layer-0, PPO smoke + BPTT chunk-invariant tests, training TOML `msr_aller_mamba_ppo_train.toml`.

### Phase 4c -- Full Mamba block (deferred, not guaranteed)
- [ ] Conv1d pre-filter + SiLU gating + in/out expansion linears + block residual. Would ship as `LayerSpec::MambaBlock` distinct from `LayerSpec::Mamba`.
```

- [ ] **Step 2: Commit**

```bash
git add TODO.md
git commit -m "docs(todo): mark Phase 4a Mamba SSM MVP in progress on feature/mamba-ssm-mvp"
```

---

## Task 1: Rust `softplus` and `expm1_over_x` free helpers

**Files:**
- Modify: `src/rust/src/data/neural.rs` (add two `pub(crate)` free functions near the top, after imports, before the `Activation` enum at line 64)
- Test: `src/rust/src/data/neural.rs` (`#[cfg(test)] mod tests` block)

**Why these live as free functions:** `softplus` and `expm1_over_x` are numerical primitives used by `MambaLayer::forward`. Making them `pub(crate)` free functions (not methods) keeps them unit-testable and reusable by any future selective-SSM variant.

- [ ] **Step 1: Write failing tests for both helpers**

Append to the existing `#[cfg(test)] mod tests` block in `src/rust/src/data/neural.rs`:

```rust
#[test]
fn softplus_matches_stable_form_small_x() {
    // softplus(0) = log(2) ≈ 0.6931471805599453
    assert!((softplus(0.0) - std::f64::consts::LN_2).abs() < 1e-15);
    // softplus(1) = log(1 + e) ≈ 1.3132616875182228
    assert!((softplus(1.0) - 1.3132616875182228).abs() < 1e-14);
    // softplus(-1) = log(1 + 1/e) ≈ 0.3132616875182228
    assert!((softplus(-1.0) - 0.3132616875182228).abs() < 1e-14);
}

#[test]
fn softplus_no_overflow_at_large_magnitude() {
    // For x = 100, softplus(x) must stay finite and ≈ x (not Inf from naive exp).
    let y = softplus(100.0);
    assert!(y.is_finite());
    assert!((y - 100.0).abs() < 1e-10);
    // For x = -100, softplus(x) ≈ exp(-100) ≈ 3.72e-44, still finite.
    let y_neg = softplus(-100.0);
    assert!(y_neg.is_finite());
    assert!(y_neg > 0.0);
    assert!(y_neg < 1e-40);
}

#[test]
fn expm1_over_x_matches_exact_for_moderate_z() {
    // For |z| >= 1e-8, use expm1(z) / z directly.
    for &z in &[0.5, -0.5, 1.0, -1.0, 5.0, -5.0, 0.01, -0.01] {
        let expected = z.exp_m1() / z;
        let got = expm1_over_x(z);
        assert!((got - expected).abs() < 1e-15, "z={z}: got {got}, expected {expected}");
    }
}

#[test]
fn expm1_over_x_taylor_branch_at_tiny_z() {
    // Taylor: 1 + z/2 + z^2/6 (error ~ z^3/24)
    // At z = 1e-10, Taylor and exact should agree to machine epsilon.
    let z = 1e-10;
    let taylor = 1.0 + z * 0.5 + z * z / 6.0;
    let got = expm1_over_x(z);
    assert!((got - taylor).abs() < 1e-16, "z=1e-10: got {got}, taylor {taylor}");
    // At z = 0, result should be 1.0 (the limit).
    assert_eq!(expm1_over_x(0.0), 1.0);
}

#[test]
fn expm1_over_x_crossover_is_smooth() {
    // Adjacent values across the crossover should not jump.
    let z1 = 0.99e-8;
    let z2 = 1.01e-8;
    let y1 = expm1_over_x(z1);
    let y2 = expm1_over_x(z2);
    assert!((y1 - y2).abs() < 1e-14, "crossover jump: y1={y1}, y2={y2}");
}
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cargo test --manifest-path src/rust/Cargo.toml softplus_matches
cargo test --manifest-path src/rust/Cargo.toml expm1_over_x
```

Expected: FAIL with `cannot find function softplus in this scope` / `cannot find function expm1_over_x in this scope`.

- [ ] **Step 3: Implement both helpers**

Add near the top of `src/rust/src/data/neural.rs`, after the existing `use` statements and before the `Activation` enum (around line 62):

```rust
/// Numerically stable softplus: `log(1 + exp(x))`.
///
/// Uses `max(x, 0) + log1p(exp(-|x|))` to avoid overflow for large positive x
/// and underflow for large negative x. The Python mirror in `rl/layers/mamba.py`
/// uses the identical manual form (NOT `torch.nn.functional.softplus`, which has a
/// `threshold=20` linear-branch fallback we do not want for bit-equivalence).
pub(crate) fn softplus(x: f64) -> f64 {
    let a = x.abs();
    x.max(0.0) + (-a).exp().ln_1p()
}

/// Stable `(exp(z) - 1) / z` with Taylor fallback for |z| < 1e-8.
///
/// For |z| < 1e-8 the exact form suffers from catastrophic cancellation and we
/// use `1 + z/2 + z^2/6` (Taylor expansion, error ~ z^3/24 which is machine
/// epsilon at |z| < 1e-5). The Python mirror uses `torch.where` to switch
/// between the same two branches.
pub(crate) fn expm1_over_x(z: f64) -> f64 {
    if z.abs() < 1e-8 {
        1.0 + z * 0.5 + z * z / 6.0
    } else {
        z.exp_m1() / z
    }
}
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cargo test --manifest-path src/rust/Cargo.toml softplus_
cargo test --manifest-path src/rust/Cargo.toml expm1_over_x
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Confirm no regressions in existing tests**

```bash
cargo test --manifest-path src/rust/Cargo.toml --lib data::neural
```

Expected: all existing `data::neural` tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/data/neural.rs
git commit -m "feat(nn): add softplus and expm1_over_x free helpers for Mamba"
```

---

## Task 2: Rust `MambaLayer` struct + `LayerSpec::Mamba` + `Layer::Mamba` variant

**Files:**
- Modify: `src/rust/src/data/neural.rs` (add `MambaLayer` struct around line 460, before `pub enum Layer`; extend `LayerSpec` enum; extend `Layer` enum)

- [ ] **Step 1: Add the `MambaLayer` struct definition**

Insert after the `TransformerLayer` struct (after line ~490, find by searching for `pub struct TransformerLayer`), before `pub enum Layer`:

```rust
/// Selective SSM core (Mamba S6) -- Phase 4a PSO-only MVP.
///
/// Per-tick forward computes input-dependent Δ, B, C from x via a fused `x_proj`
/// linear projection, discretizes A via ZOH (`A = -exp(a_log)`, diagonal),
/// updates per-channel state `h: (input_size, d_state)`, and emits
/// `y = h @ C + D * x` (skip residual per channel).
///
/// No conv1d, no SiLU gating -- those are the full Mamba block, deferred to
/// Phase 4c. No in/out expansion linears -- user stacks Dense before/after.
#[derive(Debug, Clone)]
pub struct MambaLayer {
    /// d_inner in the paper. Layer fan-in = fan-out = input_size.
    pub input_size: usize,
    /// N in the paper. SSM state dim per channel.
    pub d_state: usize,
    /// Bottleneck rank for the Δ projection (paper default: max(1, input_size / 16)).
    pub dt_rank: usize,

    /// Fused (Δ_pre, B, C) projection. Shape: (dt_rank + 2*d_state, input_size).
    pub x_proj_w: nalgebra::DMatrix<f64>,
    /// Δ lift projection. Shape: (input_size, dt_rank).
    pub dt_proj_w: nalgebra::DMatrix<f64>,
    /// Δ bias (critical init: inv_softplus(uniform(dt_min, dt_max)) per channel).
    pub dt_proj_b: nalgebra::DVector<f64>,
    /// HiPPO log-space reparameterization of A. Physical A = -exp(a_log).
    /// Shape: (input_size, d_state). Strictly negative A ensures stable contraction.
    pub a_log: nalgebra::DMatrix<f64>,
    /// Per-channel skip-residual scalar. Paper default init: 1.0.
    pub d_skip: nalgebra::DVector<f64>,
}
```

- [ ] **Step 2: Extend the `LayerSpec` enum**

Find `pub enum LayerSpec` (around line 866) and add a `Mamba` variant after `Transformer`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LayerSpec {
    Dense {
        input_size: usize,
        output_size: usize,
        activation: Activation,
    },
    Gru {
        input_size: usize,
        hidden_size: usize,
    },
    Lstm {
        input_size: usize,
        hidden_size: usize,
    },
    Window {
        input_size: usize,
        n_steps: usize,
    },
    Transformer {
        d_model: usize,
        n_heads: usize,
        d_ffn: usize,
        n_seq: usize,
    },
    Mamba {
        input_size: usize,
        d_state: usize,
        dt_rank: usize,
    },
}
```

- [ ] **Step 3: Extend the `Layer` enum + `Layer::input_size`**

Find `pub enum Layer` (around line 497) and add the `Mamba` variant + dispatch arm:

```rust
#[derive(Debug, Clone)]
pub enum Layer {
    Dense(DenseLayer),
    Gru(GruLayer),
    Lstm(LstmLayer),
    Window(WindowLayer),
    Transformer(Box<TransformerLayer>),
    // Boxed: MambaLayer with typical dims (input_size=32, d_state=16, dt_rank=2)
    // carries ~14 kB of weights (nalgebra DMatrix heap allocation). Boxing keeps
    // enum size uniform, matching Phase 3a Transformer's `large_enum_variant` fix.
    Mamba(Box<MambaLayer>),
}

impl Layer {
    pub fn input_size(&self) -> usize {
        match self {
            Layer::Dense(d) => {
                if d.w.is_empty() {
                    0
                } else {
                    d.w[0].len()
                }
            }
            Layer::Gru(g) => g.input_size,
            Layer::Lstm(l) => l.input_size,
            Layer::Window(w) => w.input_size,
            Layer::Transformer(t) => t.d_model,
            Layer::Mamba(m) => m.input_size,
        }
    }
}
```

- [ ] **Step 4: Run cargo check to confirm compilation**

```bash
cargo check --manifest-path src/rust/Cargo.toml --lib
```

Expected: compile errors in `data::neural` complaining about missing `MambaLayer` arms in `LayerWeights for Layer`, `NeuralNetModel::save_json`, `from_v2_json`, `from_flat_weights_v2` -- we'll fix these in Tasks 4-7. At this point the **struct and enum variants** should parse cleanly.

If `cargo check` reports no errors related to the new variant yet, that means we haven't hit a non-exhaustive match. That's fine -- later tasks will hit them.

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/data/neural.rs
git commit -m "feat(nn): add MambaLayer struct + LayerSpec/Layer enum variants"
```

---

## Task 3: `MambaLayer::forward` implementation

**Files:**
- Modify: `src/rust/src/data/neural.rs` (add `impl MambaLayer` block after the `MambaLayer` struct)
- Test: `src/rust/src/data/neural.rs` (`#[cfg(test)] mod tests`)

- [ ] **Step 1: Write hand-verified 2-step trajectory test (will fail)**

Append to the `#[cfg(test)] mod tests` block:

```rust
#[test]
fn mamba_forward_two_step_hand_verified() {
    use nalgebra::{DMatrix, DVector};

    // Minimal layer: d_inner=2, d_state=2, dt_rank=1
    // x_proj: (1 + 2*2, 2) = (5, 2) -- rows [dt_pre; B_0; B_1; C_0; C_1]
    // dt_proj: (2, 1), bias (2,)
    // a_log: (2, 2) -> A = -exp(a_log), here a_log = 0 -> A = -1.0
    // d_skip: (2,)

    let x_proj_w = DMatrix::from_row_slice(5, 2, &[
        0.0, 0.0,   // dt_pre row: proj dt_pre = 0 for any x -> Δ_lift = dt_proj_b
        1.0, 0.0,   // B_0
        0.0, 1.0,   // B_1
        1.0, 0.0,   // C_0
        0.0, 1.0,   // C_1
    ]);
    let dt_proj_w = DMatrix::from_row_slice(2, 1, &[0.0, 0.0]);
    // dt_proj_b such that softplus(b) = 0.5 -> b = inv_softplus(0.5) = log(e^0.5 - 1) ≈ -0.4328
    let b_val = (0.5_f64.exp() - 1.0).ln();
    let dt_proj_b = DVector::from_row_slice(&[b_val, b_val]);
    let a_log = DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 0.0]);  // A = -1
    let d_skip = DVector::from_row_slice(&[0.0, 0.0]);  // no skip, isolate SSM

    let layer = crate::data::neural::MambaLayer {
        input_size: 2, d_state: 2, dt_rank: 1,
        x_proj_w, dt_proj_w, dt_proj_b, a_log, d_skip,
    };

    let mut h = DMatrix::<f64>::zeros(2, 2);
    let x = [1.0, 0.0];

    // Expected Step 1:
    //   Δ = softplus(-0.4328) = 0.5 (per channel)
    //   A = -1 (per (d, n))
    //   Ā = exp(Δ·A) = exp(-0.5) ≈ 0.6065306597126334
    //   B̄ = Δ * B * expm1_over_x(Δ·A) = 0.5 * B * (exp(-0.5) - 1) / (-0.5)
    //      = 0.5 * B * 0.7869386805747332
    //   B_0 = [1, 0] (from x_proj row), B_1 = [0, 1]
    //   For x = [1, 0]:
    //     B̄[0, :] = 0.5 * [1, 0] * 0.7869 = [0.39347, 0]
    //     B̄[1, :] = 0.5 * [0, 1] * 0.7869 = [0, 0.39347]
    //   Wait -- B is the same vector for both d (B comes from x_proj, not per-d):
    //     proj = x_proj @ x = 5 outputs depend on x only.
    //     For x = [1, 0]: proj = [0, 1, 0, 1, 0] -> dt_pre=0, B=[1, 0], C=[1, 0]
    //   So B = [1, 0] (shape d_state,), C = [1, 0]
    //   B̄[d, n] = Δ[d] * B[n] * expm1_over_x(Δ[d] * A[d, n])
    //   For (d=0, n=0): 0.5 * 1 * 0.7869 = 0.39347
    //   For (d=0, n=1): 0.5 * 0 * 0.7869 = 0
    //   For (d=1, n=0): 0.5 * 1 * 0.7869 = 0.39347
    //   For (d=1, n=1): 0.5 * 0 * 0.7869 = 0
    //   h_new[d, n] = Ā[d, n] * h[d, n] + B̄[d, n] * x[d]
    //   For x=[1, 0]: h_new[0, 0] = 0 + 0.39347 * 1 = 0.39347; h_new[1, 0] = 0 + 0.39347 * 0 = 0
    //   y[d] = Σ_n h_new[d, n] * C[n] + D[d] * x[d]
    //   y[0] = h_new[0, 0] * 1 + h_new[0, 1] * 0 + 0 = 0.39347
    //   y[1] = h_new[1, 0] * 1 + h_new[1, 1] * 0 + 0 = 0

    let y = layer.forward(&x, &mut h);
    assert!((y[0] - 0.3934693402873666).abs() < 1e-12, "y[0] = {}", y[0]);
    assert!((y[1] - 0.0).abs() < 1e-15, "y[1] = {}", y[1]);

    // Expected Step 2: h is now [[0.39347, 0], [0, 0]]. Feed x = [0, 1]:
    //   proj = x_proj @ [0, 1] = [0, 0, 1, 0, 1] -> dt_pre=0, B=[0, 1], C=[0, 1]
    //   Δ = 0.5, A = -1 still
    //   B̄[0, 0] = 0.5 * 0 * 0.7869 = 0
    //   B̄[0, 1] = 0.5 * 1 * 0.7869 = 0.39347
    //   B̄[1, 0] = 0.5 * 0 * 0.7869 = 0
    //   B̄[1, 1] = 0.5 * 1 * 0.7869 = 0.39347
    //   h_new[0, 0] = 0.6065 * 0.39347 + 0 * 0 = 0.23865 (state decays)
    //   h_new[0, 1] = 0 + 0.39347 * 0 = 0
    //   h_new[1, 0] = 0 + 0 = 0
    //   h_new[1, 1] = 0 + 0.39347 * 1 = 0.39347
    //   y[0] = h_new[0, 0] * 0 + h_new[0, 1] * 1 = 0
    //   y[1] = h_new[1, 0] * 0 + h_new[1, 1] * 1 = 0.39347
    let x2 = [0.0, 1.0];
    let y2 = layer.forward(&x2, &mut h);
    assert!((y2[0] - 0.0).abs() < 1e-15, "y2[0] = {}", y2[0]);
    assert!((y2[1] - 0.3934693402873666).abs() < 1e-12, "y2[1] = {}", y2[1]);
    // State h[0, 0] should now be ~0.23865 (exp(-0.5) * prev)
    assert!((h[(0, 0)] - 0.2386512185453707).abs() < 1e-12, "h[0, 0] = {}", h[(0, 0)]);
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cargo test --manifest-path src/rust/Cargo.toml mamba_forward_two_step
```

Expected: FAIL with `no method named forward found for struct MambaLayer` (we haven't implemented it yet).

- [ ] **Step 3: Implement `MambaLayer::forward`**

Append to `src/rust/src/data/neural.rs`, after the `MambaLayer` struct definition:

```rust
impl MambaLayer {
    /// Single-tick forward. Mutates `h` in place (state update), returns `y`.
    ///
    /// Shapes: `x: [f64; input_size]`, `h: DMatrix<f64> (input_size, d_state)`,
    /// returns `Vec<f64>` length `input_size`.
    pub fn forward(&self, x: &[f64], h: &mut nalgebra::DMatrix<f64>) -> Vec<f64> {
        debug_assert_eq!(x.len(), self.input_size);
        debug_assert_eq!(h.nrows(), self.input_size);
        debug_assert_eq!(h.ncols(), self.d_state);

        let x_vec = nalgebra::DVector::from_row_slice(x);

        // 1. Fused x_proj: produces (dt_rank + 2*d_state,)
        let proj = &self.x_proj_w * &x_vec;
        // Split into (Δ_pre, B, C)
        let dt_pre: Vec<f64> = (0..self.dt_rank).map(|i| proj[i]).collect();
        let b_vec: Vec<f64> = (0..self.d_state).map(|i| proj[self.dt_rank + i]).collect();
        let c_vec: Vec<f64> = (0..self.d_state)
            .map(|i| proj[self.dt_rank + self.d_state + i])
            .collect();

        // 2. dt_proj + softplus -> per-channel positive Δ
        let dt_pre_v = nalgebra::DVector::from_row_slice(&dt_pre);
        let dt_lifted = &self.dt_proj_w * &dt_pre_v + &self.dt_proj_b;
        let delta: Vec<f64> = (0..self.input_size)
            .map(|i| softplus(dt_lifted[i]))
            .collect();

        // 3. ZOH discretization + state update, per channel, per state dim.
        //    A = -exp(a_log), diagonal per (d, n)
        //    Ā[d, n] = exp(Δ[d] * A[d, n])
        //    B̄[d, n] = Δ[d] * B[n] * expm1_over_x(Δ[d] * A[d, n])
        //    h_new[d, n] = Ā[d, n] * h[d, n] + B̄[d, n] * x[d]
        //    y[d]        = Σ_n (h_new[d, n] * C[n])  +  D[d] * x[d]
        let mut y = vec![0.0_f64; self.input_size];
        for d in 0..self.input_size {
            let delta_d = delta[d];
            let x_d = x[d];
            let mut acc = 0.0;
            for n in 0..self.d_state {
                let a_dn = -self.a_log[(d, n)].exp();
                let za = delta_d * a_dn;
                let a_bar = za.exp();
                let b_bar = delta_d * b_vec[n] * expm1_over_x(za);
                h[(d, n)] = a_bar * h[(d, n)] + b_bar * x_d;
                acc += h[(d, n)] * c_vec[n];
            }
            y[d] = acc + self.d_skip[d] * x_d;
        }
        y
    }
}
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
cargo test --manifest-path src/rust/Cargo.toml mamba_forward_two_step
```

Expected: PASS.

- [ ] **Step 5: Add a finite-output proptest**

Append to the `#[cfg(test)] mod tests` block:

```rust
proptest::proptest! {
    #[test]
    fn mamba_forward_finite_on_finite_inputs(
        d_inner in 1usize..=4,
        d_state in 1usize..=4,
        dt_rank in 1usize..=3,
        seed in 0u64..1000,
    ) {
        use nalgebra::{DMatrix, DVector};
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let rand_vec = |n: usize, rng: &mut rand::rngs::StdRng| -> Vec<f64> {
            (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect()
        };
        let x_proj_w = DMatrix::from_row_slice(dt_rank + 2 * d_state, d_inner,
            &rand_vec((dt_rank + 2 * d_state) * d_inner, &mut rng));
        let dt_proj_w = DMatrix::from_row_slice(d_inner, dt_rank,
            &rand_vec(d_inner * dt_rank, &mut rng));
        let dt_proj_b = DVector::from_row_slice(&rand_vec(d_inner, &mut rng));
        let a_log = DMatrix::from_row_slice(d_inner, d_state, &rand_vec(d_inner * d_state, &mut rng));
        let d_skip = DVector::from_row_slice(&rand_vec(d_inner, &mut rng));

        let layer = crate::data::neural::MambaLayer {
            input_size: d_inner, d_state, dt_rank,
            x_proj_w, dt_proj_w, dt_proj_b, a_log, d_skip,
        };
        let x: Vec<f64> = rand_vec(d_inner, &mut rng);
        let mut h = DMatrix::<f64>::zeros(d_inner, d_state);

        for _ in 0..50 {
            let y = layer.forward(&x, &mut h);
            for v in &y {
                proptest::prop_assert!(v.is_finite(), "y not finite: {v}");
            }
            for i in 0..d_inner {
                for j in 0..d_state {
                    proptest::prop_assert!(h[(i, j)].is_finite(), "h[{i}, {j}] not finite");
                }
            }
        }
    }
}
```

- [ ] **Step 6: Run proptest**

```bash
cargo test --manifest-path src/rust/Cargo.toml mamba_forward_finite
```

Expected: PASS (256 proptest cases by default).

- [ ] **Step 7: Commit**

```bash
git add src/rust/src/data/neural.rs
git commit -m "feat(nn): MambaLayer::forward with ZOH discretization + hand-verified test"
```

---

## Task 4: `LayerWeights for MambaLayer` (flat round-trip)

**Files:**
- Modify: `src/rust/src/data/neural.rs` (add `impl LayerWeights for MambaLayer` block after the existing `impl LayerWeights for TransformerLayer`)
- Test: `src/rust/src/data/neural.rs` (`#[cfg(test)] mod tests`)

- [ ] **Step 1: Write round-trip test (will fail)**

Append to the `#[cfg(test)] mod tests` block:

```rust
#[test]
fn mamba_to_flat_from_flat_roundtrip() {
    use nalgebra::{DMatrix, DVector};
    use rand::{Rng, SeedableRng};

    let (input_size, d_state, dt_rank) = (8, 4, 2);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let gen = |n: usize, rng: &mut rand::rngs::StdRng| -> Vec<f64> {
        (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect()
    };

    let original = crate::data::neural::MambaLayer {
        input_size, d_state, dt_rank,
        x_proj_w: DMatrix::from_row_slice(dt_rank + 2 * d_state, input_size, &gen((dt_rank + 2 * d_state) * input_size, &mut rng)),
        dt_proj_w: DMatrix::from_row_slice(input_size, dt_rank, &gen(input_size * dt_rank, &mut rng)),
        dt_proj_b: DVector::from_row_slice(&gen(input_size, &mut rng)),
        a_log: DMatrix::from_row_slice(input_size, d_state, &gen(input_size * d_state, &mut rng)),
        d_skip: DVector::from_row_slice(&gen(input_size, &mut rng)),
    };

    let expected_n = input_size * (3 * d_state + 2 * dt_rank + 2);
    assert_eq!(original.n_params(), expected_n);

    let flat = original.to_flat();
    assert_eq!(flat.len(), expected_n);

    let spec = crate::data::neural::LayerSpec::Mamba { input_size, d_state, dt_rank };
    let (reconstructed, cursor) = crate::data::neural::MambaLayer::from_flat(&spec, &flat)
        .expect("from_flat failed");
    assert_eq!(cursor, expected_n);

    // Element-by-element comparison
    assert_eq!(reconstructed.input_size, original.input_size);
    assert_eq!(reconstructed.d_state, original.d_state);
    assert_eq!(reconstructed.dt_rank, original.dt_rank);
    for i in 0..reconstructed.x_proj_w.nrows() {
        for j in 0..reconstructed.x_proj_w.ncols() {
            assert_eq!(reconstructed.x_proj_w[(i, j)], original.x_proj_w[(i, j)]);
        }
    }
    for i in 0..reconstructed.dt_proj_w.nrows() {
        for j in 0..reconstructed.dt_proj_w.ncols() {
            assert_eq!(reconstructed.dt_proj_w[(i, j)], original.dt_proj_w[(i, j)]);
        }
    }
    for i in 0..input_size {
        assert_eq!(reconstructed.dt_proj_b[i], original.dt_proj_b[i]);
        assert_eq!(reconstructed.d_skip[i], original.d_skip[i]);
    }
    for i in 0..input_size {
        for j in 0..d_state {
            assert_eq!(reconstructed.a_log[(i, j)], original.a_log[(i, j)]);
        }
    }
}

#[test]
fn mamba_from_flat_rejects_short_slice() {
    let spec = crate::data::neural::LayerSpec::Mamba { input_size: 4, d_state: 2, dt_rank: 1 };
    let expected_n = 4 * (3 * 2 + 2 * 1 + 2);  // = 40
    let too_short = vec![0.0_f64; expected_n - 1];
    let result = crate::data::neural::MambaLayer::from_flat(&spec, &too_short);
    assert!(result.is_err(), "should reject short flat slice");
}
```

- [ ] **Step 2: Run to confirm failure**

```bash
cargo test --manifest-path src/rust/Cargo.toml mamba_to_flat
cargo test --manifest-path src/rust/Cargo.toml mamba_from_flat_rejects
```

Expected: FAIL (LayerWeights not impl'd, `from_flat` not visible).

- [ ] **Step 3: Implement `LayerWeights for MambaLayer`**

Find `impl LayerWeights for TransformerLayer` (around line 679 in `src/rust/src/data/neural.rs`) and append after it:

```rust
impl LayerWeights for MambaLayer {
    fn n_params(&self) -> usize {
        self.input_size * (3 * self.d_state + 2 * self.dt_rank + 2)
    }

    fn to_flat(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n_params());
        // 1. x_proj_w row-major
        for i in 0..self.x_proj_w.nrows() {
            for j in 0..self.x_proj_w.ncols() {
                out.push(self.x_proj_w[(i, j)]);
            }
        }
        // 2. dt_proj_w row-major
        for i in 0..self.dt_proj_w.nrows() {
            for j in 0..self.dt_proj_w.ncols() {
                out.push(self.dt_proj_w[(i, j)]);
            }
        }
        // 3. dt_proj_b
        for i in 0..self.dt_proj_b.len() {
            out.push(self.dt_proj_b[i]);
        }
        // 4. a_log row-major
        for i in 0..self.a_log.nrows() {
            for j in 0..self.a_log.ncols() {
                out.push(self.a_log[(i, j)]);
            }
        }
        // 5. d_skip
        for i in 0..self.d_skip.len() {
            out.push(self.d_skip[i]);
        }
        out
    }

    fn from_flat(spec: &LayerSpec, flat: &[f64]) -> Result<(Self, usize), String> {
        let LayerSpec::Mamba { input_size, d_state, dt_rank } = spec else {
            return Err("from_flat called with non-Mamba spec".into());
        };
        let (input_size, d_state, dt_rank) = (*input_size, *d_state, *dt_rank);
        let expected = input_size * (3 * d_state + 2 * dt_rank + 2);
        if flat.len() < expected {
            return Err(format!(
                "Mamba: flat slice too short (need {expected}, got {})",
                flat.len()
            ));
        }

        let mut cursor = 0;
        // 1. x_proj_w
        let rows = dt_rank + 2 * d_state;
        let cols = input_size;
        let x_proj_w = nalgebra::DMatrix::from_row_slice(rows, cols, &flat[cursor..cursor + rows * cols]);
        cursor += rows * cols;
        // 2. dt_proj_w
        let dt_proj_w = nalgebra::DMatrix::from_row_slice(input_size, dt_rank, &flat[cursor..cursor + input_size * dt_rank]);
        cursor += input_size * dt_rank;
        // 3. dt_proj_b
        let dt_proj_b = nalgebra::DVector::from_row_slice(&flat[cursor..cursor + input_size]);
        cursor += input_size;
        // 4. a_log
        let a_log = nalgebra::DMatrix::from_row_slice(input_size, d_state, &flat[cursor..cursor + input_size * d_state]);
        cursor += input_size * d_state;
        // 5. d_skip
        let d_skip = nalgebra::DVector::from_row_slice(&flat[cursor..cursor + input_size]);
        cursor += input_size;

        Ok((
            MambaLayer {
                input_size, d_state, dt_rank,
                x_proj_w, dt_proj_w, dt_proj_b, a_log, d_skip,
            },
            cursor,
        ))
    }
}
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
cargo test --manifest-path src/rust/Cargo.toml mamba_to_flat
cargo test --manifest-path src/rust/Cargo.toml mamba_from_flat_rejects
```

Expected: PASS.

- [ ] **Step 5: Add proptest for random chromosome round-trip**

Append to `#[cfg(test)] mod tests`:

```rust
proptest::proptest! {
    #[test]
    fn mamba_flat_roundtrip_proptest(
        d_inner in 1usize..=8,
        d_state in 1usize..=8,
        dt_rank in 1usize..=4,
        seed in 0u64..200,
    ) {
        use nalgebra::{DMatrix, DVector};
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let n = d_inner * (3 * d_state + 2 * dt_rank + 2);
        let flat: Vec<f64> = (0..n).map(|_| rng.gen_range(-5.0..5.0)).collect();

        let spec = crate::data::neural::LayerSpec::Mamba { input_size: d_inner, d_state, dt_rank };
        let (layer, cursor) = crate::data::neural::MambaLayer::from_flat(&spec, &flat).unwrap();
        proptest::prop_assert_eq!(cursor, n);

        let back = layer.to_flat();
        for i in 0..n {
            proptest::prop_assert_eq!(back[i], flat[i]);
        }
    }
}
```

- [ ] **Step 6: Run proptest**

```bash
cargo test --manifest-path src/rust/Cargo.toml mamba_flat_roundtrip_proptest
```

Expected: PASS (256 cases).

- [ ] **Step 7: Commit**

```bash
git add src/rust/src/data/neural.rs
git commit -m "feat(nn): LayerWeights for MambaLayer with canonical flat ordering"
```

---

## Task 5: `LayerState::Mamba` + `LayerWeights for Layer` dispatch

**Files:**
- Modify: `src/rust/src/data/nn_state.rs` (add `Mamba` variant + `for_layer` + `reset` arms)
- Modify: `src/rust/src/data/neural.rs` (extend `LayerWeights for Layer` match at ~line 755)

- [ ] **Step 1: Add `LayerState::Mamba` variant**

Open `src/rust/src/data/nn_state.rs`. Find `pub enum LayerState` (around line 12) and add:

```rust
#[derive(Debug, Clone)]
pub enum LayerState {
    None,
    Gru {
        h: Vec<f64>,
    },
    Lstm {
        h: Vec<f64>,
        c: Vec<f64>,
    },
    Window {
        buffer: VecDeque<Vec<f64>>,
    },
    Transformer {
        k_cache: VecDeque<Vec<f64>>,
        v_cache: VecDeque<Vec<f64>>,
    },
    /// Mamba SSM state: shape (input_size, d_state). Zero-initialized at episode start.
    /// Single 2D tensor per layer (unlike LSTM's tuple state or Window's deque).
    Mamba {
        h: nalgebra::DMatrix<f64>,
    },
}
```

- [ ] **Step 2: Extend `LayerState::for_layer`**

Find `impl LayerState { pub fn for_layer` (around line 33) and add the `Layer::Mamba` arm:

```rust
impl LayerState {
    pub fn for_layer(layer: &Layer) -> Self {
        match layer {
            Layer::Dense(_) => LayerState::None,
            Layer::Gru(g) => LayerState::Gru {
                h: vec![0.0; g.hidden_size],
            },
            Layer::Lstm(l) => LayerState::Lstm {
                h: vec![0.0; l.hidden_size],
                c: vec![0.0; l.hidden_size],
            },
            Layer::Window(w) => {
                let mut buffer = VecDeque::with_capacity(w.n_steps);
                for _ in 0..w.n_steps {
                    buffer.push_back(vec![0.0; w.input_size]);
                }
                LayerState::Window { buffer }
            }
            Layer::Transformer(_) => LayerState::Transformer {
                k_cache: VecDeque::new(),
                v_cache: VecDeque::new(),
            },
            Layer::Mamba(m) => LayerState::Mamba {
                h: nalgebra::DMatrix::<f64>::zeros(m.input_size, m.d_state),
            },
        }
    }
    // ... rest unchanged
}
```

- [ ] **Step 3: Extend `LayerState::reset`**

Add the `LayerState::Mamba` arm to the `reset` match:

```rust
    pub fn reset(&mut self) {
        match self {
            LayerState::None => {}
            LayerState::Gru { h } => {
                for v in h.iter_mut() { *v = 0.0; }
            }
            LayerState::Lstm { h, c } => {
                for v in h.iter_mut() { *v = 0.0; }
                for v in c.iter_mut() { *v = 0.0; }
            }
            LayerState::Window { buffer } => {
                for slot in buffer.iter_mut() {
                    for v in slot.iter_mut() { *v = 0.0; }
                }
            }
            LayerState::Transformer { k_cache, v_cache } => {
                k_cache.clear();
                v_cache.clear();
            }
            LayerState::Mamba { h } => {
                h.fill(0.0);
            }
        }
    }
```

- [ ] **Step 4: Extend `LayerWeights for Layer` in `neural.rs`**

Open `src/rust/src/data/neural.rs`, find `impl LayerWeights for Layer` (around line 755), and add `Layer::Mamba` arms to `n_params`, `to_flat`, `from_flat`:

```rust
impl LayerWeights for Layer {
    fn n_params(&self) -> usize {
        match self {
            Layer::Dense(l) => l.n_params(),
            Layer::Gru(l) => l.n_params(),
            Layer::Lstm(l) => l.n_params(),
            Layer::Window(l) => l.n_params(),
            Layer::Transformer(l) => l.n_params(),
            Layer::Mamba(l) => l.n_params(),
        }
    }

    fn to_flat(&self) -> Vec<f64> {
        match self {
            Layer::Dense(l) => l.to_flat(),
            Layer::Gru(l) => l.to_flat(),
            Layer::Lstm(l) => l.to_flat(),
            Layer::Window(l) => l.to_flat(),
            Layer::Transformer(l) => l.to_flat(),
            Layer::Mamba(l) => l.to_flat(),
        }
    }

    fn from_flat(spec: &LayerSpec, flat: &[f64]) -> Result<(Self, usize), String> {
        match spec {
            LayerSpec::Dense { .. } => {
                let (l, n) = DenseLayer::from_flat(spec, flat)?;
                Ok((Layer::Dense(l), n))
            }
            LayerSpec::Gru { .. } => {
                let (l, n) = GruLayer::from_flat(spec, flat)?;
                Ok((Layer::Gru(l), n))
            }
            LayerSpec::Lstm { .. } => {
                let (l, n) = LstmLayer::from_flat(spec, flat)?;
                Ok((Layer::Lstm(l), n))
            }
            LayerSpec::Window { .. } => {
                let (l, n) = WindowLayer::from_flat(spec, flat)?;
                Ok((Layer::Window(l), n))
            }
            LayerSpec::Transformer { .. } => {
                let (l, n) = TransformerLayer::from_flat(spec, flat)?;
                Ok((Layer::Transformer(Box::new(l)), n))
            }
            LayerSpec::Mamba { .. } => {
                let (l, n) = MambaLayer::from_flat(spec, flat)?;
                Ok((Layer::Mamba(Box::new(l)), n))
            }
        }
    }
}
```

Note: read the existing arms first before overwriting -- the structure above matches what's already there; only `Mamba` is new. Use Edit with narrow context to add only the new arm to each of the three method match blocks.

- [ ] **Step 5: Run `cargo check`**

```bash
cargo check --manifest-path src/rust/Cargo.toml --lib
```

Expected: compilation errors in `NeuralNetModel::save_json` / `from_v2_json` / `from_flat_weights_v2` / `forward` -- these we address in Tasks 6-7. No new errors should remain in `LayerState` / `LayerWeights for Layer`.

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/data/neural.rs src/rust/src/data/nn_state.rs
git commit -m "feat(nn): LayerState::Mamba + LayerWeights for Layer dispatch arms"
```

---

## Task 6: `TomlLayerSpec::Mamba` + config validator

**Files:**
- Modify: `src/rust/src/config.rs` (add `Mamba` variant to `TomlLayerSpec` + `to_layer_spec` arm with `dt_rank` resolution)
- Test: `src/rust/src/config.rs` (`#[cfg(test)] mod tests`)

- [ ] **Step 1: Locate `TomlLayerSpec` and `to_layer_spec`**

```bash
grep -n "enum TomlLayerSpec\|fn to_layer_spec" src/rust/src/config.rs
```

- [ ] **Step 2: Write failing test for `dt_rank` auto-resolution**

Append to the `#[cfg(test)] mod tests` block in `src/rust/src/config.rs`:

```rust
#[test]
fn mamba_toml_resolves_dt_rank_from_input_size() {
    // input_size=32, omitted dt_rank -> max(1, 32/16) = 2
    let toml = r#"
        [[network.architecture]]
        type = "mamba"
        input_size = 32
        d_state = 16
    "#;
    // ... parse toml, extract Mamba spec
    // This test requires your existing TomlLayerSpec parse path; adapt to the
    // actual helper used by other layer tests (e.g. load_network_config).
    let parsed: TomlLayerSpec = toml::from_str(
        r#"type = "mamba"
input_size = 32
d_state = 16
"#
    ).unwrap();
    let spec = parsed.to_layer_spec().unwrap();
    match spec {
        LayerSpec::Mamba { input_size, d_state, dt_rank } => {
            assert_eq!(input_size, 32);
            assert_eq!(d_state, 16);
            assert_eq!(dt_rank, 2);  // max(1, 32/16) = 2
        }
        _ => panic!("expected Mamba spec"),
    }
}

#[test]
fn mamba_toml_explicit_dt_rank_overrides_default() {
    let parsed: TomlLayerSpec = toml::from_str(
        r#"type = "mamba"
input_size = 32
d_state = 16
dt_rank = 8
"#
    ).unwrap();
    let spec = parsed.to_layer_spec().unwrap();
    match spec {
        LayerSpec::Mamba { dt_rank, .. } => assert_eq!(dt_rank, 8),
        _ => panic!("expected Mamba spec"),
    }
}

#[test]
fn mamba_toml_rejects_dt_rank_larger_than_input_size() {
    let parsed: TomlLayerSpec = toml::from_str(
        r#"type = "mamba"
input_size = 8
d_state = 4
dt_rank = 16
"#
    ).unwrap();
    let result = parsed.to_layer_spec();
    assert!(result.is_err());
    let msg = result.unwrap_err();
    assert!(msg.contains("dt_rank"), "error message should mention dt_rank: {msg}");
}

#[test]
fn mamba_toml_rejects_zero_dims() {
    // d_state = 0 must fail
    let parsed: TomlLayerSpec = toml::from_str(
        r#"type = "mamba"
input_size = 8
d_state = 0
"#
    ).unwrap();
    assert!(parsed.to_layer_spec().is_err());
}

#[test]
fn mamba_toml_defaults_dt_rank_to_one_for_small_input() {
    // input_size=8, omitted dt_rank -> max(1, 8/16) = max(1, 0) = 1
    let parsed: TomlLayerSpec = toml::from_str(
        r#"type = "mamba"
input_size = 8
d_state = 4
"#
    ).unwrap();
    let spec = parsed.to_layer_spec().unwrap();
    match spec {
        LayerSpec::Mamba { dt_rank, .. } => assert_eq!(dt_rank, 1),
        _ => panic!("expected Mamba"),
    }
}
```

- [ ] **Step 3: Run tests to confirm failure**

```bash
cargo test --manifest-path src/rust/Cargo.toml mamba_toml
```

Expected: FAIL with "no variant `mamba`" or similar.

- [ ] **Step 4: Add `TomlLayerSpec::Mamba` variant**

Find `pub enum TomlLayerSpec` in `src/rust/src/config.rs`. Add a `Mamba` variant mirroring `Transformer`:

```rust
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TomlLayerSpec {
    Dense {
        input_size: usize,
        output_size: usize,
        activation: Activation,
    },
    Gru {
        input_size: usize,
        hidden_size: usize,
    },
    Lstm {
        input_size: usize,
        hidden_size: usize,
    },
    Window {
        input_size: usize,
        n_steps: usize,
    },
    Transformer {
        d_model: usize,
        n_heads: usize,
        d_ffn: usize,
        n_seq: usize,
    },
    Mamba {
        input_size: usize,
        d_state: usize,
        #[serde(default)]
        dt_rank: Option<usize>,
    },
}
```

- [ ] **Step 5: Add `to_layer_spec` Mamba arm**

Find `impl TomlLayerSpec { pub fn to_layer_spec(&self) -> Result<LayerSpec, String>` and add:

```rust
TomlLayerSpec::Mamba { input_size, d_state, dt_rank } => {
    if *input_size == 0 {
        return Err("Mamba: input_size must be > 0".into());
    }
    if *d_state == 0 {
        return Err("Mamba: d_state must be > 0".into());
    }
    let resolved = dt_rank.unwrap_or_else(|| (*input_size / 16).max(1));
    if resolved == 0 {
        return Err("Mamba: dt_rank must be > 0".into());
    }
    if resolved > *input_size {
        return Err(format!(
            "Mamba: dt_rank ({resolved}) must be <= input_size ({input_size})"
        ));
    }
    Ok(LayerSpec::Mamba {
        input_size: *input_size,
        d_state: *d_state,
        dt_rank: resolved,
    })
}
```

- [ ] **Step 6: Run tests to confirm pass**

```bash
cargo test --manifest-path src/rust/Cargo.toml mamba_toml
```

Expected: all 5 new tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/rust/src/config.rs
git commit -m "feat(nn): TomlLayerSpec::Mamba with dt_rank auto-resolution"
```

---

## Task 7: `NeuralNetModel` JSON + flat-weights Mamba arms

**Files:**
- Modify: `src/rust/src/data/neural.rs` (add `Mamba` arms to `save_json`, `from_v2_json`, `from_flat_weights_v2`, and the internal `layer_sizes` derivation)

- [ ] **Step 1: Locate the three functions**

```bash
grep -n "fn save_json\|fn from_v2_json\|fn from_flat_weights_v2" src/rust/src/data/neural.rs
```

- [ ] **Step 2: Read the existing Transformer arm in `save_json`**

```bash
grep -n "Transformer" src/rust/src/data/neural.rs | head -20
```

Use Read tool on the relevant lines to understand the existing Transformer arm structure. Each function has a match over `&self.layers[i]` (save) or `&architecture[i]` (load); you'll mirror the Transformer arm's structure for Mamba.

- [ ] **Step 3: Add `Mamba` arm to `save_json`**

In `NeuralNetModel::save_json`, find the `Layer::Transformer` arm that writes its `weights` dict entry. Add immediately after it:

```rust
Layer::Mamba(m) => {
    // Flat weights dict matching Rust NnLayerWeights schema.
    // Top-level keys, NOT nested (following Transformer's precedent).
    let mut layer_weights = std::collections::BTreeMap::new();
    // x_proj_w: Vec<Vec<f64>> (row-major, 2D)
    let mut x_proj_rows: Vec<Vec<f64>> = Vec::with_capacity(m.x_proj_w.nrows());
    for i in 0..m.x_proj_w.nrows() {
        let mut row = Vec::with_capacity(m.x_proj_w.ncols());
        for j in 0..m.x_proj_w.ncols() {
            row.push(m.x_proj_w[(i, j)]);
        }
        x_proj_rows.push(row);
    }
    // dt_proj_w: Vec<Vec<f64>>
    let mut dt_proj_rows: Vec<Vec<f64>> = Vec::with_capacity(m.dt_proj_w.nrows());
    for i in 0..m.dt_proj_w.nrows() {
        let mut row = Vec::with_capacity(m.dt_proj_w.ncols());
        for j in 0..m.dt_proj_w.ncols() {
            row.push(m.dt_proj_w[(i, j)]);
        }
        dt_proj_rows.push(row);
    }
    // a_log: Vec<Vec<f64>>
    let mut a_log_rows: Vec<Vec<f64>> = Vec::with_capacity(m.a_log.nrows());
    for i in 0..m.a_log.nrows() {
        let mut row = Vec::with_capacity(m.a_log.ncols());
        for j in 0..m.a_log.ncols() {
            row.push(m.a_log[(i, j)]);
        }
        a_log_rows.push(row);
    }
    // dt_proj_b, d_skip: Vec<f64>
    let dt_proj_b_vec: Vec<f64> = m.dt_proj_b.iter().copied().collect();
    let d_skip_vec: Vec<f64> = m.d_skip.iter().copied().collect();

    let nlw = NnLayerWeights {
        w: None, b: None,
        // We add five extra fields; the NnLayerWeights struct needs these too.
        // See Step 4 below.
        x_proj_w: Some(x_proj_rows),
        dt_proj_w: Some(dt_proj_rows),
        dt_proj_b: Some(dt_proj_b_vec),
        a_log: Some(a_log_rows),
        d_skip: Some(d_skip_vec),
        // ... plus whatever Transformer uses, default to None
        ..Default::default()
    };
    weights.insert(format!("layer_{i}"), nlw);
}
```

**NOTE:** The exact key used (`layer_{i}` vs the layer-type name) depends on the existing scheme. Read the existing Transformer save_json arm to confirm the key format, then mirror it.

- [ ] **Step 4: Extend `NnLayerWeights` struct with Mamba fields**

Open `src/rust/src/data/neural.rs`. Find `struct NnLayerWeights` (around line 820-860 based on the grep output showing `ln2_beta: Option<Vec<f64>>` at line 860). Add five new optional fields with `#[serde(default)]`:

```rust
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct NnLayerWeights {
    // ... existing fields (w, b, weight_ih, weight_hh, bias_ih, bias_hh, ln1_gamma, ln1_beta, ln2_gamma, ln2_beta, w_q, b_q, ...) ...

    // Mamba fields (Phase 4a)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    x_proj_w: Option<Vec<Vec<f64>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    dt_proj_w: Option<Vec<Vec<f64>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    dt_proj_b: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    a_log: Option<Vec<Vec<f64>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    d_skip: Option<Vec<f64>>,
}
```

Read the existing struct first to see its field list; add the five Mamba fields after the last existing one.

- [ ] **Step 5: Add `Mamba` arm to `from_v2_json`**

Find `NeuralNetModel::from_v2_json` and the `LayerSpec::Transformer` arm. Add:

```rust
LayerSpec::Mamba { input_size, d_state, dt_rank } => {
    let key = format!("layer_{i}");
    let nlw = file.weights.get(&key).ok_or_else(|| {
        DataError(format!("Mamba layer {i}: missing weights entry '{key}'"))
    })?;
    // Expected shapes:
    //   x_proj_w:  (dt_rank + 2*d_state, input_size)
    //   dt_proj_w: (input_size, dt_rank)
    //   dt_proj_b: (input_size,)
    //   a_log:     (input_size, d_state)
    //   d_skip:    (input_size,)
    let x_proj_rows = nlw.x_proj_w.as_ref().ok_or_else(|| {
        DataError(format!("Mamba layer {i}: missing x_proj_w"))
    })?;
    let expected_xp_rows = dt_rank + 2 * d_state;
    if x_proj_rows.len() != expected_xp_rows || x_proj_rows.iter().any(|r| r.len() != *input_size) {
        return Err(DataError(format!(
            "Mamba layer {i}: x_proj_w shape mismatch (expected {}x{}, got {}x{})",
            expected_xp_rows, input_size, x_proj_rows.len(),
            x_proj_rows.first().map_or(0, |r| r.len())
        )));
    }
    let mut x_proj_w = nalgebra::DMatrix::<f64>::zeros(expected_xp_rows, *input_size);
    for (r, row) in x_proj_rows.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            x_proj_w[(r, c)] = v;
        }
    }

    let dt_proj_rows = nlw.dt_proj_w.as_ref().ok_or_else(|| {
        DataError(format!("Mamba layer {i}: missing dt_proj_w"))
    })?;
    if dt_proj_rows.len() != *input_size || dt_proj_rows.iter().any(|r| r.len() != *dt_rank) {
        return Err(DataError(format!(
            "Mamba layer {i}: dt_proj_w shape mismatch (expected {}x{})",
            input_size, dt_rank
        )));
    }
    let mut dt_proj_w = nalgebra::DMatrix::<f64>::zeros(*input_size, *dt_rank);
    for (r, row) in dt_proj_rows.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            dt_proj_w[(r, c)] = v;
        }
    }

    let dt_proj_b_v = nlw.dt_proj_b.as_ref().ok_or_else(|| {
        DataError(format!("Mamba layer {i}: missing dt_proj_b"))
    })?;
    if dt_proj_b_v.len() != *input_size {
        return Err(DataError(format!(
            "Mamba layer {i}: dt_proj_b length {} != input_size {}",
            dt_proj_b_v.len(), input_size
        )));
    }
    let dt_proj_b = nalgebra::DVector::from_row_slice(dt_proj_b_v);

    let a_log_rows = nlw.a_log.as_ref().ok_or_else(|| {
        DataError(format!("Mamba layer {i}: missing a_log"))
    })?;
    if a_log_rows.len() != *input_size || a_log_rows.iter().any(|r| r.len() != *d_state) {
        return Err(DataError(format!(
            "Mamba layer {i}: a_log shape mismatch (expected {}x{})",
            input_size, d_state
        )));
    }
    let mut a_log = nalgebra::DMatrix::<f64>::zeros(*input_size, *d_state);
    for (r, row) in a_log_rows.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            a_log[(r, c)] = v;
        }
    }

    let d_skip_v = nlw.d_skip.as_ref().ok_or_else(|| {
        DataError(format!("Mamba layer {i}: missing d_skip"))
    })?;
    if d_skip_v.len() != *input_size {
        return Err(DataError(format!(
            "Mamba layer {i}: d_skip length {} != input_size {}",
            d_skip_v.len(), input_size
        )));
    }
    let d_skip = nalgebra::DVector::from_row_slice(d_skip_v);

    layers.push(Layer::Mamba(Box::new(MambaLayer {
        input_size: *input_size, d_state: *d_state, dt_rank: *dt_rank,
        x_proj_w, dt_proj_w, dt_proj_b, a_log, d_skip,
    })));
    layer_sizes.push(*input_size);
}
```

The exact symbol for `DataError` and the local variable names (`layers`, `layer_sizes`, `file.weights`) must match what's already in `from_v2_json`. Read the Transformer arm first to confirm names.

- [ ] **Step 6: Add `Mamba` arm to `from_flat_weights_v2`**

Find `NeuralNetModel::from_flat_weights_v2` and the `LayerSpec::Transformer` arm. Add:

```rust
LayerSpec::Mamba { .. } => {
    let (layer, n) = MambaLayer::from_flat(spec, &flat[cursor..])
        .map_err(DataError)?;
    cursor += n;
    layer_sizes.push(layer.input_size);
    layers.push(Layer::Mamba(Box::new(layer)));
}
```

Again, confirm symbol names match the existing Transformer arm.

- [ ] **Step 7: Update `layer_sizes.push` logic for output size**

The existing logic typically pushes `layer.output_size()` AFTER appending the layer, but some layers only contribute one element to `layer_sizes`. For Mamba `I=O=input_size`, so we push `input_size` once in Step 5/6 above. Verify by reading the Gru/Lstm arms (they have the same `I=O` property -- wait, GRU/LSTM have `hidden_size` != `input_size` typically; Mamba is the first `I=O` constrained layer other than Window).

Actually Window has `output = n_steps * input_size`, Transformer has `I=O=d_model`. So **Transformer's arm** is the right template. Double-check the `layer_sizes` push logic mirrors Transformer.

- [ ] **Step 8: Extend `NeuralNetModel::forward` dispatch**

Find `NeuralNetModel::forward` (or `forward_with_state`, whichever is the canonical entry point). Add a `Layer::Mamba` arm to the main `match layer` inside the per-layer loop:

```rust
Layer::Mamba(m) => {
    let LayerState::Mamba { h } = state_for_layer else {
        return Err(DataError(format!(
            "Mamba layer {i}: state mismatch (expected LayerState::Mamba, got other)"
        )));
    };
    current = m.forward(&current, h);
}
```

The exact variable names (`current`, `state_for_layer`, loop structure) depend on the existing code -- read the Transformer arm first.

- [ ] **Step 9: Run `cargo check` + `cargo clippy`**

```bash
cargo check --manifest-path src/rust/Cargo.toml --lib
cargo clippy --manifest-path src/rust/Cargo.toml --lib -- -D warnings
```

Expected: zero errors, zero clippy warnings. If clippy fires on `large_enum_variant` because of `Layer::Mamba`, verify the `Box<MambaLayer>` wrapping is correct (it's already in the enum definition from Task 2).

- [ ] **Step 10: Run the full Rust test suite**

```bash
cargo test --manifest-path src/rust/Cargo.toml --lib
```

Expected: all tests PASS, including existing golden regressions (Mamba is additive -- no existing behavior changes).

- [ ] **Step 11: Commit**

```bash
git add src/rust/src/data/neural.rs
git commit -m "feat(nn): NeuralNetModel Mamba arms in save_json/from_v2_json/from_flat_weights_v2"
```

---

## Task 8: Python `MambaLayer` torch module

**Files:**
- Create: `src/python/aerocapture/training/rl/layers/mamba.py`
- Test: `tests/test_python_mamba_layer.py`

- [ ] **Step 1: Write the failing unit test**

Create `tests/test_python_mamba_layer.py`:

```python
"""Unit test for the Python MambaLayer torch mirror.

Validates the forward contract in isolation (Python-side only). The full
cross-language equivalence test vs Rust runtime lives in
test_rust_python_mamba_equivalence.py.
"""
from __future__ import annotations

import math

import pytest
import torch

from aerocapture.training.rl.layers.mamba import MambaLayer


@pytest.fixture
def tiny_layer():
    layer = MambaLayer(input_size=2, d_state=2, dt_rank=1)
    layer.double()
    # Override weights to match the Rust hand-verified test fixture exactly.
    with torch.no_grad():
        layer.x_proj_w.copy_(torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=torch.float64))
        layer.dt_proj_w.copy_(torch.zeros(2, 1, dtype=torch.float64))
        # inv_softplus(0.5) = log(e^0.5 - 1)
        b_val = math.log(math.exp(0.5) - 1.0)
        layer.dt_proj_b.copy_(torch.full((2,), b_val, dtype=torch.float64))
        layer.a_log.zero_()   # A = -exp(0) = -1
        layer.d_skip.zero_()  # no skip
    return layer


def test_mamba_forward_step_zero_state(tiny_layer):
    x = torch.tensor([1.0, 0.0], dtype=torch.float64)
    h = torch.zeros(2, 2, dtype=torch.float64)
    y, h_new = tiny_layer(x, h)

    # Matches Rust hand-verified expectations: y[0] ≈ 0.39347, y[1] = 0
    assert abs(y[0].item() - 0.3934693402873666) < 1e-12
    assert abs(y[1].item() - 0.0) < 1e-15


def test_mamba_forward_two_step_state_evolution(tiny_layer):
    x1 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    h0 = torch.zeros(2, 2, dtype=torch.float64)
    y1, h1 = tiny_layer(x1, h0)

    x2 = torch.tensor([0.0, 1.0], dtype=torch.float64)
    y2, h2 = tiny_layer(x2, h1)

    assert abs(y2[0].item() - 0.0) < 1e-15
    assert abs(y2[1].item() - 0.3934693402873666) < 1e-12
    # State h[0, 0] decays by exp(-0.5) from step 1's 0.39347
    assert abs(h2[0, 0].item() - 0.2386512185453707) < 1e-12


def test_mamba_new_state_dtype_matches_parameters():
    layer = MambaLayer(input_size=4, d_state=3, dt_rank=1)
    layer.double()
    state = layer.new_state()
    assert state.dtype == torch.float64
    assert state.shape == (4, 3)
    assert bool(torch.all(state == 0.0))


def test_mamba_deterministic_under_repeated_input():
    torch.manual_seed(0)
    layer = MambaLayer(input_size=4, d_state=3, dt_rank=1)
    layer.double()
    x = torch.randn(4, dtype=torch.float64)
    h_a = layer.new_state()
    h_b = layer.new_state()
    for _ in range(5):
        y_a, h_a = layer(x, h_a)
        y_b, h_b = layer(x, h_b)
        assert torch.allclose(y_a, y_b, atol=0.0)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
uv run pytest tests/test_python_mamba_layer.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'aerocapture.training.rl.layers.mamba'`.

- [ ] **Step 3: Create `src/python/aerocapture/training/rl/layers/mamba.py`**

```python
"""Python torch mirror of the Rust MambaLayer (Phase 4a, PSO-only).

Consumed exclusively by the cross-language equivalence test and (in Phase 4b)
the PPO training path. PSO training bypasses this module entirely -- it goes
through `aerocapture_rs.flat_weights_to_json` + the Rust forward runtime.

The manual softplus / expm1_over_x helpers are 1-for-1 equivalents of the
Rust `pub(crate)` free functions in `src/rust/src/data/neural.rs`. Both sides
must produce bit-identical f64 output (verified by
tests/test_rust_python_mamba_equivalence.py).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


def _softplus(x: Tensor) -> Tensor:
    """Numerically stable softplus matching Rust `softplus` bit-for-bit.

    NOT `torch.nn.functional.softplus`, which has a `threshold=20` linear-branch
    fallback that would break equivalence at |x| > 20.
    """
    return x.clamp_min(0.0) + torch.log1p(torch.exp(-x.abs()))


def _expm1_over_x(z: Tensor) -> Tensor:
    """(exp(z) - 1) / z with Taylor fallback for |z| < 1e-8.

    Matches Rust `expm1_over_x`. Uses `torch.where` for branchless dispatch
    (autograd-compatible for Phase 4b PPO).
    """
    taylor = 1.0 + 0.5 * z + (z * z) / 6.0
    # Avoid division by zero in the exact branch by replacing z=0 with z=1
    # (the where-mask selects the taylor branch there anyway).
    safe_z = torch.where(z != 0.0, z, torch.ones_like(z))
    exact = torch.expm1(z) / safe_z
    return torch.where(z.abs() < 1e-8, taylor, exact)


class MambaLayer(nn.Module):
    """Selective SSM core (Mamba S6) -- PSO-only in Phase 4a.

    Parameters:
        input_size: d_inner; layer fan-in = fan-out.
        d_state:    SSM state dim per channel (N in paper).
        dt_rank:    Bottleneck rank for the Δ projection.

    State contract:
        `new_state()` -> zero-initialized `Tensor` of shape (input_size, d_state),
        dtype tracks parameter dtype (so `policy.double()` propagates).
        `forward(x, h) -> (y, h_new)` where `x: (input_size,)` and `h: (input_size, d_state)`.
    """

    def __init__(self, input_size: int, d_state: int, dt_rank: int) -> None:
        super().__init__()
        if input_size <= 0 or d_state <= 0 or dt_rank <= 0:
            raise ValueError(f"MambaLayer: all dims must be positive; got input_size={input_size}, d_state={d_state}, dt_rank={dt_rank}")
        if dt_rank > input_size:
            raise ValueError(f"MambaLayer: dt_rank ({dt_rank}) must be <= input_size ({input_size})")
        self.input_size = input_size
        self.d_state = d_state
        self.dt_rank = dt_rank

        # Parameter shapes match Rust canonical flat ordering (see spec section 3.3).
        self.x_proj_w = nn.Parameter(torch.zeros(dt_rank + 2 * d_state, input_size))
        self.dt_proj_w = nn.Parameter(torch.zeros(input_size, dt_rank))
        self.dt_proj_b = nn.Parameter(torch.zeros(input_size))
        self.a_log = nn.Parameter(torch.zeros(input_size, d_state))
        self.d_skip = nn.Parameter(torch.zeros(input_size))

    def new_state(self) -> Tensor:
        """Return a zero-initialized state tensor with parameter dtype / device."""
        return torch.zeros(
            self.input_size, self.d_state,
            dtype=self.x_proj_w.dtype,
            device=self.x_proj_w.device,
        )

    def forward(self, x: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        """Single-step forward.

        Args:
            x: (input_size,) input vector.
            h: (input_size, d_state) current state.

        Returns:
            y: (input_size,) output vector.
            h_new: (input_size, d_state) updated state.
        """
        assert x.shape == (self.input_size,), f"x shape {x.shape} != ({self.input_size},)"
        assert h.shape == (self.input_size, self.d_state), \
            f"h shape {h.shape} != ({self.input_size}, {self.d_state})"

        # 1. Fused x_proj -> split into (Δ_pre, B, C)
        proj = self.x_proj_w @ x                                   # (dt_rank + 2*d_state,)
        dt_pre = proj[: self.dt_rank]                              # (dt_rank,)
        b_vec = proj[self.dt_rank : self.dt_rank + self.d_state]   # (d_state,)
        c_vec = proj[self.dt_rank + self.d_state : self.dt_rank + 2 * self.d_state]  # (d_state,)

        # 2. dt_proj + softplus -> per-channel positive Δ
        dt_lifted = self.dt_proj_w @ dt_pre + self.dt_proj_b       # (input_size,)
        delta = _softplus(dt_lifted)                                # (input_size,)

        # 3. ZOH discretization + state update (fully vectorized over (d, n))
        a = -torch.exp(self.a_log)                                  # (input_size, d_state), A < 0
        za = delta.unsqueeze(1) * a                                 # (input_size, d_state)
        a_bar = torch.exp(za)
        b_bar = delta.unsqueeze(1) * b_vec.unsqueeze(0) * _expm1_over_x(za)
        h_new = a_bar * h + b_bar * x.unsqueeze(1)                  # (input_size, d_state)
        y = h_new @ c_vec + self.d_skip * x                         # (input_size,)
        return y, h_new
```

- [ ] **Step 4: Run test to confirm pass**

```bash
uv run pytest tests/test_python_mamba_layer.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/rl/layers/mamba.py tests/test_python_mamba_layer.py
git commit -m "feat(nn): Python MambaLayer torch mirror with manual softplus/expm1_over_x"
```

---

## Task 9: `MambaSpec` + `LayerSpec` union + PPO rejection guards

**Files:**
- Modify: `src/python/aerocapture/training/rl/schemas.py` (add `MambaSpec` + extend union)
- Modify: `src/python/aerocapture/training/rl/layers/__init__.py` (add `build_layer` dispatch)
- Modify: `src/python/aerocapture/training/model_io.py` + `src/python/aerocapture/training/rl/export.py` (whichever contains `load_policy_from_json`; guard Mamba)
- Test: `tests/test_mamba_ppo_rejection.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_mamba_ppo_rejection.py`:

```python
"""Phase 4a gate: Mamba layer must reject PPO usage at build_layer / load_policy_from_json.

PSO training bypasses build_layer entirely (it calls Rust directly). The PPO path
does call build_layer via V2Policy construction, so this rejection is load-bearing
for the "PSO-only" Phase 4a scope.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from aerocapture.training.rl.layers import build_layer
from aerocapture.training.rl.schemas import MambaSpec


def test_mamba_spec_validates():
    spec = MambaSpec(type="mamba", input_size=8, d_state=4, dt_rank=2)
    assert spec.input_size == 8
    assert spec.d_state == 4
    assert spec.dt_rank == 2


def test_mamba_spec_auto_resolves_dt_rank():
    spec = MambaSpec(type="mamba", input_size=32, d_state=16)
    assert spec.dt_rank == 2  # max(1, 32 // 16) = 2


def test_mamba_spec_auto_resolves_dt_rank_small_input():
    spec = MambaSpec(type="mamba", input_size=8, d_state=4)
    assert spec.dt_rank == 1  # max(1, 8 // 16) = 1


def test_mamba_spec_rejects_dt_rank_larger_than_input():
    with pytest.raises(ValueError, match="dt_rank"):
        MambaSpec(type="mamba", input_size=8, d_state=4, dt_rank=16)


def test_build_layer_mamba_raises_not_implemented():
    spec = MambaSpec(type="mamba", input_size=8, d_state=4, dt_rank=2)
    with pytest.raises(NotImplementedError, match="Mamba is PSO-only in Phase 4a"):
        build_layer(spec)


def test_load_policy_from_json_with_mamba_raises():
    from aerocapture.training.model_io import load_policy_from_json

    minimal_json = {
        "format_version": 2,
        "architecture": [
            {"type": "dense", "input_size": 8, "output_size": 8, "activation": "linear"},
            {"type": "mamba", "input_size": 8, "d_state": 4, "dt_rank": 2},
            {"type": "dense", "input_size": 8, "output_size": 2, "activation": "linear"},
        ],
        "weights": {
            "layer_0": {"w": [[0.0] * 8] * 8, "b": [0.0] * 8},
            "layer_1": {
                "x_proj_w": [[0.0] * 8] * 8,
                "dt_proj_w": [[0.0] * 2] * 8,
                "dt_proj_b": [0.0] * 8,
                "a_log": [[0.0] * 4] * 8,
                "d_skip": [0.0] * 8,
            },
            "layer_2": {"w": [[0.0] * 8] * 2, "b": [0.0] * 2},
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "model.json"
        p.write_text(json.dumps(minimal_json))
        with pytest.raises(NotImplementedError, match="Mamba"):
            load_policy_from_json(str(p))
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_mamba_ppo_rejection.py -v
```

Expected: FAIL with `cannot import name 'MambaSpec' from ...`.

- [ ] **Step 3: Add `MambaSpec` to `schemas.py`**

Open `src/python/aerocapture/training/rl/schemas.py`. After `TransformerSpec` (ends around line 72), add:

```python
class MambaSpec(BaseModel):
    """Selective SSM (Mamba S6) layer (Phase 4a, PSO-only).

    Input/output dims are both `input_size` (d_inner). `dt_rank` is the
    bottleneck rank for the Δ projection; if None, resolves to
    `max(1, input_size // 16)` (paper default).

    `build_layer(MambaSpec)` raises NotImplementedError -- PPO support deferred
    to Phase 4b (see docs/superpowers/specs/2026-04-24-phase-4a-mamba-ssm-mvp-design.md).
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["mamba"]
    input_size: int = Field(ge=1)
    d_state: int = Field(ge=1)
    dt_rank: int | None = None

    @model_validator(mode="after")
    def _resolve_and_validate_dt_rank(self) -> MambaSpec:
        if self.dt_rank is None:
            # Paper default: max(1, input_size // 16)
            # Cast to int since `//` on ints returns int but type checker may widen.
            resolved = max(1, self.input_size // 16)
            object.__setattr__(self, "dt_rank", resolved)
        if self.dt_rank < 1:
            raise ValueError(f"dt_rank must be >= 1, got {self.dt_rank}")
        if self.dt_rank > self.input_size:
            raise ValueError(
                f"dt_rank ({self.dt_rank}) must be <= input_size ({self.input_size})"
            )
        return self
```

Then update the `LayerSpec` union on line 75 to include `MambaSpec`:

```python
LayerSpec = Annotated[
    DenseSpec | GruSpec | LstmSpec | WindowSpec | TransformerSpec | MambaSpec,
    Discriminator("type"),
]
```

- [ ] **Step 4: Add `build_layer` dispatch in `rl/layers/__init__.py`**

Open `src/python/aerocapture/training/rl/layers/__init__.py`. Find the `build_layer` function. Add an `isinstance` branch for `MambaSpec` BEFORE the final fallback, raising `NotImplementedError`:

```python
from aerocapture.training.rl.schemas import (
    DenseSpec, GruSpec, LstmSpec, WindowSpec, TransformerSpec, MambaSpec,
)
# ... existing imports ...


def build_layer(spec):
    if isinstance(spec, DenseSpec):
        return DenseLayer(...)
    if isinstance(spec, GruSpec):
        return GruLayer(...)
    if isinstance(spec, LstmSpec):
        return LstmLayer(...)
    if isinstance(spec, WindowSpec):
        raise NotImplementedError(
            "Window is PSO-only; PPO use deferred -- see "
            "docs/superpowers/specs/2026-04-20-phase-2b-window-mlp-design.md"
        )
    if isinstance(spec, TransformerSpec):
        raise NotImplementedError(
            "Transformer is PSO-only in Phase 3a; PPO use deferred -- see "
            "docs/superpowers/specs/2026-04-22-phase-3a-transformer-mvp-design.md"
        )
    if isinstance(spec, MambaSpec):
        raise NotImplementedError(
            "Mamba is PSO-only in Phase 4a; PPO use deferred -- see "
            "docs/superpowers/specs/2026-04-24-phase-4a-mamba-ssm-mvp-design.md"
        )
    raise TypeError(f"Unknown layer spec: {type(spec).__name__}")
```

Read the existing `build_layer` function first to match its exact structure -- the code above is a reconstruction for clarity.

- [ ] **Step 5: Add Mamba guard to `load_policy_from_json`**

Find `load_policy_from_json` (either `src/python/aerocapture/training/model_io.py` or `src/python/aerocapture/training/rl/export.py` -- `grep -rn "def load_policy_from_json" src/python/`). Near the start, after the architecture is parsed, add:

```python
for layer_spec in parsed.architecture:
    if isinstance(layer_spec, MambaSpec):
        raise NotImplementedError(
            "Mamba is PSO-only in Phase 4a; load_policy_from_json not supported. "
            "See docs/superpowers/specs/2026-04-24-phase-4a-mamba-ssm-mvp-design.md"
        )
```

This should be alongside the existing `WindowSpec` / `TransformerSpec` guards.

- [ ] **Step 6: Run test to confirm pass**

```bash
uv run pytest tests/test_mamba_ppo_rejection.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/python/aerocapture/training/rl/schemas.py \
        src/python/aerocapture/training/rl/layers/__init__.py \
        src/python/aerocapture/training/model_io.py \
        src/python/aerocapture/training/rl/export.py \
        tests/test_mamba_ppo_rejection.py
git commit -m "feat(nn): MambaSpec pydantic schema + build_layer/load_policy PPO guards"
```

Note: only stage the files you actually modified (use `git status` first to see what's changed).

---

## Task 10: `_mamba_specs` + `_layer_n_params` + `_layer_output_size` + `describe_architecture`

**Files:**
- Modify: `src/python/aerocapture/training/encoding.py` (add `_mamba_specs`, update `_layer_param_specs`)
- Modify: `src/python/aerocapture/training/config.py` (add Mamba arms to `_layer_n_params`, `_layer_output_size`, `describe_architecture`)
- Test: `tests/test_mamba_encoding.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_mamba_encoding.py`:

```python
"""Tests for Mamba PSO ParamSpec generation and config arm dispatch."""
from __future__ import annotations

import math

import numpy as np
import pytest

from aerocapture.training.encoding import _layer_param_specs, nn_param_specs_from_v2
from aerocapture.training.rl.schemas import DenseSpec, MambaSpec
from aerocapture.training.config import _layer_n_params, _layer_output_size


def test_mamba_param_specs_total_count_matches_formula():
    spec = MambaSpec(type="mamba", input_size=32, d_state=16, dt_rank=2)
    specs = _layer_param_specs(spec, bound_multiplier=1.0)
    # Formula: input_size * (3*d_state + 2*dt_rank + 2) = 32 * (48 + 4 + 2) = 32 * 54 = 1728
    assert len(specs) == 1728


def test_mamba_param_specs_layout_matches_canonical_order():
    spec = MambaSpec(type="mamba", input_size=4, d_state=2, dt_rank=1)
    specs = _layer_param_specs(spec, bound_multiplier=1.0)
    # Canonical order (section 3.3):
    #   1. x_proj_w: (dt_rank + 2*d_state, input_size) = (5, 4) = 20
    #   2. dt_proj_w: (input_size, dt_rank) = (4, 1) = 4
    #   3. dt_proj_b: (input_size,) = 4
    #   4. a_log: (input_size, d_state) = (4, 2) = 8
    #   5. d_skip: (input_size,) = 4
    # Total = 40
    assert len(specs) == 40

    names = [s.name for s in specs]
    assert names[:20] == ["x_proj_w"] * 20
    assert names[20:24] == ["dt_proj_w"] * 4
    assert names[24:28] == ["dt_proj_b"] * 4
    assert names[28:36] == ["a_log"] * 8
    assert names[36:40] == ["d_skip"] * 4


def test_mamba_param_specs_hippo_centers():
    spec = MambaSpec(type="mamba", input_size=2, d_state=3, dt_rank=1)
    specs = _layer_param_specs(spec, bound_multiplier=1.0)
    # a_log starts at index 2 + 2 + 2 = 6 (after x_proj_w: 6, dt_proj_w: 2, dt_proj_b: 2)
    x_proj_n = (1 + 2 * 3) * 2                 # 14
    dt_proj_w_n = 2 * 1                         # 2
    dt_proj_b_n = 2                             # 2
    a_log_start = x_proj_n + dt_proj_w_n + dt_proj_b_n
    # For each d in [0, 2), n in [0, 3): init_center = log(n + 1)
    expected_centers = []
    for _d in range(2):
        for n in range(3):
            expected_centers.append(math.log(n + 1))
    for i, expected in enumerate(expected_centers):
        assert abs(specs[a_log_start + i].init_center - expected) < 1e-15, \
            f"a_log spec {i}: got {specs[a_log_start + i].init_center}, expected {expected}"


def test_mamba_param_specs_d_skip_centers_are_one():
    spec = MambaSpec(type="mamba", input_size=4, d_state=2, dt_rank=1)
    specs = _layer_param_specs(spec, bound_multiplier=1.0)
    d_skip_specs = [s for s in specs if s.name == "d_skip"]
    assert len(d_skip_specs) == 4
    for s in d_skip_specs:
        assert s.init_center == 1.0
        assert s.low == 0.0   # 1.0 - 1.0
        assert s.high == 2.0  # 1.0 + 1.0


def test_layer_n_params_mamba():
    spec = MambaSpec(type="mamba", input_size=32, d_state=16, dt_rank=2)
    assert _layer_n_params(spec) == 1728


def test_layer_output_size_mamba_equals_input_size():
    spec = MambaSpec(type="mamba", input_size=32, d_state=16, dt_rank=2)
    assert _layer_output_size(spec) == 32


def test_nn_param_specs_from_v2_handles_mamba():
    arch = [
        DenseSpec(type="dense", input_size=23, output_size=8, activation="tanh"),
        MambaSpec(type="mamba", input_size=8, d_state=4, dt_rank=2),
        DenseSpec(type="dense", input_size=8, output_size=2, activation="linear"),
    ]
    specs = nn_param_specs_from_v2(arch, bound_multiplier=1.0)
    # Dense(23->8): 23*8+8=192; Mamba(8,4,2): 8*(12+4+2)=144; Dense(8->2): 18
    assert len(specs) == 192 + 144 + 18
```

- [ ] **Step 2: Run test to confirm failure**

```bash
uv run pytest tests/test_mamba_encoding.py -v
```

Expected: most tests FAIL (Mamba dispatch not added yet).

- [ ] **Step 3: Add `_MAMBA_DT_BIAS_SEED` + `_mamba_specs` to `encoding.py`**

Open `src/python/aerocapture/training/encoding.py`. At module level (near other seed constants; if absent add near the top after imports):

```python
# Deterministic sub-seed for dt_proj_b center draw. Matched between _mamba_specs
# (ParamSpec bounds) and _init_mamba_layer (initial population values) so both
# agree on the center each ParamSpec window is centered around.
_MAMBA_DT_BIAS_SEED: int = 0xDE17A  # "delta" in hex-ish; arbitrary constant, keep stable
```

Then add the `_mamba_specs` helper:

```python
def _mamba_specs(spec, bound_multiplier: float) -> list["ParamSpec"]:
    """Generate PSO ParamSpec list for a Mamba layer in canonical flat order.

    Order matches Rust `LayerWeights for MambaLayer::to_flat`:
      1. x_proj_w  (dt_rank + 2*d_state, input_size) row-major -- Xavier bounds
      2. dt_proj_w (input_size, dt_rank)              row-major -- Xavier * dt_rank^{-0.5}
      3. dt_proj_b (input_size,)                                -- inv_softplus(U(1e-3, 1e-1)) centers
      4. a_log     (input_size, d_state)              row-major -- HiPPO log(n+1) centers
      5. d_skip    (input_size,)                                -- 1.0 centers

    Per-slice bounds are `[center - bound_multiplier, center + bound_multiplier]`.
    """
    import math as _math
    import numpy as np

    d_inner = spec.input_size
    d_state = spec.d_state
    dt_rank = spec.dt_rank

    specs: list[ParamSpec] = []

    # 1. x_proj_w: Xavier uniform bounds, center 0
    fan_in_xp = d_inner
    fan_out_xp = dt_rank + 2 * d_state
    bound_xp = _math.sqrt(6.0 / (fan_in_xp + fan_out_xp)) * bound_multiplier
    for _ in range(fan_out_xp * d_inner):
        specs.append(ParamSpec(name="x_proj_w", low=-bound_xp, high=+bound_xp, init_center=0.0, scale="linear"))

    # 2. dt_proj_w: Xavier * dt_rank^{-0.5}, center 0
    fan_in_dt = dt_rank
    fan_out_dt = d_inner
    bound_dt = _math.sqrt(6.0 / (fan_in_dt + fan_out_dt)) / _math.sqrt(max(dt_rank, 1)) * bound_multiplier
    for _ in range(d_inner * dt_rank):
        specs.append(ParamSpec(name="dt_proj_w", low=-bound_dt, high=+bound_dt, init_center=0.0, scale="linear"))

    # 3. dt_proj_b: per-channel inv_softplus(U(1e-3, 1e-1)) centers
    local = np.random.default_rng(_MAMBA_DT_BIAS_SEED)
    dt_draws = local.uniform(1e-3, 1e-1, size=d_inner)
    for d in range(d_inner):
        dt = float(dt_draws[d])
        center = _math.log(_math.expm1(dt))
        specs.append(ParamSpec(
            name="dt_proj_b",
            low=center - bound_multiplier,
            high=center + bound_multiplier,
            init_center=center,
            scale="linear",
        ))

    # 4. a_log: HiPPO log(n+1) centers, broadcast across d_inner (outer loop d, inner n)
    for _d in range(d_inner):
        for n in range(d_state):
            center = _math.log(n + 1)
            specs.append(ParamSpec(
                name="a_log",
                low=center - bound_multiplier,
                high=center + bound_multiplier,
                init_center=center,
                scale="linear",
            ))

    # 5. d_skip: 1.0 centers
    for _ in range(d_inner):
        specs.append(ParamSpec(
            name="d_skip",
            low=1.0 - bound_multiplier,
            high=1.0 + bound_multiplier,
            init_center=1.0,
            scale="linear",
        ))

    return specs
```

Adapt `ParamSpec(...)` to the actual class name and field set. Read existing helpers like `_dense_specs` / `_gru_specs` to match the style.

- [ ] **Step 4: Update `_layer_param_specs` dispatch**

Find `def _layer_param_specs` in `encoding.py`. Add a Mamba branch:

```python
def _layer_param_specs(spec, bound_multiplier: float) -> list[ParamSpec]:
    if isinstance(spec, DenseSpec):
        return _dense_specs(spec, bound_multiplier)
    if isinstance(spec, GruSpec):
        return _gru_specs(spec, bound_multiplier)
    if isinstance(spec, LstmSpec):
        return _lstm_specs(spec, bound_multiplier)
    if isinstance(spec, WindowSpec):
        return []  # zero trainable params
    if isinstance(spec, TransformerSpec):
        return _transformer_specs(spec, bound_multiplier)
    if isinstance(spec, MambaSpec):
        return _mamba_specs(spec, bound_multiplier)
    raise TypeError(f"Unknown layer spec: {type(spec).__name__}")
```

Import `MambaSpec` at the top of the file if not already imported.

- [ ] **Step 5: Add Mamba arms in `config.py`**

Open `src/python/aerocapture/training/config.py`. Find `_layer_n_params`, `_layer_output_size`, `describe_architecture`. Add `MambaSpec` branches:

```python
def _layer_n_params(layer_spec) -> int:
    if isinstance(layer_spec, DenseSpec):
        return layer_spec.input_size * layer_spec.output_size + layer_spec.output_size
    if isinstance(layer_spec, GruSpec):
        return 3 * layer_spec.hidden_size * (layer_spec.input_size + layer_spec.hidden_size) + 6 * layer_spec.hidden_size
    if isinstance(layer_spec, LstmSpec):
        return 4 * layer_spec.hidden_size * (layer_spec.input_size + layer_spec.hidden_size) + 8 * layer_spec.hidden_size
    if isinstance(layer_spec, WindowSpec):
        return 0
    if isinstance(layer_spec, TransformerSpec):
        # ... existing formula ...
    if isinstance(layer_spec, MambaSpec):
        return layer_spec.input_size * (3 * layer_spec.d_state + 2 * layer_spec.dt_rank + 2)
    raise TypeError(f"unknown layer spec: {type(layer_spec).__name__}")


def _layer_output_size(layer_spec) -> int:
    if isinstance(layer_spec, DenseSpec):
        return layer_spec.output_size
    if isinstance(layer_spec, GruSpec):
        return layer_spec.hidden_size
    if isinstance(layer_spec, LstmSpec):
        return layer_spec.hidden_size
    if isinstance(layer_spec, WindowSpec):
        return layer_spec.n_steps * layer_spec.input_size
    if isinstance(layer_spec, TransformerSpec):
        return layer_spec.d_model
    if isinstance(layer_spec, MambaSpec):
        return layer_spec.input_size
    raise TypeError(f"unknown layer spec: {type(layer_spec).__name__}")


def describe_architecture(arch) -> str:
    parts = []
    for spec in arch:
        if isinstance(spec, DenseSpec):
            parts.append(f"Dense({spec.input_size}->{spec.output_size}, {spec.activation})")
        elif isinstance(spec, GruSpec):
            parts.append(f"Gru(I={spec.input_size}, H={spec.hidden_size})")
        elif isinstance(spec, LstmSpec):
            parts.append(f"Lstm(I={spec.input_size}, H={spec.hidden_size})")
        elif isinstance(spec, WindowSpec):
            parts.append(f"Window(I={spec.input_size}, n_steps={spec.n_steps})")
        elif isinstance(spec, TransformerSpec):
            parts.append(f"Transformer(d_model={spec.d_model}, n_heads={spec.n_heads}, d_ffn={spec.d_ffn}, n_seq={spec.n_seq})")
        elif isinstance(spec, MambaSpec):
            parts.append(f"Mamba(d_inner={spec.input_size}, d_state={spec.d_state}, dt_rank={spec.dt_rank})")
        else:
            parts.append(f"Unknown({type(spec).__name__})")
    return " -> ".join(parts)
```

Adapt to the actual existing function signatures. Import `MambaSpec` at the top if not present.

- [ ] **Step 6: Run tests to confirm pass**

```bash
uv run pytest tests/test_mamba_encoding.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/python/aerocapture/training/encoding.py \
        src/python/aerocapture/training/config.py \
        tests/test_mamba_encoding.py
git commit -m "feat(nn): _mamba_specs + Mamba arms in _layer_n_params/output_size/describe"
```

---

## Task 11: `_init_mamba_layer` + `init_v2_population` dispatch

**Files:**
- Modify: `src/python/aerocapture/training/initialization_v2.py` (add `_init_mamba_layer` helper + dispatch)
- Test: `tests/test_init_v2_mamba.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_init_v2_mamba.py`:

```python
"""Tests for init_v2_population Mamba arm diversity and center-agreement invariants."""
from __future__ import annotations

import math

import numpy as np
import pytest

from aerocapture.training.initialization_v2 import init_v2_population
from aerocapture.training.encoding import _MAMBA_DT_BIAS_SEED, _layer_param_specs
from aerocapture.training.rl.schemas import DenseSpec, MambaSpec


def test_mamba_init_produces_correct_param_count():
    arch = [MambaSpec(type="mamba", input_size=4, d_state=2, dt_rank=1)]
    rng = np.random.default_rng(0)
    pop = init_v2_population(arch, n_pop=8, bound_multiplier=1.0, rng=rng)
    # n_params = 4 * (3*2 + 2*1 + 2) = 4 * 10 = 40
    assert pop.shape == (8, 40)


def test_mamba_init_all_individuals_differ_on_every_slice():
    """Regression test against the bug where dt_proj_b / a_log / d_skip are
    identical across the PSO population (killing exploration)."""
    arch = [MambaSpec(type="mamba", input_size=4, d_state=2, dt_rank=1)]
    rng = np.random.default_rng(123)
    pop = init_v2_population(arch, n_pop=16, bound_multiplier=1.0, rng=rng)

    x_proj_n = 5 * 4       # 20
    dt_proj_w_n = 4 * 1    # 4
    dt_proj_b_n = 4        # 4
    a_log_n = 4 * 2        # 8
    d_skip_n = 4           # 4

    slices = {
        "x_proj_w":  slice(0, x_proj_n),
        "dt_proj_w": slice(x_proj_n, x_proj_n + dt_proj_w_n),
        "dt_proj_b": slice(x_proj_n + dt_proj_w_n, x_proj_n + dt_proj_w_n + dt_proj_b_n),
        "a_log":     slice(x_proj_n + dt_proj_w_n + dt_proj_b_n,
                           x_proj_n + dt_proj_w_n + dt_proj_b_n + a_log_n),
        "d_skip":    slice(x_proj_n + dt_proj_w_n + dt_proj_b_n + a_log_n,
                           x_proj_n + dt_proj_w_n + dt_proj_b_n + a_log_n + d_skip_n),
    }
    for name, sl in slices.items():
        std_across_pop = pop[:, sl].std(axis=0)
        assert (std_across_pop > 1e-9).all(), f"slice {name} has zero-variance columns"


def test_mamba_init_centers_agree_with_param_spec_bounds():
    """Load-bearing invariant: each init value must fall inside [low, high] from _layer_param_specs."""
    spec = MambaSpec(type="mamba", input_size=4, d_state=2, dt_rank=1)
    arch = [spec]
    rng = np.random.default_rng(42)
    pop = init_v2_population(arch, n_pop=32, bound_multiplier=1.0, rng=rng)

    ps = _layer_param_specs(spec, bound_multiplier=1.0)
    assert pop.shape[1] == len(ps)

    for i, param_spec in enumerate(ps):
        col = pop[:, i]
        assert (col >= param_spec.low - 1e-9).all(), f"param {i} ({param_spec.name}): below low"
        assert (col <= param_spec.high + 1e-9).all(), f"param {i} ({param_spec.name}): above high"


def test_mamba_init_a_log_mean_is_hippo():
    """Population mean of a_log slice should converge to HiPPO centers as n_pop grows."""
    spec = MambaSpec(type="mamba", input_size=2, d_state=4, dt_rank=1)
    arch = [spec]
    rng = np.random.default_rng(7)
    pop = init_v2_population(arch, n_pop=10000, bound_multiplier=1.0, rng=rng)

    x_proj_n = (1 + 2 * 4) * 2
    dt_proj_w_n = 2 * 1
    dt_proj_b_n = 2
    a_log_start = x_proj_n + dt_proj_w_n + dt_proj_b_n

    a_log_mean = pop[:, a_log_start : a_log_start + 2 * 4].mean(axis=0)
    # HiPPO: for each d in [0, 2), n in [0, 4): center = log(n+1)
    expected = np.array([math.log(n + 1) for _d in range(2) for n in range(4)])
    assert np.allclose(a_log_mean, expected, atol=0.01)  # jitter_std=0.01, 10000 samples


def test_mamba_init_d_skip_mean_is_one():
    spec = MambaSpec(type="mamba", input_size=4, d_state=2, dt_rank=1)
    arch = [spec]
    rng = np.random.default_rng(99)
    pop = init_v2_population(arch, n_pop=10000, bound_multiplier=1.0, rng=rng)

    d_skip_start = 5 * 4 + 4 * 1 + 4 + 4 * 2
    d_skip = pop[:, d_skip_start : d_skip_start + 4]
    assert np.allclose(d_skip.mean(axis=0), 1.0, atol=0.01)
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_init_v2_mamba.py -v
```

Expected: FAIL -- `init_v2_population` has no MambaSpec branch.

- [ ] **Step 3: Add `_init_mamba_layer` helper + dispatch**

Open `src/python/aerocapture/training/initialization_v2.py`. Add near the top (after imports):

```python
_INIT_JITTER_STD = 0.01  # matches Phase 2a LSTM forget-bias jitter convention
```

Import `MambaSpec` and `_MAMBA_DT_BIAS_SEED` at the top:

```python
from aerocapture.training.encoding import _MAMBA_DT_BIAS_SEED  # noqa: F401 if already imported elsewhere
from aerocapture.training.rl.schemas import DenseSpec, GruSpec, LstmSpec, WindowSpec, TransformerSpec, MambaSpec
```

Then add the `_init_mamba_layer` function:

```python
def _init_mamba_layer(
    spec: MambaSpec,
    n_pop: int,
    bound_multiplier: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Initialize `n_pop` flat chromosomes for a Mamba layer.

    Canonical flat order (matches Rust `LayerWeights for MambaLayer::to_flat`):
      1. x_proj_w  -- Xavier uniform around 0 (per-individual)
      2. dt_proj_w -- Xavier * dt_rank^{-0.5} around 0 (per-individual)
      3. dt_proj_b -- shared center (inv_softplus(U(1e-3, 1e-1))) + per-individual jitter
      4. a_log     -- HiPPO log(n+1) broadcast across d_inner + per-individual jitter
      5. d_skip    -- 1.0 + per-individual jitter

    Per-individual jitter (std = 0.01 * bound_multiplier) ensures PSO population
    diversity even on slices whose centers are shared across individuals.
    """
    d_inner = spec.input_size
    d_state = spec.d_state
    dt_rank = spec.dt_rank
    n_params = d_inner * (3 * d_state + 2 * dt_rank + 2)
    pop = np.empty((n_pop, n_params), dtype=np.float64)

    # Shared per-channel dt_proj_b centers (same seed _mamba_specs uses)
    local = np.random.default_rng(_MAMBA_DT_BIAS_SEED)
    dt_bias_centers = np.log(np.expm1(local.uniform(1e-3, 1e-1, size=d_inner)))  # (d_inner,)

    # HiPPO a_log centers, broadcast across d_inner, row-major flatten
    a_log_centers = np.broadcast_to(
        np.log(np.arange(d_state) + 1.0), (d_inner, d_state)
    ).copy().ravel()

    jitter_std = _INIT_JITTER_STD * bound_multiplier

    for i in range(n_pop):
        buf = []
        # 1. x_proj_w: Xavier uniform around 0 (per-individual)
        fan_in_xp = d_inner
        fan_out_xp = dt_rank + 2 * d_state
        bound_xp = math.sqrt(6.0 / (fan_in_xp + fan_out_xp))
        buf.append(rng.uniform(-bound_xp, +bound_xp, size=fan_out_xp * d_inner))
        # 2. dt_proj_w: Xavier * dt_rank^{-0.5} (per-individual)
        fan_in_dt = dt_rank
        fan_out_dt = d_inner
        bound_dt = math.sqrt(6.0 / (fan_in_dt + fan_out_dt)) / math.sqrt(max(dt_rank, 1))
        buf.append(rng.uniform(-bound_dt, +bound_dt, size=d_inner * dt_rank))
        # 3. dt_proj_b: shared centers + per-individual jitter
        buf.append(dt_bias_centers + rng.normal(0.0, jitter_std, size=d_inner))
        # 4. a_log: HiPPO centers + per-individual jitter
        buf.append(a_log_centers + rng.normal(0.0, jitter_std, size=d_inner * d_state))
        # 5. d_skip: 1.0 + per-individual jitter
        buf.append(np.full(d_inner, 1.0) + rng.normal(0.0, jitter_std, size=d_inner))

        pop[i] = np.concatenate(buf)

    return pop
```

Import `math` at the top of the file if not already imported.

- [ ] **Step 4: Add Mamba branch to `init_v2_population` dispatch**

Find the per-layer loop in `init_v2_population` that dispatches on spec type. Add a `MambaSpec` branch alongside the others:

```python
def init_v2_population(architecture, n_pop, bound_multiplier, rng):
    slabs = []
    for spec in architecture:
        if isinstance(spec, DenseSpec):
            slabs.append(_init_dense_layer(spec, n_pop, bound_multiplier, rng))
        elif isinstance(spec, GruSpec):
            slabs.append(_init_gru_layer(spec, n_pop, bound_multiplier, rng))
        elif isinstance(spec, LstmSpec):
            slabs.append(_init_lstm_layer(spec, n_pop, bound_multiplier, rng))
        elif isinstance(spec, WindowSpec):
            continue  # zero-param, no slab
        elif isinstance(spec, TransformerSpec):
            slabs.append(_init_transformer_layer(spec, n_pop, bound_multiplier, rng))
        elif isinstance(spec, MambaSpec):
            slabs.append(_init_mamba_layer(spec, n_pop, bound_multiplier, rng))
        else:
            raise TypeError(f"init_v2_population: unknown layer spec {type(spec).__name__}")
    return np.concatenate(slabs, axis=1) if slabs else np.empty((n_pop, 0), dtype=np.float64)
```

Match the real structure of the existing function -- this is a reconstruction.

- [ ] **Step 5: Run tests to confirm pass**

```bash
uv run pytest tests/test_init_v2_mamba.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/initialization_v2.py tests/test_init_v2_mamba.py
git commit -m "feat(nn): _init_mamba_layer with HiPPO/inv_softplus centers + per-individual jitter"
```

---

## Task 12: `export_v2_policy_to_json` Mamba branch + obs-norm guard

**Files:**
- Modify: `src/python/aerocapture/training/rl/export.py` (add Mamba branch to `export_v2_policy_to_json`; extend obs-norm guard)
- Test: `tests/test_export_v2_mamba.py`

**Note:** PSO does NOT use `export_v2_policy_to_json` -- it goes through the Rust `flat_weights_to_json` PyO3 helper. This task is primarily to keep the Phase 4a/4b seam clean: when Phase 4b lands PPO Mamba, this code is already ready for the PyTorch `V2Policy` export path.

- [ ] **Step 1: Write failing test**

Create `tests/test_export_v2_mamba.py`:

```python
"""Tests for PyTorch V2Policy Mamba export (Phase 4a -- read-ready for Phase 4b)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from aerocapture.training.rl.layers.mamba import MambaLayer
from aerocapture.training.rl.schemas import DenseSpec, MambaSpec


# Note: this test assumes V2Policy can be instantiated from an architecture list
# that includes a Mamba layer -- but `build_layer(MambaSpec)` raises.
# We therefore construct the export target manually, skipping V2Policy.


def test_export_v2_mamba_layer_emits_flat_keys():
    """Export a hand-constructed MambaLayer + surrounding dummy Dense layers,
    verify the JSON v2 weights dict has the 5 flat Mamba keys at layer level.
    """
    from aerocapture.training.rl.export import _serialize_mamba_layer  # helper under test

    m = MambaLayer(input_size=4, d_state=2, dt_rank=1)
    m.double()
    with torch.no_grad():
        m.x_proj_w.normal_(0, 0.1)
        m.dt_proj_w.normal_(0, 0.1)
        m.dt_proj_b.normal_(0, 0.1)
        m.a_log.normal_(0, 0.1)
        m.d_skip.normal_(0, 0.1)

    weights_dict = _serialize_mamba_layer(m)
    # Flat at layer level -- NOT nested under "weights"
    assert set(weights_dict.keys()) == {"x_proj_w", "dt_proj_w", "dt_proj_b", "a_log", "d_skip"}
    assert len(weights_dict["x_proj_w"]) == 5      # dt_rank + 2*d_state
    assert len(weights_dict["x_proj_w"][0]) == 4   # input_size
    assert len(weights_dict["dt_proj_w"]) == 4     # input_size
    assert len(weights_dict["dt_proj_w"][0]) == 1  # dt_rank
    assert len(weights_dict["dt_proj_b"]) == 4
    assert len(weights_dict["a_log"]) == 4
    assert len(weights_dict["a_log"][0]) == 2
    assert len(weights_dict["d_skip"]) == 4


def test_obs_norm_bake_in_rejects_mamba_as_layer_zero():
    """Phase 0 invariant: obs-normalizer bake-in is only safe into a Dense layer 0.
    Mamba's first op is x_proj @ x but subsequent softplus + A = -exp(a_log) are
    nonlinear, so absorbing an affine input transform isn't closed-form.
    """
    from aerocapture.training.rl.export import _check_obs_norm_bake_compatibility

    arch = [
        MambaSpec(type="mamba", input_size=8, d_state=4, dt_rank=2),
        DenseSpec(type="dense", input_size=8, output_size=2, activation="linear"),
    ]
    with pytest.raises(NotImplementedError, match="Mamba"):
        _check_obs_norm_bake_compatibility(arch, obs_normalizer_active=True)
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_export_v2_mamba.py -v
```

Expected: FAIL with import errors.

- [ ] **Step 3: Add `_serialize_mamba_layer` helper + Mamba dispatch to `export_v2_policy_to_json`**

Open `src/python/aerocapture/training/rl/export.py`. Add a helper:

```python
def _serialize_mamba_layer(layer: MambaLayer) -> dict:
    """Serialize a MambaLayer to the flat-at-layer-level Mamba weights dict."""
    import numpy as np
    def to_list(t: torch.Tensor) -> list:
        return t.detach().cpu().numpy().astype(np.float64).tolist()
    return {
        "x_proj_w":  to_list(layer.x_proj_w),   # (dt_rank + 2*d_state, input_size)
        "dt_proj_w": to_list(layer.dt_proj_w),  # (input_size, dt_rank)
        "dt_proj_b": to_list(layer.dt_proj_b),  # (input_size,)
        "a_log":     to_list(layer.a_log),      # (input_size, d_state)
        "d_skip":    to_list(layer.d_skip),     # (input_size,)
    }
```

Import `MambaLayer` at the top:

```python
from aerocapture.training.rl.layers.mamba import MambaLayer
```

Find the main `export_v2_policy_to_json` function's per-layer dispatch and add:

```python
elif isinstance(layer, MambaLayer):
    weights_dict[f"layer_{i}"] = _serialize_mamba_layer(layer)
```

- [ ] **Step 4: Add Mamba obs-norm guard**

Find the existing obs-normalizer bake-in function (e.g. `_check_obs_norm_bake_compatibility` or inline in `export_v2_policy_to_json`). Add:

```python
def _check_obs_norm_bake_compatibility(architecture, obs_normalizer_active: bool) -> None:
    if not obs_normalizer_active:
        return
    if not architecture:
        return
    first = architecture[0]
    if isinstance(first, (WindowSpec, TransformerSpec)):
        raise NotImplementedError(
            f"obs-normalizer bake-in into {type(first).__name__} as layer 0 is not supported; "
            "current bake-in assumes a Dense first layer."
        )
    if isinstance(first, MambaSpec):
        raise NotImplementedError(
            "obs-normalizer bake-in into MambaSpec as layer 0 is not supported. "
            "Mamba's x_proj + softplus + A = -exp(a_log) is nonlinear in x; absorbing "
            "an affine input transform would require shifting dt_proj_b through softplus "
            "(not closed-form). Deferred to Phase 4b. See "
            "docs/superpowers/specs/2026-04-24-phase-4a-mamba-ssm-mvp-design.md"
        )
```

Import `MambaSpec` at the top of `export.py`.

- [ ] **Step 5: Run tests to confirm pass**

```bash
uv run pytest tests/test_export_v2_mamba.py -v
```

Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/rl/export.py tests/test_export_v2_mamba.py
git commit -m "feat(nn): export_v2_policy_to_json Mamba branch + obs-norm guard"
```

---

## Task 13: Training config + compare_guidance + train_all.sh

**Files:**
- Create: `configs/training/msr_aller_mamba_pso_train.toml`
- Modify: `src/python/aerocapture/training/compare_guidance.py` (add scheme entry)
- Modify: `train_all.sh` (add alias group)

- [ ] **Step 1: Read `msr_aller_transformer_pso_train.toml` as template**

```bash
cat configs/training/msr_aller_transformer_pso_train.toml
```

- [ ] **Step 2: Create `configs/training/msr_aller_mamba_pso_train.toml`**

Create the file with the architecture defined in the spec:

```toml
base = "common.toml"

# --- Simulation ---
[simulation]
guidance = "neural_network"

# --- Architecture ---
# Dense(23 -> 32, swish) -> Mamba(32, 16) -> Mamba(32, 16) -> Dense(32 -> 2, asinh)
# Param count: 768 + 1728 + 1728 + 66 = 4290

[[network.architecture]]
type = "dense"
input_size = 23
output_size = 32
activation = "swish"

[[network.architecture]]
type = "mamba"
input_size = 32
d_state = 16
# dt_rank omitted -> max(1, 32/16) = 2

[[network.architecture]]
type = "mamba"
input_size = 32
d_state = 16

[[network.architecture]]
type = "dense"
input_size = 32
output_size = 2
activation = "asinh"

# --- Optimizer ---
[optimizer]
algorithm = "pso"
n_pop = 64
n_gen = 2000
seed_strategy = "adaptive"
training_n_sims = 20
validation_n_sims = 1000
seed_pool_interval = 50
curation_sample_size = 1000
curation_top_k = 5
```

Compare to `msr_aller_transformer_pso_train.toml` and ensure consistency of any cross-cutting fields (e.g. `[data]`, `[network] input_mask`, `[mission]` overrides) that the base already supplies; add only what's Mamba-specific.

- [ ] **Step 3: Sanity-check config parsing with Rust**

```bash
./src/rust/target/release/aerocapture configs/training/msr_aller_mamba_pso_train.toml --dry-run 2>&1 | head -20
```

If the Rust binary supports `--dry-run` or equivalent, confirm the config parses without error. If no dry-run flag exists, use a Python-side TOML check:

```bash
uv run python -c "from aerocapture.training.toml_utils import load_toml_with_bases; print(load_toml_with_bases('configs/training/msr_aller_mamba_pso_train.toml'))" | head -30
```

- [ ] **Step 4: Register scheme in `compare_guidance.py`**

Open `src/python/aerocapture/training/compare_guidance.py`. Find `SCHEMES` dict and `_NN_DEPLOY_SCHEMES` set. Add:

```python
SCHEMES = {
    # ... existing entries ...
    "neural_network_mamba_pso": "Mamba SSM (PSO)",
}

_NN_DEPLOY_SCHEMES = {
    # ... existing entries ...
    "neural_network_mamba_pso",
}

SCHEME_TRAINING_CONFIGS = {
    # ... existing entries ...
    "neural_network_mamba_pso": "configs/training/msr_aller_mamba_pso_train.toml",
}
```

Match the actual dict/set names used in the file.

- [ ] **Step 5: Add `train_all.sh` aliases**

Open `train_all.sh`. Find the alias-resolution block (case-statement that maps user-friendly names to scheme names) and the training-order block. Add:

```bash
# In the alias resolution case statement:
mamba_pso | nn_mamba_pso | mamba)
    scheme="neural_network_mamba_pso"
    ;;
```

Also add `neural_network_mamba_pso` to the default "run all" ordering. Place it after `piecewise_constant` (the dependency) and near the other NN-PSO schemes.

- [ ] **Step 6: Smoke-test the alias**

```bash
./train_all.sh --help 2>&1 | grep -i mamba
```

If there's no `--help`, do a dry-run equivalent; or invoke with an immediate Ctrl+C after it prints the chosen scheme.

- [ ] **Step 7: Commit**

```bash
git add configs/training/msr_aller_mamba_pso_train.toml \
        src/python/aerocapture/training/compare_guidance.py \
        train_all.sh
git commit -m "feat(nn): Mamba PSO training config + compare_guidance + train_all.sh aliases"
```

---

## Task 14: Cross-language equivalence test (THE gate)

**Files:**
- Create: `tests/test_rust_python_mamba_equivalence.py`

This is the load-bearing test. If it passes, all of Tasks 1-12 are correct end-to-end.

- [ ] **Step 1: Rebuild the PyO3 bindings**

```bash
uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml
```

From repo root, not the subcrate. This is important per the user's `PyO3 rebuild` preference memory.

- [ ] **Step 2: Create the test file**

```python
"""Cross-language bit-equivalence gate for Mamba SSM layer.

Architecture: Dense(4 -> 8, tanh) -> Mamba(8, 4, 2) -> Dense(8 -> 2, linear)
Total params: 40 + 144 + 18 = 202.

Exports a Python V2Policy to JSON v2, loads it in Rust via
`aerocapture_rs.nn_forward_sequence`, and feeds 100 random f64 inputs.
Asserts max abs diff < 1e-14 (target: machine epsilon, ~1e-16).

If this test fails:
  - Constant drift (both ~1e-12): flat-weight ordering mismatch
  - Growing drift over sequence: state update bug
  - Only fails past step N: warm-up / state-init bug
  - Any NaN / Inf: numerical stability regression in softplus or expm1_over_x
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("aerocapture_rs")

import aerocapture_rs  # noqa: E402

from aerocapture.training.rl.layers.dense import DenseLayer  # noqa: E402
from aerocapture.training.rl.layers.mamba import MambaLayer  # noqa: E402


@pytest.mark.slow
def test_mamba_rust_python_equivalence_100_steps():
    torch.manual_seed(0)

    # Build the 3-layer network manually in Python f64
    dense0 = DenseLayer(input_size=4, output_size=8, activation="tanh").double()
    mamba = MambaLayer(input_size=8, d_state=4, dt_rank=2).double()
    dense1 = DenseLayer(input_size=8, output_size=2, activation="linear").double()

    # Randomize weights with small scale (keeps outputs in a tight range)
    with torch.no_grad():
        for layer in (dense0, dense1):
            for p in layer.parameters():
                p.uniform_(-0.3, 0.3)
        mamba.x_proj_w.uniform_(-0.3, 0.3)
        mamba.dt_proj_w.uniform_(-0.3, 0.3)
        mamba.dt_proj_b.uniform_(-0.5, 0.5)  # wider -- centers matter for softplus
        mamba.a_log.uniform_(0.0, 2.0)        # HiPPO-ish range
        mamba.d_skip.fill_(1.0)

    # Serialize to JSON v2 manually (bypass V2Policy since build_layer rejects Mamba)
    def to_list(t):
        return t.detach().cpu().numpy().astype(np.float64).tolist()

    json_payload = {
        "format_version": 2,
        "architecture": [
            {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
            {"type": "mamba", "input_size": 8, "d_state": 4, "dt_rank": 2},
            {"type": "dense", "input_size": 8, "output_size": 2, "activation": "linear"},
        ],
        "weights": {
            "layer_0": {
                "w": to_list(dense0.w),
                "b": to_list(dense0.b),
            },
            "layer_1": {
                "x_proj_w":  to_list(mamba.x_proj_w),
                "dt_proj_w": to_list(mamba.dt_proj_w),
                "dt_proj_b": to_list(mamba.dt_proj_b),
                "a_log":     to_list(mamba.a_log),
                "d_skip":    to_list(mamba.d_skip),
            },
            "layer_2": {
                "w": to_list(dense1.w),
                "b": to_list(dense1.b),
            },
        },
        "input_mask": list(range(4)),
    }

    # Adapt `dense0.w` / `dense0.b` to the actual DenseLayer attribute names.
    # If the existing DenseLayer uses `weight` / `bias`, change to that.

    with tempfile.TemporaryDirectory() as tmp:
        json_path = Path(tmp) / "model.json"
        json_path.write_text(json.dumps(json_payload))

        # Generate test input sequence
        rng = np.random.default_rng(1234)
        input_seq = rng.uniform(-1.0, 1.0, size=(100, 4)).astype(np.float64)

        # Rust forward (stateful, threads NnState across all 100 steps).
        # Convert numpy array to list-of-lists (matches Phase 3a transformer test pattern).
        rust_out_array = np.asarray(
            aerocapture_rs.nn_forward_sequence(str(json_path), [row.tolist() for row in input_seq]),
            dtype=np.float64,
        )
        assert rust_out_array.shape == (100, 2)

        # Python forward (thread h manually)
        h_mamba = mamba.new_state()
        py_outs = np.empty_like(rust_out_array)
        for i in range(100):
            x = torch.tensor(input_seq[i], dtype=torch.float64)
            y0 = dense0(x)
            y1, h_mamba = mamba(y0, h_mamba)
            y2 = dense1(y1)
            py_outs[i] = y2.detach().cpu().numpy()

        max_diff = np.max(np.abs(rust_out_array - py_outs))
        print(f"Mamba cross-language max abs diff over 100 steps: {max_diff:.3e}")
        assert max_diff < 1e-14, f"cross-language drift: {max_diff:.3e} >= 1e-14"
```

**NOTE:** The exact attribute names on `DenseLayer` (`w` / `b` vs `weight` / `bias`) and the PyO3 `nn_forward_sequence` signature need to be confirmed against the existing code:

```bash
grep -n "class DenseLayer" src/python/aerocapture/training/rl/layers/dense.py
grep -n "nn_forward_sequence" src/rust/aerocapture-py/src/lib.rs
```

Adjust the test accordingly. The Phase 3a Transformer equivalence test (`tests/test_rust_python_transformer_equivalence.py`) has the right skeleton to copy.

- [ ] **Step 3: Run the test**

```bash
uv run pytest tests/test_rust_python_mamba_equivalence.py -v -s
```

Expected: PASS with `max abs diff` in the 1e-16 to 1e-15 range (well below the 1e-14 gate).

If it fails, debug in this order:
1. Print the flat chromosome length — does it match `input_size * (3*d_state + 2*dt_rank + 2)`?
2. Print `rust_out[0]` and `py_out[0]` — single-step disagreement narrows it to the forward pass.
3. Print `rust_out[N]` for increasing N — drift over time means state update mismatch.
4. Unit-test softplus + expm1_over_x at the specific weight values — narrows to helpers.

- [ ] **Step 4: Commit**

```bash
git add tests/test_rust_python_mamba_equivalence.py
git commit -m "test(nn): cross-language Mamba equivalence gate (< 1e-14 f64)"
```

---

## Task 15: Warm-up + PSO smoke + Rust golden regression tests

**Files:**
- Create: `tests/test_mamba_warmup.py`
- Create: `tests/test_mamba_pso_smoke.py`
- (PPO rejection test already created in Task 9.)

- [ ] **Step 1: Create `tests/test_mamba_warmup.py`**

```python
"""Mamba state warm-up test: catches state-init bugs early.

Drives the same architecture for multiple steps and verifies:
  - Step 0 output (zero state) differs from step 1 output (non-zero state)
  - Repeated runs with identical input sequences produce identical output
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("aerocapture_rs")

import aerocapture_rs  # noqa: E402


def _build_test_model_json(tmp_path: Path) -> Path:
    rng = np.random.default_rng(7)
    json_payload = {
        "format_version": 2,
        "architecture": [
            {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
            {"type": "mamba", "input_size": 8, "d_state": 4, "dt_rank": 2},
            {"type": "dense", "input_size": 8, "output_size": 2, "activation": "linear"},
        ],
        "weights": {
            "layer_0": {
                "w": rng.uniform(-0.3, 0.3, size=(8, 4)).tolist(),
                "b": rng.uniform(-0.1, 0.1, size=(8,)).tolist(),
            },
            "layer_1": {
                "x_proj_w":  rng.uniform(-0.3, 0.3, size=(8, 8)).tolist(),
                "dt_proj_w": rng.uniform(-0.3, 0.3, size=(8, 2)).tolist(),
                "dt_proj_b": rng.uniform(-0.5, 0.5, size=(8,)).tolist(),
                "a_log":     rng.uniform(0.0, 2.0, size=(8, 4)).tolist(),
                "d_skip":    [1.0] * 8,
            },
            "layer_2": {
                "w": rng.uniform(-0.3, 0.3, size=(2, 8)).tolist(),
                "b": rng.uniform(-0.1, 0.1, size=(2,)).tolist(),
            },
        },
        "input_mask": list(range(4)),
    }
    path = tmp_path / "model.json"
    path.write_text(json.dumps(json_payload))
    return path


@pytest.mark.slow
def test_mamba_state_differs_from_step_0_to_step_1():
    with tempfile.TemporaryDirectory() as tmp:
        model_path = _build_test_model_json(Path(tmp))
        # Feed the SAME input twice -- outputs must differ because state accumulates
        x = np.array([0.5, -0.3, 0.1, 0.8], dtype=np.float64)
        seq = np.stack([x, x])
        out = aerocapture_rs.nn_forward_sequence(str(model_path), seq)
        assert np.max(np.abs(out[0] - out[1])) > 1e-10, \
            "state evolution: step 0 and step 1 outputs should differ"


@pytest.mark.slow
def test_mamba_forward_is_deterministic_across_runs():
    with tempfile.TemporaryDirectory() as tmp:
        model_path = _build_test_model_json(Path(tmp))
        rng = np.random.default_rng(0)
        seq = rng.uniform(-1.0, 1.0, size=(20, 4)).astype(np.float64)
        out_a = aerocapture_rs.nn_forward_sequence(str(model_path), seq)
        out_b = aerocapture_rs.nn_forward_sequence(str(model_path), seq)
        assert np.array_equal(out_a, out_b), "nondeterministic forward!"
```

- [ ] **Step 2: Create `tests/test_mamba_pso_smoke.py`**

```python
"""End-to-end PSO smoke test for Mamba. Runs 2 short gens and asserts outputs are sane."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("aerocapture_rs")

import aerocapture_rs  # noqa: E402


@pytest.mark.slow
def test_mamba_pso_smoke_2_gens(tmp_path: Path):
    # Minimal reduced-arch config
    config = tmp_path / "mamba_smoke.toml"
    config.write_text("""
base = "../../configs/training/common.toml"

[simulation]
guidance = "neural_network"

[[network.architecture]]
type = "dense"
input_size = 23
output_size = 8
activation = "tanh"

[[network.architecture]]
type = "mamba"
input_size = 8
d_state = 4
dt_rank = 1

[[network.architecture]]
type = "dense"
input_size = 8
output_size = 2
activation = "linear"

[optimizer]
algorithm = "pso"
n_pop = 4
n_gen = 2
seed_strategy = "fixed"
training_n_sims = 4
validation_n_sims = 4
""")

    # Adapt `base` path to an absolute reference if the TOML loader needs it.
    # The Phase 3a transformer smoke test has the precise pattern.

    output_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable, "-m", "aerocapture.training.train",
            str(config),
            "--no-tui", "--skip-report",
            "--output-dir", str(output_dir),
        ],
        check=False, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"train crashed:\n{result.stderr}"

    best_model_path = output_dir / "best_model.json"
    assert best_model_path.exists(), "best_model.json not written"
    payload = json.loads(best_model_path.read_text())
    assert payload["format_version"] == 2
    types = [layer["type"] for layer in payload["architecture"]]
    assert types == ["dense", "mamba", "dense"]

    # Rust runtime sanity: forward with a zero input must return finite (2,) tuple
    zero = [0.0] * 23
    result_tuple = aerocapture_rs.nn_forward(str(best_model_path), zero)
    assert len(result_tuple) == 2
    import math
    assert all(math.isfinite(v) for v in result_tuple)
```

This test is slow (runs actual PSO); mark `@pytest.mark.slow` and gate in CI to the `python-pyo3` job only.

- [ ] **Step 3: Run both tests**

```bash
uv run pytest tests/test_mamba_warmup.py tests/test_mamba_pso_smoke.py -v -m slow
```

Expected: both PASS. The PSO smoke test may take 30-90 seconds depending on physics sim speed.

- [ ] **Step 4: Run the full Rust golden regression suite**

```bash
cargo test --manifest-path src/rust/Cargo.toml --release --lib
cargo test --manifest-path src/rust/Cargo.toml --release --test guidance_golden_regression
```

Expected: all 10 guidance golden regression CSVs bit-identical (Mamba is additive; no existing behavior changes).

- [ ] **Step 5: Commit**

```bash
git add tests/test_mamba_warmup.py tests/test_mamba_pso_smoke.py
git commit -m "test(nn): Mamba warm-up determinism + PSO 2-gen smoke test"
```

---

## Task 16: CI wiring

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Read the existing CI workflow**

```bash
cat .github/workflows/ci.yml
```

Identify the three jobs: `rust`, `python` (main), `python-pyo3`.

- [ ] **Step 2: Add Mamba tests to `python-pyo3` job**

Find the `python-pyo3` job's `run: uv run pytest ...` line. Append the three @slow Mamba tests:

```yaml
- name: Run PyO3 tests
  run: >-
    uv run pytest
    tests/test_pyo3.py
    tests/test_v2_rust_python_equivalence.py
    tests/test_gru_pso_smoke.py
    tests/test_gru_ppo_smoke.py
    tests/test_rust_python_transformer_equivalence.py
    tests/test_transformer_warmup.py
    tests/test_transformer_pso_smoke.py
    tests/test_rust_python_mamba_equivalence.py
    tests/test_mamba_warmup.py
    tests/test_mamba_pso_smoke.py
    --tb=short
```

Adapt to the actual current syntax.

- [ ] **Step 3: Ensure PPO rejection + encoding + Python mirror tests run in main `python` job**

The main `python` job runs the full `pytest tests/` suite by default. Confirm `test_mamba_ppo_rejection.py`, `test_python_mamba_layer.py`, `test_mamba_encoding.py`, `test_init_v2_mamba.py`, `test_export_v2_mamba.py` are NOT marked `@pytest.mark.slow` and will execute there. Grep to confirm:

```bash
grep -l "@pytest.mark.slow" tests/test_mamba_ppo_rejection.py tests/test_python_mamba_layer.py tests/test_mamba_encoding.py tests/test_init_v2_mamba.py tests/test_export_v2_mamba.py 2>/dev/null
```

Expected: no output (no slow markers on these fast tests).

- [ ] **Step 4: Lint check**

```bash
./lint_code.sh
```

Expected: zero ruff errors, zero mypy errors. If `lint_code.sh` complains about untyped imports in the new files, add them to ignore lists or add proper type annotations.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci(nn): add Mamba cross-language + warm-up + PSO smoke tests to python-pyo3 job"
```

---

## Task 17: CLAUDE.md update

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add a Phase 4a paragraph mirroring Phase 3a style**

Open `CLAUDE.md`. Find the Phase 3a Transformer MVP paragraph (search for "Phase 3a Transformer MVP"). Append an analogous Phase 4a paragraph immediately after it. The paragraph structure (matching Phase 3a / 2b format):

```markdown
**Phase 4a Mamba Selective SSM MVP (branch `feature/mamba-ssm-mvp`, 2026-04-24)** adds the sixth stateful layer type: a 1-layer selective SSM core (Mamba S6) with diagonal A, HiPPO-style init, ZOH discretization, and input-dependent Δ/B/C projections. PSO-only; PPO deferred to Phase 4b. Registered as `neural_network_mamba_pso`.
- **Rust**: `MambaLayer` struct (`x_proj_w [dt_rank + 2*d_state, input_size]` no bias, `dt_proj_w [input_size, dt_rank]` + `dt_proj_b [input_size]`, `a_log [input_size, d_state]` with `A = -exp(a_log)` reparameterization, `d_skip [input_size]` per-channel residual), `Layer::Mamba(Box<MambaLayer>)` boxed for `large_enum_variant` uniformity, `LayerSpec::Mamba { input_size, d_state, dt_rank }`, `LayerState::Mamba { h: DMatrix<f64> }` zero-initialized `(input_size, d_state)`, `LayerWeights for MambaLayer` canonical flat order `[x_proj_w row-major, dt_proj_w row-major, dt_proj_b, a_log row-major, d_skip]`, pub(crate) free helpers `softplus` (stable `max(x,0) + log1p(exp(-|x|))`) and `expm1_over_x` (Taylor fallback at `|z| < 1e-8`). TOML `[[network.architecture]] type = "mamba"` with `dt_rank` auto-resolved to `max(1, input_size // 16)` when omitted.
- **Python**: `MambaLayer` torch module (manual softplus matching Rust bit-for-bit, NOT `F.softplus` which has a `threshold=20` branch; manual `_expm1_over_x` via `torch.where` for autograd compatibility; state contract `forward(x, h) -> (y, h_new)` uniform with GRU/LSTM/Transformer), `MambaSpec` pydantic schema with `model_validator(mode='after')` resolving `dt_rank` from `input_size`, `LayerSpec` discriminated union extended. `build_layer(MambaSpec)` + `load_policy_from_json` both raise `NotImplementedError` with pointers to the Phase 4a spec (PPO gate). `_mamba_specs` ParamSpec generator: Xavier on `x_proj_w`, Xavier * `dt_rank^{-0.5}` on `dt_proj_w`, inv_softplus(U(1e-3, 1e-1)) centers on `dt_proj_b` (shared per-channel via `_MAMBA_DT_BIAS_SEED`), HiPPO `log(n+1)` centers on `a_log`, 1.0 centers on `d_skip`. `_init_mamba_layer` writes population with per-individual `N(0, 0.01*bound_multiplier)` jitter around the shared centers -- prevents PSO init collapse on the non-zero-centered slices (load-bearing; mirrors Phase 2a LSTM forget-bias).
- **Training**: `configs/training/msr_aller_mamba_pso_train.toml` -- Dense(23 -> 32, swish) -> Mamba(d_inner=32, d_state=16, dt_rank=2) -> Mamba(32, 16, 2) -> Dense(32 -> 2, asinh), 4290 trainable params, PSO `n_pop=64 n_gen=2000 seed_strategy="adaptive"`. Registered as `neural_network_mamba_pso` in `compare_guidance.SCHEMES` + `_NN_DEPLOY_SCHEMES`; `train_all.sh` aliases `mamba_pso` / `nn_mamba_pso` / `mamba`.
- **Gates**: cross-language Mamba equivalence (100-step sequence through `nn_forward_sequence`, Dense(4->8,tanh) -> Mamba(8, 4, 2) -> Dense(8->2, linear), max abs diff target 1e-14, expected actual ~1e-16), warm-up test (state starts at zero and evolves deterministically, no step-0-vs-step-1 collapse), PSO smoke test on reduced ~338-param arch (2 gens @slow), PPO-rejection test (@fast, `build_layer` + `load_policy_from_json` both raise). All 10 Rust guidance golden regressions bit-identical.

Full spec: `docs/superpowers/specs/2026-04-24-phase-4a-mamba-ssm-mvp-design.md`. Plan: `docs/superpowers/plans/2026-04-24-phase-4a-mamba-ssm-mvp-plan.md`.
```

- [ ] **Step 2: Update the "Extensibility" paragraph to mention Mamba's role**

Find the "post-Phase-2b contract" / extensibility paragraph in `CLAUDE.md` (search for "Extensibility (post-Phase"). Add a note that Mamba establishes the **first 2D single-tensor state** layer (distinct from GRU's flat 1D state, LSTM's tuple of two 1D states, Transformer's paired VecDeque K/V caches, and Window's single-VecDeque). State:

> **2D single-tensor state** (Phase 4a Mamba precedent) is the third state-shape pattern: a single `DMatrix<f64>` of shape `(d_inner, d_state)`. For Phase 4b (Mamba PPO), the rollout buffer stores it as `(T, B, d_inner, d_state)` with `ndim == 4`, requiring new dispatch in `ppo_update_bptt`, `hidden_shapes`, and `_np_state_to_torch` / `_torch_state_to_np`. The existing `_zero_state_where_done` tensor-branch handles single-tensor states regardless of dimensionality, so Mamba's 2D state slots in cleanly when unboxed.

- [ ] **Step 3: Update the **Testing** count**

In the `**Testing**` bullet (Python + Rust), bump the approximate test counts to reflect the new tests added:
- Python: `tests/test_python_mamba_layer.py` (4 tests), `tests/test_mamba_encoding.py` (7), `tests/test_init_v2_mamba.py` (5), `tests/test_export_v2_mamba.py` (2), `tests/test_mamba_ppo_rejection.py` (6). Total: 24 new fast tests + 3 slow tests.
- Rust: softplus (2 tests), expm1_over_x (3), mamba_forward (2 + proptest), mamba_flat_roundtrip (1 + proptest), toml parsing (5). Total: 13 new tests + 2 proptests.

Update the text accordingly (don't fixate on exact numbers -- the existing paragraph uses approximate counts).

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude): Phase 4a Mamba SSM MVP paragraph + 2D-state extensibility note"
```

---

## Task 18: Final validation and smart-commit

**Files:**
- Full branch review

- [ ] **Step 1: Run the full Rust test suite**

```bash
cargo test --manifest-path src/rust/Cargo.toml --release --lib
```

Expected: 100% pass.

- [ ] **Step 2: Run the full Python test suite**

```bash
uv run pytest tests/ --tb=short
```

Expected: 100% pass (aside from any tests already marked `xfail` or `skip`).

- [ ] **Step 3: Run Rust fmt + clippy**

```bash
cargo fmt --manifest-path src/rust/Cargo.toml --check
cargo clippy --manifest-path src/rust/Cargo.toml --lib -- -D warnings
```

Expected: zero output (clean).

- [ ] **Step 4: Run Python lint**

```bash
./lint_code.sh
```

Expected: zero ruff + mypy errors.

- [ ] **Step 5: Check all guidance golden regressions**

```bash
cargo test --manifest-path src/rust/Cargo.toml --release guidance_golden
```

Expected: all 10 pass bit-identically.

- [ ] **Step 6: Invoke the smart-commit skill to finalize**

Invoke the `smart-commit` skill with the instruction to review the entire `feature/mamba-ssm-mvp` branch, update `CLAUDE.md` and `README.md` with any remaining doc drift, and commit any straggler changes. This is per the user's global CLAUDE.md requirement that implementation plans always end with `smart-commit`.

Tell smart-commit: "Review the entire feature/mamba-ssm-mvp branch against the spec (docs/superpowers/specs/2026-04-24-phase-4a-mamba-ssm-mvp-design.md) and the plan (docs/superpowers/plans/2026-04-24-phase-4a-mamba-ssm-mvp-plan.md). Sync CLAUDE.md + README.md with any drift, commit any straggler files, and produce a clean final state suitable for PR review."

---

## Completion Criteria

All of the following must be true before declaring Phase 4a landed:

1. All 18 tasks above have their boxes checked.
2. CI is green (`rust`, `python`, `python-pyo3` jobs all pass on `feature/mamba-ssm-mvp`).
3. All 10 Rust guidance golden regressions bit-identical vs main.
4. `uv run python -m aerocapture.training.compare_guidance --schemes neural_network_transformer_pso neural_network_mamba_pso --n-sims 500` produces finite cost and three-way classification output.
5. `./train_all.sh mamba` runs end-to-end (e.g. `n_gen=5` override) producing `best_model.json`, `report.pdf`, `final_eval.parquet`.
6. `CLAUDE.md` + `TODO.md` reflect the shipped state; Phase 4a box checked; Phase 4b box present.

Post-merge, Phase 4b (Mamba PPO-BPTT) is unblocked and well-defined.
