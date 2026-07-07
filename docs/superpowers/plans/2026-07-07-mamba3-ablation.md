# Mamba-3 Ablation Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a PSO-only `Mamba3` NN layer with two orthogonal recurrence flags (exponential-trapezoidal discretization, complex-diagonal rotational state) plus a 2x2 ablation experiment that reports tail DV with σ_run error bars.

**Architecture:** A new `Mamba3Layer` clones the proven `MambaLayer` scaffolding across both languages and adds two independent modes. `discretization ∈ {euler, trapezoidal}`, `state_mode ∈ {real, complex}`. `euler`+`real` is defined bit-identical to `MambaLayer`. PSO-only: the Python torch mirror exists solely for the cross-language equivalence gate; `build_layer`/`load_policy_from_json` raise `NotImplementedError` (same gate as Mamba). Complex arithmetic uses explicit (re, im) real ops on both sides to preserve bit-identity.

**Tech Stack:** Rust (edition 2024, nalgebra), PyO3/maturin, Python 3.14 (torch mirror, pymoo PSO), pytest, cargo test.

## Global Constraints

- Never commit to `main`. Work stays on `feature/mamba3-ablation`.
- Real-mode recurrence MUST reuse `helpers::expm1_over_x` (with `f64::exp_m1`), NOT the new complex helper — this is what makes `real`+`euler` bit-identical to `MambaLayer`.
- Complex arithmetic: explicit `(re, im)` scalars on both sides. Never a native complex dtype (breaks cross-language bit-identity).
- `x_proj` shape stays `(dt_rank + 2*d_state, input_size)` in all modes (`B`, `C` stay real).
- Flat-weight order (conditional): `x_proj_w → dt_proj_w → dt_proj_b → a_log → [a_imag if complex] → [lambda_logit if trapezoidal] → d_skip`.
- `n_params = input_size·(3·d_state + 2·dt_rank + 2) + [complex: input_size·d_state] + [trapz: input_size]`.
- PyO3 rebuild is ALWAYS from repo root: `uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml`.
- Rust: `./check_all.sh` (fmt + clippy + test + release) must pass. Python: `uv run ruff check`, `uv run ruff format`, `uv run mypy`, `uv run pytest tests` must pass.
- `MAMBA3_EVAL_SEED_OFFSET = 10_000_000` (1M-9M already reserved).

---

## Task 1: Rust `expm1_over_x_complex` helper

**Files:**
- Modify: `src/rust/src/data/neural/layers/mamba3.rs` (new file — created here, extended in Task 2)
- Test: inline `#[cfg(test)]` in the same file

**Interfaces:**
- Produces: `pub(super) fn expm1_over_x_complex(zr: f64, zi: f64) -> (f64, f64)` — complex `(exp(z)-1)/z`, explicit (re,im), Taylor fallback for `|z| < 1e-8`.

- [ ] **Step 1: Create the file with the helper and a failing test**

Create `src/rust/src/data/neural/layers/mamba3.rs`:

```rust
//! Mamba-3 ablation layer: euler|trapezoidal x real|complex. PSO-only spike.

/// Complex `(exp(z) - 1) / z` with Taylor fallback for |z| < 1e-8.
/// z = zr + i·zi. Returns (re, im). Explicit real arithmetic for
/// cross-language bit-identity with the Python mirror.
pub(super) fn expm1_over_x_complex(zr: f64, zi: f64) -> (f64, f64) {
    let mag = (zr * zr + zi * zi).sqrt();
    if mag < 1e-8 {
        // Taylor 1 + z/2 + z^2/6; z^2 = (zr^2 - zi^2) + i(2 zr zi)
        let z2r = zr * zr - zi * zi;
        let z2i = 2.0 * zr * zi;
        (1.0 + 0.5 * zr + z2r / 6.0, 0.5 * zi + z2i / 6.0)
    } else {
        // exp(z) = e^zr (cos zi + i sin zi)
        let er = zr.exp();
        let ez_r = er * zi.cos();
        let ez_i = er * zi.sin();
        let num_r = ez_r - 1.0;
        let num_i = ez_i;
        // (num) / (zr + i zi) = num·conj(z) / |z|^2
        let denom = zr * zr + zi * zi;
        (
            (num_r * zr + num_i * zi) / denom,
            (num_i * zr - num_r * zi) / denom,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_reduces_to_real_on_real_axis() {
        // On the real axis (zi=0), Re matches the real (exp(z)-1)/z form; Im ~ 0.
        for zr in [-2.0, -0.5, 0.3, 1.5] {
            let (re, im) = expm1_over_x_complex(zr, 0.0);
            let expected = (zr.exp() - 1.0) / zr;
            assert!((re - expected).abs() < 1e-12, "zr={zr}");
            assert!(im.abs() < 1e-15, "zr={zr}");
        }
    }

    #[test]
    fn complex_taylor_branch_finite_at_zero() {
        let (re, im) = expm1_over_x_complex(0.0, 0.0);
        assert!((re - 1.0).abs() < 1e-15);
        assert!(im.abs() < 1e-15);
    }
}
```

- [ ] **Step 2: Register the module and run the test to verify it compiles + passes**

Add `pub mod mamba3;` to `src/rust/src/data/neural/layers/mod.rs` (alongside the existing `pub mod mamba;`).

Run: `cd src/rust && cargo test -p aerocapture mamba3::tests`
Expected: 2 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add src/rust/src/data/neural/layers/mamba3.rs src/rust/src/data/neural/layers/mod.rs
git commit -m "feat(mamba3): complex expm1_over_x helper"
```

---

## Task 2: `Mamba3Layer` struct + `LayerWeights` flat round-trip

**Files:**
- Modify: `src/rust/src/data/neural/layers/mamba3.rs`
- Test: inline

**Interfaces:**
- Consumes: `expm1_over_x_complex` (Task 1), `super::super::LayerWeights` trait.
- Produces: `pub struct Mamba3Layer { input_size, d_state, dt_rank, trapezoidal: bool, complex: bool, x_proj_w, dt_proj_w, dt_proj_b, a_log, a_imag: Option<DMatrix>, lambda_logit: Option<DVector>, d_skip }`; `impl LayerWeights for Mamba3Layer`.

- [ ] **Step 1: Write the struct + LayerWeights impl**

Prepend to `mamba3.rs` (above the helper is fine; keep the helper). Mirror `MambaLayer` fields, adding `trapezoidal`/`complex` flags and the two conditional weight slabs. Use `nalgebra::{DMatrix, DVector}`.

```rust
use super::super::LayerWeights;
use super::helpers::expm1_over_x; // real path reuses the existing helper (bit-identity anchor)

#[derive(Debug, Clone)]
pub struct Mamba3Layer {
    pub input_size: usize,
    pub d_state: usize,
    pub dt_rank: usize,
    pub trapezoidal: bool,
    pub complex: bool,
    pub x_proj_w: nalgebra::DMatrix<f64>,   // (dt_rank + 2*d_state, input_size)
    pub dt_proj_w: nalgebra::DMatrix<f64>,  // (input_size, dt_rank)
    pub dt_proj_b: nalgebra::DVector<f64>,  // (input_size,)
    pub a_log: nalgebra::DMatrix<f64>,      // (input_size, d_state)
    pub a_imag: Option<nalgebra::DMatrix<f64>>,   // (input_size, d_state) iff complex
    pub lambda_logit: Option<nalgebra::DVector<f64>>, // (input_size,) iff trapezoidal
    pub d_skip: nalgebra::DVector<f64>,     // (input_size,)
}

impl LayerWeights for Mamba3Layer {
    fn n_params(&self) -> usize {
        let base = self.input_size * (3 * self.d_state + 2 * self.dt_rank + 2);
        base + if self.complex { self.input_size * self.d_state } else { 0 }
            + if self.trapezoidal { self.input_size } else { 0 }
    }

    fn to_flat(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n_params());
        let push_mat = |out: &mut Vec<f64>, m: &nalgebra::DMatrix<f64>| {
            for i in 0..m.nrows() {
                for j in 0..m.ncols() {
                    out.push(m[(i, j)]);
                }
            }
        };
        push_mat(&mut out, &self.x_proj_w);
        push_mat(&mut out, &self.dt_proj_w);
        out.extend(self.dt_proj_b.iter().copied());
        push_mat(&mut out, &self.a_log);
        if let Some(ai) = &self.a_imag {
            push_mat(&mut out, ai);
        }
        if let Some(ll) = &self.lambda_logit {
            out.extend(ll.iter().copied());
        }
        out.extend(self.d_skip.iter().copied());
        out
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_flat(&mut self, flat: &[f64]) -> usize {
        let mut c = 0;
        let xr = self.dt_rank + 2 * self.d_state;
        self.x_proj_w = nalgebra::DMatrix::from_row_slice(xr, self.input_size, &flat[c..c + xr * self.input_size]);
        c += xr * self.input_size;
        self.dt_proj_w = nalgebra::DMatrix::from_row_slice(self.input_size, self.dt_rank, &flat[c..c + self.input_size * self.dt_rank]);
        c += self.input_size * self.dt_rank;
        self.dt_proj_b = nalgebra::DVector::from_row_slice(&flat[c..c + self.input_size]);
        c += self.input_size;
        self.a_log = nalgebra::DMatrix::from_row_slice(self.input_size, self.d_state, &flat[c..c + self.input_size * self.d_state]);
        c += self.input_size * self.d_state;
        if self.complex {
            self.a_imag = Some(nalgebra::DMatrix::from_row_slice(self.input_size, self.d_state, &flat[c..c + self.input_size * self.d_state]));
            c += self.input_size * self.d_state;
        } else {
            self.a_imag = None;
        }
        if self.trapezoidal {
            self.lambda_logit = Some(nalgebra::DVector::from_row_slice(&flat[c..c + self.input_size]));
            c += self.input_size;
        } else {
            self.lambda_logit = None;
        }
        self.d_skip = nalgebra::DVector::from_row_slice(&flat[c..c + self.input_size]);
        c += self.input_size;
        c
    }
}

impl Mamba3Layer {
    /// Zero-weight constructor for the given shape + flags (weights filled by `from_flat`).
    pub fn zeros(input_size: usize, d_state: usize, dt_rank: usize, trapezoidal: bool, complex: bool) -> Self {
        Self {
            input_size, d_state, dt_rank, trapezoidal, complex,
            x_proj_w: nalgebra::DMatrix::zeros(dt_rank + 2 * d_state, input_size),
            dt_proj_w: nalgebra::DMatrix::zeros(input_size, dt_rank),
            dt_proj_b: nalgebra::DVector::zeros(input_size),
            a_log: nalgebra::DMatrix::zeros(input_size, d_state),
            a_imag: if complex { Some(nalgebra::DMatrix::zeros(input_size, d_state)) } else { None },
            lambda_logit: if trapezoidal { Some(nalgebra::DVector::zeros(input_size)) } else { None },
            d_skip: nalgebra::DVector::zeros(input_size),
        }
    }
}
```

Silence the `expm1_over_x` unused-import warning for now with `#[allow(unused_imports)]` on the `use` line; Task 3 consumes it.

- [ ] **Step 2: Write the round-trip test (add to the `tests` module)**

```rust
    #[test]
    fn flat_roundtrip_all_flag_combos() {
        for &(trap, cplx) in &[(false, false), (true, false), (false, true), (true, true)] {
            let mut m = Mamba3Layer::zeros(4, 3, 2, trap, cplx);
            let n = m.n_params();
            let slab: Vec<f64> = (0..n).map(|i| 0.01 * (i as f64 + 1.0)).collect();
            let consumed = m.from_flat(&slab);
            assert_eq!(consumed, n, "trap={trap} cplx={cplx}");
            assert_eq!(m.to_flat(), slab, "trap={trap} cplx={cplx}");
        }
    }
```

- [ ] **Step 3: Run the tests**

Run: `cd src/rust && cargo test -p aerocapture mamba3::tests`
Expected: 3 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/data/neural/layers/mamba3.rs
git commit -m "feat(mamba3): Mamba3Layer struct + conditional flat round-trip"
```

---

## Task 3: `Mamba3Layer::forward` (all 4 modes) + correctness anchors

**Files:**
- Modify: `src/rust/src/data/neural/layers/mamba3.rs`
- Test: inline (real-euler-≡-mamba anchor, trapz→euler, complex warmup)

**Interfaces:**
- Consumes: `MambaLayer` (for the anchor test; import `super::MambaLayer`).
- Produces: `pub fn forward(&self, x: &[f64], h_re: &mut DMatrix, h_im: &mut DMatrix, x_prev: &mut DVector, b_prev: &mut DVector) -> Vec<f64>`. State args are all `(input_size, d_state)` / `(input_size,)` / `(d_state,)`; unused slabs stay zero.

- [ ] **Step 1: Write forward**

Add to `impl Mamba3Layer`. Compute `Δ`, `B`, `C` exactly as `MambaLayer::forward`, then branch on `complex`; within each, branch on `trapezoidal`. Real path uses `expm1_over_x`; complex uses `expm1_over_x_complex`.

```rust
    pub fn forward(
        &self,
        x: &[f64],
        h_re: &mut nalgebra::DMatrix<f64>,
        h_im: &mut nalgebra::DMatrix<f64>,
        x_prev: &mut nalgebra::DVector<f64>,
        b_prev: &mut nalgebra::DVector<f64>,
    ) -> Vec<f64> {
        let x_vec = nalgebra::DVector::from_row_slice(x);
        let proj = &self.x_proj_w * &x_vec;
        let dt_pre: Vec<f64> = (0..self.dt_rank).map(|i| proj[i]).collect();
        let b_vec: Vec<f64> = (0..self.d_state).map(|i| proj[self.dt_rank + i]).collect();
        let c_vec: Vec<f64> = (0..self.d_state).map(|i| proj[self.dt_rank + self.d_state + i]).collect();

        let dt_pre_v = nalgebra::DVector::from_row_slice(&dt_pre);
        let dt_lifted = &self.dt_proj_w * &dt_pre_v + &self.dt_proj_b;
        let delta: Vec<f64> = (0..self.input_size).map(|i| super::helpers::softplus(dt_lifted[i])).collect();

        let mut y = vec![0.0_f64; self.input_size];
        for d in 0..self.input_size {
            let dd = delta[d];
            let xd = x[d];
            let lam = self.lambda_logit.as_ref().map_or(1.0, |ll| 1.0 / (1.0 + (-ll[d]).exp()));
            let xp = x_prev[d];
            let mut acc = 0.0;
            for n in 0..self.d_state {
                let ar = -self.a_log[(d, n)].exp();
                let za_r = dd * ar;
                if self.complex {
                    let ai = self.a_imag.as_ref().unwrap()[(d, n)];
                    let za_i = dd * ai;
                    let r = za_r.exp();
                    let (alpha_r, alpha_i) = (r * za_i.cos(), r * za_i.sin());
                    let (ex_r, ex_i) = expm1_over_x_complex(za_r, za_i);
                    // current-input drive: b_bar = Δ·B[n]·expm1_over_x_complex(za)
                    let bb_r = dd * b_vec[n] * ex_r;
                    let bb_i = dd * b_vec[n] * ex_i;
                    // state update: h = α·h + λ·bb·x + (1-λ)·Δ·α·B_prev[n]·x_prev  (trapz cross term)
                    let hr = h_re[(d, n)];
                    let hi = h_im[(d, n)];
                    let mut nr = alpha_r * hr - alpha_i * hi + lam * bb_r * xd;
                    let mut ni = alpha_r * hi + alpha_i * hr + lam * bb_i * xd;
                    if self.trapezoidal {
                        let cross = (1.0 - lam) * dd * b_prev[n] * xp;
                        nr += alpha_r * cross;
                        ni += alpha_i * cross;
                    }
                    h_re[(d, n)] = nr;
                    h_im[(d, n)] = ni;
                    acc += nr * c_vec[n]; // readout reads Re(h)
                } else {
                    let alpha = za_r.exp();
                    let bb = dd * b_vec[n] * expm1_over_x(za_r);
                    let mut nr = alpha * h_re[(d, n)] + lam * bb * xd;
                    if self.trapezoidal {
                        nr += (1.0 - lam) * dd * alpha * b_prev[n] * xp;
                    }
                    h_re[(d, n)] = nr;
                    acc += nr * c_vec[n];
                }
            }
            y[d] = acc + self.d_skip[d] * xd;
        }
        if self.trapezoidal {
            *x_prev = x_vec;
            *b_prev = nalgebra::DVector::from_row_slice(&b_vec);
        }
        y
    }
```

Remove the `#[allow(unused_imports)]` added in Task 2 now that `expm1_over_x` is used.

- [ ] **Step 2: Write the anchor + reduction + warmup tests**

```rust
    fn mamba_ref(input_size: usize, d_state: usize, dt_rank: usize) -> (super::MambaLayer, Mamba3Layer, Vec<f64>) {
        let mut m3 = Mamba3Layer::zeros(input_size, d_state, dt_rank, false, false);
        let n = m3.n_params();
        let slab: Vec<f64> = (0..n).map(|i| 0.05 * ((i % 7) as f64 - 3.0)).collect();
        m3.from_flat(&slab);
        // MambaLayer has the identical real+euler layout; load the same slab.
        let mut m = super::MambaLayer {
            input_size, d_state, dt_rank,
            x_proj_w: nalgebra::DMatrix::zeros(dt_rank + 2 * d_state, input_size),
            dt_proj_w: nalgebra::DMatrix::zeros(input_size, dt_rank),
            dt_proj_b: nalgebra::DVector::zeros(input_size),
            a_log: nalgebra::DMatrix::zeros(input_size, d_state),
            d_skip: nalgebra::DVector::zeros(input_size),
        };
        m.from_flat(&slab);
        (m, m3, slab)
    }

    #[test]
    fn real_euler_bit_identical_to_mamba() {
        let (m, m3, _) = mamba_ref(4, 3, 2);
        let mut h = nalgebra::DMatrix::zeros(4, 3);
        let mut hr = nalgebra::DMatrix::zeros(4, 3);
        let mut hi = nalgebra::DMatrix::zeros(4, 3);
        let mut xp = nalgebra::DVector::zeros(4);
        let mut bp = nalgebra::DVector::zeros(3);
        for t in 0..20 {
            let x: Vec<f64> = (0..4).map(|d| 0.1 * (d as f64 + 1.0) * (t as f64 + 1.0).sin()).collect();
            let ym = m.forward(&x, &mut h);
            let y3 = m3.forward(&x, &mut hr, &mut hi, &mut xp, &mut bp);
            for d in 0..4 {
                assert_eq!(ym[d], y3[d], "t={t} d={d}"); // BIT-identical
            }
        }
    }

    #[test]
    fn trapezoidal_reduces_to_euler_at_high_lambda() {
        let mut euler = Mamba3Layer::zeros(4, 3, 2, false, false);
        let n_e = euler.n_params();
        let slab: Vec<f64> = (0..n_e).map(|i| 0.05 * ((i % 5) as f64 - 2.0)).collect();
        euler.from_flat(&slab);
        // trapezoidal layout = euler + lambda_logit(input_size) inserted before d_skip.
        let mut trap = Mamba3Layer::zeros(4, 3, 2, true, false);
        let split = n_e - 4; // everything up to (but not incl.) d_skip
        let mut tslab = slab[..split].to_vec();
        tslab.extend(std::iter::repeat(30.0).take(4)); // lambda_logit -> sigmoid ~ 1
        tslab.extend(&slab[split..]);                  // d_skip
        trap.from_flat(&tslab);
        let mut he = nalgebra::DMatrix::zeros(4, 3);
        let (mut hr, mut hi) = (nalgebra::DMatrix::zeros(4, 3), nalgebra::DMatrix::zeros(4, 3));
        let mut xp = nalgebra::DVector::zeros(4);
        let mut bp = nalgebra::DVector::zeros(3);
        let mut he0 = nalgebra::DMatrix::zeros(4, 3);
        let (mut z1, mut z2) = (nalgebra::DVector::zeros(4), nalgebra::DVector::zeros(3));
        for t in 0..15 {
            let x: Vec<f64> = (0..4).map(|d| 0.2 * (d as f64 - 1.0) * (t as f64).cos()).collect();
            let ye = euler.forward(&x, &mut he, &mut he0, &mut z1, &mut z2);
            let yt = trap.forward(&x, &mut hr, &mut hi, &mut xp, &mut bp);
            for d in 0..4 {
                assert!((ye[d] - yt[d]).abs() < 1e-12, "t={t} d={d} {} vs {}", ye[d], yt[d]);
            }
        }
    }

    #[test]
    fn complex_warmup_deterministic() {
        let mut m = Mamba3Layer::zeros(3, 4, 1, false, true);
        let n = m.n_params();
        let slab: Vec<f64> = (0..n).map(|i| 0.03 * ((i % 9) as f64 - 4.0)).collect();
        m.from_flat(&slab);
        let run = || {
            let (mut hr, mut hi) = (nalgebra::DMatrix::zeros(3, 4), nalgebra::DMatrix::zeros(3, 4));
            let (mut xp, mut bp) = (nalgebra::DVector::zeros(3), nalgebra::DVector::zeros(4));
            let mut last = vec![];
            for t in 0..10 {
                let x: Vec<f64> = (0..3).map(|d| 0.15 * (d as f64 + t as f64).sin()).collect();
                last = m.forward(&x, &mut hr, &mut hi, &mut xp, &mut bp);
            }
            last
        };
        assert_eq!(run(), run());
        assert!(run().iter().all(|v| v.is_finite()));
    }
```

- [ ] **Step 3: Run**

Run: `cd src/rust && cargo test -p aerocapture mamba3::tests`
Expected: 6 tests PASS (esp. `real_euler_bit_identical_to_mamba`).

- [ ] **Step 4: Commit**

```bash
git add src/rust/src/data/neural/layers/mamba3.rs
git commit -m "feat(mamba3): forward (4 modes) + real-euler bit-identity anchor"
```

---

## Task 4: Rust enum wiring — `LayerSpec` / `Layer` / `LayerState`

**Files:**
- Modify: `src/rust/src/data/neural/mod.rs` (LayerSpec enum ~line 477; Layer enum ~296; `LayerSpec::io` ~508; `Layer` LayerWeights dispatch arms ~355-385; export at line 11)
- Modify: `src/rust/src/data/nn_state.rs` (LayerState enum ~12; `for_layer` ~39; `reset` ~68)
- Test: inline in nn_state.rs (for_layer shape)

**Interfaces:**
- Consumes: `Mamba3Layer` (Task 1-3).
- Produces: `LayerSpec::Mamba3 { input_size, d_state, dt_rank, trapezoidal: bool, complex: bool }`; `Layer::Mamba3(Box<Mamba3Layer>)`; `LayerState::Mamba3 { h_re, h_im, x_prev, b_prev }`.

- [ ] **Step 1: Export the type**

`src/rust/src/data/neural/mod.rs:11` — add `Mamba3Layer` to the `pub use layers::{...}` list.

- [ ] **Step 2: Add `LayerSpec::Mamba3`**

In the `LayerSpec` enum (`#[serde(tag = "type")]`, ~line 477), mirror the `Mamba` variant. Serde tag = `"mamba3"`:

```rust
    #[serde(rename = "mamba3")]
    Mamba3 {
        input_size: usize,
        d_state: usize,
        dt_rank: usize,
        #[serde(default)]
        trapezoidal: bool,
        #[serde(default)]
        complex: bool,
    },
```

In `LayerSpec::io` (~508), add: `LayerSpec::Mamba3 { input_size, .. } => (*input_size, *input_size, "mamba3"),`

- [ ] **Step 3: Add `Layer::Mamba3` + dispatch arms**

In `enum Layer` (~296), after `Mamba(Box<MambaLayer>)`, add `Mamba3(Box<Mamba3Layer>),`.

In the four `Layer` match sites (io/input_size ~325, to_flat ~359, from_flat ~371, n_params ~382), add a `Layer::Mamba3(m) => ...` arm identical in shape to the `Layer::Mamba(m)` arm directly above each (`m.input_size` / `m.to_flat()` / `m.from_flat(flat)` / `m.n_params()`).

- [ ] **Step 4: Add `LayerState::Mamba3`**

`nn_state.rs` LayerState enum (~12):

```rust
    /// Mamba-3 state: complex (h_re, h_im) + trapezoidal previous-input carry.
    Mamba3 {
        h_re: nalgebra::DMatrix<f64>,
        h_im: nalgebra::DMatrix<f64>,
        x_prev: nalgebra::DVector<f64>,
        b_prev: nalgebra::DVector<f64>,
    },
```

`for_layer` (~39) — add:

```rust
    Layer::Mamba3(m) => LayerState::Mamba3 {
        h_re: nalgebra::DMatrix::zeros(m.input_size, m.d_state),
        h_im: nalgebra::DMatrix::zeros(m.input_size, m.d_state),
        x_prev: nalgebra::DVector::zeros(m.input_size),
        b_prev: nalgebra::DVector::zeros(m.d_state),
    },
```

`reset` (~68) — add: `LayerState::Mamba3 { h_re, h_im, x_prev, b_prev } => { h_re.fill(0.0); h_im.fill(0.0); x_prev.fill(0.0); b_prev.fill(0.0); }`

- [ ] **Step 5: Wire `forward` dispatch in `NeuralNetModel::forward`**

`neural/mod.rs` forward match (~1543, next to the `(Layer::Mamba(m), LayerState::Mamba { h })` arm ~1574):

```rust
    (Layer::Mamba3(m), LayerState::Mamba3 { h_re, h_im, x_prev, b_prev }) => {
        current = m.forward(&current, h_re, h_im, x_prev, b_prev);
    }
```

- [ ] **Step 6: Build + for_layer shape test**

Add to nn_state.rs tests:

```rust
    #[test]
    fn layer_state_mamba3_for_layer_shapes() {
        let layer = crate::data::neural::Layer::Mamba3(Box::new(
            crate::data::neural::Mamba3Layer::zeros(5, 4, 1, true, true),
        ));
        match LayerState::for_layer(&layer) {
            LayerState::Mamba3 { h_re, h_im, x_prev, b_prev } => {
                assert_eq!(h_re.shape(), (5, 4));
                assert_eq!(h_im.shape(), (5, 4));
                assert_eq!(x_prev.len(), 5);
                assert_eq!(b_prev.len(), 4);
            }
            _ => panic!("expected Mamba3"),
        }
    }
```

Run: `cd src/rust && cargo build -p aerocapture && cargo test -p aerocapture mamba3 layer_state_mamba3`
Expected: builds, tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/rust/src/data/neural/mod.rs src/rust/src/data/nn_state.rs
git commit -m "feat(mamba3): LayerSpec/Layer/LayerState enum wiring + forward dispatch"
```

---

## Task 5: `TomlLayerSpec::Mamba3` + JSON round-trip

**Files:**
- Modify: `src/rust/src/config.rs` (TomlLayerSpec ~204, to_layer_spec ~237/306)
- Modify: `src/rust/src/data/neural/mod.rs` (`from_v2_json` ~1274, `save_json` ~1482, `from_flat_weights_v2` Mamba arm)
- Test: inline round-trip in neural/mod.rs

**Interfaces:**
- Consumes: `LayerSpec::Mamba3`, `Mamba3Layer`.
- Produces: TOML `type = "mamba3"` parsing with `discretization`/`state_mode` string flags; v2 JSON save/load for Mamba3.

- [ ] **Step 1: TomlLayerSpec::Mamba3**

`config.rs` TomlLayerSpec enum (~204), mirror `Mamba` (~228) but accept string flags:

```rust
    #[serde(rename = "mamba3")]
    Mamba3 {
        input_size: usize,
        d_state: usize,
        dt_rank: Option<usize>,
        #[serde(default = "default_discretization")]
        discretization: String,
        #[serde(default = "default_state_mode")]
        state_mode: String,
    },
```

Add free fns near the enum: `fn default_discretization() -> String { "euler".into() }`, `fn default_state_mode() -> String { "real".into() }`.

`to_layer_spec` (~306, after the `Mamba` arm) — validate and map:

```rust
    TomlLayerSpec::Mamba3 { input_size, d_state, dt_rank, discretization, state_mode } => {
        if *input_size == 0 { return Err(ParseError("Mamba3: input_size must be > 0".into())); }
        if *d_state == 0 { return Err(ParseError("Mamba3: d_state must be > 0".into())); }
        let resolved = dt_rank.unwrap_or_else(|| (*input_size / 16).max(1));
        if resolved == 0 || resolved > *input_size {
            return Err(ParseError(format!("Mamba3: dt_rank ({resolved}) must be in 1..={input_size}")));
        }
        let trapezoidal = match discretization.as_str() {
            "euler" => false, "trapezoidal" => true,
            other => return Err(ParseError(format!("Mamba3: discretization must be euler|trapezoidal, got {other}"))),
        };
        let complex = match state_mode.as_str() {
            "real" => false, "complex" => true,
            other => return Err(ParseError(format!("Mamba3: state_mode must be real|complex, got {other}"))),
        };
        Ok(LayerSpec::Mamba3 { input_size: *input_size, d_state: *d_state, dt_rank: resolved, trapezoidal, complex })
    }
```

- [ ] **Step 2: `from_v2_json` Mamba3 arm**

`neural/mod.rs` `from_v2_json` (~1274, the Mamba arm). Add a `"mamba3"` case that reads `input_size`/`d_state`/`dt_rank`/`trapezoidal`/`complex` from the layer JSON, pushes `LayerSpec::Mamba3`, and constructs `Layer::Mamba3(Box::new({ let mut m = Mamba3Layer::zeros(...); m.from_flat(&flat_slab); m }))` from the layer's flat weights (mirror how the Mamba arm at ~1391 builds `Layer::Mamba`). The weights dict for a Mamba3 layer is a single flat array under the same key convention the Mamba arm uses.

- [ ] **Step 3: `save_json` Mamba3 arm**

`save_json` (~1482, the `Layer::Mamba(m)` arm). Add `Layer::Mamba3(m) => { ... }` writing `type: "mamba3"`, `input_size`, `d_state`, `dt_rank`, `trapezoidal`, `complex`, and the flat weight array (via `m.to_flat()`), mirroring the Mamba arm's JSON object shape.

- [ ] **Step 4: `from_flat_weights_v2` Mamba3 arm**

In `from_flat_weights_v2` (the `LayerSpec::Mamba { .. }` arm), add `LayerSpec::Mamba3 { input_size, d_state, dt_rank, trapezoidal, complex } => { let mut m = Mamba3Layer::zeros(*input_size, *d_state, *dt_rank, *trapezoidal, *complex); let used = m.from_flat(&flat[cursor..]); cursor += used; layers.push(Layer::Mamba3(Box::new(m))); }`.

- [ ] **Step 5: Rust JSON round-trip test**

Add to neural/mod.rs tests (mirror the Mamba round-trip test):

```rust
    #[test]
    fn mamba3_json_roundtrip_all_flags() {
        for &(disc, sm) in &[("euler","real"),("trapezoidal","real"),("euler","complex"),("trapezoidal","complex")] {
            let arch_json = format!(r#"{{"format_version":2,"architecture":[
                {{"type":"dense","input_size":4,"output_size":3,"activation":"tanh"}},
                {{"type":"mamba3","input_size":3,"d_state":2,"dt_rank":1,"trapezoidal":{},"complex":{}}},
                {{"type":"dense","input_size":3,"output_size":2,"activation":"linear"}}
            ]}}"#, disc=="trapezoidal", sm=="complex");
            // Build via from_flat_weights_v2 from a known slab, save_json to a temp, reload, compare flats.
            // (Use the same helper the Mamba round-trip test uses; assert to_flat() equal pre/post reload.)
        }
    }
```

Fill the body by copying the Mamba round-trip test's save/reload mechanics. Assert `model.layers[1]` flat weights survive save->load bit-identically for all 4 flag combos.

- [ ] **Step 6: Run**

Run: `cd src/rust && cargo test -p aerocapture mamba3`
Expected: all Mamba3 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/rust/src/config.rs src/rust/src/data/neural/mod.rs
git commit -m "feat(mamba3): TOML parser + v2 JSON save/load round-trip"
```

---

## Task 6: Python `Mamba3Layer` torch mirror

**Files:**
- Create: `src/python/aerocapture/training/rl/layers/mamba3.py`
- Test: `tests/test_python_mamba3_layer.py`

**Interfaces:**
- Produces: `Mamba3Layer(nn.Module)` with `__init__(input_size, d_state, dt_rank, trapezoidal, complex)`, `forward_unbatched(x, state) -> (y, state)` where `state = (h_re, h_im, x_prev, b_prev)` tuple of tensors, `new_state()`, `to_flat()`, `from_flat(slab)`.

- [ ] **Step 1: Write the mirror**

Mirror `mamba.py`'s helpers and structure, but for a spike we only need the UNBATCHED forward (the equivalence gate runs per-step; PPO is out of scope). Complex ops explicit. The `_softplus` helper is identical to mamba.py's. Add `_expm1_over_x_real` (mamba.py's `_expm1_over_x`) and `_expm1_over_x_complex` matching the Rust helper.

```python
"""Python torch mirror of the Rust Mamba3Layer (PSO-only spike).

Consumed ONLY by the cross-language equivalence test. PSO training goes through
the Rust runtime (flat_weights_to_json + nn_forward). build_layer raises for PPO.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import Tensor, nn


def _softplus(x: Tensor) -> Tensor:
    return x.clamp_min(0.0) + torch.log1p(torch.exp(-x.abs()))


def _expm1_over_x_real(z: Tensor) -> Tensor:
    taylor = 1.0 + 0.5 * z + (z * z) / 6.0
    safe = torch.where(z.abs() < 1e-8, torch.ones_like(z), z)
    exact = torch.expm1(z) / safe
    gate = (z.abs() >= 1e-8).to(z.dtype)
    return taylor + (exact - taylor) * gate


def _expm1_over_x_complex(zr: Tensor, zi: Tensor) -> tuple[Tensor, Tensor]:
    """Mirror of Rust expm1_over_x_complex: same (exp(z)-1)/z form, NOT torch.expm1."""
    mag = torch.sqrt(zr * zr + zi * zi)
    small = mag < 1e-8
    # Taylor
    z2r = zr * zr - zi * zi
    z2i = 2.0 * zr * zi
    t_r = 1.0 + 0.5 * zr + z2r / 6.0
    t_i = 0.5 * zi + z2i / 6.0
    # Exact
    er = torch.exp(zr)
    ez_r = er * torch.cos(zi)
    ez_i = er * torch.sin(zi)
    num_r = ez_r - 1.0
    num_i = ez_i
    denom = torch.where(small, torch.ones_like(zr), zr * zr + zi * zi)
    e_r = (num_r * zr + num_i * zi) / denom
    e_i = (num_i * zr - num_r * zi) / denom
    g = (~small).to(zr.dtype)
    return t_r + (e_r - t_r) * g, t_i + (e_i - t_i) * g


class Mamba3Layer(nn.Module):
    def __init__(self, input_size: int, d_state: int, dt_rank: int, trapezoidal: bool, complex: bool) -> None:
        super().__init__()
        self.input_size, self.d_state, self.dt_rank = input_size, d_state, dt_rank
        self.trapezoidal, self.complex = trapezoidal, complex
        self.x_proj_w = nn.Parameter(torch.zeros(dt_rank + 2 * d_state, input_size))
        self.dt_proj_w = nn.Parameter(torch.zeros(input_size, dt_rank))
        self.dt_proj_b = nn.Parameter(torch.zeros(input_size))
        self.a_log = nn.Parameter(torch.zeros(input_size, d_state))
        self.a_imag = nn.Parameter(torch.zeros(input_size, d_state)) if complex else None
        self.lambda_logit = nn.Parameter(torch.zeros(input_size)) if trapezoidal else None
        self.d_skip = nn.Parameter(torch.zeros(input_size))

    def new_state(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        dt = self.x_proj_w.dtype
        return (
            torch.zeros(self.input_size, self.d_state, dtype=dt),
            torch.zeros(self.input_size, self.d_state, dtype=dt),
            torch.zeros(self.input_size, dtype=dt),
            torch.zeros(self.d_state, dtype=dt),
        )

    def forward_unbatched(self, x: Tensor, state: tuple[Tensor, Tensor, Tensor, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:
        h_re, h_im, x_prev, b_prev = state
        proj = self.x_proj_w @ x
        dt_pre = proj[: self.dt_rank]
        b_vec = proj[self.dt_rank : self.dt_rank + self.d_state]
        c_vec = proj[self.dt_rank + self.d_state : self.dt_rank + 2 * self.d_state]
        delta = _softplus(self.dt_proj_w @ dt_pre + self.dt_proj_b)  # (input_size,)
        lam = torch.sigmoid(self.lambda_logit) if self.trapezoidal else torch.ones(self.input_size, dtype=x.dtype)

        ar = -torch.exp(self.a_log)                 # (in, N)
        za_r = delta.unsqueeze(1) * ar              # (in, N)
        if self.complex:
            za_i = delta.unsqueeze(1) * self.a_imag
            r = torch.exp(za_r)
            alpha_r, alpha_i = r * torch.cos(za_i), r * torch.sin(za_i)
            ex_r, ex_i = _expm1_over_x_complex(za_r, za_i)
            bb_r = delta.unsqueeze(1) * b_vec.unsqueeze(0) * ex_r
            bb_i = delta.unsqueeze(1) * b_vec.unsqueeze(0) * ex_i
            nr = alpha_r * h_re - alpha_i * h_im + lam.unsqueeze(1) * bb_r * x.unsqueeze(1)
            ni = alpha_r * h_im + alpha_i * h_re + lam.unsqueeze(1) * bb_i * x.unsqueeze(1)
            if self.trapezoidal:
                cross = (1.0 - lam).unsqueeze(1) * delta.unsqueeze(1) * b_prev.unsqueeze(0) * x_prev.unsqueeze(1)
                nr = nr + alpha_r * cross
                ni = ni + alpha_i * cross
            h_re, h_im = nr, ni
            y = (nr * c_vec.unsqueeze(0)).sum(dim=1) + self.d_skip * x
        else:
            alpha = torch.exp(za_r)
            bb = delta.unsqueeze(1) * b_vec.unsqueeze(0) * _expm1_over_x_real(za_r)
            nr = alpha * h_re + lam.unsqueeze(1) * bb * x.unsqueeze(1)
            if self.trapezoidal:
                nr = nr + (1.0 - lam).unsqueeze(1) * delta.unsqueeze(1) * alpha * b_prev.unsqueeze(0) * x_prev.unsqueeze(1)
            h_re = nr
            y = (nr * c_vec.unsqueeze(0)).sum(dim=1) + self.d_skip * x
        if self.trapezoidal:
            x_prev, b_prev = x.clone(), b_vec.detach().clone()
        return y, (h_re, h_im, x_prev, b_prev)

    def to_flat(self) -> np.ndarray:
        parts = [self.x_proj_w, self.dt_proj_w, self.dt_proj_b, self.a_log]
        if self.complex:
            parts.append(self.a_imag)
        if self.trapezoidal:
            parts.append(self.lambda_logit)
        parts.append(self.d_skip)
        return np.concatenate([p.detach().cpu().numpy().astype(np.float64).ravel() for p in parts])

    def from_flat(self, slab: np.ndarray) -> None:
        c = 0
        def take(param: nn.Parameter, shape: tuple[int, ...]) -> None:
            nonlocal c
            n = int(np.prod(shape))
            with torch.no_grad():
                param.copy_(torch.from_numpy(np.ascontiguousarray(slab[c : c + n]).reshape(shape)).to(param.dtype))
            c += n
        take(self.x_proj_w, (self.dt_rank + 2 * self.d_state, self.input_size))
        take(self.dt_proj_w, (self.input_size, self.dt_rank))
        take(self.dt_proj_b, (self.input_size,))
        take(self.a_log, (self.input_size, self.d_state))
        if self.complex:
            take(self.a_imag, (self.input_size, self.d_state))
        if self.trapezoidal:
            take(self.lambda_logit, (self.input_size,))
        take(self.d_skip, (self.input_size,))
```

Note the `# noqa` on `complex` shadowing the builtin if ruff flags it (`A002`); add `# noqa: A002` on the `__init__` signature line if needed to match repo lint settings.

- [ ] **Step 2: Write the torch-only unit test**

`tests/test_python_mamba3_layer.py`:

```python
import numpy as np
import torch
from aerocapture.training.rl.layers.mamba3 import Mamba3Layer


def test_flat_roundtrip_all_flags():
    for trap in (False, True):
        for cplx in (False, True):
            m = Mamba3Layer(4, 3, 2, trap, cplx).double()
            n = len(m.to_flat())
            slab = np.linspace(-0.5, 0.5, n)
            m.from_flat(slab)
            assert np.array_equal(m.to_flat(), slab)


def test_forward_finite():
    m = Mamba3Layer(3, 4, 1, True, True).double()
    m.from_flat(np.linspace(-0.3, 0.3, len(m.to_flat())))
    st = m.new_state()
    for t in range(10):
        x = torch.tensor([0.1 * (d + t) for d in range(3)], dtype=torch.float64)
        y, st = m.forward_unbatched(x, st)
    assert torch.isfinite(y).all()
```

- [ ] **Step 3: Run**

Run: `uv run pytest tests/test_python_mamba3_layer.py -v`
Expected: 2 PASS.

- [ ] **Step 4: Commit**

```bash
git add src/python/aerocapture/training/rl/layers/mamba3.py tests/test_python_mamba3_layer.py
git commit -m "feat(mamba3): python torch mirror (unbatched)"
```

---

## Task 7: `Mamba3Spec` schema + union + PPO-rejection guards

**Files:**
- Modify: `src/python/aerocapture/training/rl/schemas.py` (MambaSpec ~76, union ~106)
- Modify: `src/python/aerocapture/training/rl/layers/__init__.py` (build_layer dispatch)
- Modify: `src/python/aerocapture/training/model_io.py` (load_policy_from_json)
- Test: `tests/test_mamba3_ppo_rejection.py`

**Interfaces:**
- Produces: `Mamba3Spec` (fields `type: Literal["mamba3"]`, `input_size`, `d_state`, `dt_rank: int | None`, `discretization: Literal["euler","trapezoidal"] = "euler"`, `state_mode: Literal["real","complex"] = "real"`); appended to the `LayerSpec` discriminated union.

- [ ] **Step 1: Add `Mamba3Spec`**

In `schemas.py`, after `MambaSpec`, add a Pydantic model mirroring `MambaSpec` (including the `resolve_and_validate_dt_rank` model_validator that resolves `dt_rank = max(1, input_size // 16)` when None) plus the two `Literal` flag fields. Append `Mamba3Spec` to the union at line 106:

```python
LayerSpec = Annotated[DenseSpec | GruSpec | LstmSpec | WindowSpec | TransformerSpec | MambaSpec | Mamba3Spec, Discriminator("type")]
```

- [ ] **Step 2: `build_layer` PPO-rejection guard**

In `rl/layers/__init__.py`, add a `Mamba3Spec` branch that raises:

```python
    if isinstance(spec, Mamba3Spec):
        raise NotImplementedError("Mamba3 is PSO-only (Phase 4a spike); PPO path not implemented. See docs/superpowers/specs/2026-07-07-mamba3-ablation-design.md")
```

(Import `Mamba3Spec` at the top.)

- [ ] **Step 3: `load_policy_from_json` guard**

In `model_io.py::load_policy_from_json`, wherever it dispatches on layer `type`, add a `"mamba3"` case raising the same `NotImplementedError` (mirror the existing `"mamba"` rejection).

- [ ] **Step 4: Rejection test**

`tests/test_mamba3_ppo_rejection.py`:

```python
import pytest
from aerocapture.training.rl.layers import build_layer
from aerocapture.training.rl.schemas import Mamba3Spec


def test_build_layer_rejects_mamba3():
    spec = Mamba3Spec(type="mamba3", input_size=8, d_state=4, dt_rank=1)
    with pytest.raises(NotImplementedError, match="PSO-only"):
        build_layer(spec)
```

- [ ] **Step 5: Run**

Run: `uv run pytest tests/test_mamba3_ppo_rejection.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/rl/schemas.py src/python/aerocapture/training/rl/layers/__init__.py src/python/aerocapture/training/model_io.py tests/test_mamba3_ppo_rejection.py
git commit -m "feat(mamba3): Mamba3Spec schema + PSO-only PPO guards"
```

---

## Task 8: `_mamba3_specs` + config sizing arms

**Files:**
- Modify: `src/python/aerocapture/training/encoding.py` (`_mamba_specs` ~260, dispatch ~124)
- Modify: `src/python/aerocapture/training/config.py` (`_layer_n_params`, `_layer_output_size`, `describe_architecture`)
- Test: `tests/test_mamba3_encoding.py`

**Interfaces:**
- Consumes: `Mamba3Spec`.
- Produces: `_mamba3_specs(layer, layer_idx, bound_multiplier) -> list[ParamSpec]` in the flat order of the Global Constraints.

- [ ] **Step 1: `_mamba3_specs`**

In `encoding.py`, add `_mamba3_specs` that reuses `_mamba_specs`' base bounds (Xavier `x_proj_w`, Xavier·dt_rank^-0.5 `dt_proj_w`, inv_softplus dt-bias centers via `_MAMBA_DT_BIAS_SEED ^ layer_idx`, HiPPO `a_log`, `d_skip`=1), then appends (in order): if `complex`, an `a_imag` block (bounds ±π, center 0 — rotation frequency); if `trapezoidal`, a `lambda_logit` block (center +4, tight ±0.1·bound_multiplier — near-euler start). Dispatch in `_layer_param_specs` (~124): `if isinstance(layer, Mamba3Spec): return _mamba3_specs(layer, layer_idx, bound_multiplier)`.

- [ ] **Step 2: config sizing arms**

In `config.py`, add `Mamba3Spec`/`"mamba3"` arms to `_layer_n_params` (base Mamba formula + `input_size*d_state` if complex + `input_size` if trapezoidal), `_layer_output_size` (= `input_size`), and `describe_architecture` (`f"mamba3({input_size},{d_state},{dt_rank},{disc},{sm})"`). Reuse `resolve_mamba_dt_rank`.

- [ ] **Step 3: Encoding test**

`tests/test_mamba3_encoding.py`:

```python
from aerocapture.training.encoding import _mamba3_specs
from aerocapture.training.rl.schemas import Mamba3Spec


def _n_expected(input_size, d_state, dt_rank, trap, cplx):
    base = input_size * (3 * d_state + 2 * dt_rank + 2)
    return base + (input_size * d_state if cplx else 0) + (input_size if trap else 0)


def test_mamba3_specs_length_matches_n_params():
    for disc, trap in (("euler", False), ("trapezoidal", True)):
        for sm, cplx in (("real", False), ("complex", True)):
            spec = Mamba3Spec(type="mamba3", input_size=16, d_state=8, dt_rank=1, discretization=disc, state_mode=sm)
            specs = _mamba3_specs(spec, 0, 2.0)
            assert len(specs) == _n_expected(16, 8, 1, trap, cplx), (disc, sm)
```

- [ ] **Step 4: Run**

Run: `uv run pytest tests/test_mamba3_encoding.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/encoding.py src/python/aerocapture/training/config.py tests/test_mamba3_encoding.py
git commit -m "feat(mamba3): PSO param specs + config sizing arms"
```

---

## Task 9: `_init_mamba3_layer` + `init_v2_population` dispatch

**Files:**
- Modify: `src/python/aerocapture/training/initialization_v2.py` (`_init_mamba_layer` ~176, `init_v2_population` dispatch)
- Test: `tests/test_init_v2_mamba3.py`

**Interfaces:**
- Produces: `_init_mamba3_layer(entry, n_pop, bound_multiplier, rng, layer_idx) -> np.ndarray (n_pop, n_params)` in the flat order.

- [ ] **Step 1: `_init_mamba3_layer`**

Mirror `_init_mamba_layer` (HiPPO a_log centers, shared dt-bias centers via `_MAMBA_DT_BIAS_SEED ^ layer_idx`, Xavier x_proj/dt_proj, d_skip=1, per-individual `N(0, 0.01·bound_multiplier)` jitter), inserting after `a_log`: `a_imag` (S4D-Lin ramp center `π·(n+1)/d_state` per state column, small jitter) when complex; `lambda_logit` (center +4, small jitter) when trapezoidal. Add a `Mamba3Spec`/`"mamba3"` dispatch branch in `init_v2_population`.

- [ ] **Step 2: Init test**

`tests/test_init_v2_mamba3.py`:

```python
import numpy as np
from aerocapture.training.initialization_v2 import init_v2_population


def test_init_v2_mamba3_shape_and_finite():
    arch = [
        {"type": "dense", "input_size": 23, "output_size": 16, "activation": "swish"},
        {"type": "mamba3", "input_size": 16, "d_state": 8, "dt_rank": 1, "discretization": "trapezoidal", "state_mode": "complex"},
        {"type": "dense", "input_size": 16, "output_size": 2, "activation": "linear"},
    ]
    pop = init_v2_population(arch, n_pop=8, bound_multiplier=2.0, rng=np.random.default_rng(0))
    assert pop.shape[0] == 8
    assert np.isfinite(pop).all()
```

- [ ] **Step 3: Run**

Run: `uv run pytest tests/test_init_v2_mamba3.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/python/aerocapture/training/initialization_v2.py tests/test_init_v2_mamba3.py
git commit -m "feat(mamba3): activation-aware population init"
```

---

## Task 10: Rebuild PyO3 + cross-language equivalence gate (THE gate)

**Files:**
- Test: `tests/test_rust_python_mamba3_equivalence.py`

**Interfaces:**
- Consumes: `aerocapture_rs.nn_forward` (v2 JSON loader), the Python `Mamba3Layer` mirror, the Rust runtime.

- [ ] **Step 1: Rebuild the PyO3 module (Rust changes must be compiled in)**

Run: `uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml`
Expected: builds `aerocapture_rs`.

- [ ] **Step 2: Write the equivalence test (all 4 flag combos)**

`tests/test_rust_python_mamba3_equivalence.py`. Build a 3-layer `Dense -> Mamba3 -> Dense` model in torch (`.double()`), export to a v2 JSON (write the Mamba3 layer flat weights under the same key convention the Mamba equivalence test uses — reference `tests/test_rust_python_mamba_equivalence.py` for the JSON writer helper), feed 100 random f64 inputs through `aerocapture_rs.nn_forward` (stateless — Rust resets per call) vs the Python mirror stepped per-input with per-step reset, assert max abs diff < 1e-12.

```python
import json
import numpy as np
import torch
import aerocapture_rs
from aerocapture.training.rl.layers.mamba3 import Mamba3Layer

# Mirror the Dense mirror the existing mamba equivalence test uses; a minimal
# Dense is fine here (weight matmul + activation). Reuse the helper from
# tests/test_rust_python_mamba_equivalence.py if it exposes one.

def _build(disc, sm, tmp_path):
    torch.manual_seed(0)
    d_in, d_mid, d_state = 4, 6, 4
    # ... construct Dense(4->6) -> Mamba3(6, d_state, dt_rank=1, flags) -> Dense(6->2),
    # random-init all, write a format_version=2 JSON with each layer's flat weights.
    ...

def test_mamba3_rust_python_equivalence(tmp_path):
    for disc in ("euler", "trapezoidal"):
        for sm in ("real", "complex"):
            path, torch_layers = _build(disc, sm, tmp_path)
            xs = np.random.default_rng(1).standard_normal((100, 4))
            # Rust: stateless per-call forward accumulates internal state across the 100 inputs
            rust_out = np.array([aerocapture_rs.nn_forward(str(path), x.tolist()) for x in xs])
            # Python: step the mirror with persistent state, reset once at start
            py_out = _run_python(torch_layers, xs)
            diff = np.abs(rust_out - py_out).max()
            assert diff < 1e-12, f"{disc}/{sm}: max abs diff {diff}"
```

Fill `_build` / `_run_python` by copying the mechanics from `tests/test_rust_python_mamba_equivalence.py` (which already exercises Dense + Mamba across `nn_forward`). The ONLY new part is the Mamba3 layer JSON block + the tuple-state stepping.

- [ ] **Step 3: Run**

Run: `uv run pytest tests/test_rust_python_mamba3_equivalence.py -v`
Expected: PASS, max abs diff ~1e-14 (well under 1e-12).

- [ ] **Step 4: Commit**

```bash
git add tests/test_rust_python_mamba3_equivalence.py
git commit -m "test(mamba3): cross-language equivalence gate (4 flag combos)"
```

---

## Task 11: PSO smoke test

**Files:**
- Test: `tests/test_mamba3_pso_smoke.py`

- [ ] **Step 1: Write the smoke test**

Mirror `tests/test_mamba_pso_smoke.py` (@pytest.mark.slow). Build a reduced `Dense -> Mamba3(trapezoidal, complex) -> Dense` arch, run 2 PSO gens with `training_n_sims=2` via `aerocapture.training.train`, assert `best_model.json` is v2, contains a `"mamba3"` layer with `trapezoidal`/`complex` keys, and `aerocapture_rs.nn_forward` returns a finite 2-tuple.

```python
import pytest

@pytest.mark.slow
def test_mamba3_pso_2gen_smoke(tmp_path):
    # Write a reduced mamba3 training TOML (n_pop=4, n_gen=2, training_n_sims=2),
    # subprocess `python -m aerocapture.training.train <toml> --n-gen 2 --no-tui --skip-report --output-dir <tmp>`,
    # then assert best_model.json exists, is format_version 2, has a mamba3 layer, and nn_forward is finite.
    ...
```

Fill from `test_mamba_pso_smoke.py`.

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_mamba3_pso_smoke.py -v -m slow`
Expected: PASS (may take ~1-2 min).

- [ ] **Step 3: Commit**

```bash
git add tests/test_mamba3_pso_smoke.py
git commit -m "test(mamba3): PSO 2-gen smoke"
```

---

## Task 12: Experiment script `mamba3_ablation.py`

**Files:**
- Create: `src/python/aerocapture/training/experiments/mamba3_ablation.py`
- Create: `src/python/aerocapture/training/experiments/__init__.py` (if the dir is new)
- Modify: `src/python/aerocapture/training/evaluate.py` (add `MAMBA3_EVAL_SEED_OFFSET = 10_000_000`)
- Test: `tests/test_mamba3_ablation.py`

**Interfaces:**
- Produces: `ARMS: dict[str, tuple[str, str]]` (arm -> (discretization, state_mode)); `generate_configs`, `eval_arm`, `summarize` functions; a CLI with `--generate/--train/--eval/--report/--all`.

- [ ] **Step 1: Add the seed offset**

`evaluate.py` after `STRESS_EVAL_SEED_OFFSET = 9_000_000`:

```python
MAMBA3_EVAL_SEED_OFFSET = 10_000_000  # Mamba-3 ablation spike; disjoint from 1M-9M
```

- [ ] **Step 2: Write the experiment module**

`experiments/mamba3_ablation.py`. Mirror `param_sweep.py`'s structure (argparse subcommands, subprocess train, `run_batch` eval, `make_reserved_seeds`). Key pieces:

```python
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
from aerocapture.training.evaluate import MAMBA3_EVAL_SEED_OFFSET, make_reserved_seeds, compute_cost, read_cost_kwargs
from aerocapture.training.toml_utils import load_toml_with_bases

ARMS: dict[str, tuple[str, str]] = {
    "baseline": ("euler", "real"),
    "trapz": ("trapezoidal", "real"),
    "complex": ("euler", "complex"),
    "both": ("trapezoidal", "complex"),
}
BASE_CONFIG = "configs/training/msr_aller_mamba_pso_train.toml"
CONFIG_DIR = Path("configs/training/mamba3")
OUT_DIR = Path("training_output/mamba3")
BASE_SEED = 20260707


def _base_architecture() -> list[dict]:
    """Resolve the base config's [[network.architecture]] and swap mamba -> mamba3."""
    cfg = load_toml_with_bases(Path(BASE_CONFIG))
    arch = [dict(layer) for layer in cfg["network"]["architecture"]]
    for layer in arch:
        if layer.get("type") == "mamba":
            layer["type"] = "mamba3"  # flags filled per arm in generate_configs
    return arch


def generate_configs(repeats: int) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    arch = _base_architecture()
    for arm, (disc, sm) in ARMS.items():
        arm_arch = [dict(l) for l in arch]
        for layer in arm_arch:
            if layer["type"] == "mamba3":
                layer["discretization"], layer["state_mode"] = disc, sm
        for r in range(repeats):
            # Write a leaf TOML: base-inherit BASE_CONFIG, REPLACE [[network.architecture]]
            # with arm_arch, set [monte_carlo] seed = BASE_SEED + r, results_suffix per arm/rep.
            _write_leaf(CONFIG_DIR / f"{arm}_s{r}.toml", arm_arch, BASE_SEED + r)


def eval_arm(arm: str, repeats: int, n_sims: int) -> dict:
    """run_batch each repeat's best_model.json (+ best_params.json scaffolding) on the shared pool."""
    seeds = make_reserved_seeds(0, MAMBA3_EVAL_SEED_OFFSET, n_sims)
    per_rep = []
    for r in range(repeats):
        model_dir = OUT_DIR / f"{arm}_s{r}"
        # load best_model.json; overrides = _scaffolding_overrides(model_dir); run_batch on seeds;
        # dv = final_records[:, DV_COL]; captured mask; compute p50/p95/CVaR95/capture.
        per_rep.append(_score_one(model_dir, seeds))
    return _aggregate(arm, per_rep)  # mean +/- std across repeats -> sigma_run
```

`--report` prints an arm x {p50, p95, CVaR95, capture} table with `± σ_run` (std across repeats) and appends a line per tail metric: `GAP vs baseline = X; sigma_run = Y; SIGNIFICANT if |X| > Y`. CVaR95 = mean of DV above the 95th percentile.

Reuse `param_sweep.py`'s `_load_nn_scaffolding_overrides` for the `best_params.json` scaffolding (the base config inherits `scaffolding = "live"` if the mamba base does — check and mirror).

- [ ] **Step 3: Write a unit test for the pure pieces**

`tests/test_mamba3_ablation.py`:

```python
import numpy as np
from aerocapture.training.experiments.mamba3_ablation import ARMS, _cvar95


def test_arms_cover_2x2():
    assert set(ARMS.values()) == {("euler","real"),("trapezoidal","real"),("euler","complex"),("trapezoidal","complex")}


def test_cvar95_is_tail_mean():
    x = np.arange(100.0)  # p95 = 94.05; CVaR95 = mean of top 5%
    assert _cvar95(x) > np.percentile(x, 95)
```

(Extract `_cvar95(arr)` as a module-level pure helper so it is unit-testable.)

- [ ] **Step 4: Run**

Run: `uv run pytest tests/test_mamba3_ablation.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/experiments/ tests/test_mamba3_ablation.py src/python/aerocapture/training/evaluate.py
git commit -m "feat(mamba3): 2x2 ablation experiment script + seed offset"
```

---

## Task 13: `--generate` configs + tiny end-to-end dry run

**Files:**
- Create: `configs/training/mamba3/*.toml` (generated)

- [ ] **Step 1: Generate the arm configs**

Run: `uv run python -m aerocapture.training.experiments.mamba3_ablation --generate --repeats 1`
Expected: writes `configs/training/mamba3/{baseline,trapz,complex,both}_s0.toml`.

- [ ] **Step 2: Sanity-load one generated config through Rust**

Run: `uv run python -c "import aerocapture_rs; aerocapture_rs.load_config('configs/training/mamba3/both_s0.toml')"`
Expected: no error (proves the `mamba3` TOML parses + the flags validate through `to_layer_spec`).

- [ ] **Step 3: Tiny dry run (NOT the real budget)**

Run: `uv run python -m aerocapture.training.experiments.mamba3_ablation --all --repeats 1 --n-gen 5 --n-sims 20`
Expected: trains each arm 5 gens, evals on 20 sims, prints the comparison table. Numbers are meaningless (tiny budget) — this proves the generate->train->eval->report pipeline is wired end to end.

- [ ] **Step 4: Commit generated configs**

```bash
git add configs/training/mamba3/
git commit -m "chore(mamba3): generated 2x2 arm configs (repeats=1)"
```

---

## Task 14: CLAUDE.md update + full regression sweep

**Files:**
- Modify: `CLAUDE.md` (Stateful NN Runtime Infrastructure section)

- [ ] **Step 1: Add a Mamba-3 paragraph to CLAUDE.md**

Under the Phase 4a Mamba section, add a short paragraph documenting the `Mamba3` layer: PSO-only spike, two flags (`discretization`/`state_mode`), nested trapezoidal (λ→1 == euler), complex-diagonal state (real B/C, Re readout), flat-weight order, `n_params` formula, `MAMBA3_EVAL_SEED_OFFSET = 10M`, the experiment script, and the deliberate simplifications (constant λ, S4D-Lin θ). Reference the spec + plan paths.

- [ ] **Step 2: Full Rust check**

Run: `./check_all.sh`
Expected: fmt clean, clippy clean, all Rust tests PASS (incl. all `mamba3` tests + the 6 guidance goldens bit-identical), release build OK.

- [ ] **Step 3: Full Python check**

Run: `./lint_code.sh && uv run pytest tests`
Expected: ruff + mypy clean; all tests PASS (excluding/including slow per repo default).

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(mamba3): CLAUDE.md Mamba-3 ablation paragraph"
```

---

## Task 15: Final validation + smart-commit

- [ ] **Step 1: Confirm branch state**

Run: `git log --oneline main..feature/mamba3-ablation`
Expected: the spec + plan + all Task commits, no `main` commits.

- [ ] **Step 2: Invoke smart-commit over the whole branch**

Invoke the `smart-commit` skill, telling it to take the whole `feature/mamba3-ablation` branch into account (syncs CLAUDE.md / README with the code, then commits anything outstanding).

---

## Self-Review

**Spec coverage:**
- Packaging (new Mamba3 type, PSO-only) — Tasks 4, 5, 7.
- Trapezoidal (nested, λ_logit init +4) — Tasks 3, 8, 9.
- Complex (a_imag, Re readout, explicit re/im) — Tasks 1, 3, 6.
- Flat-weight layout / n_params — Tasks 2, 8.
- LayerState {h_re,h_im,x_prev,b_prev} — Task 4.
- Init (S4D-Lin θ, λ +4) — Task 9.
- Experiment (2x2, 3 repeats, shared pool, σ_run gate) — Tasks 12, 13.
- Gates (real-euler anchor, trapz→euler, complex warmup, 4-combo equivalence <1e-12, PSO smoke, PPO rejection, n_params) — Tasks 3, 5, 6, 7, 8, 10, 11.
- Goldens bit-identical — Task 14.
- smart-commit final step — Task 15.

**Placeholder scan:** The two cross-language/PSO-smoke tests (Tasks 10, 11) and the experiment leaf-writer (Task 12) reference "copy the mechanics from `test_rust_python_mamba_equivalence.py` / `test_mamba_pso_smoke.py` / `param_sweep.py`" rather than inlining the full harness. This is deliberate — those harnesses are large and already exist; the novel delta (Mamba3 JSON block, tuple-state stepping, arch swap) is specified. Everything else is complete inline code.

**Type consistency:** `Mamba3Layer::forward(x, h_re, h_im, x_prev, b_prev)` signature is consistent across Task 3 (def), Task 4 (dispatch call), and the state shapes in `LayerState::Mamba3` (Task 4). Python `forward_unbatched(x, state)` tuple-state is consistent across Tasks 6 and 10. Flag names `trapezoidal`/`complex` (Rust bool) map to `discretization`/`state_mode` (TOML/Spec string) consistently in Tasks 4, 5, 7, 8.
