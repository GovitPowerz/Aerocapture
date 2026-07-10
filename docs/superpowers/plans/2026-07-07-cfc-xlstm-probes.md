# CfC + xLSTM Probes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three new PSO-only NN layer types (`cfc`, `slstm`, `mlstm`) wired through the full Rust+Python contract, plus two experiment scripts (`cfc_probe.py`, `xlstm_probe.py`) that run matched-budget controlled arms with sigma_run error bars.

**Architecture:** Cell-only layers following the mamba3 extension contract on branch `feature/cfc-xlstm` (stacked on `feature/mamba3-ablation`). Rust owns the runtime (scalar f64 loops); Python owns PSO specs/init and unbatched torch mirrors used only by cross-language equivalence tests. `build_layer` / `load_policy_from_json` raise `NotImplementedError` (PSO-only gate, mamba3 pattern verbatim).

**Tech Stack:** Rust (edition 2024, nalgebra, serde), Python 3.14 (pydantic v2, torch, numpy), PyO3/maturin, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-07-07-cfc-xlstm-probes-design.md` — equations, flat orders, and param budgets there are authoritative.

## Global Constraints

- Branch: `feature/cfc-xlstm`. Never commit to main. Commit after every task with the `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer.
- All existing behavior additive-only: existing layers, golden regressions, and deployed champions must be untouched.
- Rust: `cargo fmt` + `cargo clippy -- -D warnings` clean per task. Run cargo from repo root via `--manifest-path src/rust/Cargo.toml` (never bare `cd`).
- PyO3 rebuild ALWAYS from repo root: `uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml` (subcrate builds go stale).
- Python: ruff line-length 160, mypy strict. Run via `uv run pytest tests/<file> -v`.
- Canonical flat weight order per layer MUST be identical in: Rust `to_flat`/`from_flat`, Python `encoding.py` ParamSpecs, `initialization_v2.py` fills, and torch-mirror parameter layout. A mismatch scrambles PSO chromosomes silently.
- Gate order conventions: sLSTM 4H axis = (i, f, z, o). CfC head order = (ff1, ff2, ta, tb). mLSTM slab order = (q, k, v, o, i, f).
- `CFC_DT = 1.0` (one guidance tick, absorbed into learned time heads).
- Stabilizer init: `m = 0.0`. Forget-bias init center `+2.0` (sLSTM bias f-slice, mLSTM `b_f`), ParamSpec bound `3.0 * bound_multiplier` on those slices.

---

### Task 1: Rust shared helpers (`lecun_tanh`, `stabilized_exp_gates`, flat-copy helpers)

**Files:**
- Modify: `src/rust/src/data/neural/layers/helpers.rs`

**Interfaces:**
- Produces: `pub(crate) fn lecun_tanh(z: f64) -> f64`; `pub(crate) fn stabilized_exp_gates(i_pre: f64, f_pre: f64, m_prev: f64) -> (f64, f64, f64)` returning `(i_gate, f_gate, m_new)`; `pub(crate) fn copy_mat_from_flat(mat: &mut [Vec<f64>], flat: &[f64], idx: &mut usize)`; `pub(crate) fn copy_vec_from_flat(v: &mut [f64], flat: &[f64], idx: &mut usize)`.
- Consumes: nothing new.

- [ ] **Step 1: Write failing unit tests** — append to the `#[cfg(test)] mod tests` at the bottom of `helpers.rs` (create the mod if the file has none; check first with `rg -n "mod tests" src/rust/src/data/neural/layers/helpers.rs`):

```rust
#[test]
fn lecun_tanh_matches_definition() {
    assert!((lecun_tanh(0.0)).abs() < 1e-15);
    let z = 0.7;
    let expected = 1.7159 * (2.0 * z / 3.0).tanh();
    assert_eq!(lecun_tanh(z), expected);
    assert_eq!(lecun_tanh(-z), -expected);
}

#[test]
fn stabilized_exp_gates_both_args_nonpositive() {
    // Stabilizer guarantees exp arguments <= 0, so gates are in (0, 1].
    for (i_pre, f_pre, m_prev) in [(0.0, 0.0, 0.0), (50.0, -50.0, 10.0), (-300.0, 300.0, -5.0)] {
        let (ig, fg, m_new) = stabilized_exp_gates(i_pre, f_pre, m_prev);
        assert!(ig.is_finite() && fg.is_finite() && m_new.is_finite());
        assert!(ig > 0.0 && ig <= 1.0, "ig={ig}");
        assert!(fg > 0.0 && fg <= 1.0, "fg={fg}");
        assert_eq!(m_new, (f_pre + m_prev).max(i_pre));
        // One of the two gates is exactly exp(0) = 1 (the max branch).
        assert!(ig == 1.0 || fg == 1.0);
    }
}

#[test]
fn copy_helpers_advance_cursor() {
    let flat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut mat = vec![vec![0.0; 2]; 2];
    let mut v = vec![0.0; 2];
    let mut idx = 0;
    copy_mat_from_flat(&mut mat, &flat, &mut idx);
    copy_vec_from_flat(&mut v, &flat, &mut idx);
    assert_eq!(idx, 6);
    assert_eq!(mat, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    assert_eq!(v, vec![5.0, 6.0]);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --manifest-path src/rust/Cargo.toml lecun_tanh 2>&1 | tail -5`
Expected: compile error (`lecun_tanh` not found).

- [ ] **Step 3: Implement** — add to `helpers.rs` below the existing helpers:

```rust
/// LeCun-scaled tanh used by the CfC backbone: 1.7159 * tanh(2z/3).
/// Constant order matches the Python mirror (`1.7159 * torch.tanh(2.0 * z / 3.0)`).
pub(crate) fn lecun_tanh(z: f64) -> f64 {
    1.7159 * (2.0 * z / 3.0).tanh()
}

/// xLSTM stabilized exponential gating (Beck et al. 2024, eq. 15-17).
/// m_new = max(f_pre + m_prev, i_pre); both exp arguments are <= 0 by
/// construction, so the returned gates are finite for arbitrarily large
/// preactivations. Returns (i_gate, f_gate, m_new).
pub(crate) fn stabilized_exp_gates(i_pre: f64, f_pre: f64, m_prev: f64) -> (f64, f64, f64) {
    let m_new = (f_pre + m_prev).max(i_pre);
    ((i_pre - m_new).exp(), (f_pre + m_prev - m_new).exp(), m_new)
}

/// Copy a row-major matrix slab out of `flat`, advancing the cursor.
pub(crate) fn copy_mat_from_flat(mat: &mut [Vec<f64>], flat: &[f64], idx: &mut usize) {
    for row in mat.iter_mut() {
        let n = row.len();
        row.copy_from_slice(&flat[*idx..*idx + n]);
        *idx += n;
    }
}

/// Copy a vector slab out of `flat`, advancing the cursor.
pub(crate) fn copy_vec_from_flat(v: &mut [f64], flat: &[f64], idx: &mut usize) {
    let n = v.len();
    v.copy_from_slice(&flat[*idx..*idx + n]);
    *idx += n;
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --manifest-path src/rust/Cargo.toml helpers 2>&1 | tail -5`
Expected: PASS (3 new tests).

- [ ] **Step 5: fmt + clippy + commit**

```bash
cargo fmt --manifest-path src/rust/Cargo.toml
cargo clippy --manifest-path src/rust/Cargo.toml -- -D warnings
git add src/rust/src/data/neural/layers/helpers.rs
git commit -m "feat(cfc-xlstm): shared helpers -- lecun_tanh, stabilized exp gates, flat-copy cursors

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Rust CfC layer end-to-end

**Files:**
- Create: `src/rust/src/data/neural/layers/cfc.rs`
- Modify: `src/rust/src/data/neural/layers/mod.rs` (register), `src/rust/src/data/neural/mod.rs` (enum arms), `src/rust/src/data/nn_state.rs` (state variant), `src/rust/src/config.rs` (TOML spec)
- Test: inline `#[cfg(test)]` in `cfc.rs` + round-trip test in `src/rust/src/data/neural/tests.rs`

**Interfaces:**
- Consumes: Task 1 helpers.
- Produces: `pub struct CfcLayer { input_size, hidden_size, backbone_units, w_bb, b_bb, w_ff1, b_ff1, w_ff2, b_ff2, w_ta, b_ta, w_tb, b_tb }` with `CfcLayer::zeros(input_size, hidden_size, backbone_units) -> Self` and `forward(&self, x: &[f64], h: &mut [f64]) -> Vec<f64>`; `LayerSpec::Cfc { input_size, hidden_size, backbone_units }` (serde tag `"cfc"`); `Layer::Cfc(Box<CfcLayer>)`; `LayerState::Cfc { h: Vec<f64> }`; JSON weight keys `w_bb, b_bb, w_ff1, b_ff1, w_ff2, b_ff2, w_ta, b_ta, w_tb, b_tb`.

- [ ] **Step 1: Write the layer with inline failing tests** — create `src/rust/src/data/neural/layers/cfc.rs`:

```rust
//! CfC (closed-form continuous-time) cell -- ncps "default" mode, cell-only.
//!
//! Forward (dt fixed at one guidance tick, absorbed into the learned time heads):
//!   cat = [x, h]
//!   xb  = lecun_tanh(W_bb @ cat + b_bb)
//!   g   = sigmoid(-(W_ta @ xb + b_ta) * CFC_DT + (W_tb @ xb + b_tb))
//!   h'  = (1 - g) * tanh(W_ff1 @ xb + b_ff1) + g * tanh(W_ff2 @ xb + b_ff2)
//! Output = h', bounded in (-1, 1) by construction.
//!
//! Canonical flat order (LayerWeights + PSO chromosome + torch mirror):
//!   w_bb, b_bb, w_ff1, b_ff1, w_ff2, b_ff2, w_ta, b_ta, w_tb, b_tb
//! (matrices row-major, interleaved matrix/bias pairs).

use super::super::{Activation, LayerWeights};
use super::helpers::{copy_mat_from_flat, copy_vec_from_flat, dot_plus_bias, lecun_tanh};

/// Fixed per-tick dt: guidance cadence is constant, so dt is absorbed into
/// the learned time heads t_a / t_b (spec: deliberate simplification).
const CFC_DT: f64 = 1.0;

#[derive(Debug, Clone)]
pub struct CfcLayer {
    pub input_size: usize,
    pub hidden_size: usize,
    pub backbone_units: usize,
    pub w_bb: Vec<Vec<f64>>,  // [B, I+H]
    pub b_bb: Vec<f64>,       // [B]
    pub w_ff1: Vec<Vec<f64>>, // [H, B]
    pub b_ff1: Vec<f64>,      // [H]
    pub w_ff2: Vec<Vec<f64>>, // [H, B]
    pub b_ff2: Vec<f64>,      // [H]
    pub w_ta: Vec<Vec<f64>>,  // [H, B]
    pub b_ta: Vec<f64>,       // [H]
    pub w_tb: Vec<Vec<f64>>,  // [H, B]
    pub b_tb: Vec<f64>,       // [H]
}

impl CfcLayer {
    pub fn zeros(input_size: usize, hidden_size: usize, backbone_units: usize) -> Self {
        let cat = input_size + hidden_size;
        Self {
            input_size,
            hidden_size,
            backbone_units,
            w_bb: vec![vec![0.0; cat]; backbone_units],
            b_bb: vec![0.0; backbone_units],
            w_ff1: vec![vec![0.0; backbone_units]; hidden_size],
            b_ff1: vec![0.0; hidden_size],
            w_ff2: vec![vec![0.0; backbone_units]; hidden_size],
            b_ff2: vec![0.0; hidden_size],
            w_ta: vec![vec![0.0; backbone_units]; hidden_size],
            b_ta: vec![0.0; hidden_size],
            w_tb: vec![vec![0.0; backbone_units]; hidden_size],
            b_tb: vec![0.0; hidden_size],
        }
    }

    /// One step: reads x + h, overwrites h with h_new, returns h_new as output.
    pub fn forward(&self, x: &[f64], h: &mut [f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.input_size);
        assert_eq!(h.len(), self.hidden_size);
        let mut cat = Vec::with_capacity(self.input_size + self.hidden_size);
        cat.extend_from_slice(x);
        cat.extend_from_slice(h);
        let xb: Vec<f64> = (0..self.backbone_units)
            .map(|j| lecun_tanh(dot_plus_bias(&self.w_bb[j], &cat, self.b_bb[j])))
            .collect();
        let mut h_new = vec![0.0; self.hidden_size];
        for i in 0..self.hidden_size {
            let ff1 = dot_plus_bias(&self.w_ff1[i], &xb, self.b_ff1[i]).tanh();
            let ff2 = dot_plus_bias(&self.w_ff2[i], &xb, self.b_ff2[i]).tanh();
            let t_a = dot_plus_bias(&self.w_ta[i], &xb, self.b_ta[i]);
            let t_b = dot_plus_bias(&self.w_tb[i], &xb, self.b_tb[i]);
            let g = Activation::Sigmoid.apply(-t_a * CFC_DT + t_b);
            h_new[i] = (1.0 - g) * ff1 + g * ff2;
        }
        h.copy_from_slice(&h_new);
        h_new
    }
}

impl LayerWeights for CfcLayer {
    fn to_flat(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.n_params());
        for (mat, bias) in [
            (&self.w_bb, &self.b_bb),
            (&self.w_ff1, &self.b_ff1),
            (&self.w_ff2, &self.b_ff2),
            (&self.w_ta, &self.b_ta),
            (&self.w_tb, &self.b_tb),
        ] {
            for row in mat.iter() {
                v.extend_from_slice(row);
            }
            v.extend_from_slice(bias);
        }
        v
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_flat(&mut self, flat: &[f64]) -> usize {
        let mut idx = 0;
        copy_mat_from_flat(&mut self.w_bb, flat, &mut idx);
        copy_vec_from_flat(&mut self.b_bb, flat, &mut idx);
        copy_mat_from_flat(&mut self.w_ff1, flat, &mut idx);
        copy_vec_from_flat(&mut self.b_ff1, flat, &mut idx);
        copy_mat_from_flat(&mut self.w_ff2, flat, &mut idx);
        copy_vec_from_flat(&mut self.b_ff2, flat, &mut idx);
        copy_mat_from_flat(&mut self.w_ta, flat, &mut idx);
        copy_vec_from_flat(&mut self.b_ta, flat, &mut idx);
        copy_mat_from_flat(&mut self.w_tb, flat, &mut idx);
        copy_vec_from_flat(&mut self.b_tb, flat, &mut idx);
        idx
    }

    fn n_params(&self) -> usize {
        let cat = self.input_size + self.hidden_size;
        self.backbone_units * cat
            + self.backbone_units
            + 4 * (self.hidden_size * self.backbone_units + self.hidden_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn patterned(input_size: usize, hidden_size: usize, backbone_units: usize) -> CfcLayer {
        let mut l = CfcLayer::zeros(input_size, hidden_size, backbone_units);
        let n = l.n_params();
        let flat: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01 - 0.3).collect();
        assert_eq!(l.from_flat(&flat), n);
        l
    }

    #[test]
    fn flat_round_trip_is_bit_identical() {
        let l = patterned(3, 4, 5);
        let flat = l.to_flat();
        assert_eq!(flat.len(), l.n_params());
        let mut l2 = CfcLayer::zeros(3, 4, 5);
        assert_eq!(l2.from_flat(&flat), flat.len());
        assert_eq!(l2.to_flat(), flat);
    }

    #[test]
    fn n_params_formula() {
        // B(I+H) + B + 4(HB + H) = 5*7 + 5 + 4*(4*5 + 4) = 136
        assert_eq!(CfcLayer::zeros(3, 4, 5).n_params(), 136);
    }

    #[test]
    fn output_is_bounded_and_state_evolves() {
        let l = patterned(3, 4, 5);
        let mut h = vec![0.0; 4];
        let mut prev = h.clone();
        for t in 0..50 {
            let x = vec![0.5 * (t as f64).sin(), -0.2, 0.9];
            let out = l.forward(&x, &mut h);
            assert_eq!(out, h);
            for &v in &out {
                assert!(v.is_finite() && v.abs() < 1.0, "unbounded output {v}");
            }
            if t == 1 {
                assert_ne!(h, prev, "state must evolve");
            }
            prev.clone_from(&h);
        }
    }

    #[test]
    fn zero_weights_give_neutral_gate_output() {
        // All-zero weights: ff1 = ff2 = tanh(0) = 0 -> h' = 0 regardless of g.
        let l = CfcLayer::zeros(2, 3, 2);
        let mut h = vec![0.0; 3];
        let out = l.forward(&[1.0, -1.0], &mut h);
        assert_eq!(out, vec![0.0; 3]);
    }
}
```

- [ ] **Step 2: Register the module** — in `src/rust/src/data/neural/layers/mod.rs` add (alphabetical placement):

```rust
pub(crate) mod cfc;
```
and
```rust
pub use cfc::CfcLayer;
```

- [ ] **Step 3: Run inline tests to verify they pass**

Run: `cargo test --manifest-path src/rust/Cargo.toml cfc:: 2>&1 | tail -5`
Expected: 4 PASS.

- [ ] **Step 4: Wire the enums.** In `src/rust/src/data/neural/mod.rs`:

(a) Import: extend the `layers::` import list (line ~12) with `CfcLayer`.

(b) `Layer` enum — add after `Mamba3(...)`:
```rust
    // Boxed for enum-variant size uniformity (5 matrix + 5 bias vectors, ~264 bytes unboxed).
    Cfc(Box<CfcLayer>),
```

(c) `Layer::input_size()` — add arm:
```rust
            Layer::Cfc(l) => l.input_size,
```

(d) `impl LayerWeights for Layer` — add to all three matches:
```rust
            Layer::Cfc(l) => l.to_flat(),
            Layer::Cfc(l) => l.from_flat(flat),
            Layer::Cfc(l) => l.n_params(),
```

(e) `NnLayerWeights` struct — add the CfC JSON fields after the mamba3 fields:
```rust
    // CfC fields (cfc-xlstm probes)
    #[serde(skip_serializing_if = "Option::is_none", default)]
    w_bb: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    b_bb: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    w_ff1: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    b_ff1: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    w_ff2: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    b_ff2: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    w_ta: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    b_ta: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    w_tb: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    b_tb: Option<Vec<f64>>,
```

(f) `LayerSpec` enum — add variant (serde snake_case tag gives `"cfc"`):
```rust
    Cfc {
        input_size: usize,
        hidden_size: usize,
        backbone_units: usize,
    },
```

(g) `LayerSpec::io()` — add arm:
```rust
            LayerSpec::Cfc {
                input_size,
                hidden_size,
                ..
            } => (*input_size, *hidden_size, "cfc"),
```

(h) `from_v2_json` layer-build match — add arm (slab-assembly style, mirrors Mamba3):
```rust
                LayerSpec::Cfc {
                    input_size,
                    hidden_size,
                    backbone_units,
                } => {
                    if *input_size == 0 || *hidden_size == 0 || *backbone_units == 0 {
                        return Err(DataError(format!(
                            "Layer {i} (cfc) input_size, hidden_size, backbone_units must be positive in {path}"
                        )));
                    }
                    if i == 0 {
                        layer_sizes.push(*input_size);
                    }
                    layer_sizes.push(*hidden_size);

                    let key = format!("layer_{}", i);
                    let lw = file.weights.get(&key).ok_or_else(|| {
                        DataError(format!("Missing {} in weights in {}", key, path))
                    })?;

                    let flat_mat =
                        |name: &str, m: &Option<Vec<Vec<f64>>>| -> Result<Vec<f64>, DataError> {
                            let rows = m.as_ref().ok_or_else(|| {
                                DataError(format!("Layer {i} (cfc) missing {name} in {path}"))
                            })?;
                            Ok(rows.iter().flat_map(|r| r.iter().copied()).collect())
                        };
                    let flat_vec =
                        |name: &str, v: &Option<Vec<f64>>| -> Result<Vec<f64>, DataError> {
                            v.as_ref().cloned().ok_or_else(|| {
                                DataError(format!("Layer {i} (cfc) missing {name} in {path}"))
                            })
                        };

                    let mut slab = Vec::new();
                    slab.extend(flat_mat("w_bb", &lw.w_bb)?);
                    slab.extend(flat_vec("b_bb", &lw.b_bb)?);
                    slab.extend(flat_mat("w_ff1", &lw.w_ff1)?);
                    slab.extend(flat_vec("b_ff1", &lw.b_ff1)?);
                    slab.extend(flat_mat("w_ff2", &lw.w_ff2)?);
                    slab.extend(flat_vec("b_ff2", &lw.b_ff2)?);
                    slab.extend(flat_mat("w_ta", &lw.w_ta)?);
                    slab.extend(flat_vec("b_ta", &lw.b_ta)?);
                    slab.extend(flat_mat("w_tb", &lw.w_tb)?);
                    slab.extend(flat_vec("b_tb", &lw.b_tb)?);

                    let mut l = CfcLayer::zeros(*input_size, *hidden_size, *backbone_units);
                    if slab.len() != l.n_params() {
                        return Err(DataError(format!(
                            "Layer {i} (cfc) weight count {} != expected {} in {path}",
                            slab.len(),
                            l.n_params()
                        )));
                    }
                    l.from_flat(&slab);
                    layers.push(Layer::Cfc(Box::new(l)));
                }
```

(i) `save_json` — add arm:
```rust
                Layer::Cfc(l) => NnLayerWeights {
                    w_bb: Some(l.w_bb.clone()),
                    b_bb: Some(l.b_bb.clone()),
                    w_ff1: Some(l.w_ff1.clone()),
                    b_ff1: Some(l.b_ff1.clone()),
                    w_ff2: Some(l.w_ff2.clone()),
                    b_ff2: Some(l.b_ff2.clone()),
                    w_ta: Some(l.w_ta.clone()),
                    b_ta: Some(l.b_ta.clone()),
                    w_tb: Some(l.w_tb.clone()),
                    b_tb: Some(l.b_tb.clone()),
                    ..NnLayerWeights::default()
                },
```

(j) `forward` dispatch — add arm:
```rust
                (Layer::Cfc(l), LayerState::Cfc { h }) => {
                    current = l.forward(&current, h);
                }
```

(k) `from_flat_weights_v2` — add arm:
```rust
                LayerSpec::Cfc {
                    input_size,
                    hidden_size,
                    backbone_units,
                } => {
                    if *input_size == 0 || *hidden_size == 0 || *backbone_units == 0 {
                        return Err(DataError(format!(
                            "from_flat_weights_v2: Cfc layer {} dims must be positive (input_size={}, hidden_size={}, backbone_units={})",
                            i, input_size, hidden_size, backbone_units
                        )));
                    }
                    if i == 0 {
                        layer_sizes.push(*input_size);
                    }
                    layer_sizes.push(*hidden_size);
                    Layer::Cfc(Box::new(CfcLayer::zeros(
                        *input_size,
                        *hidden_size,
                        *backbone_units,
                    )))
                }
```

- [ ] **Step 5: State variant.** In `src/rust/src/data/nn_state.rs`:

`LayerState` enum:
```rust
    /// CfC hidden state, flat like GRU. Reset zeros it.
    Cfc {
        h: Vec<f64>,
    },
```
`for_layer`:
```rust
            Layer::Cfc(l) => LayerState::Cfc {
                h: vec![0.0; l.hidden_size],
            },
```
`reset`:
```rust
            LayerState::Cfc { h } => {
                for v in h.iter_mut() {
                    *v = 0.0;
                }
            }
```
Append a test to the nn_state tests mod:
```rust
    #[test]
    fn layer_state_cfc_for_layer_and_reset() {
        use crate::data::neural::CfcLayer;
        let layer = Layer::Cfc(Box::new(CfcLayer::zeros(3, 4, 5)));
        let mut state = LayerState::for_layer(&layer);
        if let LayerState::Cfc { h } = &mut state {
            assert_eq!(h.len(), 4);
            h[0] = 9.0;
        } else {
            panic!("expected LayerState::Cfc");
        }
        state.reset();
        if let LayerState::Cfc { h } = &state {
            assert!(h.iter().all(|&v| v == 0.0));
        } else {
            panic!("expected LayerState::Cfc after reset");
        }
    }
```

- [ ] **Step 6: TOML spec.** In `src/rust/src/config.rs`, `TomlLayerSpec` enum add:
```rust
    Cfc {
        input_size: usize,
        hidden_size: usize,
        backbone_units: usize,
    },
```
`to_layer_spec()` add:
```rust
            TomlLayerSpec::Cfc {
                input_size,
                hidden_size,
                backbone_units,
            } => {
                if *input_size == 0 || *hidden_size == 0 || *backbone_units == 0 {
                    return Err(ParseError(format!(
                        "Cfc: input_size, hidden_size, backbone_units must be > 0 (got {input_size}, {hidden_size}, {backbone_units})"
                    )));
                }
                Ok(LayerSpec::Cfc {
                    input_size: *input_size,
                    hidden_size: *hidden_size,
                    backbone_units: *backbone_units,
                })
            }
```

- [ ] **Step 7: JSON round-trip test.** Append to `src/rust/src/data/neural/tests.rs` (mirror the mamba3 round-trip test naming there):

```rust
#[test]
fn cfc_json_round_trip_bit_identical() {
    use crate::data::neural::{LayerSpec, NeuralNetModel};
    let arch = vec![
        LayerSpec::Dense {
            input_size: 3,
            output_size: 4,
            activation: crate::data::neural::Activation::Tanh,
        },
        LayerSpec::Cfc {
            input_size: 4,
            hidden_size: 4,
            backbone_units: 5,
        },
        LayerSpec::Dense {
            input_size: 4,
            output_size: 2,
            activation: crate::data::neural::Activation::Linear,
        },
    ];
    let n: usize = 3 * 4 + 4 + (5 * 8 + 5 + 4 * (4 * 5 + 4)) + (4 * 2 + 2);
    let flat: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001 - 0.2).collect();
    let model = NeuralNetModel::from_flat_weights_v2(&arch, &flat, None, None).unwrap();
    assert_eq!(model.n_params(), n);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cfc_rt.json");
    model.save_json(path.to_str().unwrap()).unwrap();
    let loaded = NeuralNetModel::from_json_file(path.to_str().unwrap()).unwrap();
    assert_eq!(loaded.to_flat_weights(), model.to_flat_weights());
}
```

NOTE: check the exact signatures used by the existing mamba3 round-trip test in `tests.rs` first (`rg -n "round_trip" src/rust/src/data/neural/tests.rs`) and copy its construction/load idiom verbatim (`from_flat_weights_v2` arg list and the JSON-load entry point name differ across phases; the mamba3 test is the authority).

- [ ] **Step 8: Full Rust test run**

Run: `cargo test --manifest-path src/rust/Cargo.toml 2>&1 | tail -5`
Expected: all tests pass (existing + new). The `forward` dispatch `unreachable!` message string does not need updating (it names the invariant, not every variant).

- [ ] **Step 9: fmt + clippy + commit**

```bash
cargo fmt --manifest-path src/rust/Cargo.toml
cargo clippy --manifest-path src/rust/Cargo.toml -- -D warnings
git add src/rust/src/data/neural/layers/cfc.rs src/rust/src/data/neural/layers/mod.rs src/rust/src/data/neural/mod.rs src/rust/src/data/neural/tests.rs src/rust/src/data/nn_state.rs src/rust/src/config.rs
git commit -m "feat(cfc): Rust CfC cell -- struct, forward, LayerWeights, enum + TOML + state wiring

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Rust sLSTM layer end-to-end

**Files:**
- Create: `src/rust/src/data/neural/layers/slstm.rs`
- Modify: `src/rust/src/data/neural/layers/mod.rs`, `src/rust/src/data/neural/mod.rs`, `src/rust/src/data/nn_state.rs`, `src/rust/src/config.rs`
- Test: inline `#[cfg(test)]` in `slstm.rs` + round-trip in `src/rust/src/data/neural/tests.rs`

**Interfaces:**
- Consumes: Task 1 `stabilized_exp_gates`, `copy_mat_from_flat`, `copy_vec_from_flat`.
- Produces: `pub struct SlstmLayer { input_size, hidden_size, weight_ih, weight_hh, bias }` with `zeros(input_size, hidden_size)` and `forward(&self, x: &[f64], h: &mut [f64], c: &mut [f64], n: &mut [f64], m: &mut [f64]) -> Vec<f64>`; `LayerSpec::Slstm { input_size, hidden_size }` (tag `"slstm"`); `Layer::Slstm(SlstmLayer)` (NOT boxed — 2 matrices + 1 vec, same footprint class as GruLayer); `LayerState::Slstm { h, c, n, m: Vec<f64> }`; JSON keys `weight_ih, weight_hh, bias` (new `bias` field on NnLayerWeights).

- [ ] **Step 1: Create `src/rust/src/data/neural/layers/slstm.rs`:**

```rust
//! sLSTM cell (xLSTM, Beck et al. 2024) -- scalar state, exponential gating,
//! max-stabilizer. Cell-only: full recurrent matrices, single head, single bias.
//!
//! Gate order on the 4H axis: (i, f, z, o).
//!   (i~, f~, z~, o~) = W_ih @ x + W_hh @ h + b        per-unit row slices
//!   m' = max(f~ + m, i~)
//!   i' = exp(i~ - m');  f' = exp(f~ + m - m')
//!   c' = f'*c + i'*tanh(z~);   n' = f'*n + i'
//!   h' = sigmoid(o~) * c' / n'
//! No div-by-zero at t=0: n_1 = i' > 0 and every later step adds a positive i'.
//!
//! Canonical flat order: weight_ih row-major [4H, I], weight_hh row-major [4H, H], bias [4H].

use super::super::{Activation, LayerWeights};
use super::helpers::{copy_mat_from_flat, copy_vec_from_flat, dot_plus_bias, stabilized_exp_gates};

#[derive(Debug, Clone)]
pub struct SlstmLayer {
    pub input_size: usize,
    pub hidden_size: usize,
    pub weight_ih: Vec<Vec<f64>>, // [4H, I]
    pub weight_hh: Vec<Vec<f64>>, // [4H, H]
    pub bias: Vec<f64>,           // [4H]
}

impl SlstmLayer {
    pub fn zeros(input_size: usize, hidden_size: usize) -> Self {
        let four_h = 4 * hidden_size;
        Self {
            input_size,
            hidden_size,
            weight_ih: vec![vec![0.0; input_size]; four_h],
            weight_hh: vec![vec![0.0; hidden_size]; four_h],
            bias: vec![0.0; four_h],
        }
    }

    /// One step: reads x + state, updates (h, c, n, m) in place, returns h_new.
    pub fn forward(
        &self,
        x: &[f64],
        h: &mut [f64],
        c: &mut [f64],
        n: &mut [f64],
        m: &mut [f64],
    ) -> Vec<f64> {
        assert_eq!(x.len(), self.input_size);
        assert_eq!(h.len(), self.hidden_size);
        let hs = self.hidden_size;
        // All 4H preactivations against the PREVIOUS h, before any mutation.
        // Add order matches the torch mirror: (W_ih@x + b) + W_hh@h.
        let mut pre = vec![0.0; 4 * hs];
        for (r, p) in pre.iter_mut().enumerate() {
            *p = dot_plus_bias(&self.weight_ih[r], x, self.bias[r])
                + dot_plus_bias(&self.weight_hh[r], h, 0.0);
        }
        let mut h_new = vec![0.0; hs];
        for i in 0..hs {
            let i_pre = pre[i];
            let f_pre = pre[i + hs];
            let z = pre[i + 2 * hs].tanh();
            let o = Activation::Sigmoid.apply(pre[i + 3 * hs]);
            let (i_g, f_g, m_new) = stabilized_exp_gates(i_pre, f_pre, m[i]);
            c[i] = f_g * c[i] + i_g * z;
            n[i] = f_g * n[i] + i_g;
            m[i] = m_new;
            h_new[i] = o * (c[i] / n[i]);
        }
        h.copy_from_slice(&h_new);
        h_new
    }
}

impl LayerWeights for SlstmLayer {
    fn to_flat(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.n_params());
        for row in &self.weight_ih {
            v.extend_from_slice(row);
        }
        for row in &self.weight_hh {
            v.extend_from_slice(row);
        }
        v.extend_from_slice(&self.bias);
        v
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_flat(&mut self, flat: &[f64]) -> usize {
        let mut idx = 0;
        copy_mat_from_flat(&mut self.weight_ih, flat, &mut idx);
        copy_mat_from_flat(&mut self.weight_hh, flat, &mut idx);
        copy_vec_from_flat(&mut self.bias, flat, &mut idx);
        idx
    }

    fn n_params(&self) -> usize {
        4 * self.hidden_size * self.input_size
            + 4 * self.hidden_size * self.hidden_size
            + 4 * self.hidden_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn patterned(input_size: usize, hidden_size: usize) -> SlstmLayer {
        let mut l = SlstmLayer::zeros(input_size, hidden_size);
        let n = l.n_params();
        let flat: Vec<f64> = (0..n).map(|i| ((i % 17) as f64) * 0.05 - 0.4).collect();
        l.from_flat(&flat);
        l
    }

    #[test]
    fn flat_round_trip_is_bit_identical() {
        let l = patterned(3, 4);
        let flat = l.to_flat();
        assert_eq!(flat.len(), l.n_params());
        let mut l2 = SlstmLayer::zeros(3, 4);
        assert_eq!(l2.from_flat(&flat), flat.len());
        assert_eq!(l2.to_flat(), flat);
    }

    #[test]
    fn n_params_formula() {
        // 4HI + 4HH + 4H = 48 + 64 + 16 = 128
        assert_eq!(SlstmLayer::zeros(3, 4).n_params(), 128);
    }

    #[test]
    fn first_step_from_zero_state_is_finite() {
        // n starts at 0; the first update must not divide by zero (n_1 = i' > 0).
        let l = patterned(3, 4);
        let (mut h, mut c, mut n, mut m) = (vec![0.0; 4], vec![0.0; 4], vec![0.0; 4], vec![0.0; 4]);
        let out = l.forward(&[0.3, -0.7, 1.1], &mut h, &mut c, &mut n, &mut m);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(n.iter().all(|&v| v > 0.0), "n must be strictly positive after step 1");
    }

    #[test]
    fn stabilizer_survives_huge_preactivations() {
        // Bias +-300 drives i~/f~ far beyond exp overflow without the stabilizer.
        let mut l = SlstmLayer::zeros(2, 2);
        for j in 0..2 {
            l.bias[j] = 300.0; // i gates
            l.bias[j + 2] = -300.0; // f gates
        }
        let (mut h, mut c, mut n, mut m) = (vec![0.0; 2], vec![0.0; 2], vec![0.0; 2], vec![0.0; 2]);
        for _ in 0..10 {
            let out = l.forward(&[1.0, -1.0], &mut h, &mut c, &mut n, &mut m);
            assert!(out.iter().all(|v| v.is_finite()), "stabilizer failed: {out:?}");
        }
        // Flip: huge forget, huge negative input gate.
        let mut l2 = SlstmLayer::zeros(2, 2);
        for j in 0..2 {
            l2.bias[j] = -300.0;
            l2.bias[j + 2] = 300.0;
        }
        let (mut h, mut c, mut n, mut m) = (vec![0.0; 2], vec![0.0; 2], vec![0.0; 2], vec![0.0; 2]);
        for _ in 0..10 {
            let out = l2.forward(&[1.0, -1.0], &mut h, &mut c, &mut n, &mut m);
            assert!(out.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn state_evolves_deterministically() {
        let l = patterned(3, 4);
        let run = || {
            let (mut h, mut c, mut n, mut m) =
                (vec![0.0; 4], vec![0.0; 4], vec![0.0; 4], vec![0.0; 4]);
            let mut last = Vec::new();
            for t in 0..20 {
                last = l.forward(&[(t as f64) * 0.1, 0.5, -0.5], &mut h, &mut c, &mut n, &mut m);
            }
            last
        };
        assert_eq!(run(), run());
    }
}
```

- [ ] **Step 2: Register + wire.** Same six wiring sites as Task 2, with these exact arms:

`layers/mod.rs`: `pub(crate) mod slstm;` + `pub use slstm::SlstmLayer;`

`neural/mod.rs` import list: add `SlstmLayer`.

`Layer` enum:
```rust
    Slstm(SlstmLayer),
```
`Layer::input_size()`: `Layer::Slstm(l) => l.input_size,`
`LayerWeights for Layer` (3 matches): `Layer::Slstm(l) => l.to_flat(),` / `l.from_flat(flat),` / `l.n_params(),`

`NnLayerWeights`: one new field (weight_ih/weight_hh already exist for GRU/LSTM):
```rust
    // sLSTM single-bias field (GRU/LSTM use the bias_ih/bias_hh pair instead)
    #[serde(skip_serializing_if = "Option::is_none", default)]
    bias: Option<Vec<f64>>,
```

`LayerSpec` enum:
```rust
    Slstm {
        input_size: usize,
        hidden_size: usize,
    },
```
`LayerSpec::io()`:
```rust
            LayerSpec::Slstm {
                input_size,
                hidden_size,
            } => (*input_size, *hidden_size, "slstm"),
```

`from_v2_json` arm (slab-assembly, same closure style as the Cfc arm in Task 2 step 4h — redefine the `flat_mat`/`flat_vec` closures locally with "slstm" in the error strings):
```rust
                LayerSpec::Slstm {
                    input_size,
                    hidden_size,
                } => {
                    if *input_size == 0 || *hidden_size == 0 {
                        return Err(DataError(format!(
                            "Layer {i} (slstm) input_size and hidden_size must be positive in {path}"
                        )));
                    }
                    if i == 0 {
                        layer_sizes.push(*input_size);
                    }
                    layer_sizes.push(*hidden_size);

                    let key = format!("layer_{}", i);
                    let lw = file.weights.get(&key).ok_or_else(|| {
                        DataError(format!("Missing {} in weights in {}", key, path))
                    })?;
                    let flat_mat =
                        |name: &str, m: &Option<Vec<Vec<f64>>>| -> Result<Vec<f64>, DataError> {
                            let rows = m.as_ref().ok_or_else(|| {
                                DataError(format!("Layer {i} (slstm) missing {name} in {path}"))
                            })?;
                            Ok(rows.iter().flat_map(|r| r.iter().copied()).collect())
                        };
                    let flat_vec =
                        |name: &str, v: &Option<Vec<f64>>| -> Result<Vec<f64>, DataError> {
                            v.as_ref().cloned().ok_or_else(|| {
                                DataError(format!("Layer {i} (slstm) missing {name} in {path}"))
                            })
                        };

                    let mut slab = Vec::new();
                    slab.extend(flat_mat("weight_ih", &lw.weight_ih)?);
                    slab.extend(flat_mat("weight_hh", &lw.weight_hh)?);
                    slab.extend(flat_vec("bias", &lw.bias)?);

                    let mut l = SlstmLayer::zeros(*input_size, *hidden_size);
                    if slab.len() != l.n_params() {
                        return Err(DataError(format!(
                            "Layer {i} (slstm) weight count {} != expected {} in {path}",
                            slab.len(),
                            l.n_params()
                        )));
                    }
                    l.from_flat(&slab);
                    layers.push(Layer::Slstm(l));
                }
```

`save_json` arm:
```rust
                Layer::Slstm(l) => NnLayerWeights {
                    weight_ih: Some(l.weight_ih.clone()),
                    weight_hh: Some(l.weight_hh.clone()),
                    bias: Some(l.bias.clone()),
                    ..NnLayerWeights::default()
                },
```

`forward` dispatch arm:
```rust
                (Layer::Slstm(l), LayerState::Slstm { h, c, n, m }) => {
                    current = l.forward(&current, h, c, n, m);
                }
```

`from_flat_weights_v2` arm:
```rust
                LayerSpec::Slstm {
                    input_size,
                    hidden_size,
                } => {
                    if *input_size == 0 || *hidden_size == 0 {
                        return Err(DataError(format!(
                            "from_flat_weights_v2: Slstm layer {} dims must be positive (input_size={}, hidden_size={})",
                            i, input_size, hidden_size
                        )));
                    }
                    if i == 0 {
                        layer_sizes.push(*input_size);
                    }
                    layer_sizes.push(*hidden_size);
                    Layer::Slstm(SlstmLayer::zeros(*input_size, *hidden_size))
                }
```

- [ ] **Step 3: State variant.** `nn_state.rs`:

```rust
    /// sLSTM state: hidden h, cell c, normalizer n, stabilizer m. All zero-init
    /// (m_0 = 0 per the xLSTM reference; no div-by-zero since n_1 = i' > 0).
    Slstm {
        h: Vec<f64>,
        c: Vec<f64>,
        n: Vec<f64>,
        m: Vec<f64>,
    },
```
`for_layer`:
```rust
            Layer::Slstm(l) => LayerState::Slstm {
                h: vec![0.0; l.hidden_size],
                c: vec![0.0; l.hidden_size],
                n: vec![0.0; l.hidden_size],
                m: vec![0.0; l.hidden_size],
            },
```
`reset`:
```rust
            LayerState::Slstm { h, c, n, m } => {
                for vec in [h, c, n, m] {
                    for v in vec.iter_mut() {
                        *v = 0.0;
                    }
                }
            }
```
Test in nn_state tests mod:
```rust
    #[test]
    fn layer_state_slstm_for_layer_and_reset() {
        use crate::data::neural::SlstmLayer;
        let layer = Layer::Slstm(SlstmLayer::zeros(3, 4));
        let mut state = LayerState::for_layer(&layer);
        if let LayerState::Slstm { h, c, n, m } = &mut state {
            assert!(h.len() == 4 && c.len() == 4 && n.len() == 4 && m.len() == 4);
            h[0] = 1.0;
            c[1] = 2.0;
            n[2] = 3.0;
            m[3] = 4.0;
        } else {
            panic!("expected LayerState::Slstm");
        }
        state.reset();
        if let LayerState::Slstm { h, c, n, m } = &state {
            for vec in [h, c, n, m] {
                assert!(vec.iter().all(|&v| v == 0.0));
            }
        } else {
            panic!("expected LayerState::Slstm after reset");
        }
    }
```

- [ ] **Step 4: TOML spec.** `config.rs` `TomlLayerSpec`:
```rust
    Slstm {
        input_size: usize,
        hidden_size: usize,
    },
```
`to_layer_spec()`:
```rust
            TomlLayerSpec::Slstm {
                input_size,
                hidden_size,
            } => {
                if *input_size == 0 || *hidden_size == 0 {
                    return Err(ParseError(format!(
                        "Slstm: input_size and hidden_size must be > 0 (got {input_size}, {hidden_size})"
                    )));
                }
                Ok(LayerSpec::Slstm {
                    input_size: *input_size,
                    hidden_size: *hidden_size,
                })
            }
```

- [ ] **Step 5: JSON round-trip test** in `neural/tests.rs` (same idiom as the Task 2 cfc round-trip; copy the mamba3 test's construction/load entry points):

```rust
#[test]
fn slstm_json_round_trip_bit_identical() {
    use crate::data::neural::{LayerSpec, NeuralNetModel};
    let arch = vec![
        LayerSpec::Dense {
            input_size: 3,
            output_size: 4,
            activation: crate::data::neural::Activation::Tanh,
        },
        LayerSpec::Slstm {
            input_size: 4,
            hidden_size: 4,
        },
        LayerSpec::Dense {
            input_size: 4,
            output_size: 2,
            activation: crate::data::neural::Activation::Linear,
        },
    ];
    let n: usize = (3 * 4 + 4) + (4 * 4 * 4 + 4 * 4 * 4 + 4 * 4) + (4 * 2 + 2);
    let flat: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001 - 0.15).collect();
    let model = NeuralNetModel::from_flat_weights_v2(&arch, &flat, None, None).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("slstm_rt.json");
    model.save_json(path.to_str().unwrap()).unwrap();
    let loaded = NeuralNetModel::from_json_file(path.to_str().unwrap()).unwrap();
    assert_eq!(loaded.to_flat_weights(), model.to_flat_weights());
}
```

- [ ] **Step 6: Run, fmt, clippy, commit**

Run: `cargo test --manifest-path src/rust/Cargo.toml slstm 2>&1 | tail -5` -> all PASS, then the full suite `cargo test --manifest-path src/rust/Cargo.toml 2>&1 | tail -3`.

```bash
cargo fmt --manifest-path src/rust/Cargo.toml
cargo clippy --manifest-path src/rust/Cargo.toml -- -D warnings
git add src/rust/src/data/neural/layers/slstm.rs src/rust/src/data/neural/layers/mod.rs src/rust/src/data/neural/mod.rs src/rust/src/data/neural/tests.rs src/rust/src/data/nn_state.rs src/rust/src/config.rs
git commit -m "feat(slstm): Rust sLSTM cell -- exponential gating + stabilizer, full wiring

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Rust mLSTM layer end-to-end

**Files:**
- Create: `src/rust/src/data/neural/layers/mlstm.rs`
- Modify: `src/rust/src/data/neural/layers/mod.rs`, `src/rust/src/data/neural/mod.rs`, `src/rust/src/data/nn_state.rs`, `src/rust/src/config.rs`
- Test: inline `#[cfg(test)]` in `mlstm.rs` + round-trip in `src/rust/src/data/neural/tests.rs`

**Interfaces:**
- Consumes: Task 1 helpers.
- Produces: `pub struct MlstmLayer { input_size, hidden_size, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o, w_i, b_i, w_f, b_f }` with `zeros(input_size, hidden_size)` and `forward(&self, x: &[f64], c: &mut nalgebra::DMatrix<f64>, n: &mut [f64], m: &mut f64) -> Vec<f64>`; `LayerSpec::Mlstm { input_size, hidden_size }` (tag `"mlstm"`); `Layer::Mlstm(Box<MlstmLayer>)`; `LayerState::Mlstm { c: DMatrix<f64>, n: Vec<f64>, m: f64 }`; JSON keys reuse `w_q/b_q/w_k/b_k/w_v/b_v/w_o/b_o` (Transformer fields) plus new `w_i` (vec), `b_i` (f64), `w_f` (vec), `b_f` (f64).

- [ ] **Step 1: Create `src/rust/src/data/neural/layers/mlstm.rs`:**

```rust
//! mLSTM cell (xLSTM, Beck et al. 2024) -- matrix memory, covariance update,
//! exponential gating with scalar stabilizer. Cell-only, single head, d_qk = d_v = H.
//! No recurrent weights (paper-faithful: all gates and projections read x only).
//!
//!   q = W_q x + b_q;  k = (W_k x + b_k)/sqrt(H);  v = W_v x + b_v
//!   i~ = w_i . x + b_i (scalar);  f~ = w_f . x + b_f (scalar)
//!   m' = max(f~ + m, i~);  i' = exp(i~ - m');  f' = exp(f~ + m - m')
//!   C' = f' C + i' (v k^T);   n' = f' n + i' k
//!   h' = sigmoid(W_o x + b_o) * (C' q) / max(|n' . q|, 1)
//!
//! Canonical flat order: w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o, w_i, b_i, w_f, b_f
//! (matrices row-major, scalars as single elements).

use super::super::{Activation, LayerWeights};
use super::helpers::{copy_mat_from_flat, copy_vec_from_flat, dot_plus_bias, stabilized_exp_gates};

#[derive(Debug, Clone)]
pub struct MlstmLayer {
    pub input_size: usize,
    pub hidden_size: usize,
    pub w_q: Vec<Vec<f64>>, // [H, I]
    pub b_q: Vec<f64>,      // [H]
    pub w_k: Vec<Vec<f64>>, // [H, I]
    pub b_k: Vec<f64>,      // [H]
    pub w_v: Vec<Vec<f64>>, // [H, I]
    pub b_v: Vec<f64>,      // [H]
    pub w_o: Vec<Vec<f64>>, // [H, I]
    pub b_o: Vec<f64>,      // [H]
    pub w_i: Vec<f64>,      // [I]
    pub b_i: f64,
    pub w_f: Vec<f64>, // [I]
    pub b_f: f64,
}

impl MlstmLayer {
    pub fn zeros(input_size: usize, hidden_size: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            w_q: vec![vec![0.0; input_size]; hidden_size],
            b_q: vec![0.0; hidden_size],
            w_k: vec![vec![0.0; input_size]; hidden_size],
            b_k: vec![0.0; hidden_size],
            w_v: vec![vec![0.0; input_size]; hidden_size],
            b_v: vec![0.0; hidden_size],
            w_o: vec![vec![0.0; input_size]; hidden_size],
            b_o: vec![0.0; hidden_size],
            w_i: vec![0.0; input_size],
            b_i: 0.0,
            w_f: vec![0.0; input_size],
            b_f: 0.0,
        }
    }

    /// One step: reads x, updates (C, n, m) in place, returns h_new.
    pub fn forward(
        &self,
        x: &[f64],
        c: &mut nalgebra::DMatrix<f64>,
        n: &mut [f64],
        m: &mut f64,
    ) -> Vec<f64> {
        assert_eq!(x.len(), self.input_size);
        let hs = self.hidden_size;
        let sqrt_h = (hs as f64).sqrt();
        let q: Vec<f64> = (0..hs)
            .map(|j| dot_plus_bias(&self.w_q[j], x, self.b_q[j]))
            .collect();
        let k: Vec<f64> = (0..hs)
            .map(|j| dot_plus_bias(&self.w_k[j], x, self.b_k[j]) / sqrt_h)
            .collect();
        let v: Vec<f64> = (0..hs)
            .map(|j| dot_plus_bias(&self.w_v[j], x, self.b_v[j]))
            .collect();
        let i_pre = dot_plus_bias(&self.w_i, x, self.b_i);
        let f_pre = dot_plus_bias(&self.w_f, x, self.b_f);
        let (i_g, f_g, m_new) = stabilized_exp_gates(i_pre, f_pre, *m);
        *m = m_new;
        // C' = f' C + i' (v k^T); association i' * (v_r * k_col) matches
        // torch `ig * torch.outer(v, k)` in the Python mirror.
        for r in 0..hs {
            for col in 0..hs {
                c[(r, col)] = f_g * c[(r, col)] + i_g * (v[r] * k[col]);
            }
        }
        for (j, nj) in n.iter_mut().enumerate() {
            *nj = f_g * *nj + i_g * k[j];
        }
        let nq: f64 = n.iter().zip(&q).map(|(a, b)| a * b).sum();
        let denom = nq.abs().max(1.0);
        let mut out = vec![0.0; hs];
        for (r, o_r) in out.iter_mut().enumerate() {
            let cq: f64 = (0..hs).map(|col| c[(r, col)] * q[col]).sum();
            let o = Activation::Sigmoid.apply(dot_plus_bias(&self.w_o[r], x, self.b_o[r]));
            *o_r = o * (cq / denom);
        }
        out
    }
}

impl LayerWeights for MlstmLayer {
    fn to_flat(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.n_params());
        for (mat, bias) in [
            (&self.w_q, &self.b_q),
            (&self.w_k, &self.b_k),
            (&self.w_v, &self.b_v),
            (&self.w_o, &self.b_o),
        ] {
            for row in mat.iter() {
                v.extend_from_slice(row);
            }
            v.extend_from_slice(bias);
        }
        v.extend_from_slice(&self.w_i);
        v.push(self.b_i);
        v.extend_from_slice(&self.w_f);
        v.push(self.b_f);
        v
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_flat(&mut self, flat: &[f64]) -> usize {
        let mut idx = 0;
        copy_mat_from_flat(&mut self.w_q, flat, &mut idx);
        copy_vec_from_flat(&mut self.b_q, flat, &mut idx);
        copy_mat_from_flat(&mut self.w_k, flat, &mut idx);
        copy_vec_from_flat(&mut self.b_k, flat, &mut idx);
        copy_mat_from_flat(&mut self.w_v, flat, &mut idx);
        copy_vec_from_flat(&mut self.b_v, flat, &mut idx);
        copy_mat_from_flat(&mut self.w_o, flat, &mut idx);
        copy_vec_from_flat(&mut self.b_o, flat, &mut idx);
        copy_vec_from_flat(&mut self.w_i, flat, &mut idx);
        self.b_i = flat[idx];
        idx += 1;
        copy_vec_from_flat(&mut self.w_f, flat, &mut idx);
        self.b_f = flat[idx];
        idx += 1;
        idx
    }

    fn n_params(&self) -> usize {
        4 * (self.hidden_size * self.input_size + self.hidden_size) + 2 * (self.input_size + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn patterned(input_size: usize, hidden_size: usize) -> MlstmLayer {
        let mut l = MlstmLayer::zeros(input_size, hidden_size);
        let n = l.n_params();
        let flat: Vec<f64> = (0..n).map(|i| ((i % 13) as f64) * 0.06 - 0.36).collect();
        l.from_flat(&flat);
        l
    }

    fn zero_state(hs: usize) -> (nalgebra::DMatrix<f64>, Vec<f64>, f64) {
        (nalgebra::DMatrix::zeros(hs, hs), vec![0.0; hs], 0.0)
    }

    #[test]
    fn flat_round_trip_is_bit_identical() {
        let l = patterned(3, 4);
        let flat = l.to_flat();
        assert_eq!(flat.len(), l.n_params());
        let mut l2 = MlstmLayer::zeros(3, 4);
        assert_eq!(l2.from_flat(&flat), flat.len());
        assert_eq!(l2.to_flat(), flat);
    }

    #[test]
    fn n_params_formula() {
        // 4(HI + H) + 2(I + 1) = 4*(12 + 4) + 2*4 = 72
        assert_eq!(MlstmLayer::zeros(3, 4).n_params(), 72);
    }

    #[test]
    fn denominator_clamp_path_is_finite() {
        // Tiny weights -> |n . q| << 1 -> denominator clamps to 1.0.
        let mut l = MlstmLayer::zeros(3, 4);
        let n_par = l.n_params();
        let flat: Vec<f64> = (0..n_par).map(|i| ((i % 7) as f64) * 1e-6).collect();
        l.from_flat(&flat);
        let (mut c, mut n, mut m) = zero_state(4);
        let out = l.forward(&[0.4, -0.2, 0.8], &mut c, &mut n, &mut m);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn stabilizer_survives_huge_gate_preactivations() {
        let mut l = patterned(3, 4);
        l.b_i = 300.0;
        l.b_f = -300.0;
        let (mut c, mut n, mut m) = zero_state(4);
        for _ in 0..10 {
            let out = l.forward(&[1.0, -1.0, 0.5], &mut c, &mut n, &mut m);
            assert!(out.iter().all(|v| v.is_finite()));
        }
        l.b_i = -300.0;
        l.b_f = 300.0;
        let (mut c, mut n, mut m) = zero_state(4);
        for _ in 0..10 {
            let out = l.forward(&[1.0, -1.0, 0.5], &mut c, &mut n, &mut m);
            assert!(out.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn hundred_steps_finite_and_deterministic() {
        let l = patterned(4, 6);
        let run = || {
            let (mut c, mut n, mut m) = zero_state(6);
            let mut last = Vec::new();
            for t in 0..100 {
                let x = vec![(t as f64 * 0.07).sin(), 0.3, -0.9, 0.1];
                last = l.forward(&x, &mut c, &mut n, &mut m);
                assert!(last.iter().all(|v| v.is_finite()));
            }
            last
        };
        assert_eq!(run(), run());
    }
}
```

- [ ] **Step 2: Register + wire** (same sites as Tasks 2/3):

`layers/mod.rs`: `pub(crate) mod mlstm;` + `pub use mlstm::MlstmLayer;`
`neural/mod.rs` import: add `MlstmLayer`.

`Layer` enum:
```rust
    // Boxed for enum-variant size uniformity (4 matrix + 4 bias + 2 gate vectors).
    Mlstm(Box<MlstmLayer>),
```
`input_size()`: `Layer::Mlstm(l) => l.input_size,`
`LayerWeights for Layer`: `Layer::Mlstm(l) => l.to_flat(),` / `l.from_flat(flat),` / `l.n_params(),`

`NnLayerWeights` — add the four gate fields (w_q..b_o already exist from Transformer):
```rust
    // mLSTM scalar-gate fields (cfc-xlstm probes)
    #[serde(skip_serializing_if = "Option::is_none", default)]
    w_i: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    b_i: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    w_f: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    b_f: Option<f64>,
```

`LayerSpec`:
```rust
    Mlstm {
        input_size: usize,
        hidden_size: usize,
    },
```
`io()`:
```rust
            LayerSpec::Mlstm {
                input_size,
                hidden_size,
            } => (*input_size, *hidden_size, "mlstm"),
```

`from_v2_json` arm:
```rust
                LayerSpec::Mlstm {
                    input_size,
                    hidden_size,
                } => {
                    if *input_size == 0 || *hidden_size == 0 {
                        return Err(DataError(format!(
                            "Layer {i} (mlstm) input_size and hidden_size must be positive in {path}"
                        )));
                    }
                    if i == 0 {
                        layer_sizes.push(*input_size);
                    }
                    layer_sizes.push(*hidden_size);

                    let key = format!("layer_{}", i);
                    let lw = file.weights.get(&key).ok_or_else(|| {
                        DataError(format!("Missing {} in weights in {}", key, path))
                    })?;
                    let flat_mat =
                        |name: &str, m: &Option<Vec<Vec<f64>>>| -> Result<Vec<f64>, DataError> {
                            let rows = m.as_ref().ok_or_else(|| {
                                DataError(format!("Layer {i} (mlstm) missing {name} in {path}"))
                            })?;
                            Ok(rows.iter().flat_map(|r| r.iter().copied()).collect())
                        };
                    let flat_vec =
                        |name: &str, v: &Option<Vec<f64>>| -> Result<Vec<f64>, DataError> {
                            v.as_ref().cloned().ok_or_else(|| {
                                DataError(format!("Layer {i} (mlstm) missing {name} in {path}"))
                            })
                        };
                    let scalar = |name: &str, v: &Option<f64>| -> Result<f64, DataError> {
                        v.ok_or_else(|| {
                            DataError(format!("Layer {i} (mlstm) missing {name} in {path}"))
                        })
                    };

                    let mut slab = Vec::new();
                    slab.extend(flat_mat("w_q", &lw.w_q)?);
                    slab.extend(flat_vec("b_q", &lw.b_q)?);
                    slab.extend(flat_mat("w_k", &lw.w_k)?);
                    slab.extend(flat_vec("b_k", &lw.b_k)?);
                    slab.extend(flat_mat("w_v", &lw.w_v)?);
                    slab.extend(flat_vec("b_v", &lw.b_v)?);
                    slab.extend(flat_mat("w_o", &lw.w_o)?);
                    slab.extend(flat_vec("b_o", &lw.b_o)?);
                    slab.extend(flat_vec("w_i", &lw.w_i)?);
                    slab.push(scalar("b_i", &lw.b_i)?);
                    slab.extend(flat_vec("w_f", &lw.w_f)?);
                    slab.push(scalar("b_f", &lw.b_f)?);

                    let mut l = MlstmLayer::zeros(*input_size, *hidden_size);
                    if slab.len() != l.n_params() {
                        return Err(DataError(format!(
                            "Layer {i} (mlstm) weight count {} != expected {} in {path}",
                            slab.len(),
                            l.n_params()
                        )));
                    }
                    l.from_flat(&slab);
                    layers.push(Layer::Mlstm(Box::new(l)));
                }
```

`save_json` arm:
```rust
                Layer::Mlstm(l) => NnLayerWeights {
                    w_q: Some(l.w_q.clone()),
                    b_q: Some(l.b_q.clone()),
                    w_k: Some(l.w_k.clone()),
                    b_k: Some(l.b_k.clone()),
                    w_v: Some(l.w_v.clone()),
                    b_v: Some(l.b_v.clone()),
                    w_o: Some(l.w_o.clone()),
                    b_o: Some(l.b_o.clone()),
                    w_i: Some(l.w_i.clone()),
                    b_i: Some(l.b_i),
                    w_f: Some(l.w_f.clone()),
                    b_f: Some(l.b_f),
                    ..NnLayerWeights::default()
                },
```

`forward` dispatch arm:
```rust
                (Layer::Mlstm(l), LayerState::Mlstm { c, n, m }) => {
                    current = l.forward(&current, c, n, m);
                }
```

`from_flat_weights_v2` arm:
```rust
                LayerSpec::Mlstm {
                    input_size,
                    hidden_size,
                } => {
                    if *input_size == 0 || *hidden_size == 0 {
                        return Err(DataError(format!(
                            "from_flat_weights_v2: Mlstm layer {} dims must be positive (input_size={}, hidden_size={})",
                            i, input_size, hidden_size
                        )));
                    }
                    if i == 0 {
                        layer_sizes.push(*input_size);
                    }
                    layer_sizes.push(*hidden_size);
                    Layer::Mlstm(Box::new(MlstmLayer::zeros(*input_size, *hidden_size)))
                }
```

- [ ] **Step 3: State variant.** `nn_state.rs`:

```rust
    /// mLSTM state: matrix memory C (H x H), normalizer n (H,), scalar stabilizer m.
    /// Reset zeros all three.
    Mlstm {
        c: nalgebra::DMatrix<f64>,
        n: Vec<f64>,
        m: f64,
    },
```
`for_layer`:
```rust
            Layer::Mlstm(l) => LayerState::Mlstm {
                c: nalgebra::DMatrix::<f64>::zeros(l.hidden_size, l.hidden_size),
                n: vec![0.0; l.hidden_size],
                m: 0.0,
            },
```
`reset`:
```rust
            LayerState::Mlstm { c, n, m } => {
                c.fill(0.0);
                for v in n.iter_mut() {
                    *v = 0.0;
                }
                *m = 0.0;
            }
```
Test:
```rust
    #[test]
    fn layer_state_mlstm_for_layer_and_reset() {
        use crate::data::neural::MlstmLayer;
        let layer = Layer::Mlstm(Box::new(MlstmLayer::zeros(3, 4)));
        let mut state = LayerState::for_layer(&layer);
        if let LayerState::Mlstm { c, n, m } = &mut state {
            assert_eq!(c.shape(), (4, 4));
            assert_eq!(n.len(), 4);
            c[(0, 0)] = 5.0;
            n[1] = 2.0;
            *m = 7.0;
        } else {
            panic!("expected LayerState::Mlstm");
        }
        state.reset();
        if let LayerState::Mlstm { c, n, m } = &state {
            assert!(c.iter().all(|&v| v == 0.0));
            assert!(n.iter().all(|&v| v == 0.0));
            assert_eq!(*m, 0.0);
        } else {
            panic!("expected LayerState::Mlstm after reset");
        }
    }
```

- [ ] **Step 4: TOML spec.** `config.rs`:
```rust
    Mlstm {
        input_size: usize,
        hidden_size: usize,
    },
```
```rust
            TomlLayerSpec::Mlstm {
                input_size,
                hidden_size,
            } => {
                if *input_size == 0 || *hidden_size == 0 {
                    return Err(ParseError(format!(
                        "Mlstm: input_size and hidden_size must be > 0 (got {input_size}, {hidden_size})"
                    )));
                }
                Ok(LayerSpec::Mlstm {
                    input_size: *input_size,
                    hidden_size: *hidden_size,
                })
            }
```

- [ ] **Step 5: JSON round-trip test** in `neural/tests.rs`:

```rust
#[test]
fn mlstm_json_round_trip_bit_identical() {
    use crate::data::neural::{LayerSpec, NeuralNetModel};
    let arch = vec![
        LayerSpec::Dense {
            input_size: 3,
            output_size: 4,
            activation: crate::data::neural::Activation::Tanh,
        },
        LayerSpec::Mlstm {
            input_size: 4,
            hidden_size: 4,
        },
        LayerSpec::Dense {
            input_size: 4,
            output_size: 2,
            activation: crate::data::neural::Activation::Linear,
        },
    ];
    let n: usize = (3 * 4 + 4) + (4 * (4 * 4 + 4) + 2 * (4 + 1)) + (4 * 2 + 2);
    let flat: Vec<f64> = (0..n).map(|i| (i as f64) * 0.002 - 0.25).collect();
    let model = NeuralNetModel::from_flat_weights_v2(&arch, &flat, None, None).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mlstm_rt.json");
    model.save_json(path.to_str().unwrap()).unwrap();
    let loaded = NeuralNetModel::from_json_file(path.to_str().unwrap()).unwrap();
    assert_eq!(loaded.to_flat_weights(), model.to_flat_weights());
}
```

- [ ] **Step 6: Run, fmt, clippy, commit**

Run: `cargo test --manifest-path src/rust/Cargo.toml mlstm 2>&1 | tail -5` then full suite.

```bash
cargo fmt --manifest-path src/rust/Cargo.toml
cargo clippy --manifest-path src/rust/Cargo.toml -- -D warnings
git add src/rust/src/data/neural/layers/mlstm.rs src/rust/src/data/neural/layers/mod.rs src/rust/src/data/neural/mod.rs src/rust/src/data/neural/tests.rs src/rust/src/data/nn_state.rs src/rust/src/config.rs
git commit -m "feat(mlstm): Rust mLSTM cell -- matrix memory, covariance update, full wiring

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Python schemas, sizing, encoding, init

**Files:**
- Modify: `src/python/aerocapture/training/rl/schemas.py`, `src/python/aerocapture/training/config.py`, `src/python/aerocapture/training/encoding.py`, `src/python/aerocapture/training/initialization_v2.py`
- Test: `tests/test_cfc_encoding.py`, `tests/test_xlstm_encoding.py`, `tests/test_init_v2_cfc_xlstm.py`

**Interfaces:**
- Consumes: flat orders defined in Tasks 2-4 (they are the canonical layouts these specs/fills MUST match element-for-element).
- Produces: `CfcSpec`/`SlstmSpec`/`MlstmSpec` pydantic classes in the `LayerSpec` union; `_layer_n_params`/`_layer_output_size` arms (`cfc -> B(I+H)+B+4(HB+H)`, `slstm -> 4HI+4HH+4H`, `mlstm -> 4(HI+H)+2(I+1)`; output size = hidden_size for all three); `_cfc_specs`/`_slstm_specs`/`_mlstm_specs` ParamSpec generators; `_fill_cfc`/`_fill_slstm`/`_fill_mlstm` population fills.

- [ ] **Step 1: Write failing tests** — `tests/test_cfc_encoding.py`:

```python
"""ParamSpec width + bound checks for the CfC probe layer."""

from __future__ import annotations

from aerocapture.training.config import _layer_n_params, _layer_output_size
from aerocapture.training.encoding import _layer_param_specs
from aerocapture.training.rl.schemas import CfcSpec


def test_cfc_spec_width_matches_n_params() -> None:
    spec = CfcSpec(type="cfc", input_size=3, hidden_size=4, backbone_units=5)
    specs = _layer_param_specs(spec, layer_idx=0, bound_multiplier=2.0)
    # B(I+H) + B + 4(HB + H) = 5*7 + 5 + 4*24 = 136
    assert len(specs) == 136
    assert _layer_n_params(spec.model_dump()) == 136
    assert _layer_output_size(spec.model_dump()) == 4


def test_cfc_spec_order_starts_with_backbone() -> None:
    spec = CfcSpec(type="cfc", input_size=3, hidden_size=4, backbone_units=5)
    specs = _layer_param_specs(spec, layer_idx=0, bound_multiplier=1.0)
    assert specs[0].name.startswith("w_bb")
    assert specs[5 * 7].name.startswith("b_bb")
    assert specs[-1].name.startswith("b_tb")
```

`tests/test_xlstm_encoding.py`:

```python
"""ParamSpec width + forget-bound checks for the sLSTM/mLSTM probe layers."""

from __future__ import annotations

from aerocapture.training.config import _layer_n_params, _layer_output_size
from aerocapture.training.encoding import _layer_param_specs
from aerocapture.training.rl.schemas import MlstmSpec, SlstmSpec


def test_slstm_width_and_forget_slice_bounds() -> None:
    spec = SlstmSpec(type="slstm", input_size=3, hidden_size=4)
    specs = _layer_param_specs(spec, layer_idx=0, bound_multiplier=1.0)
    # 4HI + 4HH + 4H = 48 + 64 + 16 = 128
    assert len(specs) == 128
    assert _layer_n_params(spec.model_dump()) == 128
    assert _layer_output_size(spec.model_dump()) == 4
    bias = specs[48 + 64 :]
    h = 4
    for j, ps in enumerate(bias):
        if h <= j < 2 * h:  # forget slice (gate order i, f, z, o)
            assert ps.p_max == 3.0, f"forget bias {j} bound {ps.p_max}"
        else:
            assert ps.p_max == 0.1, f"bias {j} bound {ps.p_max}"


def test_mlstm_width_and_forget_bound() -> None:
    spec = MlstmSpec(type="mlstm", input_size=3, hidden_size=4)
    specs = _layer_param_specs(spec, layer_idx=0, bound_multiplier=1.0)
    # 4(HI + H) + 2(I + 1) = 64 + 8 = 72
    assert len(specs) == 72
    assert _layer_n_params(spec.model_dump()) == 72
    assert _layer_output_size(spec.model_dump()) == 4
    assert specs[-1].name.startswith("b_f")
    assert specs[-1].p_max == 3.0  # wide bound for the +2.0 forget center
    assert specs[-(3 + 1) - 1].name.startswith("b_i")  # b_i sits before w_f (len I=3) + b_f
    assert specs[-5].p_max == 0.1
```

`tests/test_init_v2_cfc_xlstm.py`:

```python
"""init_v2_population centers for the cfc/slstm/mlstm probe layers."""

from __future__ import annotations

import numpy as np
from aerocapture.training.initialization_v2 import init_v2_population


def _arch(mid: dict) -> list[dict]:
    return [
        {"type": "dense", "input_size": 3, "output_size": 4, "activation": "tanh"},
        mid,
        {"type": "dense", "input_size": 4, "output_size": 2, "activation": "linear"},
    ]


def test_cfc_population_finite_and_shaped() -> None:
    arch = _arch({"type": "cfc", "input_size": 4, "hidden_size": 4, "backbone_units": 5})
    pop = init_v2_population(arch, n_pop=8, bound_multiplier=2.0, rng=np.random.default_rng(0))
    # dense 16, cfc 5*8+5+4*(4*5+4) = 141, dense 10 -> 167  (dense1 = 3*4+4 = 16, dense2 = 4*2+2 = 10)
    assert pop.shape == (8, 16 + 141 + 10)
    assert np.all(np.isfinite(pop))


def test_slstm_forget_bias_centered_at_two() -> None:
    h, i = 4, 4
    arch = _arch({"type": "slstm", "input_size": i, "hidden_size": h})
    pop = init_v2_population(arch, n_pop=64, bound_multiplier=2.0, rng=np.random.default_rng(1))
    dense1 = 3 * 4 + 4
    b0 = dense1 + 4 * h * i + 4 * h * h  # bias start inside the slstm slab
    forget = pop[:, b0 + h : b0 + 2 * h]
    other = pop[:, b0 : b0 + h]
    assert abs(float(forget.mean()) - 2.0) < 0.1
    assert abs(float(other.mean())) < 0.1


def test_mlstm_forget_bias_centered_at_two() -> None:
    h, i = 4, 4
    arch = _arch({"type": "mlstm", "input_size": i, "hidden_size": h})
    pop = init_v2_population(arch, n_pop=64, bound_multiplier=2.0, rng=np.random.default_rng(2))
    dense1 = 3 * 4 + 4
    b_f_idx = dense1 + 4 * (h * i + h) + 2 * (i + 1) - 1  # last mlstm element
    assert abs(float(pop[:, b_f_idx].mean()) - 2.0) < 0.1
    b_i_idx = dense1 + 4 * (h * i + h) + i  # w_i (I) then b_i
    assert abs(float(pop[:, b_i_idx].mean())) < 0.1
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cfc_encoding.py tests/test_xlstm_encoding.py tests/test_init_v2_cfc_xlstm.py -x 2>&1 | tail -5`
Expected: ImportError (`CfcSpec` not defined).

- [ ] **Step 3: schemas.py** — add after `Mamba3Spec` (docstring pattern: PSO-only, pointer to this spec):

```python
class CfcSpec(BaseModel):
    """CfC (closed-form continuous-time) cell -- PSO-only probe layer.

    ncps "default" mode, one backbone layer, dt fixed at one guidance tick.
    See docs/superpowers/specs/2026-07-07-cfc-xlstm-probes-design.md.
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["cfc"]
    input_size: int = Field(ge=1)
    hidden_size: int = Field(ge=1)
    backbone_units: int = Field(ge=1)


class SlstmSpec(BaseModel):
    """sLSTM cell (xLSTM) -- PSO-only probe layer. Exponential gating + stabilizer.

    See docs/superpowers/specs/2026-07-07-cfc-xlstm-probes-design.md.
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["slstm"]
    input_size: int = Field(ge=1)
    hidden_size: int = Field(ge=1)


class MlstmSpec(BaseModel):
    """mLSTM cell (xLSTM) -- PSO-only probe layer. Matrix memory, single head,
    d_qk = d_v = hidden_size.

    See docs/superpowers/specs/2026-07-07-cfc-xlstm-probes-design.md.
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["mlstm"]
    input_size: int = Field(ge=1)
    hidden_size: int = Field(ge=1)
```

Extend the union:
```python
LayerSpec = Annotated[
    DenseSpec | GruSpec | LstmSpec | WindowSpec | TransformerSpec | MambaSpec | Mamba3Spec | CfcSpec | SlstmSpec | MlstmSpec,
    Discriminator("type"),
]
```

- [ ] **Step 4: config.py** — in `_layer_n_params` (before the final `raise ValueError`):

```python
    if ltype == "cfc":
        h = int(entry["hidden_size"])
        i = int(entry["input_size"])
        b = int(entry["backbone_units"])
        return b * (i + h) + b + 4 * (h * b + h)
    if ltype == "slstm":
        h = int(entry["hidden_size"])
        i = int(entry["input_size"])
        return 4 * h * i + 4 * h * h + 4 * h
    if ltype == "mlstm":
        h = int(entry["hidden_size"])
        i = int(entry["input_size"])
        return 4 * (h * i + h) + 2 * (i + 1)
```

In `_layer_output_size` (before the final `raise`):
```python
    if ltype in ("cfc", "slstm", "mlstm"):
        return int(entry["hidden_size"])
```

In `describe_architecture` tail dispatch:
```python
                elif ltype == "cfc":
                    tail = f"hidden_size={entry['hidden_size']}, backbone_units={entry['backbone_units']}"
                elif ltype in ("slstm", "mlstm"):
                    tail = f"hidden_size={entry['hidden_size']}"
```

- [ ] **Step 5: encoding.py** — extend the schemas import with `CfcSpec, MlstmSpec, SlstmSpec`; add dispatch lines in `_layer_param_specs` before the `raise`:

```python
    if isinstance(layer, CfcSpec):
        return _cfc_specs(layer, layer_idx, bound_multiplier)
    if isinstance(layer, SlstmSpec):
        return _slstm_specs(layer, layer_idx, bound_multiplier)
    if isinstance(layer, MlstmSpec):
        return _mlstm_specs(layer, layer_idx, bound_multiplier)
```

Add the three generators after `_mamba3_specs`:

```python
def _cfc_specs(layer: CfcSpec, layer_idx: int, bound_multiplier: float) -> list[ParamSpec]:
    """Canonical flat order (Rust CfcLayer::to_flat): interleaved matrix/bias pairs
    w_bb, b_bb, w_ff1, b_ff1, w_ff2, b_ff2, w_ta, b_ta, w_tb, b_tb.

    Bounds: tanh-Xavier on w_bb/w_ff1/w_ff2 (feed lecun_tanh / tanh), plain
    Xavier ("linear") on the time heads w_ta/w_tb, tight 0.1*mul biases.
    """
    i, h, b = layer.input_size, layer.hidden_size, layer.backbone_units
    cat = i + h
    bb_bound = bound_multiplier * compute_layer_bound(cat, b, "tanh")
    ff_bound = bound_multiplier * compute_layer_bound(b, h, "tanh")
    t_bound = bound_multiplier * compute_layer_bound(b, h, "linear")
    bias_bound = 0.1 * bound_multiplier
    li = layer_idx

    specs: list[ParamSpec] = []
    for name, rows, cols, w_bound in (
        ("w_bb", b, cat, bb_bound),
        ("w_ff1", h, b, ff_bound),
        ("w_ff2", h, b, ff_bound),
        ("w_ta", h, b, t_bound),
        ("w_tb", h, b, t_bound),
    ):
        for j in range(rows * cols):
            specs.append(ParamSpec(f"{name}{li}_{j}", -w_bound, w_bound, 0.0))
        bias_name = name.replace("w_", "b_")
        for j in range(rows):
            specs.append(ParamSpec(f"{bias_name}{li}_{j}", -bias_bound, bias_bound, 0.0))
    return specs


def _slstm_specs(layer: SlstmSpec, layer_idx: int, bound_multiplier: float) -> list[ParamSpec]:
    """Canonical flat order (Rust SlstmLayer::to_flat): weight_ih [4H,I] row-major,
    weight_hh [4H,H] row-major, bias [4H]. Gate order (i, f, z, o).

    The forget slice of `bias` (rows [H:2H]) gets the wide 3.0*mul bound to hold
    the +2.0 exp-gating forget-bias init center (LSTM forget-bias-1 precedent,
    scaled for the exponential gate).
    """
    h = layer.hidden_size
    four_h = 4 * h
    w_ih_bound = bound_multiplier * compute_layer_bound(layer.input_size, four_h, "tanh")
    w_hh_bound = bound_multiplier * compute_layer_bound(h, four_h, "tanh")
    tight = 0.1 * bound_multiplier
    forget = 3.0 * bound_multiplier
    li = layer_idx

    specs: list[ParamSpec] = []
    for j in range(four_h * layer.input_size):
        specs.append(ParamSpec(f"w_ih{li}_{j}", -w_ih_bound, w_ih_bound, 0.0))
    for j in range(four_h * h):
        specs.append(ParamSpec(f"w_hh{li}_{j}", -w_hh_bound, w_hh_bound, 0.0))
    for j in range(four_h):
        if h <= j < 2 * h:
            specs.append(ParamSpec(f"b{li}_{j}", -forget, forget, 0.0))
        else:
            specs.append(ParamSpec(f"b{li}_{j}", -tight, tight, 0.0))
    return specs


def _mlstm_specs(layer: MlstmSpec, layer_idx: int, bound_multiplier: float) -> list[ParamSpec]:
    """Canonical flat order (Rust MlstmLayer::to_flat): w_q, b_q, w_k, b_k, w_v,
    b_v, w_o, b_o, w_i, b_i, w_f, b_f. Xavier ("linear") on projections and gate
    vectors; b_f wide (3.0*mul, +2.0 init center); every other bias tight.
    """
    i, h = layer.input_size, layer.hidden_size
    proj_bound = bound_multiplier * compute_layer_bound(i, h, "linear")
    gate_bound = bound_multiplier * compute_layer_bound(i, 1, "linear")
    tight = 0.1 * bound_multiplier
    forget = 3.0 * bound_multiplier
    li = layer_idx

    specs: list[ParamSpec] = []
    for name in ("w_q", "w_k", "w_v", "w_o"):
        for j in range(h * i):
            specs.append(ParamSpec(f"{name}{li}_{j}", -proj_bound, proj_bound, 0.0))
        bias_name = name.replace("w_", "b_")
        for j in range(h):
            specs.append(ParamSpec(f"{bias_name}{li}_{j}", -tight, tight, 0.0))
    for j in range(i):
        specs.append(ParamSpec(f"w_i{li}_{j}", -gate_bound, gate_bound, 0.0))
    specs.append(ParamSpec(f"b_i{li}", -tight, tight, 0.0))
    for j in range(i):
        specs.append(ParamSpec(f"w_f{li}_{j}", -gate_bound, gate_bound, 0.0))
    specs.append(ParamSpec(f"b_f{li}", -forget, forget, 0.0))
    return specs
```

- [ ] **Step 6: initialization_v2.py** — dispatch lines in `_fill_layer` before the `raise`:

```python
    elif t == "cfc":
        _fill_cfc(entry, slab, bound_multiplier, rng)
    elif t == "slstm":
        _fill_slstm(entry, slab, bound_multiplier, rng)
    elif t == "mlstm":
        _fill_mlstm(entry, slab, bound_multiplier, rng)
```

Fills (append after `_fill_mamba3`):

```python
def _fill_cfc(entry: dict, slab: np.ndarray, bound_multiplier: float, rng: np.random.Generator) -> None:
    """CfC slab in canonical flat order (interleaved matrix/bias pairs):
    w_bb, b_bb, w_ff1, b_ff1, w_ff2, b_ff2, w_ta, b_ta, w_tb, b_tb.
    Xavier-uniform matrices (tanh gain on bb/ff, linear on time heads),
    N(0, BIAS_NOISE_STD * mul) biases.
    """
    i = int(entry["input_size"])
    h = int(entry["hidden_size"])
    b = int(entry["backbone_units"])
    cat = i + h
    bias_std = BIAS_NOISE_STD * bound_multiplier
    pop_n = slab.shape[0]
    c = 0
    for rows, cols, act in ((b, cat, "tanh"), (h, b, "tanh"), (h, b, "tanh"), (h, b, "linear"), (h, b, "linear")):
        w_bound = bound_multiplier * compute_layer_bound(cols, rows, act)
        n_w = rows * cols
        slab[:, c : c + n_w] = rng.uniform(-w_bound, w_bound, size=(pop_n, n_w))
        c += n_w
        slab[:, c : c + rows] = rng.normal(0.0, bias_std, size=(pop_n, rows))
        c += rows
    assert c == slab.shape[1], f"cfc slab width mismatch: filled {c}, have {slab.shape[1]}"


def _fill_slstm(entry: dict, slab: np.ndarray, bound_multiplier: float, rng: np.random.Generator) -> None:
    """sLSTM slab: weight_ih, weight_hh (tanh-Xavier), bias N(0, std) except the
    forget slice (rows [H:2H], gate order i/f/z/o) centered at +2.0 -- the
    exp-gating analogue of the LSTM forget-bias-1 init.
    """
    fan_in = int(entry["input_size"])
    hidden = int(entry["hidden_size"])
    four_h = 4 * hidden
    n_w_ih = four_h * fan_in
    n_w_hh = four_h * hidden
    w_ih_bound = bound_multiplier * compute_layer_bound(fan_in, four_h, "tanh")
    w_hh_bound = bound_multiplier * compute_layer_bound(hidden, four_h, "tanh")
    bias_std = BIAS_NOISE_STD * bound_multiplier
    pop_n = slab.shape[0]
    slab[:, :n_w_ih] = rng.uniform(-w_ih_bound, w_ih_bound, size=(pop_n, n_w_ih))
    slab[:, n_w_ih : n_w_ih + n_w_hh] = rng.uniform(-w_hh_bound, w_hh_bound, size=(pop_n, n_w_hh))
    b0 = n_w_ih + n_w_hh
    slab[:, b0 : b0 + four_h] = rng.normal(0.0, bias_std, size=(pop_n, four_h))
    slab[:, b0 + hidden : b0 + 2 * hidden] = 2.0 + rng.normal(0.0, bias_std, size=(pop_n, hidden))


def _fill_mlstm(entry: dict, slab: np.ndarray, bound_multiplier: float, rng: np.random.Generator) -> None:
    """mLSTM slab: w_q/b_q, w_k/b_k, w_v/b_v, w_o/b_o (linear-Xavier + N(0,std)
    biases), then w_i, b_i, w_f, b_f with b_f centered at +2.0.
    """
    fan_in = int(entry["input_size"])
    hidden = int(entry["hidden_size"])
    proj_bound = bound_multiplier * compute_layer_bound(fan_in, hidden, "linear")
    gate_bound = bound_multiplier * compute_layer_bound(fan_in, 1, "linear")
    bias_std = BIAS_NOISE_STD * bound_multiplier
    pop_n = slab.shape[0]
    c = 0
    for _ in range(4):
        n_w = hidden * fan_in
        slab[:, c : c + n_w] = rng.uniform(-proj_bound, proj_bound, size=(pop_n, n_w))
        c += n_w
        slab[:, c : c + hidden] = rng.normal(0.0, bias_std, size=(pop_n, hidden))
        c += hidden
    slab[:, c : c + fan_in] = rng.uniform(-gate_bound, gate_bound, size=(pop_n, fan_in))
    c += fan_in
    slab[:, c] = rng.normal(0.0, bias_std, size=pop_n)
    c += 1
    slab[:, c : c + fan_in] = rng.uniform(-gate_bound, gate_bound, size=(pop_n, fan_in))
    c += fan_in
    slab[:, c] = 2.0 + rng.normal(0.0, bias_std, size=pop_n)
    c += 1
    assert c == slab.shape[1], f"mlstm slab width mismatch: filled {c}, have {slab.shape[1]}"
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_cfc_encoding.py tests/test_xlstm_encoding.py tests/test_init_v2_cfc_xlstm.py -v 2>&1 | tail -10`
Expected: 7 PASS. Also run the existing suites to prove no regression: `uv run pytest tests/test_mamba3_encoding.py tests/test_init_v2_mamba3.py -q 2>&1 | tail -3`.

- [ ] **Step 8: Lint + commit**

```bash
./lint_code.sh
git add src/python/aerocapture/training/rl/schemas.py src/python/aerocapture/training/config.py src/python/aerocapture/training/encoding.py src/python/aerocapture/training/initialization_v2.py tests/test_cfc_encoding.py tests/test_xlstm_encoding.py tests/test_init_v2_cfc_xlstm.py
git commit -m "feat(cfc-xlstm): pydantic specs, sizing arms, PSO ParamSpecs, activation-aware init

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Torch mirrors + cross-language equivalence gates

**Files:**
- Create: `src/python/aerocapture/training/rl/layers/cfc.py`, `src/python/aerocapture/training/rl/layers/slstm.py`, `src/python/aerocapture/training/rl/layers/mlstm.py`
- Test: `tests/test_rust_python_cfc_equivalence.py`, `tests/test_rust_python_slstm_equivalence.py`, `tests/test_rust_python_mlstm_equivalence.py`

**Interfaces:**
- Consumes: Rust runtime from Tasks 2-4 via `aerocapture_rs.nn_forward_sequence(json_path, inputs) -> list[list[float]]`; `DenseLayer` torch mirror (`forward(x, None) -> (y, None)`, `.linear.weight/.bias`).
- Produces: torch modules `CfcLayer(input_size, hidden_size, backbone_units)`, `SlstmLayer(input_size, hidden_size)`, `MlstmLayer(input_size, hidden_size)`, each with `forward_unbatched(x, state) -> (y, new_state)` and `new_state()`. NOT wired into `build_layer` (Task 7 adds the explicit rejection). Parameter names match the Rust JSON field names exactly (the equivalence tests serialize them directly).

- [ ] **Step 1: `rl/layers/cfc.py`:**

```python
"""CfC cell torch mirror -- matches Rust CfcLayer bit-for-bit (unbatched).

Used ONLY by the cross-language equivalence test; the PPO path rejects CfC
(build_layer raises). Math must track src/rust/src/data/neural/layers/cfc.rs
exactly: lecun_tanh constant order, sigmoid(-t_a * CFC_DT + t_b) gate, and
(1 - g) * ff1 + g * ff2 blend.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

CFC_DT = 1.0


class CfcLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, backbone_units: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.backbone_units = backbone_units
        cat = input_size + hidden_size
        self.w_bb = nn.Parameter(torch.zeros(backbone_units, cat))
        self.b_bb = nn.Parameter(torch.zeros(backbone_units))
        self.w_ff1 = nn.Parameter(torch.zeros(hidden_size, backbone_units))
        self.b_ff1 = nn.Parameter(torch.zeros(hidden_size))
        self.w_ff2 = nn.Parameter(torch.zeros(hidden_size, backbone_units))
        self.b_ff2 = nn.Parameter(torch.zeros(hidden_size))
        self.w_ta = nn.Parameter(torch.zeros(hidden_size, backbone_units))
        self.b_ta = nn.Parameter(torch.zeros(hidden_size))
        self.w_tb = nn.Parameter(torch.zeros(hidden_size, backbone_units))
        self.b_tb = nn.Parameter(torch.zeros(hidden_size))

    def forward_unbatched(self, x: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        cat = torch.cat([x, h])
        xb = 1.7159 * torch.tanh(2.0 * (self.w_bb @ cat + self.b_bb) / 3.0)
        ff1 = torch.tanh(self.w_ff1 @ xb + self.b_ff1)
        ff2 = torch.tanh(self.w_ff2 @ xb + self.b_ff2)
        t_a = self.w_ta @ xb + self.b_ta
        t_b = self.w_tb @ xb + self.b_tb
        g = torch.sigmoid(-t_a * CFC_DT + t_b)
        h_new = (1.0 - g) * ff1 + g * ff2
        return h_new, h_new

    def new_state(self) -> Tensor:
        return torch.zeros(self.hidden_size, dtype=self.w_bb.dtype)
```

NOTE the lecun_tanh transcription: Rust computes `1.7159 * tanh(2.0 * z / 3.0)` where `z` is the pre-activation; the Python line applies the same to the whole vector. `2.0 * z / 3.0` associativity ((2z)/3) is identical on both sides.

- [ ] **Step 2: `rl/layers/slstm.py`:**

```python
"""sLSTM cell torch mirror -- matches Rust SlstmLayer bit-for-bit (unbatched).

Gate order (i, f, z, o) on the 4H axis; single bias; stabilized exponential
gating. State tuple: (h, c, n, m). Preactivation add order matches Rust:
(weight_ih @ x + bias) + weight_hh @ h.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

SlstmState = tuple[Tensor, Tensor, Tensor, Tensor]


class SlstmLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.zeros(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.zeros(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))

    def forward_unbatched(self, x: Tensor, state: SlstmState) -> tuple[Tensor, SlstmState]:
        h, c, n, m = state
        pre = (self.weight_ih @ x + self.bias) + self.weight_hh @ h
        hs = self.hidden_size
        i_pre = pre[:hs]
        f_pre = pre[hs : 2 * hs]
        z = torch.tanh(pre[2 * hs : 3 * hs])
        o = torch.sigmoid(pre[3 * hs : 4 * hs])
        m_new = torch.maximum(f_pre + m, i_pre)
        i_g = torch.exp(i_pre - m_new)
        f_g = torch.exp(f_pre + m - m_new)
        c_new = f_g * c + i_g * z
        n_new = f_g * n + i_g
        h_new = o * (c_new / n_new)
        return h_new, (h_new, c_new, n_new, m_new)

    def new_state(self) -> SlstmState:
        z = lambda: torch.zeros(self.hidden_size, dtype=self.weight_ih.dtype)  # noqa: E731
        return (z(), z(), z(), z())
```

- [ ] **Step 3: `rl/layers/mlstm.py`:**

```python
"""mLSTM cell torch mirror -- matches Rust MlstmLayer bit-for-bit (unbatched).

Single head, d_qk = d_v = H. State: (C [H,H], n [H], m scalar). Association
notes mirrored from Rust: C update uses ig * (v_r * k_col) == ig * torch.outer(v, k);
k is scaled by 1/sqrt(H) at projection time.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

MlstmState = tuple[Tensor, Tensor, Tensor]


class MlstmLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.w_q = nn.Parameter(torch.zeros(hidden_size, input_size))
        self.b_q = nn.Parameter(torch.zeros(hidden_size))
        self.w_k = nn.Parameter(torch.zeros(hidden_size, input_size))
        self.b_k = nn.Parameter(torch.zeros(hidden_size))
        self.w_v = nn.Parameter(torch.zeros(hidden_size, input_size))
        self.b_v = nn.Parameter(torch.zeros(hidden_size))
        self.w_o = nn.Parameter(torch.zeros(hidden_size, input_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))
        self.w_i = nn.Parameter(torch.zeros(input_size))
        self.b_i = nn.Parameter(torch.zeros(()))
        self.w_f = nn.Parameter(torch.zeros(input_size))
        self.b_f = nn.Parameter(torch.zeros(()))

    def forward_unbatched(self, x: Tensor, state: MlstmState) -> tuple[Tensor, MlstmState]:
        c, n, m = state
        hs = self.hidden_size
        q = self.w_q @ x + self.b_q
        k = (self.w_k @ x + self.b_k) / math.sqrt(hs)
        v = self.w_v @ x + self.b_v
        i_pre = torch.dot(self.w_i, x) + self.b_i
        f_pre = torch.dot(self.w_f, x) + self.b_f
        m_new = torch.maximum(f_pre + m, i_pre)
        i_g = torch.exp(i_pre - m_new)
        f_g = torch.exp(f_pre + m - m_new)
        c_new = f_g * c + i_g * torch.outer(v, k)
        n_new = f_g * n + i_g * k
        denom = torch.clamp(torch.abs(torch.dot(n_new, q)), min=1.0)
        o = torch.sigmoid(self.w_o @ x + self.b_o)
        h_new = o * ((c_new @ q) / denom)
        return h_new, (c_new, n_new, m_new)

    def new_state(self) -> MlstmState:
        dt = self.w_q.dtype
        return (
            torch.zeros(self.hidden_size, self.hidden_size, dtype=dt),
            torch.zeros(self.hidden_size, dtype=dt),
            torch.zeros((), dtype=dt),
        )
```

- [ ] **Step 4: Rebuild PyO3 (repo root, always):**

Run: `uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml 2>&1 | tail -3`
Expected: `Installed aerocapture_rs-...`.

- [ ] **Step 5: Equivalence tests.** `tests/test_rust_python_cfc_equivalence.py` (the slstm/mlstm files are the same skeleton with the layer, JSON fields, and arch entry swapped — write all three out):

```python
"""Cross-language bit-equivalence gate for the CfC probe layer.

Architecture: Dense(4 -> 8, tanh) -> Cfc(8, 6, 5) -> Dense(6 -> 2, linear).
Exports a Python-built v2 JSON with random f64 weights, runs 100 steps through
aerocapture_rs.nn_forward_sequence, and compares against the unbatched torch
mirror. Gate 1e-12 (observed expectation ~1e-15).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("aerocapture_rs")
import aerocapture_rs  # type: ignore[import-not-found]  # noqa: E402
from aerocapture.training.rl.layers.cfc import CfcLayer  # noqa: E402
from aerocapture.training.rl.layers.dense import DenseLayer  # noqa: E402


@pytest.mark.slow
def test_cfc_rust_python_equivalence_100_steps(tmp_path: Path) -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(seed=1234)

    dense_in = DenseLayer(input_size=4, output_size=8, activation="tanh").double()
    cfc = CfcLayer(input_size=8, hidden_size=6, backbone_units=5).double()
    dense_out = DenseLayer(input_size=6, output_size=2, activation="linear").double()

    with torch.no_grad():
        for lin in (dense_in.linear, dense_out.linear):
            torch.nn.init.uniform_(lin.weight, -0.3, 0.3)
            torch.nn.init.uniform_(lin.bias, -0.3, 0.3)
        for p in cfc.parameters():
            torch.nn.init.uniform_(p, -0.6, 0.6)

    model_json = {
        "format_version": 2,
        "architecture": [
            {"type": "dense", "input_size": 4, "output_size": 8, "activation": "tanh"},
            {"type": "cfc", "input_size": 8, "hidden_size": 6, "backbone_units": 5},
            {"type": "dense", "input_size": 6, "output_size": 2, "activation": "linear"},
        ],
        "weights": {
            "layer_0": {"w": dense_in.linear.weight.detach().tolist(), "b": dense_in.linear.bias.detach().tolist()},
            "layer_1": {
                "w_bb": cfc.w_bb.detach().tolist(),
                "b_bb": cfc.b_bb.detach().tolist(),
                "w_ff1": cfc.w_ff1.detach().tolist(),
                "b_ff1": cfc.b_ff1.detach().tolist(),
                "w_ff2": cfc.w_ff2.detach().tolist(),
                "b_ff2": cfc.b_ff2.detach().tolist(),
                "w_ta": cfc.w_ta.detach().tolist(),
                "b_ta": cfc.b_ta.detach().tolist(),
                "w_tb": cfc.w_tb.detach().tolist(),
                "b_tb": cfc.b_tb.detach().tolist(),
            },
            "layer_2": {"w": dense_out.linear.weight.detach().tolist(), "b": dense_out.linear.bias.detach().tolist()},
        },
    }
    model_path = tmp_path / "cfc_eq.json"
    model_path.write_text(json.dumps(model_json))

    inputs = rng.standard_normal((100, 4)).astype(np.float64)
    rust_outs = np.asarray(
        aerocapture_rs.nn_forward_sequence(str(model_path), [row.tolist() for row in inputs]),
        dtype=np.float64,
    )
    assert rust_outs.shape == (100, 2)

    h = cfc.new_state()
    py_outs = np.empty((100, 2), dtype=np.float64)
    for layer in (dense_in, cfc, dense_out):
        layer.eval()
    with torch.no_grad():
        for t in range(100):
            x = torch.tensor(inputs[t], dtype=torch.float64)
            y0, _ = dense_in(x, None)
            y1, h = cfc.forward_unbatched(y0, h)
            y2, _ = dense_out(y1, None)
            py_outs[t] = y2.numpy()

    diff = float(np.abs(rust_outs - py_outs).max())
    print(f"CfC cross-language max abs diff over 100 steps: {diff:.3e}")
    assert diff < 1e-12
```

`tests/test_rust_python_slstm_equivalence.py` — same skeleton; differences:
- import `SlstmLayer` from `aerocapture.training.rl.layers.slstm`
- layer: `SlstmLayer(input_size=8, hidden_size=6).double()`; init: uniform -0.6..0.6 on `weight_ih`/`weight_hh`, uniform 0..2 on `bias` (exercises exp gates over a real range)
- arch entry: `{"type": "slstm", "input_size": 8, "hidden_size": 6}`
- layer_1 weights: `{"weight_ih": ..., "weight_hh": ..., "bias": ...}`
- state: `state = slstm.new_state()`, step `y1, state = slstm.forward_unbatched(y0, state)`
- head dense: `input_size=6`

`tests/test_rust_python_mlstm_equivalence.py` — same skeleton; differences:
- import `MlstmLayer`; layer `MlstmLayer(input_size=8, hidden_size=6).double()`
- init: uniform -0.5..0.5 on all matrices + `w_i`/`w_f`; set `b_i.fill_(0.3)`, `b_f.fill_(2.0)` inside `no_grad`
- arch entry: `{"type": "mlstm", "input_size": 8, "hidden_size": 6}`
- layer_1 weights keys: `w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o, w_i, b_i, w_f, b_f` — the scalars serialize via `float(mlstm.b_i.detach())` / `float(mlstm.b_f.detach())` (JSON f64, not 1-element list)
- state: `(c, n, m) = mlstm.new_state()` threading through `forward_unbatched`

- [ ] **Step 6: Run the gates**

Run: `uv run pytest tests/test_rust_python_cfc_equivalence.py tests/test_rust_python_slstm_equivalence.py tests/test_rust_python_mlstm_equivalence.py -v -s 2>&1 | tail -10`
Expected: 3 PASS, printed diffs ~1e-14 or better. If a gate fails: constant offset across steps = flat/field-order mismatch; growing drift = state-update bug; see the diagnostics header in the mamba3 equivalence test.

- [ ] **Step 7: Lint + commit**

```bash
./lint_code.sh
git add src/python/aerocapture/training/rl/layers/cfc.py src/python/aerocapture/training/rl/layers/slstm.py src/python/aerocapture/training/rl/layers/mlstm.py tests/test_rust_python_cfc_equivalence.py tests/test_rust_python_slstm_equivalence.py tests/test_rust_python_mlstm_equivalence.py
git commit -m "feat(cfc-xlstm): torch mirrors + 100-step cross-language equivalence gates

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: PSO-only gates + PSO plumbing smoke

**Files:**
- Modify: `src/python/aerocapture/training/rl/layers/__init__.py`, `src/python/aerocapture/training/model_io.py`
- Test: `tests/test_cfc_xlstm_ppo_rejection.py`, `tests/test_cfc_pso_smoke.py`, `tests/test_xlstm_pso_smoke.py`

**Interfaces:**
- Consumes: `CfcSpec`/`SlstmSpec`/`MlstmSpec` (Task 5), Rust `flat_weights_to_json`/`nn_forward` (Tasks 2-4), `nn_param_specs_from_v2` + `init_v2_population` (Task 5).
- Produces: `build_layer` and `load_policy_from_json` raise `NotImplementedError` for the three probe layers (message contains "PSO-only").

- [ ] **Step 1: Failing rejection tests** — `tests/test_cfc_xlstm_ppo_rejection.py`:

```python
"""cfc/slstm/mlstm are PSO-only: build_layer + load_policy_from_json must raise."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from aerocapture.training.rl.layers import build_layer
from aerocapture.training.rl.schemas import CfcSpec, MlstmSpec, SlstmSpec

SPECS = [
    CfcSpec(type="cfc", input_size=8, hidden_size=4, backbone_units=4),
    SlstmSpec(type="slstm", input_size=8, hidden_size=4),
    MlstmSpec(type="mlstm", input_size=8, hidden_size=4),
]


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.type)
def test_build_layer_rejects_probe_layers(spec) -> None:
    with pytest.raises(NotImplementedError, match="PSO-only"):
        build_layer(spec)


def test_load_policy_from_json_with_cfc_raises() -> None:
    from aerocapture.training.model_io import load_policy_from_json

    minimal_json = {
        "format_version": 2,
        "architecture": [
            {"type": "dense", "input_size": 4, "output_size": 4, "activation": "linear"},
            {"type": "cfc", "input_size": 4, "hidden_size": 4, "backbone_units": 4},
        ],
        "weights": {
            "layer_0": {"w": [[0.0] * 4] * 4, "b": [0.0] * 4},
            "layer_1": {
                "w_bb": [[0.0] * 8] * 4,
                "b_bb": [0.0] * 4,
                "w_ff1": [[0.0] * 4] * 4,
                "b_ff1": [0.0] * 4,
                "w_ff2": [[0.0] * 4] * 4,
                "b_ff2": [0.0] * 4,
                "w_ta": [[0.0] * 4] * 4,
                "b_ta": [0.0] * 4,
                "w_tb": [[0.0] * 4] * 4,
                "b_tb": [0.0] * 4,
            },
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "model.json"
        p.write_text(json.dumps(minimal_json))
        with pytest.raises(NotImplementedError):
            load_policy_from_json(str(p))
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cfc_xlstm_ppo_rejection.py -x 2>&1 | tail -5`
Expected: FAIL — `build_layer` hits `Unknown layer spec` ValueError instead of NotImplementedError.

- [ ] **Step 3: Gates.** In `rl/layers/__init__.py`: extend the schemas import with `CfcSpec, MlstmSpec, SlstmSpec`; before the final `raise ValueError` in `build_layer` add:

```python
    if isinstance(spec, (CfcSpec, SlstmSpec, MlstmSpec)):
        raise NotImplementedError(
            f"{spec.type} is PSO-only (architecture probe); the PPO/warm-start V2Policy "
            "path is not implemented. See docs/superpowers/specs/2026-07-07-cfc-xlstm-probes-design.md"
        )
```

In `model_io.py`: extend the schemas import and the rejection tuple at the `any(isinstance(...))` guard to include `CfcSpec, SlstmSpec, MlstmSpec`, and extend the message text to name them (keep the existing spec-doc pointers, append this probe's spec path).

- [ ] **Step 4: PSO smoke tests.** `tests/test_cfc_pso_smoke.py` (mirror `test_mamba3_pso_smoke.py`):

```python
"""PSO plumbing smoke for the CfC probe layer.

Arch: Dense(23 -> 8, tanh) -> Cfc(8, 6, 5) -> Dense(6 -> 2, linear).
Param count: 192 + (5*14 + 5 + 4*(6*5 + 6)) + 14 = 192 + 219 + 14 = 425.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("aerocapture_rs")
import aerocapture_rs  # type: ignore[import-not-found]  # noqa: E402


@pytest.mark.slow
def test_cfc_pso_smoke(tmp_path: Path) -> None:
    from aerocapture.training.encoding import nn_param_specs_from_v2
    from aerocapture.training.initialization_v2 import init_v2_population
    from aerocapture.training.rl.schemas import CfcSpec, DenseSpec

    architecture_specs: list[DenseSpec | CfcSpec] = [
        DenseSpec(type="dense", input_size=23, output_size=8, activation="tanh"),
        CfcSpec(type="cfc", input_size=8, hidden_size=6, backbone_units=5),
        DenseSpec(type="dense", input_size=6, output_size=2, activation="linear"),
    ]
    architecture_dicts = [s.model_dump() for s in architecture_specs]

    param_specs = nn_param_specs_from_v2(architecture_specs, bound_multiplier=2.0)
    assert len(param_specs) == 425, f"Expected 425 params, got {len(param_specs)}"

    rng = np.random.default_rng(42)
    pop = init_v2_population(architecture_dicts, n_pop=4, bound_multiplier=2.0, rng=rng)
    assert pop.shape == (4, 425)
    assert np.all(np.isfinite(pop))

    json_path = tmp_path / "cfc_pso_best.json"
    aerocapture_rs.flat_weights_to_json(pop[0].astype(np.float64).tolist(), json.dumps(architecture_dicts), str(json_path), None)

    loaded = json.loads(json_path.read_text())
    assert loaded["format_version"] == 2
    assert [e["type"] for e in loaded["architecture"]] == ["dense", "cfc", "dense"]
    layer_1 = loaded["weights"]["layer_1"]
    for key in ("w_bb", "b_bb", "w_ff1", "b_ff1", "w_ff2", "b_ff2", "w_ta", "b_ta", "w_tb", "b_tb"):
        assert key in layer_1, f"missing cfc weight key: {key!r}"

    out = np.asarray(aerocapture_rs.nn_forward(str(json_path), np.zeros(23, dtype=np.float64).tolist()), dtype=np.float64)
    assert out.shape == (2,)
    assert all(math.isfinite(v) for v in out), f"non-finite output: {out}"
```

`tests/test_xlstm_pso_smoke.py` — same skeleton, parametrized over the two cells:

```python
"""PSO plumbing smoke for the sLSTM/mLSTM probe layers.

slstm arch: Dense(23 -> 8, tanh) -> Slstm(8, 6) -> Dense(6 -> 2, linear) = 192 + 360 + 14 = 566.
mlstm arch: Dense(23 -> 8, tanh) -> Mlstm(8, 6) -> Dense(6 -> 2, linear) = 192 + 234 + 14 = 440.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("aerocapture_rs")
import aerocapture_rs  # type: ignore[import-not-found]  # noqa: E402

CASES = [
    ("slstm", {"type": "slstm", "input_size": 8, "hidden_size": 6}, 566, ("weight_ih", "weight_hh", "bias")),
    ("mlstm", {"type": "mlstm", "input_size": 8, "hidden_size": 6}, 440, ("w_q", "b_q", "w_k", "b_k", "w_v", "b_v", "w_o", "b_o", "w_i", "b_i", "w_f", "b_f")),
]


@pytest.mark.slow
@pytest.mark.parametrize(("name", "mid", "total", "keys"), CASES, ids=[c[0] for c in CASES])
def test_xlstm_pso_smoke(name: str, mid: dict, total: int, keys: tuple, tmp_path: Path) -> None:
    from pydantic import TypeAdapter

    from aerocapture.training.encoding import nn_param_specs_from_v2
    from aerocapture.training.initialization_v2 import init_v2_population
    from aerocapture.training.rl.schemas import LayerSpec

    architecture_dicts = [
        {"type": "dense", "input_size": 23, "output_size": 8, "activation": "tanh"},
        mid,
        {"type": "dense", "input_size": 6, "output_size": 2, "activation": "linear"},
    ]
    adapter = TypeAdapter(list[LayerSpec])
    architecture_specs = adapter.validate_python(architecture_dicts)

    param_specs = nn_param_specs_from_v2(architecture_specs, bound_multiplier=2.0)
    assert len(param_specs) == total, f"Expected {total} params, got {len(param_specs)}"

    rng = np.random.default_rng(42)
    pop = init_v2_population(architecture_dicts, n_pop=4, bound_multiplier=2.0, rng=rng)
    assert pop.shape == (4, total)
    assert np.all(np.isfinite(pop))

    json_path = tmp_path / f"{name}_pso_best.json"
    aerocapture_rs.flat_weights_to_json(pop[0].astype(np.float64).tolist(), json.dumps(architecture_dicts), str(json_path), None)

    loaded = json.loads(json_path.read_text())
    assert [e["type"] for e in loaded["architecture"]] == ["dense", name, "dense"]
    layer_1 = loaded["weights"]["layer_1"]
    for key in keys:
        assert key in layer_1, f"missing {name} weight key: {key!r}"

    out = np.asarray(aerocapture_rs.nn_forward(str(json_path), np.zeros(23, dtype=np.float64).tolist()), dtype=np.float64)
    assert out.shape == (2,)
    assert all(math.isfinite(v) for v in out)
```

- [ ] **Step 5: Run everything**

Run: `uv run pytest tests/test_cfc_xlstm_ppo_rejection.py tests/test_cfc_pso_smoke.py tests/test_xlstm_pso_smoke.py -v 2>&1 | tail -10`
Expected: all PASS (4 rejection + 3 smoke).

- [ ] **Step 6: Lint + commit**

```bash
./lint_code.sh
git add src/python/aerocapture/training/rl/layers/__init__.py src/python/aerocapture/training/model_io.py tests/test_cfc_xlstm_ppo_rejection.py tests/test_cfc_pso_smoke.py tests/test_xlstm_pso_smoke.py
git commit -m "feat(cfc-xlstm): PSO-only gates (build_layer/model_io) + PSO plumbing smokes

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Shared probe machinery + PROBE_EVAL_SEED_OFFSET

**Files:**
- Modify: `src/python/aerocapture/training/evaluate.py`
- Create: `src/python/aerocapture/training/experiments/probe_common.py`
- Test: `tests/test_cfc_probe.py` (shared-machinery tests live here; Task 9 adds the cfc-specific ones to the same file)

**Interfaces:**
- Consumes: `report.compute_eval_summary`, `report.read_cost_kwargs`, `report._load_nn_scaffolding_overrides(scheme_dir: Path, optimized_toml: Path)`, `charts.is_captured/_FR_DV_TOTAL/DV_FLOOR/DV_CAP`, `config._layer_n_params`.
- Produces: `PROBE_EVAL_SEED_OFFSET = 10_000_000` (with `MAMBA3_EVAL_SEED_OFFSET` as alias); `probe_common` functions: `cvar95(dv)`, `score_model(config, model, seeds, cost_kwargs, sim_timeout, extra_overrides=None) -> dict`, `aggregate(per_rep) -> dict`, `arch_toml(arch: list[dict]) -> str`, `leaf_toml(...) -> str`, `write_manifest(arms, config_dir, extra) -> dict`, `train_jobs(...)`, `eval_arms(...) -> dict`, `score_references(references, seeds, sim_timeout) -> dict`, `print_report(results, arms_order, baseline, treatments, title)`.

- [ ] **Step 1: evaluate.py offset.** Replace the `MAMBA3_EVAL_SEED_OFFSET = 10_000_000` block with:

```python
# Shared reserved eval pool for the architecture probe scripts (mamba3 2x2,
# cfc-vs-gru, lstm-vs-slstm-vs-mlstm): all probes score on ONE pool so their
# reports are directly comparable. Disjoint from every training/validation/
# final/other-eval stream above.
PROBE_EVAL_SEED_OFFSET = 10_000_000
# Legacy alias (mamba3_ablation.py imports this name).
MAMBA3_EVAL_SEED_OFFSET = PROBE_EVAL_SEED_OFFSET
```

- [ ] **Step 2: Failing tests for the shared machinery** — start `tests/test_cfc_probe.py`:

```python
"""Unit tests for the CfC probe driver + shared probe machinery."""

from __future__ import annotations

import numpy as np
from aerocapture.training.experiments.probe_common import aggregate, arch_toml, cvar95


def test_cvar95_is_worst_5pct_mean() -> None:
    x = np.arange(100.0)
    cv = cvar95(x)
    assert cv > float(np.percentile(x, 95))
    assert cv == float(np.mean(x[x >= np.percentile(x, 95)]))


def test_cvar95_empty_is_nan() -> None:
    assert np.isnan(cvar95(np.array([])))


def test_aggregate_mean_std() -> None:
    per_rep = [
        {"rms_cost": 10.0, "capture_rate": 1.0, "dv_p50": 100.0, "dv_p95": 200.0, "cvar95": 250.0},
        {"rms_cost": 12.0, "capture_rate": 0.9, "dv_p50": 110.0, "dv_p95": 220.0, "cvar95": 270.0},
    ]
    agg = aggregate(per_rep)
    assert agg["n_repeats"] == 2
    assert agg["dv_p95"]["mean"] == 210.0
    assert agg["dv_p95"]["std"] == 10.0


def test_arch_toml_renders_blocks() -> None:
    arch = [
        {"type": "dense", "input_size": 21, "output_size": 32, "activation": "swish"},
        {"type": "cfc", "input_size": 32, "hidden_size": 32, "backbone_units": 32},
    ]
    s = arch_toml(arch)
    assert s.count("[[network.architecture]]") == 2
    assert 'type = "cfc"' in s
    assert "backbone_units = 32" in s
    assert 'activation = "swish"' in s


def test_probe_offset_alias() -> None:
    from aerocapture.training.evaluate import MAMBA3_EVAL_SEED_OFFSET, PROBE_EVAL_SEED_OFFSET

    assert PROBE_EVAL_SEED_OFFSET == 10_000_000
    assert MAMBA3_EVAL_SEED_OFFSET == PROBE_EVAL_SEED_OFFSET
```

Run: `uv run pytest tests/test_cfc_probe.py -x 2>&1 | tail -3` -> ImportError (probe_common missing).

- [ ] **Step 3: Create `src/python/aerocapture/training/experiments/probe_common.py`:**

```python
"""Shared machinery for the architecture probe drivers (cfc_probe, xlstm_probe).

mamba3_ablation.py predates this module and keeps its own copies (left
untouched to avoid churn on the underlying branch); the newer probes share
this one. All probe scripts score on the SAME reserved pool
(evaluate.PROBE_EVAL_SEED_OFFSET) so their reports are directly comparable,
and every claim is gated on sigma_run from seed-repeats (project lesson:
single-run deltas are noise).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

METRIC_KEYS = ("rms_cost", "capture_rate", "dv_p50", "dv_p95", "cvar95")


def cvar95(dv: np.ndarray) -> float:
    """CVaR95 = mean of the worst 5% (DV at or above the 95th percentile).

    The propellant-sizing tail statistic: what the ergol budget must cover in
    the bad-luck cases, not the median mission.
    """
    if dv.size == 0:
        return float("nan")
    thr = float(np.percentile(dv, 95))
    tail = dv[dv >= thr]
    return float(np.mean(tail)) if tail.size else thr


def score_model(
    config: Path,
    model: Path,
    seeds: list[int],
    cost_kwargs: dict[str, Any],
    sim_timeout: float | None,
    extra_overrides: dict[str, Any] | None = None,
) -> dict[str, float]:
    """One MC batch of len(seeds) sims for a deployed model; tail-led metric dict."""
    import aerocapture_rs

    from aerocapture.training import charts
    from aerocapture.training.report import compute_eval_summary

    overrides = [
        {"simulation.n_sims": 1, "data.neural_network": str(model), "monte_carlo.seed": int(s), **(extra_overrides or {})}
        for s in seeds
    ]
    batch = aerocapture_rs.run_batch(str(config), overrides, n_threads=None, include_trajectories=False, sim_timeout_secs=sim_timeout)
    final = np.array(batch.final_records, dtype=np.float64)
    summary = compute_eval_summary(final, n_sims=len(seeds), cost_kwargs=cost_kwargs)
    captured = charts.is_captured(final)
    dv = np.clip(final[captured, charts._FR_DV_TOTAL], charts.DV_FLOOR, charts.DV_CAP)
    return {
        "rms_cost": float(summary["cost"]["rms"]),
        "capture_rate": float(summary["capture_rate"]),
        "dv_p50": float(summary["captured"]["dv"]["p50"]) if summary["captured"] else float("nan"),
        "dv_p95": float(summary["captured"]["dv"]["p95"]) if summary["captured"] else float("nan"),
        "cvar95": cvar95(dv),
    }


def aggregate(per_rep: list[dict[str, float]]) -> dict[str, Any]:
    if not per_rep:
        return {"n_repeats": 0}
    agg: dict[str, Any] = {"n_repeats": len(per_rep), "per_repeat": per_rep}
    for k in METRIC_KEYS:
        vals = np.array([d[k] for d in per_rep], dtype=np.float64)
        agg[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
    return agg


def arch_toml(arch: list[dict[str, Any]]) -> str:
    """Render an architecture list as [[network.architecture]] TOML blocks."""
    blocks = []
    for entry in arch:
        lines = ["[[network.architecture]]"]
        for k, v in entry.items():
            lines.append(f'{k} = "{v}"' if isinstance(v, str) else f"{k} = {v}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def leaf_toml(
    script: str,
    arm: str,
    arch: list[dict[str, Any]],
    seed: int,
    base_seed: int,
    out_dir: Path,
    n_gen: int,
    training_n_sims: int,
    input_mask: list[int],
) -> str:
    mask = ", ".join(str(i) for i in input_mask)
    return f"""# Auto-generated by {script}.py -- do not edit by hand.
# Arm: {arm}, seed={seed}.
base = ["../../missions/mars.toml", "../common.toml"]

[guidance]
type = "neural_network"

[data]
neural_network = "{out_dir.as_posix()}/best_model.json"
results_suffix = ".{script}_{arm}_s{seed - base_seed}"

[network]
input_mask = [{mask}]

{arch_toml(arch)}

[optimizer]
algorithm = "pso"
n_pop = 64
n_gen = {n_gen}
seed_strategy = "fixed"
training_n_sims = {training_n_sims}
validation_n_sims = 200

[monte_carlo]
seed = {seed}
"""


def write_manifest(arms: dict[str, list[dict[str, Any]]], config_dir: Path, extra: dict[str, Any]) -> dict[str, Any]:
    """Per-arm dims + exact param counts (cell = middle entry, total = all)."""
    from aerocapture.training.config import _layer_n_params

    manifest: dict[str, Any] = {"arms": {}, **extra}
    for arm, arch in arms.items():
        manifest["arms"][arm] = {
            "architecture": arch,
            "cell_params": _layer_n_params(arch[1]),
            "total_params": sum(_layer_n_params(e) for e in arch),
        }
    (config_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def train_jobs(
    arms: list[str],
    repeats: int,
    config_dir: Path,
    out_root: Path,
    n_gen: int,
    training_n_sims: int,
    sim_timeout: float | None,
    force: bool,
    from_scratch: bool,
) -> None:
    jobs = [(arm, r) for arm in arms for r in range(repeats)]
    for i, (arm, r) in enumerate(jobs, 1):
        out_dir = out_root / f"{arm}_s{r}"
        config = config_dir / f"{arm}_s{r}.toml"
        if (out_dir / "best_model.json").exists() and not force and not from_scratch:
            print(f"[{i}/{len(jobs)}] skip {arm}_s{r} (best_model.json exists; --force/--from-scratch to retrain)")
            continue
        cmd = [
            sys.executable,
            "-m",
            "aerocapture.training.train",
            str(config),
            "--n-gen",
            str(n_gen),
            "--no-tui",
            "--skip-report",
            "--training-n-sims",
            str(training_n_sims),
            "--output-dir",
            str(out_dir),
        ]
        if from_scratch:
            cmd.append("--from-scratch")
        if sim_timeout is not None:
            cmd += ["--sim-timeout", str(sim_timeout)]
        print(f"[{i}/{len(jobs)}] train {arm}_s{r}: {' '.join(cmd)}")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"  WARNING: training exited {rc} for {config}")


def eval_arms(
    arms: list[str],
    repeats: int,
    config_dir: Path,
    out_root: Path,
    seeds: list[int],
    sim_timeout: float | None,
) -> dict[str, Any]:
    from aerocapture.training.report import read_cost_kwargs

    out: dict[str, Any] = {}
    for arm in arms:
        per_rep: list[dict[str, float]] = []
        for r in range(repeats):
            out_dir = out_root / f"{arm}_s{r}"
            model = out_dir / "best_model.json"
            config = config_dir / f"{arm}_s{r}.toml"
            if not model.exists():
                print(f"  skip {arm}_s{r} (untrained: no {model})")
                continue
            per_rep.append(score_model(config, model, seeds, read_cost_kwargs(config), sim_timeout))
        out[arm] = aggregate(per_rep)
    return out


def score_references(
    references: dict[str, tuple[Path, Path]],
    seeds: list[int],
    sim_timeout: float | None,
) -> dict[str, dict[str, float]]:
    """Score deployed champions on the shared pool. NOT budget-matched -- each
    runs its own training TOML + best_model.json (+ best_params.json scaffolding
    when present). Missing artifacts skip with a notice."""
    from aerocapture.training.report import _load_nn_scaffolding_overrides, read_cost_kwargs

    out: dict[str, dict[str, float]] = {}
    for name, (toml_path, scheme_dir) in references.items():
        toml_path, scheme_dir = Path(toml_path), Path(scheme_dir)
        model = scheme_dir / "best_model.json"
        if not model.exists() or not toml_path.exists():
            missing = model if not model.exists() else toml_path
            print(f"  reference {name}: skipped (missing {missing})")
            continue
        extra = _load_nn_scaffolding_overrides(scheme_dir, scheme_dir / f"optimized_{scheme_dir.name}.toml")
        out[name] = score_model(toml_path, model, seeds, read_cost_kwargs(toml_path), sim_timeout, extra_overrides=extra)
    return out


def _row(label: str, d: dict[str, Any]) -> str:
    cap = d["capture_rate"]["mean"] * 100 if isinstance(d.get("capture_rate"), dict) else d["capture_rate"] * 100
    def _mv(k: str) -> tuple[float, float]:
        v = d[k]
        return (v["mean"], v["std"]) if isinstance(v, dict) else (v, float("nan"))
    rms, _ = _mv("rms_cost")
    p50, _ = _mv("dv_p50")
    p95, p95s = _mv("dv_p95")
    cv, cvs = _mv("cvar95")
    return f"{label:16s} {cap:6.1f} {rms:9.1f} {p50:8.1f} {p95:10.1f} +-{p95s:5.1f} {cv:10.1f} +-{cvs:5.1f}"


def print_report(
    results: dict[str, Any],
    arms_order: list[str],
    baseline: str,
    treatments: list[str],
    title: str,
) -> None:
    arms = results["arms"]
    print("\n" + "=" * 84)
    print(f"{title} -- {results['repeats']} repeats x {results['n_sims']} eval sims/arm")
    print("Tail metrics are the sizing statistics; lead with dv_p95 / CVaR95, not p50.")
    print("=" * 84)
    print(f"{'arm':16s} {'cap%':>6s} {'rms':>9s} {'dvP50':>8s} {'dvP95 +- sig':>18s} {'CVaR95 +- sig':>18s}")
    for arm in arms_order:
        a = arms.get(arm, {})
        if a.get("n_repeats", 0) == 0:
            print(f"{arm:16s} {'(untrained)':>66s}")
            continue
        print(_row(arm, a))
    refs = results.get("references", {})
    if refs:
        print("-" * 84)
        print("references (deployed champions, NOT budget-matched -- own mask/settings):")
        for name, d in refs.items():
            print(_row(name, d))

    base = arms.get(baseline, {})
    if base.get("n_repeats", 0) == 0:
        print(f"\nNo {baseline} baseline arm trained -- cannot compute significance.")
        return
    trained = [a for a in arms_order if arms.get(a, {}).get("n_repeats", 0) > 0]
    min_reps = min((arms[a]["n_repeats"] for a in trained), default=0)
    if min_reps < 2:
        print(f"\nSignificance skipped: need >=2 repeats to estimate sigma_run (got min {min_reps}). Re-run with --repeats 3+.")
        return
    print(f"\nSignificance vs {baseline} (gap clears sigma_run only if |gap| > combined std):")
    for metric in ("dv_p95", "cvar95"):
        bmean, bstd = base[metric]["mean"], base[metric]["std"]
        print(f"  [{metric}]  {baseline} = {bmean:.1f} +- {bstd:.1f}")
        for arm in treatments:
            a = arms.get(arm, {})
            if a.get("n_repeats", 0) == 0:
                continue
            gap = a[metric]["mean"] - bmean
            sig = float(np.sqrt(bstd**2 + a[metric]["std"] ** 2))
            verdict = "SIGNIFICANT" if abs(gap) > sig else "within sigma_run (noise)"
            arrow = "better" if gap < 0 else "worse"
            print(f"    {arm:12s} gap = {gap:+7.1f} ({arrow}), sigma_run = {sig:5.1f}  -> {verdict}")
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_cfc_probe.py -v 2>&1 | tail -8`
Expected: 5 PASS.

- [ ] **Step 5: Lint + commit**

```bash
./lint_code.sh
git add src/python/aerocapture/training/evaluate.py src/python/aerocapture/training/experiments/probe_common.py tests/test_cfc_probe.py
git commit -m "feat(probes): shared probe machinery + PROBE_EVAL_SEED_OFFSET (mamba3 alias kept)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: cfc_probe experiment driver

**Files:**
- Create: `src/python/aerocapture/training/experiments/cfc_probe.py`
- Test: extend `tests/test_cfc_probe.py`

**Interfaces:**
- Consumes: everything from `probe_common` (Task 8), `evaluate.make_reserved_seeds` + `evaluate.PROBE_EVAL_SEED_OFFSET`, `config._layer_n_params`.
- Produces: CLI `python -m aerocapture.training.experiments.cfc_probe --generate|--train|--eval|--report|--all [--repeats 3] [--n-gen 500] [--training-n-sims 10] [--n-sims 1000] [--sim-timeout S] [--force] [--from-scratch]`; configs in `configs/training/cfc_probe/`, outputs in `training_output/cfc_probe/`, results JSON `training_output/cfc_probe/probe_results.json`.

- [ ] **Step 1: Failing tests** — append to `tests/test_cfc_probe.py`:

```python
def test_cfc_arms_and_budget_within_2pct() -> None:
    from aerocapture.training.config import _layer_n_params
    from aerocapture.training.experiments.cfc_probe import ARMS

    assert set(ARMS) == {"gru", "cfc"}
    totals = {arm: sum(_layer_n_params(e) for e in arch) for arm, arch in ARMS.items()}
    assert totals["gru"] == 7106  # 704 + 6336 + 66
    assert totals["cfc"] == 7074  # 704 + 6304 + 66
    assert abs(totals["cfc"] - totals["gru"]) / totals["gru"] < 0.02


def test_cfc_leaf_toml_carries_layer_and_seed() -> None:
    from pathlib import Path

    from aerocapture.training.experiments.cfc_probe import ARMS, BASE_SEED, INPUT_MASK
    from aerocapture.training.experiments.probe_common import leaf_toml

    toml = leaf_toml("cfc_probe", "cfc", ARMS["cfc"], BASE_SEED + 2, BASE_SEED, Path("training_output/cfc_probe/cfc_s2"), 500, 10, INPUT_MASK)
    assert 'type = "cfc"' in toml
    assert "backbone_units = 32" in toml
    assert f"seed = {BASE_SEED + 2}" in toml
    assert 'seed_strategy = "fixed"' in toml
    assert ".cfc_probe_cfc_s2" in toml
```

Run: `uv run pytest tests/test_cfc_probe.py -x 2>&1 | tail -3` -> ImportError (cfc_probe missing).

- [ ] **Step 2: Create `src/python/aerocapture/training/experiments/cfc_probe.py`:**

```python
"""CfC probe: {gru, cfc} matched-budget controlled arms -> tail DV with sigma_run.

Hypothesis under test: input-dependent time constants (CfC) match or beat the
closest scalar-state baseline (GRU) on the sizing tail at the same param budget
(gru 7106 vs cfc 7074 total trainable, -0.5%). Both arms train on identical
fixed seeds; sigma_run comes from seed-repeats + PSO stochasticity. Deployed
GRU/Mamba champions are scored on the same reserved pool as reference rows
(NOT budget-matched -- own masks/settings).

CLI (from repo root):
    python -m aerocapture.training.experiments.cfc_probe --generate --repeats 3
    python -m aerocapture.training.experiments.cfc_probe --train  --repeats 3 --n-gen 500 --training-n-sims 10
    python -m aerocapture.training.experiments.cfc_probe --eval --report --repeats 3 --n-sims 1000
    python -m aerocapture.training.experiments.cfc_probe --all --repeats 3 --n-gen 500 --n-sims 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aerocapture.training.experiments import probe_common as pc

BASE_SEED = 20260707  # same as mamba3_ablation -- identical training seed lists across probes
CONFIG_DIR = Path("configs/training/cfc_probe")
OUT_DIR = Path("training_output/cfc_probe")
INPUT_MASK = list(range(21))

_DENSE_IN = {"type": "dense", "input_size": 21, "output_size": 32, "activation": "swish"}
_DENSE_OUT = {"type": "dense", "input_size": 32, "output_size": 2, "activation": "asinh"}

# arm -> full architecture (budget-matched: gru 6336 vs cfc 6304 cell params)
ARMS: dict[str, list[dict[str, Any]]] = {
    "gru": [_DENSE_IN, {"type": "gru", "input_size": 32, "hidden_size": 32}, _DENSE_OUT],
    "cfc": [_DENSE_IN, {"type": "cfc", "input_size": 32, "hidden_size": 32, "backbone_units": 32}, _DENSE_OUT],
}
BASELINE = "gru"
TREATMENTS = ["cfc"]

# Deployed champions scored on the same pool (reference rows, not budget-matched).
REFERENCES: dict[str, tuple[Path, Path]] = {
    "gru_champion": (Path("configs/training/msr_aller_gru_pso_train.toml"), Path("training_output/neural_network_gru_pso")),
    "mamba_champion": (Path("configs/training/msr_aller_mamba_pso_train.toml"), Path("training_output/neural_network_mamba_pso")),
}


def generate_configs(repeats: int, n_gen: int, training_n_sims: int) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for arm, arch in ARMS.items():
        for r in range(repeats):
            out_dir = OUT_DIR / f"{arm}_s{r}"
            path = CONFIG_DIR / f"{arm}_s{r}.toml"
            path.write_text(pc.leaf_toml("cfc_probe", arm, arch, BASE_SEED + r, BASE_SEED, out_dir, n_gen, training_n_sims, INPUT_MASK))
    manifest = pc.write_manifest(ARMS, CONFIG_DIR, {"repeats": repeats, "n_gen": n_gen, "training_n_sims": training_n_sims})
    print(f"Wrote {len(ARMS) * repeats} arm configs to {CONFIG_DIR}/")
    for arm, m in manifest["arms"].items():
        print(f"  {arm}: cell {m['cell_params']}, total {m['total_params']} trainable params")


def eval_all(repeats: int, n_sims: int, sim_timeout: float | None) -> dict[str, Any]:
    from aerocapture.training.evaluate import PROBE_EVAL_SEED_OFFSET, make_reserved_seeds

    seeds = make_reserved_seeds(0, PROBE_EVAL_SEED_OFFSET, n_sims)
    results: dict[str, Any] = {
        "n_sims": n_sims,
        "repeats": repeats,
        "arms": pc.eval_arms(list(ARMS), repeats, CONFIG_DIR, OUT_DIR, seeds, sim_timeout),
        "references": pc.score_references(REFERENCES, seeds, sim_timeout),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "probe_results.json").write_text(json.dumps(results, indent=2))
    print(f"Wrote eval results to {OUT_DIR / 'probe_results.json'}")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="CfC vs GRU matched-budget probe")
    p.add_argument("--generate", action="store_true", help="write arm configs + manifest")
    p.add_argument("--train", action="store_true", help="PSO-train each arm x repeat (subprocess)")
    p.add_argument("--eval", action="store_true", help="score deployed models + references on the reserved pool")
    p.add_argument("--report", action="store_true", help="print the arm comparison table + significance")
    p.add_argument("--all", action="store_true", help="generate -> train -> eval -> report")
    p.add_argument("--repeats", type=int, default=3, help="seed-repeats per arm (sigma_run sample)")
    p.add_argument("--n-gen", type=int, default=500, help="PSO generations per training run")
    p.add_argument("--training-n-sims", type=int, default=10, help="sims per individual per generation")
    p.add_argument("--n-sims", type=int, default=1000, help="reserved eval pool size")
    p.add_argument("--sim-timeout", type=float, default=None, help="per-sim wall-clock timeout (s)")
    p.add_argument("--force", action="store_true", help="retrain even if best_model.json exists")
    p.add_argument("--from-scratch", action="store_true", help="wipe checkpoints + retrain")
    args = p.parse_args()

    if not any((args.generate, args.train, args.eval, args.report, args.all)):
        p.error("pass at least one of --generate/--train/--eval/--report/--all")

    if args.generate or args.all:
        generate_configs(args.repeats, args.n_gen, args.training_n_sims)
    if args.train or args.all:
        pc.train_jobs(list(ARMS), args.repeats, CONFIG_DIR, OUT_DIR, args.n_gen, args.training_n_sims, args.sim_timeout, args.force, args.from_scratch)
    results: dict[str, Any] | None = None
    if args.eval or args.all:
        results = eval_all(args.repeats, args.n_sims, args.sim_timeout)
    if args.report or args.all:
        if results is None:
            results = json.loads((OUT_DIR / "probe_results.json").read_text())
        pc.print_report(results, list(ARMS), BASELINE, TREATMENTS, "CfC probe (cfc vs gru, matched budget)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run tests + a --generate smoke**

Run: `uv run pytest tests/test_cfc_probe.py -v 2>&1 | tail -8` -> 7 PASS.
Run: `uv run python -m aerocapture.training.experiments.cfc_probe --generate --repeats 1 2>&1 | tail -4`
Expected: writes 2 configs + manifest, prints `gru: cell 6336, total 7106` and `cfc: cell 6304, total 7074`. Then verify a generated leaf loads in Rust: `uv run python -c "import aerocapture_rs; aerocapture_rs.load_config('configs/training/cfc_probe/cfc_s0.toml')"` (should not raise). Clean up the smoke configs afterwards ONLY if repeats will differ in real runs (regenerate is idempotent -- leaving them is fine, they are gitignored under configs? CHECK: `git status configs/training/cfc_probe` -- if configs/training is tracked, commit the generated repeats-3 set instead by running --generate --repeats 3, matching the mamba3 branch which committed its arm configs).

- [ ] **Step 4: Lint + commit**

```bash
./lint_code.sh
uv run python -m aerocapture.training.experiments.cfc_probe --generate --repeats 3
git add src/python/aerocapture/training/experiments/cfc_probe.py tests/test_cfc_probe.py configs/training/cfc_probe/
git commit -m "feat(probes): cfc_probe driver -- gru vs cfc matched-budget arms + champion references

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: xlstm_probe experiment driver

**Files:**
- Create: `src/python/aerocapture/training/experiments/xlstm_probe.py`
- Test: `tests/test_xlstm_probe.py`

**Interfaces:**
- Consumes: `probe_common` (Task 8).
- Produces: CLI `python -m aerocapture.training.experiments.xlstm_probe` with the same flags as cfc_probe; configs in `configs/training/xlstm_probe/`, outputs in `training_output/xlstm_probe/`.

- [ ] **Step 1: Failing tests** — `tests/test_xlstm_probe.py`:

```python
"""Unit tests for the xLSTM probe driver."""

from __future__ import annotations

from pathlib import Path

from aerocapture.training.config import _layer_n_params
from aerocapture.training.experiments.probe_common import leaf_toml
from aerocapture.training.experiments.xlstm_probe import ARMS, BASE_SEED, BASELINE, INPUT_MASK, TREATMENTS


def test_xlstm_arms_and_budgets_within_2pct() -> None:
    assert set(ARMS) == {"lstm", "slstm", "mlstm"}
    assert BASELINE == "lstm"
    assert TREATMENTS == ["slstm", "mlstm"]
    totals = {arm: sum(_layer_n_params(e) for e in arch) for arm, arch in ARMS.items()}
    assert totals["lstm"] == 9218  # 704 + 8448 + 66
    assert totals["slstm"] == 9090  # 704 + 8320 + 66
    assert totals["mlstm"] == 9348  # 704 + 8514 + 130
    for arm in TREATMENTS:
        assert abs(totals[arm] - totals["lstm"]) / totals["lstm"] < 0.02


def test_mlstm_head_reads_64_wide() -> None:
    head = ARMS["mlstm"][-1]
    assert head["input_size"] == 64  # mlstm H=64 for budget parity


def test_xlstm_leaf_toml_carries_layer_and_seed() -> None:
    toml = leaf_toml("xlstm_probe", "slstm", ARMS["slstm"], BASE_SEED, BASE_SEED, Path("training_output/xlstm_probe/slstm_s0"), 500, 10, INPUT_MASK)
    assert 'type = "slstm"' in toml
    assert f"seed = {BASE_SEED}" in toml
    assert ".xlstm_probe_slstm_s0" in toml
```

Run: `uv run pytest tests/test_xlstm_probe.py -x 2>&1 | tail -3` -> ImportError.

- [ ] **Step 2: Create `src/python/aerocapture/training/experiments/xlstm_probe.py`** — same skeleton as `cfc_probe.py` (Task 9 step 2) with this header/constants block swapped in; `generate_configs`, `eval_all`, and `main` are IDENTICAL except: script name strings `"xlstm_probe"`, argparse description "xLSTM probe (lstm vs slstm vs mlstm, matched budget)", and the report title `"xLSTM probe (lstm vs slstm vs mlstm, matched budget)"`:

```python
"""xLSTM probe: {lstm, slstm, mlstm} matched-budget controlled arms.

Mechanism decomposition: lstm -> slstm isolates exponential gating (can the
cell sharply REVISE a stored estimate when surprise arrives -- the bounce, a
density shock); slstm -> mlstm isolates matrix memory (vs Mamba's diagonal
state). Budgets: lstm 9218 / slstm 9090 / mlstm 9348 total trainable (+-1.5%;
mlstm runs H=64 because it has no recurrent matrices). Same fixed training
seeds across arms; deployed LSTM/Mamba champions scored on the same reserved
pool as reference rows (NOT budget-matched).

CLI (from repo root):
    python -m aerocapture.training.experiments.xlstm_probe --generate --repeats 3
    python -m aerocapture.training.experiments.xlstm_probe --train  --repeats 3 --n-gen 500 --training-n-sims 10
    python -m aerocapture.training.experiments.xlstm_probe --eval --report --repeats 3 --n-sims 1000
    python -m aerocapture.training.experiments.xlstm_probe --all --repeats 3 --n-gen 500 --n-sims 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aerocapture.training.experiments import probe_common as pc

BASE_SEED = 20260707
CONFIG_DIR = Path("configs/training/xlstm_probe")
OUT_DIR = Path("training_output/xlstm_probe")
INPUT_MASK = list(range(21))

_DENSE_IN = {"type": "dense", "input_size": 21, "output_size": 32, "activation": "swish"}
_HEAD_32 = {"type": "dense", "input_size": 32, "output_size": 2, "activation": "asinh"}
_HEAD_64 = {"type": "dense", "input_size": 64, "output_size": 2, "activation": "asinh"}

ARMS: dict[str, list[dict[str, Any]]] = {
    "lstm": [_DENSE_IN, {"type": "lstm", "input_size": 32, "hidden_size": 32}, _HEAD_32],
    "slstm": [_DENSE_IN, {"type": "slstm", "input_size": 32, "hidden_size": 32}, _HEAD_32],
    "mlstm": [_DENSE_IN, {"type": "mlstm", "input_size": 32, "hidden_size": 64}, _HEAD_64],
}
BASELINE = "lstm"
TREATMENTS = ["slstm", "mlstm"]

REFERENCES: dict[str, tuple[Path, Path]] = {
    "lstm_champion": (Path("configs/training/msr_aller_lstm_pso_train.toml"), Path("training_output/neural_network_lstm_pso")),
    "mamba_champion": (Path("configs/training/msr_aller_mamba_pso_train.toml"), Path("training_output/neural_network_mamba_pso")),
}
```

(Then copy `generate_configs`, `eval_all`, `main` from cfc_probe.py with the three strings swapped. Deliberate duplication -- the ~60 shared lines are the CLI shell; the machinery already lives in probe_common.)

- [ ] **Step 3: Run tests + generate**

Run: `uv run pytest tests/test_xlstm_probe.py -v 2>&1 | tail -6` -> 3 PASS.
Run: `uv run python -m aerocapture.training.experiments.xlstm_probe --generate --repeats 3 2>&1 | tail -5` -> 9 configs + manifest with the budget table. Verify one loads: `uv run python -c "import aerocapture_rs; aerocapture_rs.load_config('configs/training/xlstm_probe/mlstm_s0.toml')"`.

- [ ] **Step 4: Lint + commit**

```bash
./lint_code.sh
git add src/python/aerocapture/training/experiments/xlstm_probe.py tests/test_xlstm_probe.py configs/training/xlstm_probe/
git commit -m "feat(probes): xlstm_probe driver -- lstm/slstm/mlstm mechanism decomposition

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 11: Docs + full verification sweep

**Files:**
- Modify: `CLAUDE.md`, `README.md`

**Interfaces:** none (documentation + gates).

- [ ] **Step 1: CLAUDE.md** — add a paragraph after the Mamba-3 ablation paragraph (find it: `rg -n "mamba3" CLAUDE.md | head -3`):

> **CfC + xLSTM probe layers (branch `feature/cfc-xlstm`, 2026-07-07)** — three experimental PSO-only cell types probing two recent recurrent families at matched budgets: `cfc` (closed-form continuous-time cell, ncps "default" mode: one lecun_tanh backbone + ff1/ff2/t_a/t_b heads, `h' = (1-g)*tanh(ff1) + g*tanh(ff2)` with `g = sigmoid(-t_a*dt + t_b)`, dt fixed at 1 guidance tick; state `(H,)` flat), `slstm` (xLSTM sLSTM: exponential gating with max-stabilizer, gate order i/f/z/o, single bias, full recurrent matrices, state `(h, c, n, m)`), and `mlstm` (xLSTM mLSTM: matrix memory `C (HxH)` + normalizer `n (H,)` + scalar stabilizer `m`, covariance update, single head `d_qk = d_v = H`, no recurrent weights). All three are cell-only (no block scaffolding), gated PSO-only (`build_layer`/`load_policy_from_json` raise, mamba3 pattern), with unbatched torch mirrors used solely by the 100-step cross-language equivalence gates (<1e-12). Forget-bias init: +2.0 center on the sLSTM bias f-slice and mLSTM `b_f` (ParamSpec bound 3.0*mul) — the exp-gating analogue of the LSTM forget-bias-1 precedent. Experiment drivers `experiments/cfc_probe.py` (arms {gru, cfc}, H=32/B=32, cell 6336 vs 6304) and `experiments/xlstm_probe.py` (arms {lstm H=32, slstm H=32, mlstm H=64}, totals 9218/9090/9348) mirror `mamba3_ablation.py` (generate/train/eval/report, fixed seeds, seed-repeats × sigma_run significance, tail-led reporting) via shared `experiments/probe_common.py`, score on the shared `PROBE_EVAL_SEED_OFFSET = 10_000_000` pool (`MAMBA3_EVAL_SEED_OFFSET` is now an alias), and print deployed GRU/LSTM/Mamba champions as reference rows (flagged not-budget-matched). Spec: `docs/superpowers/specs/2026-07-07-cfc-xlstm-probes-design.md`.

- [ ] **Step 2: README** — find the mamba3 experimental note (`rg -n "mamba3" README.md`) and add the sibling line next to it:

> - **Experimental probe layers** (`cfc`, `slstm`, `mlstm`): PSO-only CfC and xLSTM cells with matched-budget probe drivers (`experiments/cfc_probe.py`, `experiments/xlstm_probe.py`).

- [ ] **Step 3: Full gates, in order (all from repo root):**

```bash
./check_all.sh                                    # Rust: test + fmt --check + clippy + release build (includes the 6 guidance goldens)
uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml
uv run pytest tests -q -x -m "not slow" 2>&1 | tail -5          # full fast suite
uv run pytest tests/test_rust_python_cfc_equivalence.py tests/test_rust_python_slstm_equivalence.py tests/test_rust_python_mlstm_equivalence.py tests/test_cfc_pso_smoke.py tests/test_xlstm_pso_smoke.py -v 2>&1 | tail -8   # slow gates
./lint_code.sh
```

Expected: everything green. The golden regressions inside `cargo test` prove the changes are additive (no existing layer touched).

- [ ] **Step 4: Commit docs**

```bash
git add CLAUDE.md README.md
git commit -m "docs(cfc-xlstm): CLAUDE.md probe-layer paragraph + README experimental note

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 12: Final wrap-up via smart-commit

- [ ] **Step 1:** Invoke the `smart-commit` skill (Skill tool), instructing it to take the WHOLE `feature/cfc-xlstm` branch into account (not just the last diff): it re-syncs CLAUDE.md/README.md against everything the branch added and commits any doc drift. Do NOT push (user rule: never push).

---

## Execution notes

- Tasks 2-4 are independent of Tasks 5-7 only in theory; execute in order — the Python tests of Task 5+ assert against Rust-defined flat orders, and Task 6 needs the rebuilt PyO3 module.
- If `cargo clippy` fires `large_enum_variant` on `Layer::Slstm` (unboxed), box it and update the two construction sites (`from_v2_json`, `from_flat_weights_v2`) plus the plan's Interfaces note — do not silence the lint.
- The mamba3 equivalence/round-trip tests in `neural/tests.rs` and `tests/test_rust_python_mamba3_equivalence.py` are the authoritative idioms wherever this plan's test code diverges from the actual signatures on the branch (e.g. `from_flat_weights_v2` argument list, `from_json_file` entry-point name). Adapt the plan's test code to the real signatures, not the other way around.
- Training the probes is NOT part of this plan (hours of wall-time); the deliverable ends at `--generate` + green gates. Kick off `--train` runs separately.
