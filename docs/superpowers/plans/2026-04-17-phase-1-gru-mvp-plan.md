# Phase 1 PSO-GRU MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the first stateful NN architecture (GRU sandwiched between dense layers) on top of the Phase 0 infrastructure, trained with PSO, with all PSO NN output routed through Rust via the LayerWeights trait.

**Architecture:** Rust `Layer` struct splits into `Layer` enum (`Dense | Gru`) + sibling `DenseLayer` + new `GruLayer`. `LayerState::Gru { h }` plugs into the existing `NnState`. PyTorch `GruLayer` module with manual gate math mirrors the Rust cell bit-for-bit. `evaluate.py` stops writing v1 JSON directly and routes flat PSO weights through a new `aerocapture_rs.flat_weights_to_json` PyO3 helper, giving the Phase 0 `LayerWeights` trait its first production caller.

**Tech Stack:** Rust 2024 (edition), serde tagged enums, PyO3 for Python bindings, Python 3.14, PyTorch, Pydantic v2 discriminated unions, pymoo PSO, pytest.

**Spec:** `docs/superpowers/specs/2026-04-17-phase-1-gru-mvp-design.md`

---

## Task 0: Confirm branch + prep

**Files:**
- Already on `feature/gru-mvp` branch with spec committed.

- [ ] **Step 1: Verify branch state**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
git branch --show-current
git log --oneline main..HEAD
git status
```

Expected: on `feature/gru-mvp`, 1 commit ahead of main (spec), clean working tree.

- [ ] **Step 2: Update TODO.md to note Phase 1 is in progress**

Edit `TODO.md`, find the `### Phase 1 -- GRU MVP` header and add a tag after the title:

```markdown
### Phase 1 -- GRU MVP (validates the Phase 0 stack on one architecture) [IN PROGRESS on feature/gru-mvp]
```

Commit:

```bash
git add TODO.md
git commit -m "$(cat <<'EOF'
docs(todo): mark Phase 1 in progress on feature/gru-mvp

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 1: Rust -- split Layer struct into Layer enum + DenseLayer struct (pure rename, no behavior change)

**Files:**
- Modify: `src/rust/src/data/neural.rs`
- Modify: `src/rust/src/gnc/guidance/neural.rs` (test-code struct literals)

Goal: prepare for the GRU variant by splitting the existing dense-specific `Layer` struct into an enum. Zero behavior change. All forward output bit-identical pre/post.

- [ ] **Step 1: Read the current Layer struct and identify all construction sites**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
rg "Layer \{" src/rust/ -n
rg "impl.*Layer" src/rust/ -n
```

Expected: ~12-15 sites in `src/rust/src/data/neural.rs` and `src/rust/src/gnc/guidance/neural.rs` test modules. Each uses `Layer { w: ..., b: ..., activation: ... }` literal.

- [ ] **Step 2: Rename struct `Layer` to `DenseLayer` and wrap in enum**

In `src/rust/src/data/neural.rs`, find the struct definition:

```rust
#[derive(Debug, Clone)]
pub struct Layer {
    pub w: Vec<Vec<f64>>,
    pub b: Vec<f64>,
    pub activation: Activation,
}
```

Replace with:

```rust
/// A dense (fully-connected) layer: affine transform + activation.
#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub w: Vec<Vec<f64>>,
    pub b: Vec<f64>,
    pub activation: Activation,
}

/// Layer variant. Phase 1 ships Dense and Gru (added in Task 2).
#[derive(Debug, Clone)]
pub enum Layer {
    Dense(DenseLayer),
    // Phase 1 Task 2 adds: Gru(GruLayer)
    // Phases 2-4 add: Lstm, Attention, LayerNorm, Ssm, Window
}
```

- [ ] **Step 3: Move inherent `impl Layer` (if any) + `LayerWeights for Layer` onto `DenseLayer`**

The existing `impl LayerWeights for Layer { ... }` block in `src/rust/src/data/neural.rs` becomes:

```rust
impl LayerWeights for DenseLayer {
    fn to_flat(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.n_params());
        for row in &self.w {
            v.extend_from_slice(row);
        }
        v.extend_from_slice(&self.b);
        v
    }

    fn from_flat(&mut self, flat: &[f64]) -> usize {
        let n_out = self.w.len();
        let n_in = if n_out > 0 { self.w[0].len() } else { 0 };
        let mut idx = 0;
        for j in 0..n_out {
            self.w[j].copy_from_slice(&flat[idx..idx + n_in]);
            idx += n_in;
        }
        self.b.copy_from_slice(&flat[idx..idx + n_out]);
        idx += n_out;
        idx
    }

    fn n_params(&self) -> usize {
        let n_out = self.w.len();
        let n_in = if n_out > 0 { self.w[0].len() } else { 0 };
        n_out * n_in + n_out
    }
}
```

Also add a delegating `impl LayerWeights for Layer` that dispatches per variant:

```rust
impl LayerWeights for Layer {
    fn to_flat(&self) -> Vec<f64> {
        match self {
            Layer::Dense(d) => d.to_flat(),
        }
    }
    fn from_flat(&mut self, flat: &[f64]) -> usize {
        match self {
            Layer::Dense(d) => d.from_flat(flat),
        }
    }
    fn n_params(&self) -> usize {
        match self {
            Layer::Dense(d) => d.n_params(),
        }
    }
}
```

(Task 2 extends each match with a `Layer::Gru(g) => g.xxx(),` arm.)

- [ ] **Step 4: Add a helper on `Layer` for input_size**

Add inherent impl on `Layer`:

```rust
impl Layer {
    /// Returns the number of input features this layer expects.
    /// Used by NeuralNetModel::forward for the leading assert.
    pub fn input_size(&self) -> usize {
        match self {
            Layer::Dense(d) => if d.w.is_empty() { 0 } else { d.w[0].len() },
        }
    }
}
```

- [ ] **Step 5: Update `NeuralNetModel::forward` to dispatch through the enum**

Find the existing forward in `src/rust/src/data/neural.rs`:

```rust
pub fn forward(&self, state: &mut NnState, input: &[f64]) -> Vec<f64> {
    assert_eq!(input.len(), self.layer_sizes[0]);
    assert_eq!(state.layer_states.len(), self.layers.len());
    let mut current = input.to_vec();
    for (layer, _layer_state) in self.layers.iter().zip(state.layer_states.iter_mut()) {
        let n_out = layer.b.len();
        let mut next = Vec::with_capacity(n_out);
        for j in 0..n_out {
            let sum: f64 = layer.w[j].iter().zip(&current).map(|(w, x)| w * x).sum();
            next.push(layer.activation.apply(sum + layer.b[j]));
        }
        current = next;
    }
    current
}
```

Replace with:

```rust
pub fn forward(&self, state: &mut NnState, input: &[f64]) -> Vec<f64> {
    assert_eq!(input.len(), self.layer_sizes[0]);
    assert_eq!(state.layer_states.len(), self.layers.len());
    let mut current = input.to_vec();
    for (layer, layer_state) in self.layers.iter().zip(state.layer_states.iter_mut()) {
        match (layer, layer_state) {
            (Layer::Dense(d), LayerState::None) => {
                let n_out = d.b.len();
                let mut next = Vec::with_capacity(n_out);
                for j in 0..n_out {
                    let sum: f64 = d.w[j].iter().zip(&current).map(|(w, x)| w * x).sum();
                    next.push(d.activation.apply(sum + d.b[j]));
                }
                current = next;
            }
            _ => unreachable!("layer/state variant mismatch (construction invariant)"),
        }
    }
    current
}
```

Task 2 adds the `Layer::Gru / LayerState::Gru` arm.

- [ ] **Step 6: Update `from_v1_json` and `from_v2_json` to wrap in `Layer::Dense`**

Find each site in `src/rust/src/data/neural.rs` that pushes a `Layer { w, b, activation }` literal and wrap in `Layer::Dense(DenseLayer { w, b, activation })`.

Example in `from_v1_json`:

```rust
// Before:
layers.push(Layer {
    w: lw.w.clone(),
    b: lw.b.clone(),
    activation: file.architecture.activations[i],
});

// After:
layers.push(Layer::Dense(DenseLayer {
    w: lw.w.clone(),
    b: lw.b.clone(),
    activation: file.architecture.activations[i],
}));
```

Do the same in `from_v2_json`, `from_flat_weights` (the Phase 0 v1-compat wrapper), and any other constructor.

- [ ] **Step 7: Update `save_json` to emit v1-style weights dict for Dense variants**

`save_json` currently iterates `self.layers` and writes `NnLayerWeights { w: layer.w.clone(), b: layer.b.clone() }`. Update to match on the variant:

```rust
pub fn save_json(&self, path: &str) -> Result<(), DataError> {
    let mut weights = std::collections::BTreeMap::new();
    for (i, layer) in self.layers.iter().enumerate() {
        let key = format!("layer_{}", i);
        let entry = match layer {
            Layer::Dense(d) => NnLayerWeights {
                w: Some(d.w.clone()),
                b: Some(d.b.clone()),
            },
        };
        weights.insert(key, entry);
    }
    let file = NnJsonFileV2 {
        format_version: 2,
        architecture: self.architecture.clone(),
        weights,
        output_interpretation: self.output_interpretation.clone(),
        input_mask: self.input_mask.clone(),
        ablated_input: self.ablated_input,
    };
    let json = serde_json::to_string_pretty(&file)
        .map_err(|e| DataError(format!("JSON serialize error: {}", e)))?;
    std::fs::write(path, json).map_err(|e| DataError(format!("Cannot write {}: {}", path, e)))?;
    Ok(())
}
```

This requires changing `NnLayerWeights` struct's `w` and `b` to `Option<...>` so that the GRU variant (Task 3) can leave them None and use its own keys. Update `NnLayerWeights`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
struct NnLayerWeights {
    #[serde(skip_serializing_if = "Option::is_none")]
    w: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    b: Option<Vec<f64>>,
    // Task 3 adds: weight_ih, weight_hh, bias_ih, bias_hh (all Option<...>)
}
```

And `from_v1_json` / `from_v2_json` currently unwrap `lw.w` and `lw.b` directly. Change to:

```rust
let w = lw.w.as_ref().ok_or_else(|| DataError(format!("Layer {} missing w in {}", i, path)))?;
let b = lw.b.as_ref().ok_or_else(|| DataError(format!("Layer {} missing b in {}", i, path)))?;
```

- [ ] **Step 8: Update all test-code literals in `src/rust/src/gnc/guidance/neural.rs` tests and `src/rust/src/data/neural.rs` tests**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
rg "Layer \{" src/rust/ -n
```

For every `Layer { w: ..., b: ..., activation: ... }` literal (there are ~12-15 in test code), wrap in `Layer::Dense(DenseLayer { ... })`.

- [ ] **Step 9: Build and run tests**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo build -p aerocapture 2>&1 | tail -10
cargo test -p aerocapture 2>&1 | grep -E "^test result"
```

Expected: build clean, every `test result` line shows `0 failed`. All 436+ tests pass (same count as before the refactor; behavior unchanged).

- [ ] **Step 10: Run golden regression tests explicitly**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo test -p aerocapture --test guidance_regression 2>&1 | tail -5
```

Expected: 6 tests pass bit-identically.

- [ ] **Step 11: Clippy + fmt**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo clippy -p aerocapture --all-targets -- -D warnings 2>&1 | tail -5
cargo fmt -p aerocapture --check
```

Expected: both clean.

- [ ] **Step 12: Commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add src/rust/src/data/neural.rs src/rust/src/gnc/guidance/neural.rs
git commit -m "$(cat <<'EOF'
refactor(nn): split Layer struct into Layer enum + DenseLayer struct

Phase 1 prep: rename the existing dense-specific Layer struct to
DenseLayer and wrap in a Layer enum. Forward pass dispatches per
variant. Behavior unchanged -- 6 guidance-regression golden files
still pass bit-identically. Task 2 adds the Gru variant.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Rust -- add GruLayer + Layer::Gru variant + LayerSpec::Gru variant + forward dispatch

**Files:**
- Modify: `src/rust/src/data/neural.rs`

- [ ] **Step 1: Write failing test for GRU forward math**

Add to `src/rust/src/data/neural.rs` tests module:

```rust
#[test]
fn gru_forward_known_output() {
    // Minimal 2-input, 2-hidden GRU with specific weights; hand-compute output.
    // With all weights = 0 and biases = 0, h_new = (1 - 0.5) * 0 + 0.5 * h_prev = 0.5 * h_prev.
    let gru = GruLayer {
        input_size: 2,
        hidden_size: 2,
        weight_ih: vec![vec![0.0, 0.0]; 6],  // 3H=6 rows, 2 cols
        weight_hh: vec![vec![0.0, 0.0]; 6],  // 3H=6 rows, 2 cols
        bias_ih: vec![0.0; 6],
        bias_hh: vec![0.0; 6],
    };
    let h_prev = vec![1.0, 2.0];
    let x = vec![0.5, -0.5];
    let h_new = gru.forward(&h_prev, &x);
    // r=sigmoid(0)=0.5, z=sigmoid(0)=0.5, n=tanh(0+0.5*0)=0.
    // h_new[i] = (1 - 0.5) * 0 + 0.5 * h_prev[i] = 0.5 * h_prev[i].
    assert!((h_new[0] - 0.5).abs() < 1e-12);
    assert!((h_new[1] - 1.0).abs() < 1e-12);
}
```

- [ ] **Step 2: Run to confirm failure**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo test -p aerocapture --lib data::neural::tests::gru_forward_known_output 2>&1 | tail -5
```

Expected: compilation error on missing `GruLayer` type.

- [ ] **Step 3: Add LayerSpec::Gru variant**

In `src/rust/src/data/neural.rs`, extend the `LayerSpec` enum:

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
}
```

- [ ] **Step 4: Add GruLayer struct and forward impl**

Insert in `src/rust/src/data/neural.rs` after `DenseLayer`:

```rust
/// GRU cell matching PyTorch nn.GRUCell convention (two biases per gate).
///
/// Forward equations:
///   r_t = sigmoid(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)
///   z_t = sigmoid(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)
///   n_t = tanh(W_in @ x_t + b_in + r_t * (W_hn @ h_{t-1} + b_hn))
///   h_t = (1 - z_t) * n_t + z_t * h_{t-1}
///
/// Weight storage matches torch.nn.GRUCell:
///   weight_ih: [3H, input_size] with rows 0..H = W_ir, H..2H = W_iz, 2H..3H = W_in
///   weight_hh: [3H, H] with rows 0..H = W_hr, H..2H = W_hz, 2H..3H = W_hn
///   bias_ih:   [3H] in order b_ir, b_iz, b_in
///   bias_hh:   [3H] in order b_hr, b_hz, b_hn
#[derive(Debug, Clone)]
pub struct GruLayer {
    pub input_size: usize,
    pub hidden_size: usize,
    pub weight_ih: Vec<Vec<f64>>,
    pub weight_hh: Vec<Vec<f64>>,
    pub bias_ih: Vec<f64>,
    pub bias_hh: Vec<f64>,
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl GruLayer {
    /// Compute one forward step: (h_prev, x) -> h_new. Output == h_new (GRU).
    pub fn forward(&self, h_prev: &[f64], x: &[f64]) -> Vec<f64> {
        assert_eq!(h_prev.len(), self.hidden_size);
        assert_eq!(x.len(), self.input_size);
        let h_size = self.hidden_size;
        let mut h_new = vec![0.0; h_size];

        for i in 0..h_size {
            // r gate: row i
            let mut s_ih_r = self.bias_ih[i];
            for k in 0..self.input_size {
                s_ih_r += self.weight_ih[i][k] * x[k];
            }
            let mut s_hh_r = self.bias_hh[i];
            for k in 0..h_size {
                s_hh_r += self.weight_hh[i][k] * h_prev[k];
            }
            let r = sigmoid(s_ih_r + s_hh_r);

            // z gate: row i + H
            let mut s_ih_z = self.bias_ih[i + h_size];
            for k in 0..self.input_size {
                s_ih_z += self.weight_ih[i + h_size][k] * x[k];
            }
            let mut s_hh_z = self.bias_hh[i + h_size];
            for k in 0..h_size {
                s_hh_z += self.weight_hh[i + h_size][k] * h_prev[k];
            }
            let z = sigmoid(s_ih_z + s_hh_z);

            // n gate: row i + 2H
            let mut s_ih_n = self.bias_ih[i + 2 * h_size];
            for k in 0..self.input_size {
                s_ih_n += self.weight_ih[i + 2 * h_size][k] * x[k];
            }
            let mut s_hh_n = self.bias_hh[i + 2 * h_size];
            for k in 0..h_size {
                s_hh_n += self.weight_hh[i + 2 * h_size][k] * h_prev[k];
            }
            let n = (s_ih_n + r * s_hh_n).tanh();

            h_new[i] = (1.0 - z) * n + z * h_prev[i];
        }
        h_new
    }
}
```

- [ ] **Step 5: Extend Layer enum with Gru variant**

Update:

```rust
#[derive(Debug, Clone)]
pub enum Layer {
    Dense(DenseLayer),
    Gru(GruLayer),
}
```

- [ ] **Step 6: Extend `Layer::input_size`**

```rust
impl Layer {
    pub fn input_size(&self) -> usize {
        match self {
            Layer::Dense(d) => if d.w.is_empty() { 0 } else { d.w[0].len() },
            Layer::Gru(g) => g.input_size,
        }
    }
}
```

- [ ] **Step 7: Extend `LayerWeights for Layer` dispatch**

```rust
impl LayerWeights for Layer {
    fn to_flat(&self) -> Vec<f64> {
        match self {
            Layer::Dense(d) => d.to_flat(),
            Layer::Gru(g) => g.to_flat(),
        }
    }
    fn from_flat(&mut self, flat: &[f64]) -> usize {
        match self {
            Layer::Dense(d) => d.from_flat(flat),
            Layer::Gru(g) => g.from_flat(flat),
        }
    }
    fn n_params(&self) -> usize {
        match self {
            Layer::Dense(d) => d.n_params(),
            Layer::Gru(g) => g.n_params(),
        }
    }
}
```

Task 4 adds `impl LayerWeights for GruLayer`. For now, add a stub so the trait object dispatch compiles:

```rust
impl LayerWeights for GruLayer {
    fn to_flat(&self) -> Vec<f64> { unimplemented!("filled in Task 4") }
    fn from_flat(&mut self, _flat: &[f64]) -> usize { unimplemented!("filled in Task 4") }
    fn n_params(&self) -> usize {
        3 * self.hidden_size * self.input_size
            + 3 * self.hidden_size * self.hidden_size
            + 2 * 3 * self.hidden_size
    }
}
```

(Only `n_params` needs a real impl to satisfy downstream callers that compute model size; `to_flat`/`from_flat` are filled in Task 4.)

- [ ] **Step 8: Run test to verify forward math**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo test -p aerocapture --lib data::neural::tests::gru_forward_known_output 2>&1 | tail -5
```

Expected: PASS.

- [ ] **Step 9: Run all tests**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo test -p aerocapture 2>&1 | grep -E "^test result"
```

Expected: 0 failures.

- [ ] **Step 10: Clippy + fmt**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo clippy -p aerocapture --all-targets -- -D warnings 2>&1 | tail -5
cargo fmt -p aerocapture --check
```

Expected: both clean.

- [ ] **Step 11: Commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add src/rust/src/data/neural.rs
git commit -m "$(cat <<'EOF'
feat(nn): add GruLayer + Layer::Gru + LayerSpec::Gru variants

PyTorch nn.GRUCell convention (two biases per gate). Forward loop is
the dumb readable reference -- per-element, no fused matmul. For
600-step episodes at H=32 the per-tick cost is ~2500 multiplies,
negligible next to the sim step. Optimization deferred until
profiling says it matters. LayerWeights stub present; full impl in
Task 4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Rust -- add LayerState::Gru + NnState::Clone behavioral test

**Files:**
- Modify: `src/rust/src/data/nn_state.rs`

- [ ] **Step 1: Write failing behavioral Clone test**

Add to `src/rust/src/data/nn_state.rs` tests module:

```rust
#[test]
fn clone_is_behaviorally_independent_with_gru_state() {
    use crate::data::neural::{GruLayer, Layer};

    // Construct a model with a single Gru layer (bypassing the full
    // NeuralNetModel since we only need the LayerState here).
    let gru = GruLayer {
        input_size: 2,
        hidden_size: 3,
        weight_ih: vec![vec![0.0; 2]; 9],
        weight_hh: vec![vec![0.0; 3]; 9],
        bias_ih: vec![0.0; 9],
        bias_hh: vec![0.0; 9],
    };
    let layer = Layer::Gru(gru);
    let original_state = LayerState::for_layer(&layer);
    let mut cloned_state = original_state.clone();

    if let LayerState::Gru { h } = &mut cloned_state {
        h[0] = 42.0;
    } else {
        panic!("expected LayerState::Gru");
    }

    // Mutating clone must not affect original.
    if let LayerState::Gru { h } = &original_state {
        assert_eq!(h[0], 0.0);
        assert_eq!(h[1], 0.0);
        assert_eq!(h[2], 0.0);
    } else {
        panic!("expected LayerState::Gru");
    }
}
```

- [ ] **Step 2: Run to confirm failure**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo test -p aerocapture --lib data::nn_state::tests::clone_is_behaviorally_independent_with_gru_state 2>&1 | tail -5
```

Expected: compilation error -- `LayerState::Gru` variant does not exist.

- [ ] **Step 3: Add Gru variant to LayerState**

In `src/rust/src/data/nn_state.rs`:

```rust
#[derive(Debug, Clone)]
pub enum LayerState {
    None,
    Gru { h: Vec<f64> },
    // Phase 2+: Lstm { h, c }, Window { buffer }, Ssm { h }
}

impl LayerState {
    pub fn for_layer(layer: &Layer) -> Self {
        match layer {
            Layer::Dense(_) => LayerState::None,
            Layer::Gru(g) => LayerState::Gru {
                h: vec![0.0; g.hidden_size],
            },
        }
    }

    pub fn reset(&mut self) {
        match self {
            LayerState::None => {}
            LayerState::Gru { h } => {
                for v in h.iter_mut() {
                    *v = 0.0;
                }
            }
        }
    }
}
```

- [ ] **Step 4: Update `NeuralNetModel::forward` in `data/neural.rs` to dispatch Gru**

Extend the match in `forward`:

```rust
pub fn forward(&self, state: &mut NnState, input: &[f64]) -> Vec<f64> {
    assert_eq!(input.len(), self.layers[0].input_size());
    assert_eq!(state.layer_states.len(), self.layers.len());
    let mut current = input.to_vec();
    for (layer, layer_state) in self.layers.iter().zip(state.layer_states.iter_mut()) {
        match (layer, layer_state) {
            (Layer::Dense(d), LayerState::None) => {
                let n_out = d.b.len();
                let mut next = Vec::with_capacity(n_out);
                for j in 0..n_out {
                    let sum: f64 = d.w[j].iter().zip(&current).map(|(w, x)| w * x).sum();
                    next.push(d.activation.apply(sum + d.b[j]));
                }
                current = next;
            }
            (Layer::Gru(g), LayerState::Gru { h }) => {
                let h_new = g.forward(h, &current);
                *h = h_new.clone();
                current = h_new;
            }
            _ => unreachable!("layer/state variant mismatch (construction invariant)"),
        }
    }
    current
}
```

- [ ] **Step 5: Run the new test and the full data::nn_state tests**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo test -p aerocapture --lib data::nn_state 2>&1 | tail -10
```

Expected: all pass, including the new `clone_is_behaviorally_independent_with_gru_state`.

- [ ] **Step 6: Run full test suite**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo test -p aerocapture 2>&1 | grep -E "^test result"
```

Expected: 0 failures everywhere.

- [ ] **Step 7: Clippy + fmt**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo clippy -p aerocapture --all-targets -- -D warnings 2>&1 | tail -5
cargo fmt -p aerocapture --check
```

Expected: both clean.

- [ ] **Step 8: Commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add src/rust/src/data/nn_state.rs src/rust/src/data/neural.rs
git commit -m "$(cat <<'EOF'
feat(nn): add LayerState::Gru + NeuralNetModel::forward Gru dispatch

LayerState::Gru carries the [hidden_size] h vector per sim. for_layer
eagerly sizes from the GruLayer; reset zeros the vector. The new
clone_is_behaviorally_independent_with_gru_state test closes Phase 0
review carry-over #4 -- Clone coverage is now behavioral, not just
structural.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Rust -- LayerWeights for GruLayer + flat roundtrip test

**Files:**
- Modify: `src/rust/src/data/neural.rs`

- [ ] **Step 1: Write failing roundtrip test**

Add to `src/rust/src/data/neural.rs` tests module:

```rust
#[test]
fn gru_flat_weights_roundtrip() {
    // Build a GruLayer with distinct weight values so a buggy to_flat/from_flat
    // would produce visible mismatches.
    let input_size = 2;
    let hidden_size = 3;
    let three_h = 3 * hidden_size;
    let mut w_ih = Vec::with_capacity(three_h);
    let mut w_hh = Vec::with_capacity(three_h);
    for i in 0..three_h {
        w_ih.push((0..input_size).map(|k| (i * 10 + k) as f64 * 0.01).collect());
        w_hh.push((0..hidden_size).map(|k| (i * 10 + k) as f64 * 0.001).collect());
    }
    let b_ih: Vec<f64> = (0..three_h).map(|i| 0.1 + i as f64 * 0.01).collect();
    let b_hh: Vec<f64> = (0..three_h).map(|i| 0.2 + i as f64 * 0.01).collect();

    let original = GruLayer {
        input_size, hidden_size,
        weight_ih: w_ih, weight_hh: w_hh,
        bias_ih: b_ih, bias_hh: b_hh,
    };

    let flat = original.to_flat();
    assert_eq!(flat.len(), original.n_params());

    // Reconstruct an empty-shaped GruLayer and fill via from_flat.
    let mut twin = GruLayer {
        input_size, hidden_size,
        weight_ih: vec![vec![0.0; input_size]; three_h],
        weight_hh: vec![vec![0.0; hidden_size]; three_h],
        bias_ih: vec![0.0; three_h],
        bias_hh: vec![0.0; three_h],
    };
    let consumed = twin.from_flat(&flat);
    assert_eq!(consumed, flat.len());

    // Forward outputs must match on a fixed input.
    let h_prev = vec![0.1, -0.2, 0.3];
    let x = vec![0.5, -0.4];
    let out_orig = original.forward(&h_prev, &x);
    let out_twin = twin.forward(&h_prev, &x);
    for (a, b) in out_orig.iter().zip(out_twin.iter()) {
        assert!((a - b).abs() < 1e-15, "{} vs {}", a, b);
    }
}
```

- [ ] **Step 2: Run to confirm failure**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo test -p aerocapture --lib data::neural::tests::gru_flat_weights_roundtrip 2>&1 | tail -5
```

Expected: test panics with `unimplemented!("filled in Task 4")` from Task 2's stub.

- [ ] **Step 3: Replace stub with real `to_flat` / `from_flat`**

In `src/rust/src/data/neural.rs`, replace the Task 2 stub `impl LayerWeights for GruLayer` with:

```rust
impl LayerWeights for GruLayer {
    fn to_flat(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.n_params());
        for row in &self.weight_ih {
            v.extend_from_slice(row);
        }
        for row in &self.weight_hh {
            v.extend_from_slice(row);
        }
        v.extend_from_slice(&self.bias_ih);
        v.extend_from_slice(&self.bias_hh);
        v
    }

    fn from_flat(&mut self, flat: &[f64]) -> usize {
        let three_h = 3 * self.hidden_size;
        let mut idx = 0;
        for row in self.weight_ih.iter_mut() {
            row.copy_from_slice(&flat[idx..idx + self.input_size]);
            idx += self.input_size;
        }
        for row in self.weight_hh.iter_mut() {
            row.copy_from_slice(&flat[idx..idx + self.hidden_size]);
            idx += self.hidden_size;
        }
        self.bias_ih.copy_from_slice(&flat[idx..idx + three_h]);
        idx += three_h;
        self.bias_hh.copy_from_slice(&flat[idx..idx + three_h]);
        idx += three_h;
        idx
    }

    fn n_params(&self) -> usize {
        3 * self.hidden_size * self.input_size
            + 3 * self.hidden_size * self.hidden_size
            + 2 * 3 * self.hidden_size
    }
}
```

- [ ] **Step 4: Run the roundtrip test**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo test -p aerocapture --lib data::neural::tests::gru_flat_weights_roundtrip 2>&1 | tail -5
```

Expected: PASS.

- [ ] **Step 5: Run full test suite**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo test -p aerocapture 2>&1 | grep -E "^test result"
```

Expected: 0 failures.

- [ ] **Step 6: Clippy + fmt**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo clippy -p aerocapture --all-targets -- -D warnings 2>&1 | tail -5
cargo fmt -p aerocapture --check
```

Expected: both clean.

- [ ] **Step 7: Commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add src/rust/src/data/neural.rs
git commit -m "$(cat <<'EOF'
feat(nn): LayerWeights for GruLayer (flat-weight round-trip)

Canonical flat order: weight_ih row-major, then weight_hh row-major,
then bias_ih, then bias_hh. Byte-equivalent to the per-gate ordering
documented in Phase 0 spec section 3.4 when read as a flat sequence.
Roundtrip test uses distinct weights so silent dropped bytes or
swapped matrices produce visible mismatches.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Rust -- JSON v2 read+write for GRU + from_flat_weights_v2

**Files:**
- Modify: `src/rust/src/data/neural.rs`

- [ ] **Step 1: Write failing JSON roundtrip test**

Add to tests module:

```rust
#[test]
fn v2_gru_json_roundtrip() {
    let input_size = 2;
    let hidden_size = 3;
    let three_h = 9;
    let gru = GruLayer {
        input_size, hidden_size,
        weight_ih: (0..three_h).map(|i| (0..input_size).map(|k| (i + k) as f64 * 0.01).collect()).collect(),
        weight_hh: (0..three_h).map(|i| (0..hidden_size).map(|k| (i + k) as f64 * 0.02).collect()).collect(),
        bias_ih: (0..three_h).map(|i| 0.1 + i as f64 * 0.01).collect(),
        bias_hh: (0..three_h).map(|i| 0.2 + i as f64 * 0.01).collect(),
    };
    let original = NeuralNetModel {
        architecture: vec![LayerSpec::Gru { input_size, hidden_size }],
        layer_sizes: vec![input_size, hidden_size],
        layers: vec![Layer::Gru(gru)],
        output_interpretation: "atan2".to_string(),
        input_mask: None,
        ablated_input: None,
    };

    let tmpdir = std::env::temp_dir();
    let path = tmpdir.join("gru_roundtrip.json");
    original.save_json(path.to_str().unwrap()).unwrap();

    let loaded = NeuralNetModel::load(path.to_str().unwrap()).unwrap();
    assert_eq!(loaded.layers.len(), 1);
    match &loaded.layers[0] {
        Layer::Gru(g) => {
            assert_eq!(g.input_size, input_size);
            assert_eq!(g.hidden_size, hidden_size);
        }
        _ => panic!("expected Gru layer"),
    }
    // Forward parity
    use crate::data::nn_state::NnState;
    let mut s0 = NnState::for_model(&original);
    let mut s1 = NnState::for_model(&loaded);
    let x = vec![0.3, -0.4];
    let o0 = original.forward(&mut s0, &x);
    let o1 = loaded.forward(&mut s1, &x);
    for (a, b) in o0.iter().zip(o1.iter()) {
        assert!((a - b).abs() < 1e-15);
    }
}
```

- [ ] **Step 2: Run to confirm failure**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo test -p aerocapture --lib data::neural::tests::v2_gru_json_roundtrip 2>&1 | tail -10
```

Expected: FAIL, probably on `save_json` or `from_v2_json` because the GRU variant isn't handled.

- [ ] **Step 3: Extend `NnLayerWeights` to carry the GRU fields**

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
struct NnLayerWeights {
    #[serde(skip_serializing_if = "Option::is_none")]
    w: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    b: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    weight_ih: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    weight_hh: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bias_ih: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bias_hh: Option<Vec<f64>>,
}
```

- [ ] **Step 4: Extend `save_json` to emit Gru weights**

```rust
pub fn save_json(&self, path: &str) -> Result<(), DataError> {
    let mut weights = std::collections::BTreeMap::new();
    for (i, layer) in self.layers.iter().enumerate() {
        let key = format!("layer_{}", i);
        let entry = match layer {
            Layer::Dense(d) => NnLayerWeights {
                w: Some(d.w.clone()),
                b: Some(d.b.clone()),
                weight_ih: None, weight_hh: None, bias_ih: None, bias_hh: None,
            },
            Layer::Gru(g) => NnLayerWeights {
                w: None, b: None,
                weight_ih: Some(g.weight_ih.clone()),
                weight_hh: Some(g.weight_hh.clone()),
                bias_ih: Some(g.bias_ih.clone()),
                bias_hh: Some(g.bias_hh.clone()),
            },
        };
        weights.insert(key, entry);
    }
    let file = NnJsonFileV2 {
        format_version: 2,
        architecture: self.architecture.clone(),
        weights,
        output_interpretation: self.output_interpretation.clone(),
        input_mask: self.input_mask.clone(),
        ablated_input: self.ablated_input,
    };
    let json = serde_json::to_string_pretty(&file)
        .map_err(|e| DataError(format!("JSON serialize error: {}", e)))?;
    std::fs::write(path, json).map_err(|e| DataError(format!("Cannot write {}: {}", path, e)))?;
    Ok(())
}
```

- [ ] **Step 5: Extend `from_v2_json` to read Gru layers**

Find the match arm over `LayerSpec` variants in `from_v2_json`. Currently only Dense. Extend:

```rust
for (i, spec) in file.architecture.iter().enumerate() {
    let key = format!("layer_{}", i);
    let lw = file.weights.get(&key).ok_or_else(|| {
        DataError(format!("Missing {} in weights in {}", key, path))
    })?;
    match spec {
        LayerSpec::Dense { input_size, output_size, activation } => {
            if i == 0 { layer_sizes.push(*input_size); }
            layer_sizes.push(*output_size);
            let w = lw.w.as_ref().ok_or_else(|| DataError(format!("Layer {} (dense) missing w in {}", i, path)))?;
            let b = lw.b.as_ref().ok_or_else(|| DataError(format!("Layer {} (dense) missing b in {}", i, path)))?;
            if w.len() != *output_size || b.len() != *output_size {
                return Err(DataError(format!(
                    "Layer {} (dense) size mismatch: expected {}x{}, got w={}x?, b={} in {}",
                    i, output_size, input_size, w.len(), b.len(), path
                )));
            }
            for (row_idx, row) in w.iter().enumerate() {
                if row.len() != *input_size {
                    return Err(DataError(format!(
                        "Layer {} (dense) weight row {} length mismatch: expected {}, got {} in {}",
                        i, row_idx, input_size, row.len(), path
                    )));
                }
            }
            layers.push(Layer::Dense(DenseLayer {
                w: w.clone(),
                b: b.clone(),
                activation: *activation,
            }));
        }
        LayerSpec::Gru { input_size, hidden_size } => {
            if i == 0 { layer_sizes.push(*input_size); }
            layer_sizes.push(*hidden_size);
            let three_h = 3 * hidden_size;
            let w_ih = lw.weight_ih.as_ref().ok_or_else(|| DataError(format!("Layer {} (gru) missing weight_ih in {}", i, path)))?;
            let w_hh = lw.weight_hh.as_ref().ok_or_else(|| DataError(format!("Layer {} (gru) missing weight_hh in {}", i, path)))?;
            let b_ih = lw.bias_ih.as_ref().ok_or_else(|| DataError(format!("Layer {} (gru) missing bias_ih in {}", i, path)))?;
            let b_hh = lw.bias_hh.as_ref().ok_or_else(|| DataError(format!("Layer {} (gru) missing bias_hh in {}", i, path)))?;
            if w_ih.len() != three_h {
                return Err(DataError(format!("Layer {} (gru) weight_ih must have {} rows, got {} in {}", i, three_h, w_ih.len(), path)));
            }
            if w_hh.len() != three_h {
                return Err(DataError(format!("Layer {} (gru) weight_hh must have {} rows, got {} in {}", i, three_h, w_hh.len(), path)));
            }
            if b_ih.len() != three_h || b_hh.len() != three_h {
                return Err(DataError(format!("Layer {} (gru) biases must have {} elements in {}", i, three_h, path)));
            }
            for (r, row) in w_ih.iter().enumerate() {
                if row.len() != *input_size {
                    return Err(DataError(format!("Layer {} (gru) weight_ih row {} length: expected {}, got {} in {}", i, r, input_size, row.len(), path)));
                }
            }
            for (r, row) in w_hh.iter().enumerate() {
                if row.len() != *hidden_size {
                    return Err(DataError(format!("Layer {} (gru) weight_hh row {} length: expected {}, got {} in {}", i, r, hidden_size, row.len(), path)));
                }
            }
            layers.push(Layer::Gru(GruLayer {
                input_size: *input_size,
                hidden_size: *hidden_size,
                weight_ih: w_ih.clone(),
                weight_hh: w_hh.clone(),
                bias_ih: b_ih.clone(),
                bias_hh: b_hh.clone(),
            }));
        }
    }
}
```

- [ ] **Step 6: Add `from_flat_weights_v2`**

```rust
/// Construct a NeuralNetModel from a flat weight vector and v2 architecture spec.
/// Used by the PyO3 flat_weights_to_json helper that routes PSO output through Rust.
pub fn from_flat_weights_v2(
    flat: &[f64],
    architecture: &[LayerSpec],
    output_interpretation: &str,
    input_mask: Option<Vec<usize>>,
) -> Result<Self, DataError> {
    let mut layers: Vec<Layer> = Vec::with_capacity(architecture.len());
    let mut layer_sizes: Vec<usize> = Vec::with_capacity(architecture.len() + 1);
    let mut offset: usize = 0;

    for (i, spec) in architecture.iter().enumerate() {
        let mut layer = match spec {
            LayerSpec::Dense { input_size, output_size, activation } => {
                if i == 0 { layer_sizes.push(*input_size); }
                layer_sizes.push(*output_size);
                Layer::Dense(DenseLayer {
                    w: vec![vec![0.0; *input_size]; *output_size],
                    b: vec![0.0; *output_size],
                    activation: *activation,
                })
            }
            LayerSpec::Gru { input_size, hidden_size } => {
                if i == 0 { layer_sizes.push(*input_size); }
                layer_sizes.push(*hidden_size);
                let three_h = 3 * hidden_size;
                Layer::Gru(GruLayer {
                    input_size: *input_size,
                    hidden_size: *hidden_size,
                    weight_ih: vec![vec![0.0; *input_size]; three_h],
                    weight_hh: vec![vec![0.0; *hidden_size]; three_h],
                    bias_ih: vec![0.0; three_h],
                    bias_hh: vec![0.0; three_h],
                })
            }
        };
        let needed = layer.n_params();
        if offset + needed > flat.len() {
            return Err(DataError(format!(
                "from_flat_weights_v2: layer {} needs {} params but only {} remaining (total flat len {})",
                i, needed, flat.len() - offset, flat.len()
            )));
        }
        let consumed = layer.from_flat(&flat[offset..]);
        offset += consumed;
        layers.push(layer);
    }

    if offset != flat.len() {
        return Err(DataError(format!(
            "from_flat_weights_v2: weight vector length mismatch, consumed {} of {}",
            offset, flat.len()
        )));
    }

    Self::validate_mask(&input_mask, layer_sizes[0])?;

    let output_size = *layer_sizes.last().unwrap_or(&0);
    if output_interpretation != "direct" && output_size < 2 {
        return Err(DataError(format!(
            "output_interpretation '{}' requires >= 2 outputs, got {}",
            output_interpretation, output_size
        )));
    }

    Ok(NeuralNetModel {
        architecture: architecture.to_vec(),
        layer_sizes,
        layers,
        output_interpretation: output_interpretation.to_string(),
        input_mask,
        ablated_input: None,
    })
}
```

- [ ] **Step 7: Run the JSON roundtrip test + existing tests**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo test -p aerocapture --lib data::neural 2>&1 | tail -10
```

Expected: all pass, including `v2_gru_json_roundtrip`.

- [ ] **Step 8: Run full test suite**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo test -p aerocapture 2>&1 | grep -E "^test result"
```

Expected: 0 failures.

- [ ] **Step 9: Clippy + fmt**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo clippy -p aerocapture --all-targets -- -D warnings 2>&1 | tail -5
cargo fmt -p aerocapture --check
```

Expected: both clean.

- [ ] **Step 10: Commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add src/rust/src/data/neural.rs
git commit -m "$(cat <<'EOF'
feat(nn): v2 JSON read+write for Gru + from_flat_weights_v2

NnLayerWeights gains optional weight_ih/weight_hh/bias_ih/bias_hh fields
with serde skip-serializing-if-none. save_json emits per-variant keys
(w/b for Dense, weight_ih/... for Gru). from_v2_json reads and validates
both variants with informative error messages. from_flat_weights_v2 is
the new trait-backed constructor that will power the PSO PyO3 helper in
Task 7.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Rust -- TOML `[[network.architecture]]` parser

**Files:**
- Modify: `src/rust/src/config.rs`
- Modify: `src/rust/src/data/neural.rs` (constructor that uses the new spec)
- Test: inline `#[cfg(test)]` in config.rs

- [ ] **Step 1: Read current TomlNetwork definition**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
rg "TomlNetwork|struct.*Network" src/rust/src/config.rs -n -A 10
```

- [ ] **Step 2: Write failing test**

Add to `src/rust/src/config.rs` tests:

```rust
#[test]
fn network_architecture_v2_parses() {
    let toml = r#"
[network]
input_mask = [0, 1, 2]
output_interpretation = "atan2"

[[network.architecture]]
type = "dense"
input_size = 3
output_size = 4
activation = "tanh"

[[network.architecture]]
type = "gru"
input_size = 4
hidden_size = 4

[[network.architecture]]
type = "dense"
input_size = 4
output_size = 2
activation = "linear"
"#;
    let parsed: toml::Value = toml::from_str(toml).unwrap();
    // Drill into the [network] table and parse as TomlNetwork.
    let network_value = parsed.get("network").expect("network section present");
    let network: TomlNetwork = network_value.clone().try_into().expect("TomlNetwork parse");
    let arch = network.architecture.expect("architecture v2 path present");
    assert_eq!(arch.len(), 3);
    match &arch[1] {
        TomlLayerSpec::Gru { input_size, hidden_size } => {
            assert_eq!(*input_size, 4);
            assert_eq!(*hidden_size, 4);
        }
        _ => panic!("expected Gru at index 1"),
    }
}
```

- [ ] **Step 3: Run to confirm failure**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo test -p aerocapture --lib config::tests::network_architecture_v2_parses 2>&1 | tail -5
```

Expected: compilation error -- `TomlLayerSpec` and `architecture` field not defined.

- [ ] **Step 4: Extend TomlNetwork**

In `src/rust/src/config.rs`:

```rust
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TomlLayerSpec {
    Dense {
        input_size: usize,
        output_size: usize,
        activation: String,
    },
    Gru {
        input_size: usize,
        hidden_size: usize,
    },
}

// Add `architecture` optional field to the existing TomlNetwork struct:
#[derive(Debug, Clone, Deserialize)]
pub struct TomlNetwork {
    pub layer_sizes: Option<Vec<usize>>,
    pub activations: Option<Vec<String>>,
    /// v2 path: heterogeneous architecture spec. When present, layer_sizes/activations
    /// are ignored. When absent, v1 path applies.
    pub architecture: Option<Vec<TomlLayerSpec>>,
    pub input_mask: Option<Vec<usize>>,
    pub ablated_input: Option<usize>,
    #[serde(default = "default_output_interpretation")]
    pub output_interpretation: String,
}

fn default_output_interpretation() -> String {
    "atan2".to_string()
}
```

(If the struct already has a different field layout, add the `architecture` + `TomlLayerSpec` without disturbing the others.)

- [ ] **Step 5: Add conversion from TomlLayerSpec to LayerSpec**

In `config.rs`:

```rust
impl TomlLayerSpec {
    pub fn to_layer_spec(&self) -> Result<crate::data::neural::LayerSpec, ParseError> {
        use crate::data::neural::LayerSpec;
        match self {
            TomlLayerSpec::Dense { input_size, output_size, activation } => {
                let act = crate::data::neural::parse_activation(activation)
                    .map_err(|_| ParseError::InvalidField(format!("unknown activation: {}", activation)))?;
                Ok(LayerSpec::Dense {
                    input_size: *input_size,
                    output_size: *output_size,
                    activation: act,
                })
            }
            TomlLayerSpec::Gru { input_size, hidden_size } => {
                Ok(LayerSpec::Gru {
                    input_size: *input_size,
                    hidden_size: *hidden_size,
                })
            }
        }
    }
}
```

If `parse_activation` does not exist in `data/neural.rs`, add it (a simple string-to-Activation match; probably already present in the v1 loading code -- extract to a `pub` helper).

- [ ] **Step 6: Wire the architecture path into model construction**

Find where `TomlNetwork` is consumed to construct a `NeuralNetModel` or pass architecture data to the sim (likely in `config.rs::from_toml_file` or `data/mod.rs::SimData::from_toml`). The logic is:
- If `architecture` is Some, use `LayerSpec` list directly.
- Else (v1), synthesize from `layer_sizes` + `activations`.

Search the codebase for where layer_sizes and activations are currently consumed:

```bash
rg "layer_sizes|activations" src/rust/src/config.rs src/rust/src/data/ -n
```

Update to branch on `architecture.is_some()`. The network model itself is loaded from the JSON file at runtime -- TOML `[network]` only overrides architecture at training time. Verify with a grep. If the TOML-to-model plumbing already goes through `NeuralNetModel::load(path)`, the architecture field is informational at this layer (consumed by evaluate.py when generating the JSON) and only needs to be validated for consistency against the loaded model shape.

For now, keep the scope minimal: parse the field in TOML and make it available. Phase 1's evaluate.py integration (Task 9) is where it actually gets used.

- [ ] **Step 7: Run test + full suite**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo test -p aerocapture --lib config 2>&1 | tail -10
cargo test -p aerocapture 2>&1 | grep -E "^test result"
```

Expected: 0 failures, new test passes.

- [ ] **Step 8: Clippy + fmt**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo clippy -p aerocapture --all-targets -- -D warnings 2>&1 | tail -5
cargo fmt -p aerocapture --check
```

Expected: both clean.

- [ ] **Step 9: Commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add src/rust/src/config.rs src/rust/src/data/neural.rs
git commit -m "$(cat <<'EOF'
feat(config): TOML [[network.architecture]] array-of-tables parser

Adds TomlLayerSpec tagged enum (dense|gru) and optional architecture
field on TomlNetwork. When present, v2 heterogeneous architecture path
applies; when absent, v1 layer_sizes+activations path unchanged. The
field is exposed through to_layer_spec() for conversion to the Rust
LayerSpec enum; actual use by the training path lands in Task 9.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: PyO3 -- flat_weights_to_json helper

**Files:**
- Modify: `src/rust/aerocapture-py/src/lib.rs`

- [ ] **Step 1: Add the helper function**

Near the existing `nn_forward` in `src/rust/aerocapture-py/src/lib.rs`:

```rust
/// Construct a NeuralNetModel from flat PSO weights + v2 architecture (JSON string)
/// and write it as v2 JSON. All PSO NN output flows through this helper.
///
/// Params:
///   flat: flat weight vector (length must equal sum of per-layer n_params)
///   architecture_json: JSON-serialized list of LayerSpec dicts
///   path: output JSON file path
///   output_interpretation: "atan2" or "direct"
///   input_mask: optional list of input indices (length == layer[0] input_size)
#[pyfunction]
fn flat_weights_to_json(
    flat: Vec<f64>,
    architecture_json: String,
    path: String,
    output_interpretation: String,
    input_mask: Option<Vec<usize>>,
) -> PyResult<()> {
    use aerocapture::data::neural::{LayerSpec, NeuralNetModel};

    let specs: Vec<LayerSpec> = serde_json::from_str(&architecture_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!(
            "flat_weights_to_json: architecture_json parse error: {}", e
        )))?;
    let model = NeuralNetModel::from_flat_weights_v2(
        &flat, &specs, &output_interpretation, input_mask,
    ).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    model.save_json(&path)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(())
}
```

- [ ] **Step 2: Register in the pymodule**

Find the `#[pymodule] fn aerocapture_rs(...)` block and add:

```rust
m.add_function(wrap_pyfunction!(flat_weights_to_json, m)?)?;
```

- [ ] **Step 3: Rebuild bindings**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run maturin develop --manifest-path src/rust/aerocapture-py/Cargo.toml --release 2>&1 | tail -3
```

Expected: clean build, wheel installed.

- [ ] **Step 4: Smoke-check from Python**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run python -c "
import json
import tempfile
import numpy as np
import aerocapture_rs

arch = [
    {'type': 'dense', 'input_size': 3, 'output_size': 2, 'activation': 'linear'},
]
# Dense 3->2: 2*3 + 2 = 8 params.
flat = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.01, 0.02], dtype=np.float64)
with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
    path = f.name
aerocapture_rs.flat_weights_to_json(
    flat=flat.tolist(),
    architecture_json=json.dumps(arch),
    path=path,
    output_interpretation='atan2',
    input_mask=None,
)
raw = json.load(open(path))
print('format_version:', raw['format_version'])
print('arch[0]:', raw['architecture'][0])
print('weights layer_0 keys:', sorted(raw['weights']['layer_0'].keys()))
# Forward check
out = aerocapture_rs.nn_forward(path, [1.0, 1.0, 1.0])
print('forward output:', out)
"
```

Expected: prints `format_version: 2`, shows the dense spec, weights contain `w` and `b` keys, forward returns a 2-element list of finite floats.

- [ ] **Step 5: Commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add src/rust/aerocapture-py/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(pyo3): flat_weights_to_json helper (PSO NN-write through Rust)

Accepts a flat weight vector + JSON-serialized architecture spec and
produces a v2 JSON file via from_flat_weights_v2 + save_json. Closes
Phase 0 review carry-over #2: the LayerWeights trait now has a real
production caller. Python side routes through this in Task 9 instead
of writing v1 JSON directly.

JSON-string passthrough avoids adding pythonize to aerocapture-py
dependencies; serde_json is already present.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Python -- GruSpec + GruLayer torch module + discriminated union restoration

**Files:**
- Modify: `src/python/aerocapture/training/rl/schemas.py`
- Create: `src/python/aerocapture/training/rl/layers/gru.py`
- Modify: `src/python/aerocapture/training/rl/layers/__init__.py`
- Test: `tests/test_gru_layer.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_gru_layer.py`:

```python
"""Unit tests for GruLayer torch module."""

from __future__ import annotations

import torch
from aerocapture.training.rl.layers.gru import GruLayer


def test_gru_layer_shapes() -> None:
    layer = GruLayer(input_size=5, hidden_size=8)
    batch = 2
    x = torch.zeros(batch, 5)
    h = layer.new_state(batch, device="cpu")
    assert h.shape == (batch, 8)
    out, new_h = layer(x, h)
    assert out.shape == (batch, 8)
    assert new_h.shape == (batch, 8)
    # GRU output equals new hidden state (in-place semantics).
    torch.testing.assert_close(out, new_h, rtol=0, atol=0)


def test_gru_layer_zero_init_known_output() -> None:
    # All weights zeroed + h_prev non-zero => r=z=0.5, n=tanh(0)=0,
    # h_new = (1 - 0.5) * 0 + 0.5 * h_prev = 0.5 * h_prev.
    layer = GruLayer(input_size=2, hidden_size=3)
    with torch.no_grad():
        layer.weight_ih.zero_()
        layer.weight_hh.zero_()
        layer.bias_ih.zero_()
        layer.bias_hh.zero_()
    x = torch.tensor([[0.5, -0.5]])
    h_prev = torch.tensor([[1.0, 2.0, -1.0]])
    out, new_h = layer(x, h_prev)
    expected = h_prev * 0.5
    torch.testing.assert_close(out, expected, rtol=1e-12, atol=1e-12)


def test_gru_layer_n_params_closed_form() -> None:
    # Input size I=4, hidden size H=8: 3H*I + 3H*H + 2*3H = 96 + 192 + 48 = 336.
    layer = GruLayer(input_size=4, hidden_size=8)
    total = sum(p.numel() for p in layer.parameters())
    assert total == 336
```

- [ ] **Step 2: Run to confirm failure**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run pytest tests/test_gru_layer.py -v 2>&1 | tail -5
```

Expected: ModuleNotFoundError on `aerocapture.training.rl.layers.gru`.

- [ ] **Step 3: Create GruLayer module**

Create `src/python/aerocapture/training/rl/layers/gru.py`:

```python
"""GRU cell matching nn.GRUCell + Rust GruLayer bit-for-bit.

Canonical flat weight order (LayerWeights trait + PSO chromosome):
    weight_ih row-major, weight_hh row-major, bias_ih, bias_hh.

Forward equations (PyTorch nn.GRUCell convention):
    r_t = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)
    z_t = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)
    n_t = tanh(W_in @ x + b_in + r_t * (W_hn @ h + b_hn))
    h_t = (1 - z_t) * n_t + z_t * h_{t-1}
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class GruLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size))
        stdv = hidden_size ** -0.5
        for p in self.parameters():
            nn.init.uniform_(p, -stdv, stdv)

    def forward(self, x: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        H = self.hidden_size
        # x: [batch, input_size], h: [batch, hidden_size]
        gates_x = x @ self.weight_ih.t() + self.bias_ih  # [batch, 3H]
        gates_h = h @ self.weight_hh.t() + self.bias_hh  # [batch, 3H]
        r = torch.sigmoid(gates_x[:, :H] + gates_h[:, :H])
        z = torch.sigmoid(gates_x[:, H:2 * H] + gates_h[:, H:2 * H])
        n = torch.tanh(gates_x[:, 2 * H:3 * H] + r * gates_h[:, 2 * H:3 * H])
        h_new = (1 - z) * n + z * h
        return h_new, h_new

    def new_state(self, batch_size: int, device: Any) -> Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def extra_repr(self) -> str:
        return f"input_size={self.input_size}, hidden_size={self.hidden_size}"
```

- [ ] **Step 4: Add GruSpec to Pydantic schemas**

In `src/python/aerocapture/training/rl/schemas.py`:

```python
class GruSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["gru"]
    input_size: int = Field(ge=1)
    hidden_size: int = Field(ge=1)
```

Restore the discriminated union:

```python
LayerSpec = Annotated[Union[DenseSpec, GruSpec], Discriminator("type")]
```

Remove the Phase 0 single-variant alias comment.

- [ ] **Step 5: Update build_layer dispatch**

In `src/python/aerocapture/training/rl/layers/__init__.py`:

```python
from aerocapture.training.rl.layers.dense import DenseLayer
from aerocapture.training.rl.layers.gru import GruLayer
from aerocapture.training.rl.schemas import DenseSpec, GruSpec
from torch import nn

__all__ = ["DenseLayer", "GruLayer", "build_layer"]


def build_layer(spec) -> nn.Module:
    """Dispatch a LayerSpec to its torch module constructor."""
    if isinstance(spec, DenseSpec):
        return DenseLayer(spec.input_size, spec.output_size, spec.activation)
    if isinstance(spec, GruSpec):
        return GruLayer(spec.input_size, spec.hidden_size)
    raise ValueError(f"Unknown layer spec: {spec!r}")
```

The return type widens from `DenseLayer` (Phase 0) to `nn.Module` so mypy accepts both.

- [ ] **Step 6: Run tests**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run pytest tests/test_gru_layer.py -v 2>&1 | tail -5
```

Expected: 3 tests pass.

- [ ] **Step 7: Lint + mypy**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run ruff check src/python/aerocapture/training/rl/layers/ src/python/aerocapture/training/rl/schemas.py tests/test_gru_layer.py
uv run mypy src/python/aerocapture/training/rl/layers/ src/python/aerocapture/training/rl/schemas.py
```

Expected: both clean.

- [ ] **Step 8: Commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add src/python/aerocapture/training/rl/layers/gru.py src/python/aerocapture/training/rl/layers/__init__.py src/python/aerocapture/training/rl/schemas.py tests/test_gru_layer.py
git commit -m "$(cat <<'EOF'
feat(nn): Python GruLayer + GruSpec + restored LayerSpec discriminated union

Manual gate computation mirrors the Rust cell. Forward returns (h_new, h_new)
so the V2Policy step-wise contract stays uniform across layer types.
build_layer dispatches via isinstance; return type widened from DenseLayer
to nn.Module now that GruSpec joins the union. Phase 0's single-variant
LayerSpec alias workaround is reverted now that the Annotated Union has
two real variants.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Python -- load_policy_from_json + export_v2_policy_to_json GRU branches + _gru_specs + evaluate.py routing

**Files:**
- Modify: `src/python/aerocapture/training/model_io.py`
- Modify: `src/python/aerocapture/training/rl/export.py`
- Modify: `src/python/aerocapture/training/encoding.py`
- Modify: `src/python/aerocapture/training/evaluate.py`
- Test: `tests/test_v2_export.py` (extended with mixed Dense+Gru case)

- [ ] **Step 1: Write failing test for mixed-arch export+load**

Add to `tests/test_v2_export.py`:

```python
def test_export_load_roundtrip_mixed_dense_gru(tmp_path: Path) -> None:
    architecture = [
        DenseSpec(type="dense", input_size=3, output_size=4, activation="tanh"),
        GruSpec(type="gru", input_size=4, hidden_size=4),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    p = V2Policy(architecture=architecture, output_interpretation="atan2", input_mask=None)
    with torch.no_grad():
        for pp in p.parameters():
            pp.data = torch.randn_like(pp.data) * 0.1

    path = tmp_path / "mixed.json"
    export_v2_policy_to_json(p, str(path), obs_normalizer=None)
    q = load_policy_from_json(str(path), device="cpu")

    # Every parameter round-trips bit-for-bit (export writes f64, load reads f64,
    # and copy_ casts to the destination dtype which matches the original).
    for (_, a), (_, b) in zip(p.state_dict().items(), q.state_dict().items(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
```

You will need `from aerocapture.training.rl.schemas import GruSpec` added at the top of the test file.

- [ ] **Step 2: Run to confirm failure**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run pytest tests/test_v2_export.py::test_export_load_roundtrip_mixed_dense_gru -v 2>&1 | tail -5
```

Expected: fails because exporter doesn't know how to write Gru weights.

- [ ] **Step 3: Extend `export_v2_policy_to_json` with Gru branch**

In `src/python/aerocapture/training/rl/export.py`, find the per-layer loop and replace with:

```python
from aerocapture.training.rl.layers.dense import DenseLayer
from aerocapture.training.rl.layers.gru import GruLayer

def export_v2_policy_to_json(policy, path, obs_normalizer=None):
    architecture = []
    weights = {}

    for i, layer in enumerate(policy.layers):
        if isinstance(layer, DenseLayer):
            lin = layer.linear
            w = lin.weight.detach().cpu().numpy()
            b = lin.bias.detach().cpu().numpy()
            if i == 0 and obs_normalizer is not None:
                mean = obs_normalizer._mean.detach().cpu().numpy()
                std = obs_normalizer.std.detach().cpu().numpy()
                w = w / std
                b = b - (w * (mean / std) * std).sum(axis=1)  # existing Phase 0 math
                # TODO(Phase2): if a window layer precedes dense-0, tile mean/std over window slots.
            architecture.append({
                "type": "dense",
                "input_size": lin.in_features,
                "output_size": lin.out_features,
                "activation": layer.activation_name,
            })
            weights[f"layer_{i}"] = {"w": w.tolist(), "b": b.tolist()}
        elif isinstance(layer, GruLayer):
            if i == 0 and obs_normalizer is not None:
                raise NotImplementedError(
                    "obs_normalizer bake-in not supported when layer 0 is Gru. "
                    "Add a Dense embedding as layer 0 (per Phase 0 spec section 3.5 invariant)."
                )
            architecture.append({
                "type": "gru",
                "input_size": layer.input_size,
                "hidden_size": layer.hidden_size,
            })
            weights[f"layer_{i}"] = {
                "weight_ih": layer.weight_ih.detach().cpu().numpy().tolist(),
                "weight_hh": layer.weight_hh.detach().cpu().numpy().tolist(),
                "bias_ih": layer.bias_ih.detach().cpu().numpy().tolist(),
                "bias_hh": layer.bias_hh.detach().cpu().numpy().tolist(),
            }
        else:
            raise ValueError(f"Unknown layer type in export: {type(layer).__name__}")

    out = {
        "format_version": 2,
        "architecture": architecture,
        "weights": weights,
        "output_interpretation": policy.output_interpretation,
        "input_mask": policy.input_mask,
        "ablated_input": None,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
```

(Preserve the existing Phase 0 obs-normalizer math; don't rewrite it from scratch -- only the dispatch around it changes. If the existing `b - W @ (mean/std)` pattern in the committed Phase 0 code differs from what's shown here, keep the existing one.)

- [ ] **Step 4: Extend `load_policy_from_json` with Gru branch**

In `src/python/aerocapture/training/model_io.py`:

```python
from aerocapture.training.rl.layers import DenseLayer, GruLayer
from aerocapture.training.rl.schemas import DenseSpec, GruSpec

def load_policy_from_json(path: str, device) -> V2Policy:
    with open(path) as f:
        raw = json.load(f)
    if raw.get("format_version") != 2:
        raise ValueError(f"Expected format_version=2 in {path}, got {raw.get('format_version')}")
    arch = ArchitectureV2.model_validate(raw)
    policy = V2Policy(
        architecture=list(arch.architecture),
        output_interpretation=arch.output_interpretation,
        input_mask=arch.input_mask,
    ).to(device)

    for i, layer_spec in enumerate(arch.architecture):
        key = f"layer_{i}"
        lw = arch.weights[key]
        if isinstance(layer_spec, DenseSpec):
            if lw.w is None or lw.b is None:
                raise ValueError(f"Dense layer {key} missing w/b in {path}")
            w = torch.tensor(lw.w, dtype=torch.float64, device=device)
            b = torch.tensor(lw.b, dtype=torch.float64, device=device)
            layer = policy.layers[i]
            assert isinstance(layer, DenseLayer)
            with torch.no_grad():
                layer.linear.weight.copy_(w)
                layer.linear.bias.copy_(b)
        elif isinstance(layer_spec, GruSpec):
            required = {"weight_ih", "weight_hh", "bias_ih", "bias_hh"}
            missing = [k for k in required if getattr(lw, k, None) is None and k not in (lw.model_extra or {})]
            if missing:
                raise ValueError(f"Gru layer {key} missing {missing} in {path}")
            # Pydantic's extra="allow" on LayerWeights puts these in model_extra.
            extra = lw.model_extra or {}
            w_ih = torch.tensor(extra["weight_ih"], dtype=torch.float64, device=device)
            w_hh = torch.tensor(extra["weight_hh"], dtype=torch.float64, device=device)
            b_ih = torch.tensor(extra["bias_ih"], dtype=torch.float64, device=device)
            b_hh = torch.tensor(extra["bias_hh"], dtype=torch.float64, device=device)
            layer = policy.layers[i]
            assert isinstance(layer, GruLayer)
            with torch.no_grad():
                layer.weight_ih.copy_(w_ih)
                layer.weight_hh.copy_(w_hh)
                layer.bias_ih.copy_(b_ih)
                layer.bias_hh.copy_(b_hh)
        else:
            raise ValueError(f"Unknown layer spec type: {type(layer_spec).__name__}")

    return policy
```

- [ ] **Step 5: Add `_gru_specs` to encoding.py**

In `src/python/aerocapture/training/encoding.py`:

```python
from aerocapture.training.rl.schemas import DenseSpec, GruSpec


def _gru_specs(layer: GruSpec, layer_idx: int, bound_multiplier: float) -> list[ParamSpec]:
    import math
    from aerocapture.training.initialization import compute_layer_bound
    H = layer.hidden_size
    w_ih_bound = compute_layer_bound(layer.input_size, 3 * H, "tanh") * bound_multiplier
    w_hh_bound = compute_layer_bound(H, 3 * H, "tanh") * bound_multiplier
    b_bound = 0.1 * bound_multiplier
    specs: list[ParamSpec] = []
    for j in range(3 * H * layer.input_size):
        specs.append(ParamSpec(
            name=f"w_ih{layer_idx}_{j}",
            p_min=-w_ih_bound, p_max=+w_ih_bound,
            default=0.0, log_scale=False, is_integer=False,
        ))
    for j in range(3 * H * H):
        specs.append(ParamSpec(
            name=f"w_hh{layer_idx}_{j}",
            p_min=-w_hh_bound, p_max=+w_hh_bound,
            default=0.0, log_scale=False, is_integer=False,
        ))
    for j in range(3 * H):
        specs.append(ParamSpec(
            name=f"b_ih{layer_idx}_{j}",
            p_min=-b_bound, p_max=+b_bound,
            default=0.0, log_scale=False, is_integer=False,
        ))
    for j in range(3 * H):
        specs.append(ParamSpec(
            name=f"b_hh{layer_idx}_{j}",
            p_min=-b_bound, p_max=+b_bound,
            default=0.0, log_scale=False, is_integer=False,
        ))
    return specs
```

Update `_layer_param_specs` to dispatch on type and pass `layer_idx`:

```python
def _layer_param_specs(layer, layer_idx: int, bound_multiplier: float) -> list[ParamSpec]:
    if isinstance(layer, DenseSpec):
        return _dense_specs(layer, layer_idx, bound_multiplier)
    if isinstance(layer, GruSpec):
        return _gru_specs(layer, layer_idx, bound_multiplier)
    raise ValueError(f"Unknown layer type for PSO specs: {layer!r}")
```

Update `nn_param_specs_from_v2` to pass `layer_idx`:

```python
def nn_param_specs_from_v2(architecture, bound_multiplier: float = 1.0) -> list[ParamSpec]:
    specs: list[ParamSpec] = []
    for layer_idx, layer in enumerate(architecture):
        specs.extend(_layer_param_specs(layer, layer_idx, bound_multiplier))
    return specs
```

If `_dense_specs` already has a different signature from Phase 0 (perhaps taking `layer_idx` already, perhaps not), adapt minimally.

- [ ] **Step 6: Route `evaluate.py` through PyO3**

Find the current NN JSON-write function in `src/python/aerocapture/training/evaluate.py`:

```bash
rg "def write_nn_json|format_version|json\.dump" src/python/aerocapture/training/evaluate.py -n
```

Replace the body of the NN-writing function with a call to `aerocapture_rs.flat_weights_to_json`. Example:

```python
def write_nn_json(flat: np.ndarray, architecture: list, path: Path, input_mask: list[int] | None) -> None:
    """Write a PSO chromosome as a v2 NN JSON via the Rust LayerWeights trait.

    Replaces the previous Python-side v1 JSON writer. Closes Phase 0 review
    carry-over #2.
    """
    import json
    import aerocapture_rs
    arch_dicts = [spec.model_dump() if hasattr(spec, "model_dump") else dict(spec) for spec in architecture]
    aerocapture_rs.flat_weights_to_json(
        flat=flat.astype(np.float64).tolist(),
        architecture_json=json.dumps(arch_dicts),
        path=str(path),
        output_interpretation="atan2",
        input_mask=input_mask,
    )
```

If the existing function had a different signature, adapt minimally -- preserve the callers' API. Remove any now-dead Python-side JSON assembly code.

- [ ] **Step 7: Run tests**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run pytest tests/test_v2_export.py tests/test_nn_param_specs_v2.py -v 2>&1 | tail -10
```

Expected: all pass, including the new mixed-arch roundtrip test.

- [ ] **Step 8: Run full Python test suite**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run pytest 2>&1 | tail -10
```

Expected: 0 failures (aside from any pre-existing unrelated failures, e.g. `test_ppo_smoke_produces_artifacts` if the user's local config edits are still lying around).

- [ ] **Step 9: Lint + mypy**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
./lint_code.sh 2>&1 | tail -6
```

Expected: all clean.

- [ ] **Step 10: Commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add src/python/aerocapture/training/model_io.py src/python/aerocapture/training/rl/export.py src/python/aerocapture/training/encoding.py src/python/aerocapture/training/evaluate.py tests/test_v2_export.py
git commit -m "$(cat <<'EOF'
feat(nn): export/load/encoding Gru branches + evaluate.py routes via PyO3

Export and load gain isinstance dispatch on DenseSpec/GruSpec. The
export path raises NotImplementedError if obs_normalizer is provided
with a Gru at layer 0 (violates Phase 0 spec section 3.5 invariant).
_gru_specs produces Xavier-uniform bounds on the 3H-concatenated
gate matrices. evaluate.py writes PSO chromosomes through the new
aerocapture_rs.flat_weights_to_json helper instead of assembling
v1 JSON in Python -- one writer, one source of truth.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Cross-language equivalence -- GRU case + input_mask case

**Files:**
- Modify: `tests/test_v2_rust_python_equivalence.py`

- [ ] **Step 1: Write failing tests**

Extend `tests/test_v2_rust_python_equivalence.py`:

```python
def test_rust_python_gru_equivalence(tmp_path: Path) -> None:
    architecture = [
        DenseSpec(type="dense", input_size=5, output_size=8, activation="tanh"),
        GruSpec(type="gru", input_size=8, hidden_size=8),
        DenseSpec(type="dense", input_size=8, output_size=2, activation="linear"),
    ]
    policy = V2Policy(architecture=architecture, output_interpretation="atan2", input_mask=None)
    torch.manual_seed(42)
    with torch.no_grad():
        for p in policy.parameters():
            p.data = torch.randn_like(p.data) * 0.3
    policy.double()

    json_path = tmp_path / "gru_model.json"
    export_v2_policy_to_json(policy, str(json_path), obs_normalizer=None)

    rng = np.random.default_rng(7)
    inputs = rng.standard_normal((100, 5)).astype(np.float64)

    # PyTorch side: thread state across the 100-step sequence (GRU is stateful).
    py_out = np.zeros((100, 2), dtype=np.float64)
    state = policy.new_state(1, "cpu")
    for i, x in enumerate(inputs):
        y, state = policy(torch.from_numpy(x).unsqueeze(0), state)
        py_out[i] = y.detach().numpy()[0]

    # Rust side: nn_forward is stateless per-call (fresh NnState each call).
    # For the GRU path, a fair Rust<->Python equivalence test MUST use
    # independent single-step comparisons, not a threaded sequence -- otherwise
    # Python carries hidden state and Rust doesn't. Reset Python state per step too.
    py_single_out = np.zeros((100, 2), dtype=np.float64)
    for i, x in enumerate(inputs):
        single_state = policy.new_state(1, "cpu")
        y, _ = policy(torch.from_numpy(x).unsqueeze(0), single_state)
        py_single_out[i] = y.detach().numpy()[0]

    rust_out = np.array([
        aerocapture_rs.nn_forward(str(json_path), x.tolist())
        for x in inputs
    ])

    max_diff = np.max(np.abs(rust_out - py_single_out))
    assert max_diff < 1e-10, f"gru single-step max abs diff {max_diff}"


def test_rust_python_dense_equivalence_with_input_mask(tmp_path: Path) -> None:
    # Raw input is 5-wide; mask picks 3 indices for a 3-input first-layer Dense.
    architecture = [
        DenseSpec(type="dense", input_size=3, output_size=4, activation="tanh"),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    policy = V2Policy(architecture=architecture, output_interpretation="atan2", input_mask=[0, 2, 4])
    torch.manual_seed(7)
    with torch.no_grad():
        for p in policy.parameters():
            p.data = torch.randn_like(p.data) * 0.3
    policy.double()

    json_path = tmp_path / "masked.json"
    export_v2_policy_to_json(policy, str(json_path), obs_normalizer=None)

    rng = np.random.default_rng(11)
    raw_inputs = rng.standard_normal((50, 5)).astype(np.float64)
    masked_inputs = raw_inputs[:, [0, 2, 4]]

    py_out = np.zeros((50, 2), dtype=np.float64)
    state = policy.new_state(1, "cpu")
    for i, x in enumerate(masked_inputs):
        y, _ = policy(torch.from_numpy(x).unsqueeze(0), state)
        py_out[i] = y.detach().numpy()[0]

    # Rust nn_forward takes the RAW input and applies the mask internally.
    rust_out = np.array([
        aerocapture_rs.nn_forward(str(json_path), raw.tolist())
        for raw in raw_inputs
    ])

    max_diff = np.max(np.abs(rust_out - py_out))
    assert max_diff < 1e-10, f"dense+mask max abs diff {max_diff}"
```

You need `from aerocapture.training.rl.schemas import GruSpec` in the imports if not already present.

- [ ] **Step 2: Rebuild PyO3 (in case anything drifted)**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run maturin develop --manifest-path src/rust/aerocapture-py/Cargo.toml --release 2>&1 | tail -3
```

- [ ] **Step 3: Run the new tests**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run pytest tests/test_v2_rust_python_equivalence.py -v 2>&1 | tail -10
```

Expected: 3 tests pass (original dense + GRU + input_mask).

- [ ] **Step 4: Lint**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run ruff check tests/test_v2_rust_python_equivalence.py
uv run mypy src/python  # mypy covers src/python only per the project config
```

Expected: clean.

- [ ] **Step 5: Commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add tests/test_v2_rust_python_equivalence.py
git commit -m "$(cat <<'EOF'
test(nn): cross-language equivalence -- GRU + input_mask cases

Extends the Phase 0 integration gate with:
- Mixed Dense+Gru+Dense architecture comparison (single-step semantics
  since nn_forward is stateless per-call; threading state across the
  sequence would compare apples to oranges).
- Dense architecture with non-None input_mask = [0, 2, 4] selecting
  3 indices from a 5-wide raw input, validating that the Rust-side
  mask application in nn_forward matches the Python-side policy
  receiving the pre-masked input.

Closes Phase 0 review carry-overs #3 (mask coverage).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Training config + compare_guidance registration

**Files:**
- Create: `configs/training/msr_aller_gru_pso_train.toml`
- Modify: `src/python/aerocapture/training/compare_guidance.py`

- [ ] **Step 1: Write the training config**

Create `configs/training/msr_aller_gru_pso_train.toml`:

```toml
base = ["../missions/mars.toml", "common.toml"]

[guidance]
type = "neural_network"

[data]
neural_network = "training_output/neural_network_gru_pso/best_model.json"
results_suffix = ".train_gru_pso"

[network]
input_mask = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
output_interpretation = "atan2"

[[network.architecture]]
type = "dense"
input_size = 16
output_size = 32
activation = "tanh"

[[network.architecture]]
type = "gru"
input_size = 32
hidden_size = 32

[[network.architecture]]
type = "dense"
input_size = 32
output_size = 2
activation = "linear"

[optimizer]
algorithm = "pso"
n_pop = 64
n_gen = 1000
seed_strategy = "adaptive"
training_n_sims = 20
validation_n_sims = 1000
```

- [ ] **Step 2: Check current compare_guidance schemes**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
rg "neural_network" src/python/aerocapture/training/compare_guidance.py -n | head
```

Identify the existing `neural_network` registration.

- [ ] **Step 3: Register `neural_network_gru_pso`**

In `compare_guidance.py`, add a new entry alongside `neural_network`. The entry:
- scheme name: `neural_network_gru_pso`
- training config: `configs/training/msr_aller_gru_pso_train.toml`
- best-model path: `training_output/neural_network_gru_pso/best_model.json`
- Rust dispatch: `neural_network` (Rust doesn't see the scheme label; it only dispatches on layer types inside the loaded JSON).

Match the pattern used by the existing `neural_network_rl` registration.

- [ ] **Step 4: Smoke-test that the config loads**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run python -c "
from aerocapture.training.toml_utils import load_toml_with_bases
cfg = load_toml_with_bases('configs/training/msr_aller_gru_pso_train.toml')
print('architecture:', cfg['network']['architecture'])
print('optimizer:', cfg['optimizer'])
"
```

Expected: prints the architecture list (3 entries) and the optimizer block. No errors.

- [ ] **Step 5: Commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add configs/training/msr_aller_gru_pso_train.toml src/python/aerocapture/training/compare_guidance.py
git commit -m "$(cat <<'EOF'
feat(nn): msr_aller_gru_pso_train.toml + compare_guidance registration

Default architecture per spec: Dense(16->32,tanh) -> Gru(32->32) ->
Dense(32->2,linear). Input mask keeps the 16-input baseline for
apples-to-apples comparison against the existing neural_network PSO-MLP.
n_pop=64 and n_gen=1000 match the current NN training convention.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Training smoke test

**Files:**
- Create: `tests/test_gru_pso_smoke.py`

- [ ] **Step 1: Write the smoke test**

Create `tests/test_gru_pso_smoke.py`:

```python
"""5-gen PSO training on a minimal GRU config. Not a convergence test -- just
verifies the full stack (config parse, architecture construction, PSO eval,
Rust runtime, JSON write) runs end-to-end without error.

Runs in the python-pyo3 CI job (bindings required).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import tomli_w

aerocapture_rs = pytest.importorskip("aerocapture_rs")


@pytest.mark.slow
def test_gru_pso_smoke_5_gens(tmp_path: Path) -> None:
    from aerocapture.training.train import train

    # Minimal config: tiny pop, 5 gens, 2 sims per individual.
    base_config_path = Path("configs/training/msr_aller_gru_pso_train.toml")
    with open(base_config_path, "rb") as f:
        import tomllib
        base_cfg = tomllib.load(f)

    base_cfg["optimizer"]["n_pop"] = 8
    base_cfg["optimizer"]["n_gen"] = 5
    base_cfg["optimizer"]["training_n_sims"] = 2
    base_cfg["optimizer"]["validation_n_sims"] = 4

    # Reduce hidden size to shrink the search space.
    for entry in base_cfg["network"]["architecture"]:
        if entry["type"] == "dense" and entry.get("output_size") == 32:
            entry["output_size"] = 8
        elif entry["type"] == "gru":
            entry["input_size"] = 8
            entry["hidden_size"] = 8
        elif entry["type"] == "dense" and entry.get("input_size") == 32:
            entry["input_size"] = 8

    out_dir = tmp_path / "neural_network_gru_pso_smoke"
    base_cfg["data"]["neural_network"] = str(out_dir / "best_model.json")
    base_cfg["data"]["results_suffix"] = ".smoke"

    config_path = tmp_path / "smoke.toml"
    with open(config_path, "wb") as f:
        tomli_w.dump(base_cfg, f)

    # Run training
    result = train(
        config_path=str(config_path),
        n_gen=5,
        n_pop=8,
        output_dir=str(out_dir),
        no_tui=True,
        skip_report=True,
    )
    assert result is not None
    assert not result.get("interrupted", False)

    # Verify best_model.json exists and is v2 with a gru layer
    best_model = out_dir / "best_model.json"
    assert best_model.exists()
    raw = json.loads(best_model.read_text())
    assert raw["format_version"] == 2
    layer_types = [entry["type"] for entry in raw["architecture"]]
    assert "gru" in layer_types, f"expected gru in architecture, got {layer_types}"

    # Load and run a forward pass via Rust to confirm the output is runnable
    zeros_input = [0.0] * 16
    output = aerocapture_rs.nn_forward(str(best_model), zeros_input)
    assert len(output) == 2
    assert all(isinstance(v, float) for v in output)
```

- [ ] **Step 2: Check the `train` function signature**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
rg "^def train\(" src/python/aerocapture/training/train.py -n -A 15
```

Adapt the test's `train(...)` call to match the real signature. If the function returns a dict, great; if it returns a `TrainingResult` object, access attributes.

- [ ] **Step 3: Rebuild bindings, run the test**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run maturin develop --manifest-path src/rust/aerocapture-py/Cargo.toml --release 2>&1 | tail -3
uv run pytest tests/test_gru_pso_smoke.py -v 2>&1 | tail -10
```

Expected: 1 test passes. May take 30-60 seconds since it runs real sims.

- [ ] **Step 4: Register the smoke test in the PyO3 CI job**

Modify `.github/workflows/ci.yml`, find the `python-pyo3` job's pytest command and add the smoke test:

```yaml
- name: Run PyO3 tests
  run: uv run pytest tests/test_pyo3.py tests/test_v2_rust_python_equivalence.py tests/test_gru_pso_smoke.py -v
```

- [ ] **Step 5: Commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add tests/test_gru_pso_smoke.py .github/workflows/ci.yml
git commit -m "$(cat <<'EOF'
test(nn): GRU PSO training smoke test (5 gens, minimal arch)

Verifies the full PSO-GRU stack runs end-to-end: TOML parse,
architecture instantiation, PSO eval with Rust sim, JSON write via
aerocapture_rs.flat_weights_to_json, best_model.json is v2 and
contains a gru layer, nn_forward loads it and returns finite output.
Not a convergence test -- scientific gate (PSO-GRU vs PSO-MLP DV)
is a separate benchmark.

Added to the python-pyo3 CI job since it requires the PyO3 bindings.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Final verification + smart-commit

**Files:** none modified; verification only + smart-commit invocation.

- [ ] **Step 1: Full Rust stack**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
./check_all.sh 2>&1 | tail -10
```

Expected: Tests, Formatting, Clippy, Build all pass.

- [ ] **Step 2: Full Python stack**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
./lint_code.sh 2>&1 | tail -8
uv run pytest 2>&1 | tail -5
```

Expected: lint clean, all tests pass (any pre-existing failures unrelated to Phase 1 are noted and left alone).

- [ ] **Step 3: Golden regression bit-identity**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
cargo test -p aerocapture --manifest-path src/rust/Cargo.toml --test guidance_regression 2>&1 | tail -5
```

Expected: 6 golden files pass bit-identically.

- [ ] **Step 4: Audit Phase 1 extensibility invariants (success criterion #4 analogue)**

Manually inspect the diff `git diff main..HEAD` and confirm that adding a future Phase 2 layer type (e.g. LSTM) would touch only:
- `src/rust/src/data/neural.rs` (LayerSpec variant + Layer enum arm + new XxxLayer struct + LayerWeights impl + save_json arm + from_v2_json arm + from_flat_weights_v2 arm)
- `src/rust/src/data/nn_state.rs` (LayerState variant + for_layer arm + reset arm)
- `src/python/aerocapture/training/rl/layers/<type>.py` (new file)
- `src/python/aerocapture/training/rl/layers/__init__.py` (one dispatch line)
- `src/python/aerocapture/training/rl/schemas.py` (new Spec class + union entry)
- `src/python/aerocapture/training/encoding.py` (one dispatch branch)
- `src/python/aerocapture/training/rl/export.py` (one isinstance branch)
- `src/python/aerocapture/training/model_io.py` (one isinstance branch)
- `src/rust/src/config.rs` (one TomlLayerSpec variant + to_layer_spec arm)

No changes to `dispatch.rs`, `runner.rs`, `env.rs`, `problem.py`, or `train.py`.

- [ ] **Step 5: Invoke smart-commit skill**

Invoke `smart-commit` to finalize any outstanding documentation (CLAUDE.md entries for the GRU layer, README.md if user-facing features shifted) and create a final commit covering the branch. Tell it to take the whole `feature/gru-mvp` branch into account.

---

## Self-Review

**Spec coverage:**
- Section 2 (scope): Task 0 prepares branch, Tasks 1-12 cover every in-scope item. Non-goals (PPO, LSTM, Attention, SSM, Window, v1 loader widening, workspace clippy) are explicitly out of plan, matching the spec.
- Section 3 (architecture `Dense -> GRU -> Dense`, H=32): Task 11 config.
- Section 4 (GRU math, PyTorch nn.GRUCell convention): Task 2 Rust, Task 8 Python.
- Section 5 (JSON v2 schema for GRU): Tasks 5, 9.
- Section 6 (flat weight ordering): Task 4 Rust trait impl, Task 8 Python mirror, Task 10 cross-language.
- Section 7 (Rust implementation): Tasks 1, 2, 3, 4, 5, 6.
- Section 8 (Python implementation): Tasks 8, 9.
- Section 9 (training config): Task 11.
- Section 10 (scheme registration): Task 11.
- Section 11 (tests): covered across 2, 3, 4, 5, 6, 8, 9, 10, 12.
- Section 12 (success criteria): Task 13 audits engineering criteria 1-4; scientific gate (criterion 5) is informal.
- Section 13 (Phase 0 carry-overs): #2 closed by Task 7, #3 by Task 10, #4 by Task 3.

**Placeholder scan:** No TBD/TODO/implement-later in the plan steps. Two intentional TODO comments in the implementation code (Step 3 of Task 9: `TODO(Phase2)` for window-layer bake-in) are forward-looking markers, not plan placeholders.

**Type consistency check:**
- `GruLayer` Rust struct with fields `input_size, hidden_size, weight_ih, weight_hh, bias_ih, bias_hh` -- consistent across Tasks 2, 4, 5.
- `GruLayer` Python torch module with identical field names -- consistent across Tasks 8, 9, 10.
- `LayerSpec::Gru { input_size, hidden_size }` -- consistent across Tasks 2, 5, 6 (Rust), Task 8 (Python Pydantic GruSpec).
- `TomlLayerSpec::Gru { input_size, hidden_size }` matches the Rust `LayerSpec::Gru` variant.
- `aerocapture_rs.flat_weights_to_json(flat, architecture_json, path, output_interpretation, input_mask)` signature -- consistent between Task 7 Rust and Task 9 Python call site.
- `_gru_specs(layer, layer_idx, bound_multiplier)` signature matches the dispatcher pattern Task 9 Step 5 establishes.

No gaps found.
