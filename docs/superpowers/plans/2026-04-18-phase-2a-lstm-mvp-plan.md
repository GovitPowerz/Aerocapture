# Phase 2a LSTM MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land LSTM as the second stateful layer type on the Phase 0/1/1.5 stack, trained on both PSO and PPO-BPTT axes in one PR, with activation-aware initialization (including LSTM forget-bias-1 init per Jozefowicz et al 2015) folded in as the forcing function that closes the Phase 1 init carry-over.

**Architecture:** Rust `Layer` enum gains an `Lstm(LstmLayer)` variant parallel to `Gru(GruLayer)`; `LayerState::Lstm { h, c }` extends the state enum with the first multi-tensor variant; `LayerWeights for LstmLayer` uses 4H-concatenated gate order matching PyTorch `nn.LSTMCell` (i, f, g, o). Python `LstmLayer` torch module mirrors the Rust forward bit-for-bit. `_zero_state_where_done` gets a tuple dispatch branch (the real exercise of the Phase 0 extensibility contract). `init_v2_population` replaces the uniform-in-ParamSpec-bounds fallback with per-layer activation-aware init (dense Xavier/He/LeCun as before, GRU tanh-Xavier on 3H gate blocks, LSTM tanh-Xavier on 4H gate blocks plus forget-bias init to 1.0).

**Tech Stack:** Rust 2024 edition, PyO3 for Python bindings, Python 3.14, PyTorch (manual LSTM math matching `nn.LSTMCell`), Pydantic v2 discriminated unions, pymoo PSO, pytest.

**Spec:** `docs/superpowers/specs/2026-04-18-phase-2a-lstm-mvp-design.md`

---

## Task 0: Branch prep + TODO marker

**Files:**
- Already on `feature/lstm-mvp` with spec + spec correction committed.
- Modify: `TODO.md`

- [ ] **Step 1: Verify branch state**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
git branch --show-current
git log --oneline main..HEAD
git status
```

Expected: on `feature/lstm-mvp`, 2 commits ahead of main (spec + spec correction), clean working tree.

- [ ] **Step 2: Mark Phase 2a in progress in TODO.md**

Find the `### Phase 2 -- LSTM + Window-MLP` section in `TODO.md` and replace the heading + bullet with a subdivided structure:

```markdown
### Phase 2a -- LSTM MVP (PSO + PPO-BPTT) + activation-aware init [IN PROGRESS on feature/lstm-mvp]
- [ ] Rust `LstmLayer` + `Layer::Lstm` + `LayerState::Lstm { h, c }` + `TomlLayerSpec::Lstm`
- [ ] `LayerWeights for LstmLayer` 4H flat ordering + JSON v2 + PyO3 `flat_weights_to_json` branch
- [ ] Python `LstmLayer` torch module + `LstmSpec` pydantic + `_zero_state_where_done` tuple branch
- [ ] `_lstm_specs` + `config.py::_layer_n_params` lstm arm + export / load Lstm branches
- [ ] `init_v2_population`: dense Xavier/He/LeCun, GRU tanh-Xavier, LSTM tanh-Xavier + forget-bias 1.0
- [ ] Training configs `msr_aller_lstm_pso_train.toml` + `msr_aller_lstm_ppo_train.toml`
- [ ] Cross-language equivalence test + PSO-LSTM + PPO-LSTM smoke tests (@slow, python-pyo3 CI)

### Phase 2b -- Window-MLP (ring buffer, no new matmul)
- [ ] Deferred; separate spec + plan after Phase 2a lands

### Phase 3 -- Transformer
(unchanged)

### Phase 4 -- Mamba (S6)
(unchanged)
```

Commit:
```bash
cd /Users/govit/Git/Govit/Aerocapture
git add TODO.md
git commit -m "$(cat <<'EOF'
docs(todo): mark Phase 2a in progress on feature/lstm-mvp; split Phase 2 into 2a (LSTM) + 2b (Window-MLP)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 1: Rust -- add `LstmLayer` struct + `LayerSpec::Lstm` + `Layer::Lstm` variant + forward dispatch

**Files:**
- Modify: `src/rust/src/data/neural.rs`

Goal: add the LSTM variant alongside the existing Dense and Gru variants. Forward math only; flat-weight trait (Task 3) and JSON serialization (Task 4) come later. State-less dispatch requires a `LayerState::Lstm` stub, which lands in Task 2 — for now mark the `Layer::Lstm` arm of the forward dispatch as `unimplemented!` / add a placeholder branch that compiles against the existing `LayerState::Gru { h }`.

- [ ] **Step 1: Write a failing unit test for LSTM forward math with all-zero weights**

Add to the tests module at the bottom of `src/rust/src/data/neural.rs`:

```rust
#[test]
fn lstm_forward_known_output_zero_weights() {
    // Minimal 2-input, 2-hidden LSTM with all weights=0, all biases=0.
    // Then gates are all sigmoid(0)=0.5 (for i, f, o) and tanh(0)=0 (for g).
    // c_new = 0.5 * c_prev + 0.5 * 0 = 0.5 * c_prev
    // h_new = 0.5 * tanh(c_new)
    let lstm = LstmLayer {
        input_size: 2,
        hidden_size: 2,
        weight_ih: vec![vec![0.0, 0.0]; 8], // 4H=8 rows, 2 cols
        weight_hh: vec![vec![0.0, 0.0]; 8],
        bias_ih: vec![0.0; 8],
        bias_hh: vec![0.0; 8],
    };
    let h_prev = vec![0.0, 0.0];
    let c_prev = vec![2.0, -4.0];
    let x = vec![0.5, -0.5];
    let (h_new, c_new) = lstm.forward(&h_prev, &c_prev, &x);
    // c_new = f*c + i*g = 0.5*c_prev + 0.5*0 = 0.5*c_prev
    assert!((c_new[0] - 1.0).abs() < 1e-12);
    assert!((c_new[1] - (-2.0)).abs() < 1e-12);
    // h_new = o*tanh(c_new) = 0.5*tanh(c_new)
    assert!((h_new[0] - 0.5 * 1.0_f64.tanh()).abs() < 1e-12);
    assert!((h_new[1] - 0.5 * (-2.0_f64).tanh()).abs() < 1e-12);
}
```

- [ ] **Step 2: Run to confirm failure**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo test -p aerocapture --lib data::neural::tests::lstm_forward_known_output_zero_weights 2>&1 | tail -5
```

Expected: compilation error on missing `LstmLayer` type.

- [ ] **Step 3: Add `LstmLayer` struct + forward impl**

Insert in `src/rust/src/data/neural.rs` after the `GruLayer` block and before the `Layer` enum:

```rust
/// LSTM cell matching PyTorch nn.LSTMCell convention (two biases, no peepholes).
///
/// Forward equations with gate ordering (i, f, g, o):
///   i_t = sigmoid(W_ii @ x_t + b_ii + W_hi @ h_{t-1} + b_hi)
///   f_t = sigmoid(W_if @ x_t + b_if + W_hf @ h_{t-1} + b_hf)
///   g_t =    tanh(W_ig @ x_t + b_ig + W_hg @ h_{t-1} + b_hg)
///   o_t = sigmoid(W_io @ x_t + b_io + W_ho @ h_{t-1} + b_ho)
///   c_t = f_t * c_{t-1} + i_t * g_t
///   h_t = o_t * tanh(c_t)
///
/// Weight storage matches torch.nn.LSTMCell:
///   weight_ih: [4H, input_size] with rows 0..H = W_ii, H..2H = W_if, 2H..3H = W_ig, 3H..4H = W_io
///   weight_hh: [4H, H]          with rows 0..H = W_hi, H..2H = W_hf, 2H..3H = W_hg, 3H..4H = W_ho
///   bias_ih:   [4H] in order b_ii, b_if, b_ig, b_io
///   bias_hh:   [4H] in order b_hi, b_hf, b_hg, b_ho
#[derive(Debug, Clone)]
pub struct LstmLayer {
    pub input_size: usize,
    pub hidden_size: usize,
    pub weight_ih: Vec<Vec<f64>>,
    pub weight_hh: Vec<Vec<f64>>,
    pub bias_ih: Vec<f64>,
    pub bias_hh: Vec<f64>,
}

impl LstmLayer {
    /// Compute one forward step: (h_prev, c_prev, x) -> (h_new, c_new).
    pub fn forward(&self, h_prev: &[f64], c_prev: &[f64], x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        assert_eq!(h_prev.len(), self.hidden_size);
        assert_eq!(c_prev.len(), self.hidden_size);
        assert_eq!(x.len(), self.input_size);
        let h = self.hidden_size;
        let mut h_new = vec![0.0; h];
        let mut c_new = vec![0.0; h];

        for idx in 0..h {
            // i gate: row idx
            let i = Activation::Sigmoid.apply(
                dot_plus_bias(&self.weight_ih[idx], x, self.bias_ih[idx])
                    + dot_plus_bias(&self.weight_hh[idx], h_prev, self.bias_hh[idx]),
            );
            // f gate: row idx + H
            let f = Activation::Sigmoid.apply(
                dot_plus_bias(&self.weight_ih[idx + h], x, self.bias_ih[idx + h])
                    + dot_plus_bias(&self.weight_hh[idx + h], h_prev, self.bias_hh[idx + h]),
            );
            // g gate (tanh, the "cell candidate"): row idx + 2H
            let g = (dot_plus_bias(&self.weight_ih[idx + 2 * h], x, self.bias_ih[idx + 2 * h])
                + dot_plus_bias(&self.weight_hh[idx + 2 * h], h_prev, self.bias_hh[idx + 2 * h]))
                .tanh();
            // o gate: row idx + 3H
            let o = Activation::Sigmoid.apply(
                dot_plus_bias(&self.weight_ih[idx + 3 * h], x, self.bias_ih[idx + 3 * h])
                    + dot_plus_bias(&self.weight_hh[idx + 3 * h], h_prev, self.bias_hh[idx + 3 * h]),
            );

            c_new[idx] = f * c_prev[idx] + i * g;
            h_new[idx] = o * c_new[idx].tanh();
        }
        (h_new, c_new)
    }
}
```

- [ ] **Step 4: Extend `Layer` enum with `Lstm(LstmLayer)` variant**

Update the enum definition (replace the `// Phases 2-4 add:` comment):

```rust
#[derive(Debug, Clone)]
pub enum Layer {
    Dense(DenseLayer),
    Gru(GruLayer),
    Lstm(LstmLayer),
    // Phases 2b-4 add: Window, Attention, LayerNorm, Ssm
}
```

- [ ] **Step 5: Extend `Layer::input_size` dispatch**

```rust
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
        }
    }
}
```

- [ ] **Step 6: Extend `LayerSpec` enum with `Lstm` variant**

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
}
```

- [ ] **Step 7: Stub `LayerWeights for LstmLayer` (real impl in Task 3)**

```rust
impl LayerWeights for LstmLayer {
    fn to_flat(&self) -> Vec<f64> {
        unimplemented!("filled in Task 3")
    }
    #[allow(clippy::wrong_self_convention)]
    fn from_flat(&mut self, _flat: &[f64]) -> usize {
        unimplemented!("filled in Task 3")
    }
    fn n_params(&self) -> usize {
        4 * self.hidden_size * self.input_size
            + 4 * self.hidden_size * self.hidden_size
            + 2 * 4 * self.hidden_size
    }
}
```

- [ ] **Step 8: Extend `LayerWeights for Layer` dispatch**

```rust
impl LayerWeights for Layer {
    fn to_flat(&self) -> Vec<f64> {
        match self {
            Layer::Dense(d) => d.to_flat(),
            Layer::Gru(g) => g.to_flat(),
            Layer::Lstm(l) => l.to_flat(),
        }
    }
    fn from_flat(&mut self, flat: &[f64]) -> usize {
        match self {
            Layer::Dense(d) => d.from_flat(flat),
            Layer::Gru(g) => g.from_flat(flat),
            Layer::Lstm(l) => l.from_flat(flat),
        }
    }
    fn n_params(&self) -> usize {
        match self {
            Layer::Dense(d) => d.n_params(),
            Layer::Gru(g) => g.n_params(),
            Layer::Lstm(l) => l.n_params(),
        }
    }
}
```

- [ ] **Step 9: Run forward test to verify math**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo test -p aerocapture --lib data::neural::tests::lstm_forward_known_output_zero_weights 2>&1 | tail -5
```

Expected: PASS.

- [ ] **Step 10: Run full Rust suite + lint**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo test -p aerocapture 2>&1 | grep -E "^test result"
cargo clippy -p aerocapture --all-targets -- -D warnings 2>&1 | tail -5
cargo fmt -p aerocapture --check
```

Expected: 0 failures, clippy clean, fmt clean. Some existing tests that dispatch on `Layer` variants (e.g. the `Layer::input_size` callers) will now compile against the new arm; the forward dispatch in `NeuralNetModel::forward` is updated in Task 2 when `LayerState::Lstm` lands.

Note: the forward dispatch in `NeuralNetModel::forward` currently has a `(Layer::Gru(g), LayerState::Gru { h }) => { ... }` arm and a `_ => unreachable!()` or panic catch-all. After this task adds `Layer::Lstm`, the match will need an LSTM arm or a refined panic — that lands in Task 2 alongside `LayerState::Lstm`. If clippy complains about "non-exhaustive match" here, add a temporary `Layer::Lstm(_) => unimplemented!("Task 2: add Lstm state dispatch")` arm in `NeuralNetModel::forward` and keep it until Task 2 wires the real state.

- [ ] **Step 11: Commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add src/rust/src/data/neural.rs
git commit -m "$(cat <<'EOF'
feat(nn): add LstmLayer + Layer::Lstm + LayerSpec::Lstm variants

PyTorch nn.LSTMCell convention (two biases, no peepholes). Gate order
(i, f, g, o) concatenated on the 4H axis. Forward loop is the dumb
readable reference -- per-element, no fused matmul (negligible next
to sim step per Phase 1 profiling baseline). LayerWeights stub present
with n_params only; to_flat/from_flat filled in Task 3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Rust -- add `LayerState::Lstm { h, c }` + thread into `NeuralNetModel::forward`

**Files:**
- Modify: `src/rust/src/data/nn_state.rs`
- Modify: `src/rust/src/data/neural.rs` (forward dispatch arm)

Goal: land the first multi-tensor `LayerState` variant and wire it through the `NeuralNetModel::forward` dispatch. Named struct variant `LayerState::Lstm { h, c }` is chosen for grep-ability over positional tuple per the spec.

- [ ] **Step 1: Write failing behavioral test for Clone independence**

Add to the tests module at the bottom of `src/rust/src/data/nn_state.rs` (modeled after the existing GRU test):

```rust
#[test]
fn clone_is_behaviorally_independent_with_lstm_state() {
    use crate::data::neural::{Layer, LstmLayer};

    let lstm = LstmLayer {
        input_size: 2,
        hidden_size: 2,
        weight_ih: vec![vec![0.0, 0.0]; 8],
        weight_hh: vec![vec![0.0, 0.0]; 8],
        bias_ih: vec![0.0; 8],
        bias_hh: vec![0.0; 8],
    };
    let layer = Layer::Lstm(lstm);
    let original_state = LayerState::for_layer(&layer);
    let mut cloned_state = original_state.clone();

    // Mutate the clone
    if let LayerState::Lstm { h, c } = &mut cloned_state {
        h[0] = 1.0;
        h[1] = 2.0;
        c[0] = 3.0;
        c[1] = 4.0;
    } else {
        panic!("expected LayerState::Lstm");
    }

    // Original must remain zeroed
    if let LayerState::Lstm { h, c } = &original_state {
        assert_eq!(h, &vec![0.0, 0.0]);
        assert_eq!(c, &vec![0.0, 0.0]);
    } else {
        panic!("expected LayerState::Lstm");
    }
}
```

- [ ] **Step 2: Run to confirm failure**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo test -p aerocapture --lib data::nn_state::tests::clone_is_behaviorally_independent_with_lstm_state 2>&1 | tail -5
```

Expected: compilation error on missing `LayerState::Lstm` variant.

- [ ] **Step 3: Extend `LayerState` enum**

In `src/rust/src/data/nn_state.rs`, find the `LayerState` enum and add the `Lstm` variant:

```rust
#[derive(Debug, Clone)]
pub enum LayerState {
    None,
    Gru { h: Vec<f64> },
    Lstm { h: Vec<f64>, c: Vec<f64> },
}
```

- [ ] **Step 4: Extend `LayerState::for_layer`**

Update the constructor:

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
        }
    }
}
```

- [ ] **Step 5: Extend `LayerState::reset`**

```rust
impl LayerState {
    pub fn reset(&mut self) {
        match self {
            LayerState::None => {}
            LayerState::Gru { h } => {
                for v in h.iter_mut() {
                    *v = 0.0;
                }
            }
            LayerState::Lstm { h, c } => {
                for v in h.iter_mut() {
                    *v = 0.0;
                }
                for v in c.iter_mut() {
                    *v = 0.0;
                }
            }
        }
    }
}
```

- [ ] **Step 6: Extend `NeuralNetModel::forward` dispatch**

In `src/rust/src/data/neural.rs`, find the `(Layer::Gru(g), LayerState::Gru { h }) => { ... }` arm of the forward dispatch and add the LSTM arm:

```rust
(Layer::Lstm(l), LayerState::Lstm { h, c }) => {
    let (h_new, c_new) = l.forward(h, c, &x_in);
    *h = h_new.clone();
    *c = c_new;
    x_in = h_new;
}
```

Remove the `Layer::Lstm(_) => unimplemented!(...)` placeholder if one was added in Task 1 Step 10. The match arms should be:

- `(Layer::Dense(d), LayerState::None) => { ... }`
- `(Layer::Gru(g), LayerState::Gru { h }) => { ... }`
- `(Layer::Lstm(l), LayerState::Lstm { h, c }) => { ... }`
- `_ => panic!("layer / state type mismatch")` (existing)

- [ ] **Step 7: Run the clone test + full suite**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo test -p aerocapture --lib data::nn_state::tests::clone_is_behaviorally_independent_with_lstm_state 2>&1 | tail -5
cargo test -p aerocapture 2>&1 | grep -E "^test result"
```

Expected: PASS on the new test, 0 failures overall.

- [ ] **Step 8: Clippy + fmt**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo clippy -p aerocapture --all-targets -- -D warnings 2>&1 | tail -5
cargo fmt -p aerocapture --check
```

Expected: clean.

- [ ] **Step 9: Commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add src/rust/src/data/nn_state.rs src/rust/src/data/neural.rs
git commit -m "$(cat <<'EOF'
feat(nn): add LayerState::Lstm { h, c } + forward dispatch

First multi-tensor LayerState variant. Named struct variant (not
positional) for grep-ability. NeuralNetModel::forward dispatches to
LstmLayer::forward with both h and c slices; state is updated in
place with the new (h, c) pair.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Rust -- `LayerWeights for LstmLayer` + flat roundtrip test

**Files:**
- Modify: `src/rust/src/data/neural.rs`

Goal: fill in the stubbed `to_flat` / `from_flat` with the canonical 4H-gate order so PSO chromosomes can round-trip through the LSTM weights. Order matches the GRU pattern: `weight_ih` row-major -> `weight_hh` row-major -> `bias_ih` -> `bias_hh`.

- [ ] **Step 1: Write failing roundtrip test**

Add to the tests module in `src/rust/src/data/neural.rs`:

```rust
#[test]
fn lstm_flat_weights_roundtrip() {
    let original = LstmLayer {
        input_size: 3,
        hidden_size: 2,
        // 4H=8 rows, 3 cols -> 24 values
        weight_ih: (0..8)
            .map(|i| (0..3).map(|j| (i * 3 + j) as f64 * 0.01).collect())
            .collect(),
        // 4H=8 rows, 2 cols -> 16 values
        weight_hh: (0..8)
            .map(|i| (0..2).map(|j| 100.0 + (i * 2 + j) as f64 * 0.01).collect())
            .collect(),
        bias_ih: (0..8).map(|i| 200.0 + i as f64).collect(),
        bias_hh: (0..8).map(|i| 300.0 + i as f64).collect(),
    };

    let flat = original.to_flat();
    // 4H*I + 4H*H + 2*4H = 8*3 + 8*2 + 16 = 24 + 16 + 16 = 56
    assert_eq!(flat.len(), 56);
    assert_eq!(flat.len(), original.n_params());

    // Reconstruct into a zeroed LSTM of the same shape
    let mut reconstructed = LstmLayer {
        input_size: 3,
        hidden_size: 2,
        weight_ih: vec![vec![0.0; 3]; 8],
        weight_hh: vec![vec![0.0; 2]; 8],
        bias_ih: vec![0.0; 8],
        bias_hh: vec![0.0; 8],
    };
    let consumed = reconstructed.from_flat(&flat);
    assert_eq!(consumed, 56);

    // Deep equality
    assert_eq!(reconstructed.weight_ih, original.weight_ih);
    assert_eq!(reconstructed.weight_hh, original.weight_hh);
    assert_eq!(reconstructed.bias_ih, original.bias_ih);
    assert_eq!(reconstructed.bias_hh, original.bias_hh);
}
```

- [ ] **Step 2: Run to confirm failure**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo test -p aerocapture --lib data::neural::tests::lstm_flat_weights_roundtrip 2>&1 | tail -8
```

Expected: panics at `unimplemented!("filled in Task 3")`.

- [ ] **Step 3: Implement `to_flat` / `from_flat`**

Replace the stub from Task 1:

```rust
impl LayerWeights for LstmLayer {
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

    #[allow(clippy::wrong_self_convention)]
    fn from_flat(&mut self, flat: &[f64]) -> usize {
        let four_h = 4 * self.hidden_size;
        let mut idx = 0;
        for row in self.weight_ih.iter_mut() {
            row.copy_from_slice(&flat[idx..idx + self.input_size]);
            idx += self.input_size;
        }
        for row in self.weight_hh.iter_mut() {
            row.copy_from_slice(&flat[idx..idx + self.hidden_size]);
            idx += self.hidden_size;
        }
        self.bias_ih.copy_from_slice(&flat[idx..idx + four_h]);
        idx += four_h;
        self.bias_hh.copy_from_slice(&flat[idx..idx + four_h]);
        idx += four_h;
        idx
    }

    fn n_params(&self) -> usize {
        4 * self.hidden_size * self.input_size
            + 4 * self.hidden_size * self.hidden_size
            + 2 * 4 * self.hidden_size
    }
}
```

- [ ] **Step 4: Run test + full suite + lint**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo test -p aerocapture --lib data::neural::tests::lstm_flat_weights_roundtrip 2>&1 | tail -5
cargo test -p aerocapture 2>&1 | grep -E "^test result"
cargo clippy -p aerocapture --all-targets -- -D warnings 2>&1 | tail -5
cargo fmt -p aerocapture --check
```

Expected: PASS on the new test, 0 failures overall, clean clippy/fmt.

- [ ] **Step 5: Commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add src/rust/src/data/neural.rs
git commit -m "$(cat <<'EOF'
feat(nn): LayerWeights for LstmLayer with 4H flat ordering

Canonical order: weight_ih row-major -> weight_hh row-major -> bias_ih
-> bias_hh. Matches the GRU pattern scaled to 4H (i, f, g, o gates).
PSO chromosome round-trip verified.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Rust -- JSON v2 read + write for LSTM; `from_flat_weights_v2` LSTM arm

**Files:**
- Modify: `src/rust/src/data/neural.rs`

Goal: extend the three v2 JSON paths (`save_json`, `from_v2_json`, `from_flat_weights_v2`) with LSTM arms so v2 files round-trip and PSO chromosomes instantiate LSTM layers.

- [ ] **Step 1: Locate the v2 JSON code paths**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
grep -n "fn save_json\|fn from_v2_json\|fn from_flat_weights_v2\|Layer::Gru" src/rust/src/data/neural.rs | head -30
```

Note each site -- there will be a Gru arm in each of `save_json`, `from_v2_json`, `from_flat_weights_v2`. LSTM arms go next to each.

- [ ] **Step 2: Write a failing JSON v2 roundtrip test for LSTM**

Add to the tests module in `src/rust/src/data/neural.rs`:

```rust
#[test]
fn lstm_json_v2_roundtrip() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    // Build a small Dense -> LSTM -> Dense model
    let dense_in = DenseLayer {
        w: vec![vec![0.1, 0.2, 0.3]; 4], // 4 outputs from 3 inputs
        b: vec![0.01, 0.02, 0.03, 0.04],
        activation: Activation::Tanh,
    };
    let lstm = LstmLayer {
        input_size: 4,
        hidden_size: 3,
        weight_ih: (0..12)
            .map(|i| (0..4).map(|j| (i * 4 + j) as f64 * 0.001).collect())
            .collect(),
        weight_hh: (0..12)
            .map(|i| (0..3).map(|j| 1.0 + (i * 3 + j) as f64 * 0.001).collect())
            .collect(),
        bias_ih: (0..12).map(|i| 2.0 + i as f64 * 0.01).collect(),
        bias_hh: (0..12).map(|i| 3.0 + i as f64 * 0.01).collect(),
    };
    let dense_out = DenseLayer {
        w: vec![vec![0.5, -0.5, 0.25]; 2],
        b: vec![0.0, 0.1],
        activation: Activation::Linear,
    };
    let arch = vec![
        LayerSpec::Dense {
            input_size: 3,
            output_size: 4,
            activation: Activation::Tanh,
        },
        LayerSpec::Lstm {
            input_size: 4,
            hidden_size: 3,
        },
        LayerSpec::Dense {
            input_size: 3,
            output_size: 2,
            activation: Activation::Linear,
        },
    ];
    let original = NeuralNetModel {
        architecture: arch,
        layers: vec![Layer::Dense(dense_in), Layer::Lstm(lstm), Layer::Dense(dense_out)],
        output_interpretation: OutputInterpretation::Atan2,
        input_mask: None,
        ablated_input: None,
    };

    // Save to temp JSON
    let mut tmp = NamedTempFile::new().unwrap();
    let json = original.save_json().unwrap();
    tmp.write_all(json.as_bytes()).unwrap();

    // Load back
    let reloaded = NeuralNetModel::from_json_str(&json).unwrap();

    // Run both on the same input sequence and compare
    let input = vec![0.1, -0.2, 0.3];
    let mut s1 = NnState::for_model(&original);
    let mut s2 = NnState::for_model(&reloaded);
    for _ in 0..5 {
        let y1 = original.forward(&mut s1, &input);
        let y2 = reloaded.forward(&mut s2, &input);
        for (a, b) in y1.iter().zip(&y2) {
            assert!((a - b).abs() < 1e-14, "{} vs {}", a, b);
        }
    }
}
```

- [ ] **Step 3: Run to confirm failure**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo test -p aerocapture --lib data::neural::tests::lstm_json_v2_roundtrip 2>&1 | tail -10
```

Expected: serde error on unknown variant `lstm` (save succeeds, load fails), or save fails because the layer-to-weights serializer doesn't know about LSTM.

- [ ] **Step 4: Add LSTM arm to `save_json`'s weights serializer**

Open `src/rust/src/data/neural.rs`, find the block that builds the per-layer weights dict inside `save_json` (next to the `Layer::Gru(g) => { ... }` arm that writes `weight_ih`, `weight_hh`, `bias_ih`, `bias_hh`). Add after the Gru arm:

```rust
Layer::Lstm(l) => {
    let mut m = serde_json::Map::new();
    m.insert("weight_ih".to_string(), serde_json::to_value(&l.weight_ih)?);
    m.insert("weight_hh".to_string(), serde_json::to_value(&l.weight_hh)?);
    m.insert("bias_ih".to_string(), serde_json::to_value(&l.bias_ih)?);
    m.insert("bias_hh".to_string(), serde_json::to_value(&l.bias_hh)?);
    weights_map.insert(format!("layer_{}", idx), serde_json::Value::Object(m));
}
```

Match the exact style / field naming of the existing Gru arm in the file (the dict key and value variable names in the GRU path should be copied verbatim for the LSTM path).

- [ ] **Step 5: Add LSTM arm to `from_v2_json`'s layer-construction match**

Find the block that builds each `Layer` from a `LayerSpec + weights_map` entry (next to the `LayerSpec::Gru { input_size, hidden_size }` arm). Add after the Gru arm:

```rust
LayerSpec::Lstm { input_size, hidden_size } => {
    let weight_ih_raw: Vec<Vec<f64>> = serde_json::from_value(
        weights_map[&format!("layer_{}", idx)]["weight_ih"].clone(),
    )?;
    let weight_hh_raw: Vec<Vec<f64>> = serde_json::from_value(
        weights_map[&format!("layer_{}", idx)]["weight_hh"].clone(),
    )?;
    let bias_ih: Vec<f64> = serde_json::from_value(
        weights_map[&format!("layer_{}", idx)]["bias_ih"].clone(),
    )?;
    let bias_hh: Vec<f64> = serde_json::from_value(
        weights_map[&format!("layer_{}", idx)]["bias_hh"].clone(),
    )?;

    // Shape validation
    let four_h = 4 * hidden_size;
    if weight_ih_raw.len() != four_h || weight_ih_raw[0].len() != *input_size {
        return Err(DataError(format!(
            "LSTM layer {} weight_ih shape mismatch: expected [{}, {}], got [{}, {}]",
            idx,
            four_h,
            input_size,
            weight_ih_raw.len(),
            weight_ih_raw.first().map(|r| r.len()).unwrap_or(0),
        )));
    }
    if weight_hh_raw.len() != four_h || weight_hh_raw[0].len() != *hidden_size {
        return Err(DataError(format!(
            "LSTM layer {} weight_hh shape mismatch: expected [{}, {}], got [{}, {}]",
            idx,
            four_h,
            hidden_size,
            weight_hh_raw.len(),
            weight_hh_raw.first().map(|r| r.len()).unwrap_or(0),
        )));
    }
    if bias_ih.len() != four_h || bias_hh.len() != four_h {
        return Err(DataError(format!(
            "LSTM layer {} bias shape mismatch: expected {}, got bias_ih={}, bias_hh={}",
            idx,
            four_h,
            bias_ih.len(),
            bias_hh.len(),
        )));
    }

    layers.push(Layer::Lstm(LstmLayer {
        input_size: *input_size,
        hidden_size: *hidden_size,
        weight_ih: weight_ih_raw,
        weight_hh: weight_hh_raw,
        bias_ih,
        bias_hh,
    }));
}
```

Match the existing GRU arm's error-dialect style. If the GRU arm uses a different error type (e.g. returns `Result<_, serde_json::Error>` or uses `?` propagation), mirror that.

- [ ] **Step 6: Add LSTM arm to `from_flat_weights_v2`**

Find the function that constructs a `NeuralNetModel` from an architecture spec + a flat weight vector (used by the PyO3 `flat_weights_to_json` path in aerocapture-py). Next to the GRU arm, add:

```rust
LayerSpec::Lstm { input_size, hidden_size } => {
    let four_h = 4 * hidden_size;
    let mut lstm = LstmLayer {
        input_size: *input_size,
        hidden_size: *hidden_size,
        weight_ih: vec![vec![0.0; *input_size]; four_h],
        weight_hh: vec![vec![0.0; *hidden_size]; four_h],
        bias_ih: vec![0.0; four_h],
        bias_hh: vec![0.0; four_h],
    };
    let consumed = lstm.from_flat(&flat[offset..]);
    offset += consumed;
    layers.push(Layer::Lstm(lstm));
}
```

- [ ] **Step 7: Run the roundtrip test + full suite**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo test -p aerocapture --lib data::neural::tests::lstm_json_v2_roundtrip 2>&1 | tail -5
cargo test -p aerocapture 2>&1 | grep -E "^test result"
cargo clippy -p aerocapture --all-targets -- -D warnings 2>&1 | tail -5
cargo fmt -p aerocapture --check
```

Expected: PASS on the new test, 0 failures overall, clean clippy/fmt.

- [ ] **Step 8: Commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add src/rust/src/data/neural.rs
git commit -m "$(cat <<'EOF'
feat(nn): JSON v2 read+write for LSTM + from_flat_weights_v2

save_json writes weight_ih/weight_hh/bias_ih/bias_hh per layer entry.
from_v2_json validates 4H shape and surfaces clear DataError messages
on mismatch. from_flat_weights_v2 consumes n_params() flat values and
instantiates the LstmLayer inline (mirrors the Gru path).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Rust -- TOML `[[network.architecture]] type = "lstm"` parser

**Files:**
- Modify: `src/rust/src/config.rs`

Goal: extend the TOML layer spec parser so that `type = "lstm"` with `hidden_size = H` produces a `LayerSpec::Lstm { input_size, hidden_size }` (input_size inferred from the previous layer's output).

- [ ] **Step 1: Locate the TOML layer parser**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
grep -n "TomlLayerSpec\|fn to_layer_spec" src/rust/src/config.rs | head
```

Expected: `TomlLayerSpec` enum with `Dense` + `Gru` variants, and an impl block with `to_layer_spec(&self, prev_output: usize) -> LayerSpec` that maps each variant.

- [ ] **Step 2: Write failing test**

Add to `src/rust/src/config.rs` tests module (or create one at the bottom of the file if none exists):

```rust
#[test]
fn toml_layer_spec_parses_lstm() {
    let toml_str = r#"
[[network.architecture]]
type = "dense"
output_size = 4
activation = "tanh"

[[network.architecture]]
type = "lstm"
hidden_size = 8

[[network.architecture]]
type = "dense"
output_size = 2
activation = "linear"
"#;

    #[derive(serde::Deserialize)]
    struct NetworkSection {
        architecture: Vec<TomlLayerSpec>,
    }
    #[derive(serde::Deserialize)]
    struct Root {
        network: NetworkSection,
    }

    let parsed: Root = toml::from_str(toml_str).unwrap();
    assert_eq!(parsed.network.architecture.len(), 3);

    // Build LayerSpec list with input=3 feeding in
    let mut specs = Vec::new();
    let mut prev = 3usize;
    for toml_spec in &parsed.network.architecture {
        let spec = toml_spec.to_layer_spec(prev);
        prev = match &spec {
            LayerSpec::Dense { output_size, .. } => *output_size,
            LayerSpec::Gru { hidden_size, .. } => *hidden_size,
            LayerSpec::Lstm { hidden_size, .. } => *hidden_size,
        };
        specs.push(spec);
    }

    assert!(matches!(
        specs[1],
        LayerSpec::Lstm {
            input_size: 4,
            hidden_size: 8
        }
    ));
}
```

- [ ] **Step 3: Run to confirm failure**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo test -p aerocapture --lib config::tests::toml_layer_spec_parses_lstm 2>&1 | tail -8
```

Expected: serde error `unknown variant "lstm"` during `toml::from_str`.

- [ ] **Step 4: Extend `TomlLayerSpec` enum**

Find the enum (should have `Dense` and `Gru` variants) and add:

```rust
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TomlLayerSpec {
    Dense {
        output_size: usize,
        activation: String,
    },
    Gru {
        hidden_size: usize,
    },
    Lstm {
        hidden_size: usize,
    },
}
```

- [ ] **Step 5: Extend `to_layer_spec` match**

```rust
impl TomlLayerSpec {
    pub fn to_layer_spec(&self, prev_output: usize) -> LayerSpec {
        match self {
            TomlLayerSpec::Dense { output_size, activation } => LayerSpec::Dense {
                input_size: prev_output,
                output_size: *output_size,
                activation: parse_activation(activation).unwrap(),
            },
            TomlLayerSpec::Gru { hidden_size } => LayerSpec::Gru {
                input_size: prev_output,
                hidden_size: *hidden_size,
            },
            TomlLayerSpec::Lstm { hidden_size } => LayerSpec::Lstm {
                input_size: prev_output,
                hidden_size: *hidden_size,
            },
        }
    }
}
```

Match the exact style of the existing Gru arm (error handling on `parse_activation`, etc.).

- [ ] **Step 6: Run the TOML test + full suite**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo test -p aerocapture --lib config::tests::toml_layer_spec_parses_lstm 2>&1 | tail -5
cargo test -p aerocapture 2>&1 | grep -E "^test result"
cargo clippy -p aerocapture --all-targets -- -D warnings 2>&1 | tail -5
cargo fmt -p aerocapture --check
```

Expected: PASS, 0 failures, clean.

- [ ] **Step 7: Commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add src/rust/src/config.rs
git commit -m "$(cat <<'EOF'
feat(config): TOML [[network.architecture]] parses type = \"lstm\"

TomlLayerSpec::Lstm { hidden_size }; to_layer_spec threads the
previous layer's output as input_size. Mirrors the Gru arm.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: PyO3 -- `flat_weights_to_json` LSTM branch

**Files:**
- Modify: `src/rust/aerocapture-py/src/lib.rs`

Goal: the Python training stack writes LSTM weights to JSON via `aerocapture_rs.flat_weights_to_json(architecture, flat_weights, ...)` (Rust is the single source of truth for NN weight serialization since Phase 1). Extend this helper to accept LSTM layers in the architecture list.

- [ ] **Step 1: Locate the current helper**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
grep -n "flat_weights_to_json\|Layer::Gru" src/rust/aerocapture-py/src/lib.rs | head -20
```

Expected: a `#[pyfunction]` fn `flat_weights_to_json` that iterates the architecture list, consumes `n_params()` flat values per layer, and writes JSON v2. Currently has Dense + Gru arms.

- [ ] **Step 2: Write failing equivalence test (Python side)**

Before touching Rust, write the Python-side failing test. Create `tests/test_flat_weights_to_json_lstm.py`:

```python
"""Rust flat_weights_to_json handles LSTM architecture.

This locks in the contract that aerocapture_rs writes valid v2 JSON for
a Dense -> LSTM -> Dense architecture, so the Python PSO training path
can delegate JSON serialization to Rust for LSTM policies.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

aerocapture_rs = pytest.importorskip("aerocapture_rs")


def test_flat_weights_to_json_lstm_roundtrip():
    architecture = [
        {"type": "dense", "input_size": 3, "output_size": 4, "activation": "tanh"},
        {"type": "lstm", "input_size": 4, "hidden_size": 2},
        {"type": "dense", "input_size": 2, "output_size": 2, "activation": "linear"},
    ]
    # Param count: 3*4+4 = 16, 4*2*4 + 4*2*2 + 8*2 = 32 + 16 + 16 = 64, 2*2+2 = 6
    # Total = 86
    n_params = 16 + 64 + 6
    flat = np.arange(n_params, dtype=np.float64) * 0.001

    json_str = aerocapture_rs.flat_weights_to_json(
        architecture=architecture,
        flat_weights=flat,
        output_interpretation="atan2",
        input_mask=None,
        ablated_input=None,
    )
    payload = json.loads(json_str)

    assert payload["format_version"] == 2
    assert len(payload["architecture"]) == 3
    assert payload["architecture"][1]["type"] == "lstm"
    assert payload["architecture"][1]["hidden_size"] == 2

    # LSTM weights are nested under "layer_1"
    lstm_weights = payload["weights"]["layer_1"]
    assert "weight_ih" in lstm_weights
    assert "weight_hh" in lstm_weights
    assert "bias_ih" in lstm_weights
    assert "bias_hh" in lstm_weights

    # Shape check
    assert len(lstm_weights["weight_ih"]) == 8  # 4H
    assert len(lstm_weights["weight_ih"][0]) == 4  # input_size
    assert len(lstm_weights["weight_hh"]) == 8  # 4H
    assert len(lstm_weights["weight_hh"][0]) == 2  # H
    assert len(lstm_weights["bias_ih"]) == 8  # 4H
    assert len(lstm_weights["bias_hh"]) == 8  # 4H
```

- [ ] **Step 3: Run to confirm failure**

Build current PyO3 and run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
./build.sh
uv run pytest tests/test_flat_weights_to_json_lstm.py -v 2>&1 | tail -20
```

Expected: failure inside Rust (unknown layer type `"lstm"` when matching architecture entries).

- [ ] **Step 4: Find the dispatch site in aerocapture-py**

Run:
```bash
cd /Users/govit/Git/Govit/Aerocapture
grep -n "\"dense\"\|\"gru\"\|\"lstm\"\|parse_layer_type\|build_layer" src/rust/aerocapture-py/src/lib.rs
```

Expected: a match on `entry["type"].as_str()` handling `"dense"` and `"gru"`; possibly a helper function that builds a `LayerSpec` from a Python dict.

- [ ] **Step 5: Add the LSTM branch**

Next to the GRU branch (both at the LayerSpec-construction site and at the weights-serialization site), add:

At the architecture -> LayerSpec conversion site:
```rust
"lstm" => {
    let input_size: usize = entry.get_item("input_size")?.extract()?;
    let hidden_size: usize = entry.get_item("hidden_size")?.extract()?;
    LayerSpec::Lstm { input_size, hidden_size }
}
```

At the per-layer weights-to-JSON site (inside the `save_json`-equivalent code path that emits the weights dict), add whatever branch mirrors the existing Gru arm. If the helper delegates to `NeuralNetModel::save_json`, Task 4's work is already sufficient; just add the architecture-dispatch arm above.

- [ ] **Step 6: Rebuild PyO3 + rerun test**

```bash
cd /Users/govit/Git/Govit/Aerocapture
./build.sh
uv run pytest tests/test_flat_weights_to_json_lstm.py -v 2>&1 | tail -15
```

Expected: PASS.

- [ ] **Step 7: Verify no existing tests regressed**

Run the full PyO3 + pytest suite:
```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run pytest tests/test_pyo3.py tests/test_v2_rust_python_equivalence.py -q 2>&1 | tail -10
```

Expected: no regressions.

- [ ] **Step 8: Clippy + commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture/src/rust
cargo clippy -p aerocapture-py --all-targets -- -D warnings 2>&1 | tail -5
cd /Users/govit/Git/Govit/Aerocapture
git add src/rust/aerocapture-py/src/lib.rs tests/test_flat_weights_to_json_lstm.py
git commit -m "$(cat <<'EOF'
feat(pyo3): flat_weights_to_json LSTM branch

Python PSO path delegates LSTM weight serialization to Rust.
Architecture dispatch mirrors the existing Gru arm; JSON v2 output
matches the schema parsed by from_v2_json.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Python -- `LstmSpec` pydantic + `LstmLayer` torch module + `build_layer` dispatch

**Files:**
- Modify: `src/python/aerocapture/training/rl/schemas.py`
- Create: `src/python/aerocapture/training/rl/layers/lstm.py`
- Modify: `src/python/aerocapture/training/rl/layers/__init__.py`
- Create: `tests/test_lstm_layer_python.py`

Goal: Python LSTM module matching `nn.LSTMCell` bit-for-bit, plus schema + dispatch glue. The key correctness gate is "LstmLayer matches nn.LSTMCell to machine epsilon on f64."

- [ ] **Step 1: Write failing test `test_lstm_layer_matches_nn_lstmcell`**

Create `tests/test_lstm_layer_python.py`:

```python
"""LstmLayer torch module matches nn.LSTMCell bit-for-bit on f64."""

from __future__ import annotations

import torch
import torch.nn as nn

from aerocapture.training.rl.layers.lstm import LstmLayer


def test_lstm_layer_matches_nn_lstmcell():
    torch.manual_seed(0)
    I, H = 5, 4
    ours = LstmLayer(input_size=I, hidden_size=H).double()
    theirs = nn.LSTMCell(input_size=I, hidden_size=H).double()

    # Copy parameters: torch.nn.LSTMCell uses the same gate order (i, f, g, o).
    with torch.no_grad():
        ours.weight_ih.copy_(theirs.weight_ih)
        ours.weight_hh.copy_(theirs.weight_hh)
        ours.bias_ih.copy_(theirs.bias_ih)
        ours.bias_hh.copy_(theirs.bias_hh)

    B = 3
    x = torch.randn(B, I, dtype=torch.float64)
    h = torch.randn(B, H, dtype=torch.float64)
    c = torch.randn(B, H, dtype=torch.float64)

    h_ours, (h_ours_state, c_ours_state) = ours(x, (h, c))
    h_their, c_their = theirs(x, (h, c))

    assert torch.allclose(h_ours, h_their, atol=1e-12)
    assert torch.allclose(h_ours_state, h_their, atol=1e-12)
    assert torch.allclose(c_ours_state, c_their, atol=1e-12)


def test_lstm_layer_new_state_matches_dtype():
    I, H = 4, 6
    layer = LstmLayer(I, H).double()
    h, c = layer.new_state(batch_size=2, device="cpu")
    assert h.dtype == torch.float64
    assert c.dtype == torch.float64
    assert h.shape == (2, H)
    assert c.shape == (2, H)
    assert torch.all(h == 0.0)
    assert torch.all(c == 0.0)
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run pytest tests/test_lstm_layer_python.py -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError: No module named 'aerocapture.training.rl.layers.lstm'`.

- [ ] **Step 3: Create `LstmLayer` torch module**

Create `src/python/aerocapture/training/rl/layers/lstm.py`:

```python
"""LSTM layer matching torch.nn.LSTMCell and the Rust LstmLayer.

Gate order (i, f, g, o) is concatenated on the 4H axis. Two biases
(bias_ih, bias_hh) are kept separately for bit-for-bit nn.LSTMCell
parity; mathematically their sum is what enters each gate.

State contract: forward(x, state) -> (y, new_state) where
  state     = (h_prev, c_prev) with shapes (B, H) each
  new_state = (h_new, c_new)
  y         = h_new  (matches Rust convention that the layer's output
                      feeding the next layer is h)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LstmLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        H = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(4 * H, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * H, H))
        self.bias_ih = nn.Parameter(torch.empty(4 * H))
        self.bias_hh = nn.Parameter(torch.empty(4 * H))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Match nn.LSTMCell's default init (uniform in [-k, k] with k = 1/sqrt(H)).
        # init_v2_population overrides this for PSO; SAC/PPO warm-start overwrites
        # via load_state_dict or load_weights_from_json.
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for w in self.parameters():
            nn.init.uniform_(w, -stdv, stdv)

    def forward(
        self, x: Tensor, state: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        h_prev, c_prev = state
        ih = F.linear(x, self.weight_ih, self.bias_ih)      # (B, 4H)
        hh = F.linear(h_prev, self.weight_hh, self.bias_hh) # (B, 4H)
        H = self.hidden_size
        gates = ih + hh
        i = torch.sigmoid(gates[..., 0 * H : 1 * H])
        f = torch.sigmoid(gates[..., 1 * H : 2 * H])
        g = torch.tanh(   gates[..., 2 * H : 3 * H])
        o = torch.sigmoid(gates[..., 3 * H : 4 * H])
        c_new = f * c_prev + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, (h_new, c_new)

    def new_state(self, batch_size: int, device, dtype=None):
        dtype = dtype or self.weight_ih.dtype
        H = self.hidden_size
        zeros = torch.zeros(batch_size, H, device=device, dtype=dtype)
        return (zeros, zeros.clone())
```

- [ ] **Step 4: Extend `LayerSpec` discriminated union**

In `src/python/aerocapture/training/rl/schemas.py`, add the `LstmSpec` class after `GruSpec`:

```python
class LstmSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["lstm"]
    input_size: int = Field(ge=1)
    hidden_size: int = Field(ge=1)


LayerSpec = Annotated[DenseSpec | GruSpec | LstmSpec, Discriminator("type")]
```

- [ ] **Step 5: Extend `build_layer` dispatch**

Open `src/python/aerocapture/training/rl/layers/__init__.py`:

```bash
cd /Users/govit/Git/Govit/Aerocapture
cat src/python/aerocapture/training/rl/layers/__init__.py
```

Add the `LstmSpec -> LstmLayer` dispatch line. Edit:

```python
from aerocapture.training.rl.layers.lstm import LstmLayer
...

def build_layer(spec):
    if isinstance(spec, DenseSpec):
        return DenseLayer(spec.input_size, spec.output_size, spec.activation)
    if isinstance(spec, GruSpec):
        return GruLayer(spec.input_size, spec.hidden_size)
    if isinstance(spec, LstmSpec):
        return LstmLayer(spec.input_size, spec.hidden_size)
    raise ValueError(f"Unknown layer spec: {spec!r}")
```

Match the existing dispatch style in the file exactly; some projects route through a mapping rather than an if-chain. If so, add an `LstmSpec: lambda s: LstmLayer(s.input_size, s.hidden_size)` entry.

- [ ] **Step 6: Run the two LSTM layer tests**

```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run pytest tests/test_lstm_layer_python.py -v 2>&1 | tail -10
```

Expected: both pass.

- [ ] **Step 7: Run full Python test suite**

```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run pytest tests -q -x --ignore=tests/test_v2_rust_python_equivalence.py 2>&1 | tail -20
```

Expected: no failures. (The cross-language equivalence test is added in Task 10.)

- [ ] **Step 8: Commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add \
  src/python/aerocapture/training/rl/layers/lstm.py \
  src/python/aerocapture/training/rl/layers/__init__.py \
  src/python/aerocapture/training/rl/schemas.py \
  tests/test_lstm_layer_python.py
git commit -m "$(cat <<'EOF'
feat(nn): Python LstmLayer torch module + LstmSpec + build_layer dispatch

Manual nn.LSTMCell reproduction verified bit-for-bit on f64 (atol 1e-12).
LayerSpec discriminated union gains LstmSpec; build_layer dispatches.
forward returns (y, (h_new, c_new)) tuple state -- the first multi-tensor
state type in the V2Policy stack.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Python -- `_zero_state_where_done` tuple branch

**Files:**
- Modify: `src/python/aerocapture/training/rl/policy.py`
- Create: `tests/test_zero_state_where_done_tuple.py`

Goal: extend the state-zeroing helper to handle the LSTM `(h, c)` tuple. Currently raises `TypeError` on non-Tensor per the Phase 0 invariant.

- [ ] **Step 1: Write failing test**

Create `tests/test_zero_state_where_done_tuple.py`:

```python
"""_zero_state_where_done handles LSTM tuple state."""

from __future__ import annotations

import torch

from aerocapture.training.rl.policy import _zero_state_where_done


def test_zero_state_where_done_tuple_zeros_both_tensors_on_done_rows():
    B, H = 4, 3
    h = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    )
    c = torch.tensor(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]
    )
    done_mask = torch.tensor([False, True, False, True])

    state = [(h, c)]  # list of per-layer states; single LSTM layer here
    new_state = _zero_state_where_done(state, done_mask)
    assert len(new_state) == 1
    new_h, new_c = new_state[0]

    # Non-done rows (indices 0, 2) unchanged
    assert torch.equal(new_h[0], h[0])
    assert torch.equal(new_c[0], c[0])
    assert torch.equal(new_h[2], h[2])
    assert torch.equal(new_c[2], c[2])

    # Done rows (indices 1, 3) zeroed in both tensors
    assert torch.all(new_h[1] == 0.0)
    assert torch.all(new_c[1] == 0.0)
    assert torch.all(new_h[3] == 0.0)
    assert torch.all(new_c[3] == 0.0)


def test_zero_state_where_done_raises_on_non_tensor_non_tuple_entry():
    done_mask = torch.tensor([False])
    with pytest.raises(TypeError, match="unsupported state entry type"):
        _zero_state_where_done([object()], done_mask)


def test_zero_state_where_done_passes_through_none_and_tensor_entries_alongside_tuple():
    B, H = 2, 2
    tensor_state = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    h = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    c = torch.tensor([[0.5, 0.6], [0.7, 0.8]])
    done_mask = torch.tensor([True, False])

    state = [None, tensor_state, (h, c)]
    new_state = _zero_state_where_done(state, done_mask)

    assert new_state[0] is None
    assert torch.all(new_state[1][0] == 0.0)          # done
    assert torch.equal(new_state[1][1], tensor_state[1])  # not done
    new_h, new_c = new_state[2]
    assert torch.all(new_h[0] == 0.0)
    assert torch.all(new_c[0] == 0.0)
    assert torch.equal(new_h[1], h[1])
    assert torch.equal(new_c[1], c[1])


import pytest  # noqa: E402 -- kept at bottom so the module imports cleanly in test collection
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run pytest tests/test_zero_state_where_done_tuple.py -v 2>&1 | tail -10
```

Expected: first test fails with `TypeError: _zero_state_where_done: unsupported state entry type 'tuple'`.

- [ ] **Step 3: Extend `_zero_state_where_done`**

In `src/python/aerocapture/training/rl/policy.py`, replace the existing helper body:

```python
def _zero_state_where_done(state: list[Any], done_mask: Tensor) -> list[Any]:
    """Return a new state list where the entries for `done_mask` envs are zeroed.

    Contract:
      - `None` entries (dense / stateless layers) pass through unchanged.
      - Tensor entries of shape `(B, *)` get `done_mask` rows multiplied by 0.
      - Tuple entries (e.g. LSTM `(h, c)`) recurse into each element.
        Recursion terminates at the Tensor branch.
      - Any other non-None, non-Tensor, non-tuple entry raises TypeError.
        Future multi-tensor state types that aren't expressible as a tuple
        (e.g. Mamba's SSM state dict, Transformer KV cache) must add an
        explicit branch here rather than silently matmul-erroring downstream.
    """
    new_state: list[Any] = []
    keep_bool = (~done_mask).unsqueeze(-1)  # (B, 1), bool
    for s in state:
        new_state.append(_zero_entry(s, keep_bool, done_mask))
    return new_state


def _zero_entry(s: Any, keep_bool: Tensor, done_mask: Tensor) -> Any:
    if s is None:
        return None
    if isinstance(s, Tensor):
        return s * keep_bool.to(dtype=s.dtype, device=s.device)
    if isinstance(s, tuple):
        return tuple(_zero_entry(sub, keep_bool, done_mask) for sub in s)
    raise TypeError(
        f"_zero_state_where_done: unsupported state entry type {type(s).__name__!r}; "
        "only None, Tensor, or tuple supported. Non-tuple multi-tensor states "
        "(e.g. Mamba SSM, Transformer KV cache) need an explicit extension."
    )
```

- [ ] **Step 4: Run tests + full suite**

```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run pytest tests/test_zero_state_where_done_tuple.py -v 2>&1 | tail -10
uv run pytest tests -q -x --ignore=tests/test_v2_rust_python_equivalence.py 2>&1 | tail -15
```

Expected: three tuple-tests pass, no regressions elsewhere.

- [ ] **Step 5: Commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add src/python/aerocapture/training/rl/policy.py tests/test_zero_state_where_done_tuple.py
git commit -m "$(cat <<'EOF'
feat(rl): _zero_state_where_done tuple dispatch for LSTM (h, c)

Extends Phase 1.5 helper to recurse into tuple state entries. Keeps the
TypeError fall-through so future non-tuple multi-tensor states (Mamba
SSM, Transformer KV cache) must add explicit branches rather than
silently matmul-erroring.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Python -- `_lstm_specs` + `nn_param_specs_from_v2` dispatch + `_layer_n_params` lstm arm + export/load branches

**Files:**
- Modify: `src/python/aerocapture/training/encoding.py`
- Modify: `src/python/aerocapture/training/config.py`
- Modify: `src/python/aerocapture/training/rl/export.py`
- Modify: `src/python/aerocapture/training/model_io.py`
- Create: `tests/test_lstm_specs_and_n_params.py`

Goal: all Python integration glue for LSTM in one task -- PSO ParamSpec generator, v2 param-count computation, policy export/load.

- [ ] **Step 1: Write failing tests for specs + n_params**

Create `tests/test_lstm_specs_and_n_params.py`:

```python
"""_lstm_specs produces correct bounds; _layer_n_params handles LSTM."""

from __future__ import annotations

import math

from aerocapture.training.config import _layer_n_params
from aerocapture.training.encoding import _lstm_specs, nn_param_specs_from_v2
from aerocapture.training.rl.schemas import LstmSpec


def test_layer_n_params_lstm():
    entry = {"type": "lstm", "input_size": 32, "hidden_size": 32}
    # 4H*I + 4H*H + 8H = 4*32*32 + 4*32*32 + 8*32 = 4096 + 4096 + 256 = 8448
    assert _layer_n_params(entry) == 8448


def test_lstm_specs_count_matches_flat_weights():
    spec = LstmSpec(type="lstm", input_size=5, hidden_size=3)
    specs = _lstm_specs(spec, layer_idx=1, bound_multiplier=1.0)
    # 4H*I + 4H*H + 2*4H = 12*5 + 12*3 + 24 = 60 + 36 + 24 = 120
    assert len(specs) == 120


def test_lstm_specs_bounds_are_tanh_xavier():
    I, H = 5, 4
    spec = LstmSpec(type="lstm", input_size=I, hidden_size=H)
    specs = _lstm_specs(spec, layer_idx=0, bound_multiplier=1.0)

    # First 4H*I specs are weight_ih with tanh-Xavier(fan_in=I, fan_out=4H)
    # Xavier uniform bound for tanh is sqrt(6/(fan_in + fan_out)) * gain_tanh
    # compute_layer_bound handles this, so just sanity-check bound > 0 and
    # symmetric
    ps_ih_first = specs[0]
    assert ps_ih_first.lower_bound < 0
    assert ps_ih_first.upper_bound > 0
    assert math.isclose(ps_ih_first.lower_bound, -ps_ih_first.upper_bound)

    # Biases (last 2*4H specs) use tighter bounds (0.1 * bound_multiplier)
    bias_spec = specs[-1]
    assert math.isclose(bias_spec.upper_bound, 0.1, abs_tol=1e-12)


def test_nn_param_specs_from_v2_dispatches_lstm():
    architecture = [
        {"type": "dense", "input_size": 3, "output_size": 4, "activation": "tanh"},
        {"type": "lstm", "input_size": 4, "hidden_size": 2},
        {"type": "dense", "input_size": 2, "output_size": 1, "activation": "linear"},
    ]
    # We're passing raw dicts that Pydantic validates inside the function.
    specs = nn_param_specs_from_v2(architecture, bound_multiplier=1.0)
    # Dense0: 3*4+4 = 16; Lstm: 4*2*4 + 4*2*2 + 8*2 = 32+16+16 = 64; Dense2: 2+1 = 3
    assert len(specs) == 16 + 64 + 3
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run pytest tests/test_lstm_specs_and_n_params.py -v 2>&1 | tail -15
```

Expected: `ImportError: cannot import name '_lstm_specs'` and/or `_layer_n_params` raises on `"lstm"`.

- [ ] **Step 3: Add `_lstm_specs`**

In `src/python/aerocapture/training/encoding.py`, add after `_gru_specs`:

```python
def _lstm_specs(layer: LstmSpec, layer_idx: int, bound_multiplier: float) -> list[ParamSpec]:
    """Flat-weight spec order matches the Rust `LayerWeights for LstmLayer`:
    weight_ih (row-major [4H, I]) -> weight_hh (row-major [4H, H]) -> bias_ih -> bias_hh.

    Gate ordering on the 4H axis: (i, f, g, o). Forget-bias init to 1.0 is
    applied in init_v2_population, not reflected in these symmetric bounds.
    """
    h = layer.hidden_size
    four_h = 4 * h
    w_ih_bound = bound_multiplier * compute_layer_bound(layer.input_size, four_h, "tanh")
    w_hh_bound = bound_multiplier * compute_layer_bound(h, four_h, "tanh")
    b_bound = 0.1 * bound_multiplier

    specs: list[ParamSpec] = []
    for j in range(four_h * layer.input_size):
        specs.append(ParamSpec(f"w_ih{layer_idx}_{j}", -w_ih_bound, w_ih_bound, 0.0))
    for j in range(four_h * h):
        specs.append(ParamSpec(f"w_hh{layer_idx}_{j}", -w_hh_bound, w_hh_bound, 0.0))
    for j in range(four_h):
        specs.append(ParamSpec(f"b_ih{layer_idx}_{j}", -b_bound, b_bound, 0.0))
    for j in range(four_h):
        specs.append(ParamSpec(f"b_hh{layer_idx}_{j}", -b_bound, b_bound, 0.0))
    return specs
```

- [ ] **Step 4: Add `LstmSpec` import + dispatch in `_layer_param_specs`**

Top of `encoding.py`:
```python
from aerocapture.training.rl.schemas import DenseSpec, GruSpec, LayerSpec, LstmSpec
```

In `_layer_param_specs`:
```python
def _layer_param_specs(layer: LayerSpec, layer_idx: int, bound_multiplier: float) -> list[ParamSpec]:
    if isinstance(layer, DenseSpec):
        return _dense_specs(layer, layer_idx, bound_multiplier)
    if isinstance(layer, GruSpec):
        return _gru_specs(layer, layer_idx, bound_multiplier)
    if isinstance(layer, LstmSpec):
        return _lstm_specs(layer, layer_idx, bound_multiplier)
    msg = f"Unknown layer type for PSO specs: {layer!r}"
    raise ValueError(msg)
```

- [ ] **Step 5: Add LSTM arm to `config._layer_n_params`**

Open `src/python/aerocapture/training/config.py` around line 112. Currently handles dense + gru. Add:

```python
def _layer_n_params(entry: dict) -> int:
    t = entry["type"]
    if t == "dense":
        return entry["input_size"] * entry["output_size"] + entry["output_size"]
    if t == "gru":
        H = entry["hidden_size"]
        I = entry["input_size"]
        return 3 * H * I + 3 * H * H + 6 * H
    if t == "lstm":
        H = entry["hidden_size"]
        I = entry["input_size"]
        return 4 * H * I + 4 * H * H + 8 * H
    msg = f"Unknown layer type: {t!r}"
    raise ValueError(msg)
```

Match the existing style exactly — if the function uses a different parameter name or structure, mirror it.

- [ ] **Step 6: Add `export_v2_policy_to_json` LSTM branch**

Open `src/python/aerocapture/training/rl/export.py`. Find the section that iterates the policy's layers and writes each layer's weights dict (has a `GruLayer` branch). Add an `LstmLayer` branch:

```python
from aerocapture.training.rl.layers.lstm import LstmLayer
...

for idx, (layer, spec) in enumerate(zip(policy.layers, policy.architecture)):
    key = f"layer_{idx}"
    if isinstance(layer, DenseLayer):
        # existing dense branch
        ...
    elif isinstance(layer, GruLayer):
        # existing gru branch
        ...
    elif isinstance(layer, LstmLayer):
        weights_map[key] = {
            "weight_ih": layer.weight_ih.detach().cpu().numpy().tolist(),
            "weight_hh": layer.weight_hh.detach().cpu().numpy().tolist(),
            "bias_ih":   layer.bias_ih.detach().cpu().numpy().tolist(),
            "bias_hh":   layer.bias_hh.detach().cpu().numpy().tolist(),
        }
    else:
        raise ValueError(f"export: unknown layer type {type(layer).__name__!r}")
```

Also guard the obs-normalizer bake-in: if the first layer is `LstmLayer`, raise `NotImplementedError` (matches the GRU-as-first-layer guard from Phase 0):

```python
if obs_normalizer is not None and isinstance(policy.layers[0], (GruLayer, LstmLayer)):
    raise NotImplementedError(
        "obs-normalizer bake-in requires a Dense first layer; "
        f"got {type(policy.layers[0]).__name__}."
    )
```

- [ ] **Step 7: Add `load_policy_from_json` LSTM branch**

Open `src/python/aerocapture/training/model_io.py`. Find the layer-construction loop (has `GruSpec` branch building `GruLayer`). Add LSTM:

```python
from aerocapture.training.rl.layers.lstm import LstmLayer
from aerocapture.training.rl.schemas import LstmSpec
...

for idx, spec in enumerate(architecture):
    key = f"layer_{idx}"
    if isinstance(spec, DenseSpec):
        # existing dense branch: build DenseLayer + load w/b
        ...
    elif isinstance(spec, GruSpec):
        # existing gru branch
        ...
    elif isinstance(spec, LstmSpec):
        layer = LstmLayer(spec.input_size, spec.hidden_size)
        weights = payload["weights"][key]
        with torch.no_grad():
            layer.weight_ih.copy_(torch.tensor(weights["weight_ih"], dtype=torch.float64))
            layer.weight_hh.copy_(torch.tensor(weights["weight_hh"], dtype=torch.float64))
            layer.bias_ih.copy_(torch.tensor(weights["bias_ih"], dtype=torch.float64))
            layer.bias_hh.copy_(torch.tensor(weights["bias_hh"], dtype=torch.float64))
        layers.append(layer)
    else:
        raise ValueError(f"load: unknown layer spec {type(spec).__name__!r}")
```

Match the dtype + device handling of the existing GRU branch.

- [ ] **Step 8: Run tests + full suite**

```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run pytest tests/test_lstm_specs_and_n_params.py -v 2>&1 | tail -10
uv run pytest tests -q -x --ignore=tests/test_v2_rust_python_equivalence.py 2>&1 | tail -15
```

Expected: LSTM specs tests pass, no regressions elsewhere.

- [ ] **Step 9: Commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add \
  src/python/aerocapture/training/encoding.py \
  src/python/aerocapture/training/config.py \
  src/python/aerocapture/training/rl/export.py \
  src/python/aerocapture/training/model_io.py \
  tests/test_lstm_specs_and_n_params.py
git commit -m "$(cat <<'EOF'
feat(nn): Python LSTM integration (specs, n_params, export, load)

_lstm_specs: tanh-Xavier bounds on 4H gate blocks, 0.1*mul on biases.
nn_param_specs_from_v2: LstmSpec dispatch via isinstance chain.
config._layer_n_params: lstm arm (4HI + 4HH + 8H).
export_v2_policy_to_json: LstmLayer branch writes 4H weight/bias dicts;
obs-norm bake-in guard extended to reject LSTM as first layer.
load_policy_from_json: LstmSpec branch reconstructs parameters from JSON.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Cross-language equivalence test -- LSTM case (stateful multi-step with reset)

**Files:**
- Modify: `tests/test_v2_rust_python_equivalence.py`

Goal: the final gate on the LSTM forward math. Build a small Dense -> LSTM -> Dense V2Policy in f64, export to JSON v2, load through `aerocapture_rs.nn_forward`, feed a sequence of inputs with an explicit reset midway, and assert Rust and PyTorch match to machine epsilon.

- [ ] **Step 1: Add the LSTM case to the existing test file**

Open `tests/test_v2_rust_python_equivalence.py` and locate the existing GRU case (`test_rust_python_equivalence_gru` or similar). Add a new sibling test below it:

```python
def test_rust_python_equivalence_lstm_multi_step_with_reset(tmp_path):
    """Dense -> LSTM -> Dense: Rust forward matches PyTorch forward to
    machine epsilon on f64, across a 100-step sequence with a mid-sequence
    state reset.
    """
    import json
    import numpy as np
    import torch

    import aerocapture_rs
    from aerocapture.training.rl.policy import V2Policy
    from aerocapture.training.rl.export import export_v2_policy_to_json

    torch.manual_seed(42)
    np.random.seed(42)

    architecture = [
        {"type": "dense", "input_size": 5, "output_size": 4, "activation": "tanh"},
        {"type": "lstm", "input_size": 4, "hidden_size": 4},
        {"type": "dense", "input_size": 4, "output_size": 2, "activation": "linear"},
    ]
    policy = V2Policy(architecture=architecture, output_interpretation="atan2").double()

    json_path = tmp_path / "lstm_policy.json"
    export_v2_policy_to_json(policy, str(json_path), obs_normalizer=None)

    # Rust side: stateless nn_forward resets state per call; we simulate
    # multi-step by feeding sequentially and manually tracking the state
    # through load_for_sequential on the Python side. The Rust ground truth
    # is: "one call to nn_forward per step, with state persisted between
    # calls via the stateful helper that aerocapture_rs exposes."
    rs_handle = aerocapture_rs.load_policy_for_stateful_forward(str(json_path))

    # Python side: step through V2Policy manually
    py_state = [layer.new_state(batch_size=1, device="cpu") if hasattr(layer, "new_state") else None
                for layer in policy.layers]

    N_STEPS = 100
    RESET_AT = 50
    max_abs_diff = 0.0

    for t in range(N_STEPS):
        if t == RESET_AT:
            # Reset both sides
            py_state = [layer.new_state(batch_size=1, device="cpu") if hasattr(layer, "new_state") else None
                        for layer in policy.layers]
            aerocapture_rs.reset_stateful_forward(rs_handle)

        x = np.random.randn(5).astype(np.float64)

        # Python forward
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float64).unsqueeze(0)
            for idx, layer in enumerate(policy.layers):
                if isinstance(layer, torch.nn.Module) and hasattr(layer, "forward"):
                    # Dense: forward(x) -> y ; Gru/Lstm: forward(x, state) -> (y, new_state)
                    if py_state[idx] is None:
                        xt = layer(xt)
                    else:
                        xt, py_state[idx] = layer(xt, py_state[idx])
            py_out = xt.numpy().squeeze(0)

        # Rust forward (stateful)
        rs_out = np.asarray(aerocapture_rs.stateful_forward_step(rs_handle, x))

        diff = np.abs(py_out - rs_out).max()
        max_abs_diff = max(max_abs_diff, diff)

    # Target: machine epsilon like GRU's 4.4e-16. Assert strict bound.
    assert max_abs_diff < 1e-10, f"max abs diff {max_abs_diff!r} exceeds 1e-10"
```

**Note:** this test assumes `aerocapture_rs.load_policy_for_stateful_forward`, `reset_stateful_forward`, and `stateful_forward_step` exist. If they don't (Phase 1 / 1.5 may have used a different API for the GRU equivalence test), adapt to match the existing GRU test's style. Run:
```bash
grep -n "def test_rust_python_equivalence_gru\|aerocapture_rs\." tests/test_v2_rust_python_equivalence.py | head -20
```

Copy the exact Rust-side API invocations the GRU test uses and adapt the state reset location / architecture shape.

- [ ] **Step 2: Run the new test**

```bash
cd /Users/govit/Git/Govit/Aerocapture
./build.sh   # rebuild PyO3 to pick up any Rust changes since Task 6
uv run pytest tests/test_v2_rust_python_equivalence.py::test_rust_python_equivalence_lstm_multi_step_with_reset -v 2>&1 | tail -15
```

Expected: PASS with `max_abs_diff` at machine epsilon (~1e-16).

- [ ] **Step 3: Run all equivalence tests**

```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run pytest tests/test_v2_rust_python_equivalence.py -v 2>&1 | tail -15
```

Expected: all pass (Phase 0 dense case, Phase 1 GRU case, input_mask case, Phase 1.5 PPO-GRU case, new LSTM case).

- [ ] **Step 4: Commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add tests/test_v2_rust_python_equivalence.py
git commit -m "$(cat <<'EOF'
test(nn): cross-language LSTM equivalence -- 100-step sequence with reset

Dense -> Lstm(4, 4) -> Dense, f64. Exports V2Policy to JSON v2, loads
in Rust via aerocapture_rs, steps 100 inputs through both sides with a
mid-sequence reset at t=50. Asserts max abs diff < 1e-10 (target
machine epsilon, ~1e-16).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: `init_v2_population` -- per-layer activation-aware initialization

**Files:**
- Create: `src/python/aerocapture/training/initialization_v2.py`
- Create: `tests/test_init_v2_population.py`

Goal: single entry point `init_v2_population(architecture, n_pop, bound_multiplier, rng)` that dispatches per layer type. Dense reuses existing Xavier/He/LeCun logic; GRU uses tanh-Xavier on gate blocks; LSTM uses tanh-Xavier on gate blocks with forget-bias init to 1.0.

- [ ] **Step 1: Write failing tests**

Create `tests/test_init_v2_population.py`:

```python
"""init_v2_population produces activation-aware initial chromosomes per layer type.

Tests per-layer statistics: Xavier std for dense/gru/lstm weights, small bias noise,
and LSTM forget-bias-1 init (Jozefowicz et al 2015).
"""

from __future__ import annotations

import math

import numpy as np

from aerocapture.training.config import _layer_n_params
from aerocapture.training.initialization_v2 import init_v2_population


def _layer_offsets(architecture):
    """Return (start, end) flat-weight offsets per layer."""
    offsets = []
    cursor = 0
    for entry in architecture:
        n = _layer_n_params(entry)
        offsets.append((cursor, cursor + n))
        cursor += n
    return offsets


def test_init_v2_population_shape():
    architecture = [
        {"type": "dense", "input_size": 16, "output_size": 32, "activation": "tanh"},
        {"type": "lstm", "input_size": 32, "hidden_size": 32},
        {"type": "dense", "input_size": 32, "output_size": 2, "activation": "linear"},
    ]
    rng = np.random.default_rng(0)
    pop = init_v2_population(architecture, n_pop=50, bound_multiplier=1.0, rng=rng)
    n_expected = 544 + 8448 + 66  # = 9058
    assert pop.shape == (50, n_expected)


def test_init_v2_population_forget_bias_slice_init_to_one():
    architecture = [
        {"type": "dense", "input_size": 16, "output_size": 32, "activation": "tanh"},
        {"type": "lstm", "input_size": 32, "hidden_size": 32},
        {"type": "dense", "input_size": 32, "output_size": 2, "activation": "linear"},
    ]
    rng = np.random.default_rng(0)
    pop = init_v2_population(architecture, n_pop=1024, bound_multiplier=1.0, rng=rng)

    offsets = _layer_offsets(architecture)
    lstm_start, lstm_end = offsets[1]
    H = 32
    four_h = 4 * H
    I = 32

    # Flat order: weight_ih (4H*I) -> weight_hh (4H*H) -> bias_ih (4H) -> bias_hh (4H).
    bias_ih_start = lstm_start + four_h * I + four_h * H
    bias_ih_end = bias_ih_start + four_h
    bias_hh_start = bias_ih_end
    bias_hh_end = bias_hh_start + four_h

    # Forget slice on bias_ih is rows [H:2H] of the 4H axis.
    forget_ih = pop[:, bias_ih_start + H : bias_ih_start + 2 * H]
    forget_hh = pop[:, bias_hh_start + H : bias_hh_start + 2 * H]

    # Forget bias on ih should mean ~1.0 with small noise
    assert 0.9 < float(forget_ih.mean()) < 1.1
    assert 0.005 < float(forget_ih.std()) < 0.02

    # Forget bias on hh stays near 0 (+1 is only on ih; gate sum is ih + hh)
    assert -0.01 < float(forget_hh.mean()) < 0.01


def test_init_v2_population_non_forget_biases_small():
    architecture = [
        {"type": "lstm", "input_size": 8, "hidden_size": 8},
    ]
    rng = np.random.default_rng(1)
    pop = init_v2_population(architecture, n_pop=1024, bound_multiplier=1.0, rng=rng)
    H = 8
    four_h = 4 * H
    I = 8

    bias_ih_start = four_h * I + four_h * H
    # i-gate (rows [0:H]), g-gate (rows [2H:3H]), o-gate (rows [3H:4H]) are non-forget
    i_slice = pop[:, bias_ih_start : bias_ih_start + H]
    g_slice = pop[:, bias_ih_start + 2 * H : bias_ih_start + 3 * H]
    o_slice = pop[:, bias_ih_start + 3 * H : bias_ih_start + 4 * H]

    for s, name in [(i_slice, "i"), (g_slice, "g"), (o_slice, "o")]:
        assert abs(float(s.mean())) < 0.005, f"{name}-gate bias mean drifted from 0"
        assert 0.005 < float(s.std()) < 0.02, f"{name}-gate bias std out of range"


def test_init_v2_population_dense_tanh_xavier_std():
    """Dense layer with tanh activation: weight std ~= Xavier(fan_in, fan_out) for tanh."""
    I, O = 10, 20
    architecture = [
        {"type": "dense", "input_size": I, "output_size": O, "activation": "tanh"},
    ]
    rng = np.random.default_rng(2)
    pop = init_v2_population(architecture, n_pop=2048, bound_multiplier=1.0, rng=rng)
    # Dense flat order: row-major W (O*I values) then b (O values)
    weights = pop[:, : O * I]
    # Xavier uniform bound for tanh: std = gain * sqrt(2 / (fan_in + fan_out))
    # gain_tanh = 5/3, so theoretical std ~= (5/3)*sqrt(2/30) ~= 0.430
    # Uniform(-k, k) std = k/sqrt(3); for an init that uses Xavier bound as k,
    # std = k/sqrt(3) ~= 0.430/sqrt(3) ~= 0.248.
    # Check the empirical std is in a reasonable window; exact match depends on
    # whether the impl uses Gaussian Xavier or Uniform Xavier. Allow either:
    empirical_std = float(weights.std())
    assert 0.1 < empirical_std < 1.0


def test_init_v2_population_gru_tanh_xavier_bounds_respected():
    """GRU weight_ih block: init draws stay within ParamSpec tanh-Xavier bounds."""
    I, H = 16, 32
    architecture = [
        {"type": "gru", "input_size": I, "hidden_size": H},
    ]
    rng = np.random.default_rng(3)
    pop = init_v2_population(architecture, n_pop=256, bound_multiplier=1.0, rng=rng)
    three_h = 3 * H
    weight_ih = pop[:, : three_h * I]
    weight_hh = pop[:, three_h * I : three_h * I + three_h * H]
    # Finite + reasonable magnitude
    assert np.all(np.isfinite(weight_ih))
    assert np.all(np.isfinite(weight_hh))
    assert float(np.abs(weight_ih).max()) < 2.0  # well inside Xavier bound


def test_init_v2_population_dispatches_by_type_not_input_order():
    """Dense after LSTM, LSTM after Dense: per-layer offsets stay correct."""
    architecture = [
        {"type": "lstm", "input_size": 4, "hidden_size": 4},
        {"type": "dense", "input_size": 4, "output_size": 3, "activation": "linear"},
    ]
    rng = np.random.default_rng(4)
    pop = init_v2_population(architecture, n_pop=16, bound_multiplier=1.0, rng=rng)
    expected_n = _layer_n_params(architecture[0]) + _layer_n_params(architecture[1])
    assert pop.shape[1] == expected_n
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run pytest tests/test_init_v2_population.py -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError: No module named 'aerocapture.training.initialization_v2'`.

- [ ] **Step 3: Create `initialization_v2.py`**

Create `src/python/aerocapture/training/initialization_v2.py`:

```python
"""Activation-aware initialization for v2 architectures.

Dispatches per layer type to produce initial PSO chromosomes that match
the theoretical Xavier/He/LeCun distributions per activation, plus
LSTM forget-bias init to 1.0 (Jozefowicz, Zaremba & Sutskever 2015).

Flat-weight layout per layer must match:
  - Dense: row-major W (O*I) + b (O)
  - Gru:   row-major weight_ih (3H*I) + weight_hh (3H*H) + bias_ih (3H) + bias_hh (3H)
  - Lstm:  row-major weight_ih (4H*I) + weight_hh (4H*H) + bias_ih (4H) + bias_hh (4H)

Gate order on the multi-H axis matches the Rust / PyTorch contract:
  - Gru:  (r, z, n)
  - Lstm: (i, f, g, o)

Bias init convention:
  - Dense biases: N(0, bound_multiplier * compute_layer_bound / sqrt(3))
    (matches uniform-in-[-bound, +bound] std for consistency with
    create_nn_initial_population's dense path)
  - Gru biases:   N(0, 0.01 * bound_multiplier)
  - Lstm i/g/o biases: N(0, 0.01 * bound_multiplier)
  - Lstm forget bias slice on bias_ih: N(1.0, 0.01 * bound_multiplier)
  - Lstm bias_hh forget slice: N(0, 0.01 * bound_multiplier)
    (forget contribution is put on bias_ih only -- gate sum is ih + hh,
    so double-applying would give a forget gate of 2.0, not 1.0.)
"""

from __future__ import annotations

import numpy as np

from aerocapture.training.initialization import compute_layer_bound

BIAS_NOISE_STD = 0.01


def init_v2_population(
    architecture: list[dict],
    n_pop: int,
    bound_multiplier: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return (n_pop, n_params) initial chromosomes for the PSO path."""
    from aerocapture.training.config import _layer_n_params

    n_params = sum(_layer_n_params(entry) for entry in architecture)
    pop = np.empty((n_pop, n_params), dtype=np.float64)

    cursor = 0
    for entry in architecture:
        n = _layer_n_params(entry)
        slab = pop[:, cursor : cursor + n]
        _fill_layer(entry, slab, bound_multiplier, rng)
        cursor += n

    return pop


def _fill_layer(
    entry: dict, slab: np.ndarray, bound_multiplier: float, rng: np.random.Generator
) -> None:
    t = entry["type"]
    if t == "dense":
        _fill_dense(entry, slab, bound_multiplier, rng)
    elif t == "gru":
        _fill_gru(entry, slab, bound_multiplier, rng)
    elif t == "lstm":
        _fill_lstm(entry, slab, bound_multiplier, rng)
    else:
        raise ValueError(f"init_v2_population: unknown layer type {t!r}")


def _fill_dense(
    entry: dict, slab: np.ndarray, bound_multiplier: float, rng: np.random.Generator
) -> None:
    I = entry["input_size"]
    O = entry["output_size"]
    activation = entry["activation"]
    bound = bound_multiplier * compute_layer_bound(I, O, activation)

    # Uniform in [-bound, bound] to match create_nn_initial_population's existing dense path.
    n_w = O * I
    n_b = O
    slab[:, :n_w] = rng.uniform(-bound, bound, size=(slab.shape[0], n_w))
    slab[:, n_w : n_w + n_b] = rng.uniform(-bound, bound, size=(slab.shape[0], n_b))


def _fill_gru(
    entry: dict, slab: np.ndarray, bound_multiplier: float, rng: np.random.Generator
) -> None:
    I = entry["input_size"]
    H = entry["hidden_size"]
    three_h = 3 * H
    n_w_ih = three_h * I
    n_w_hh = three_h * H
    n_b = three_h

    # tanh-Xavier for gate blocks
    w_ih_bound = bound_multiplier * compute_layer_bound(I, three_h, "tanh")
    w_hh_bound = bound_multiplier * compute_layer_bound(H, three_h, "tanh")

    pop_n = slab.shape[0]
    slab[:, :n_w_ih] = rng.uniform(-w_ih_bound, w_ih_bound, size=(pop_n, n_w_ih))
    slab[:, n_w_ih : n_w_ih + n_w_hh] = rng.uniform(
        -w_hh_bound, w_hh_bound, size=(pop_n, n_w_hh)
    )
    bias_start = n_w_ih + n_w_hh
    slab[:, bias_start : bias_start + n_b] = rng.normal(
        0.0, BIAS_NOISE_STD * bound_multiplier, size=(pop_n, n_b)
    )
    slab[:, bias_start + n_b : bias_start + 2 * n_b] = rng.normal(
        0.0, BIAS_NOISE_STD * bound_multiplier, size=(pop_n, n_b)
    )


def _fill_lstm(
    entry: dict, slab: np.ndarray, bound_multiplier: float, rng: np.random.Generator
) -> None:
    I = entry["input_size"]
    H = entry["hidden_size"]
    four_h = 4 * H
    n_w_ih = four_h * I
    n_w_hh = four_h * H
    n_b = four_h

    w_ih_bound = bound_multiplier * compute_layer_bound(I, four_h, "tanh")
    w_hh_bound = bound_multiplier * compute_layer_bound(H, four_h, "tanh")

    pop_n = slab.shape[0]
    slab[:, :n_w_ih] = rng.uniform(-w_ih_bound, w_ih_bound, size=(pop_n, n_w_ih))
    slab[:, n_w_ih : n_w_ih + n_w_hh] = rng.uniform(
        -w_hh_bound, w_hh_bound, size=(pop_n, n_w_hh)
    )

    # bias_ih and bias_hh: small Gaussian except the forget slice on bias_ih
    bias_ih_start = n_w_ih + n_w_hh
    bias_hh_start = bias_ih_start + n_b

    # Start with all small Gaussian
    slab[:, bias_ih_start : bias_ih_start + n_b] = rng.normal(
        0.0, BIAS_NOISE_STD * bound_multiplier, size=(pop_n, n_b)
    )
    slab[:, bias_hh_start : bias_hh_start + n_b] = rng.normal(
        0.0, BIAS_NOISE_STD * bound_multiplier, size=(pop_n, n_b)
    )

    # Override forget slice on bias_ih (rows [H:2H] of the 4H axis) with N(1.0, 0.01)
    # Jozefowicz, Zaremba & Sutskever 2015.
    slab[:, bias_ih_start + H : bias_ih_start + 2 * H] = 1.0 + rng.normal(
        0.0, BIAS_NOISE_STD * bound_multiplier, size=(pop_n, H)
    )
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run pytest tests/test_init_v2_population.py -v 2>&1 | tail -20
```

Expected: all seven tests pass.

- [ ] **Step 5: Run full Python test suite**

```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run pytest tests -q -x --ignore=tests/test_v2_rust_python_equivalence.py 2>&1 | tail -15
```

Expected: no regressions.

- [ ] **Step 6: Commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add \
  src/python/aerocapture/training/initialization_v2.py \
  tests/test_init_v2_population.py
git commit -m "$(cat <<'EOF'
feat(init): init_v2_population with per-layer activation-aware dispatch

Dense: uniform-in-Xavier-bound (matches existing dense behavior).
Gru: tanh-Xavier on gate matrices, N(0, 0.01) bias noise.
Lstm: tanh-Xavier on gate matrices, N(0, 0.01) bias noise except
forget-bias slice on bias_ih initialized to N(1.0, 0.01) per
Jozefowicz, Zaremba & Sutskever 2015. Forget contribution is put on
bias_ih only; bias_hh forget stays near 0 to avoid double-application
through gate sum.

Closes the Phase 1 activation-aware init carry-over. GRU retroactively
uses Xavier bounds for gate matrices and small bias noise (was previously
uniform-in-ParamSpec-bounds, producing a noticeably wider bias
distribution).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Wire `init_v2_population` into `train.py`

**Files:**
- Modify: `src/python/aerocapture/training/train.py`

Goal: when the training config has `cfg.network.architecture` set (v2 path), route the initial PSO population through `init_v2_population` instead of the existing uniform-via-ParamSpec fallback. The dense-only v1 path (`cfg.network.layer_sizes + activations`) stays unchanged.

- [ ] **Step 1: Locate the PSO initial population dispatch**

```bash
cd /Users/govit/Git/Govit/Aerocapture
grep -n "create_nn_initial_population\|create_initial_population\|initial_population\|network.architecture" src/python/aerocapture/training/train.py | head -20
```

Expected: a block that chooses between `create_nn_initial_population` (dense-only v1) and `create_initial_population` (generic uniform, used for v2 archs).

- [ ] **Step 2: Write a small integration check**

Add to `tests/test_init_v2_population.py`:

```python
def test_init_v2_population_called_from_training_pipeline(monkeypatch):
    """Smoke: train.py routes v2 architecture through init_v2_population."""
    import numpy as np

    from aerocapture.training import train as train_mod

    call_log = []
    real_fn = train_mod.init_v2_population

    def spy(architecture, n_pop, bound_multiplier, rng):
        call_log.append({"arch_len": len(architecture), "n_pop": n_pop})
        return real_fn(architecture, n_pop, bound_multiplier, rng)

    monkeypatch.setattr(train_mod, "init_v2_population", spy)

    arch = [
        {"type": "dense", "input_size": 3, "output_size": 4, "activation": "tanh"},
        {"type": "lstm", "input_size": 4, "hidden_size": 4},
        {"type": "dense", "input_size": 4, "output_size": 2, "activation": "linear"},
    ]
    rng = np.random.default_rng(0)
    pop = train_mod.build_initial_population_for_v2(arch, n_pop=8, bound_multiplier=1.0, rng=rng)
    assert pop.shape[0] == 8
    assert len(call_log) == 1
    assert call_log[0]["arch_len"] == 3
```

(The test names `build_initial_population_for_v2` as the public entry in `train.py`. Adjust if the repo's naming convention differs — but the point is that the v2 path must invoke `init_v2_population`.)

- [ ] **Step 3: Run to confirm failure**

```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run pytest tests/test_init_v2_population.py::test_init_v2_population_called_from_training_pipeline -v 2>&1 | tail -10
```

Expected: `AttributeError: module 'aerocapture.training.train' has no attribute 'init_v2_population'` or similar.

- [ ] **Step 4: Add import + wire the v2 path**

In `src/python/aerocapture/training/train.py`:

```python
from aerocapture.training.initialization_v2 import init_v2_population
```

Find the branch that produces the initial population. If the existing code looks like:

```python
if cfg.network.architecture is not None:
    # v2 path
    initial_pop = create_initial_population(n_pop=cfg.optimizer.n_pop, specs=specs, rng=rng)
else:
    # v1 dense-only path
    initial_pop = create_nn_initial_population(...)
```

Change to:

```python
if cfg.network.architecture is not None:
    # v2 path: activation-aware init (Phase 2a)
    arch_as_dicts = [layer_spec_to_dict(s) for s in cfg.network.architecture]
    initial_pop = init_v2_population(
        arch_as_dicts,
        n_pop=cfg.optimizer.n_pop,
        bound_multiplier=cfg.optimizer.bound_multiplier,
        rng=rng,
    )
else:
    # v1 dense-only path (unchanged)
    initial_pop = create_nn_initial_population(...)
```

If `cfg.network.architecture` is a list of Pydantic models, `layer_spec_to_dict` is just `.model_dump()`. Add a small helper if the config structure needs converting:

```python
def build_initial_population_for_v2(architecture, n_pop, bound_multiplier, rng):
    return init_v2_population(architecture, n_pop, bound_multiplier, rng)
```

This thin wrapper is the callable the integration test spies on.

- [ ] **Step 5: Run the integration smoke**

```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run pytest tests/test_init_v2_population.py -v 2>&1 | tail -15
```

Expected: all pass including the new integration spy.

- [ ] **Step 6: Run full Python test suite**

```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run pytest tests -q -x --ignore=tests/test_v2_rust_python_equivalence.py 2>&1 | tail -15
```

Expected: no regressions. The existing Phase 1 PSO-GRU smoke test `tests/test_gru_pso_smoke.py` continues to pass (it asserts shape + finiteness + JSON structure, not specific cost values, so the init-distribution change is transparent).

- [ ] **Step 7: Commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add src/python/aerocapture/training/train.py tests/test_init_v2_population.py
git commit -m "$(cat <<'EOF'
feat(train): route v2 PSO initial population through init_v2_population

When cfg.network.architecture is set (v2 path), draw the initial
population via activation-aware init instead of the generic
uniform-in-ParamSpec-bounds fallback. Dense-only v1 path
(create_nn_initial_population) is unchanged; v1 configs train
bit-identically.

Phase 1 GRU retroactively gets tanh-Xavier gate init + small bias
noise. PSO-GRU convergence improves or stays flat; smoke-test
assertions (shape + finiteness + JSON structure) are invariant under
init changes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Training configs + `compare_guidance` + `train_all.sh`

**Files:**
- Create: `configs/training/msr_aller_lstm_pso_train.toml`
- Create: `configs/training/msr_aller_lstm_ppo_train.toml`
- Modify: `src/python/aerocapture/training/compare_guidance.py`
- Modify: `train_all.sh`

- [ ] **Step 1: Create `msr_aller_lstm_pso_train.toml`**

Mirror the GRU PSO config at `configs/training/msr_aller_gru_pso_train.toml` with the middle layer swapped to LSTM:

```bash
cd /Users/govit/Git/Govit/Aerocapture
cat configs/training/msr_aller_gru_pso_train.toml
```

Then create `configs/training/msr_aller_lstm_pso_train.toml` with the same structure. Key differences:

- Replace the Gru layer block with `type = "lstm"` and the same `hidden_size = 32`.
- Change `results_suffix = "lstm_pso"` (or whatever the suffix naming scheme is in the GRU config).
- Keep `input_mask = [0..15]` (same 16-input PSO convention as GRU).
- Keep `[optimizer]` block identical to GRU PSO: `algorithm = "pso"`, `n_pop = 64`, `n_gen = 1000`, `seed_strategy = "adaptive"`, `training_n_sims = 20`, `validation_n_sims = 1000`.

Verify param count: Dense(16->32) = 544, Lstm(32,32) = 8448, Dense(32->2) = 66. Total = **9058**. Document this in a TOML comment at the top.

- [ ] **Step 2: Create `msr_aller_lstm_ppo_train.toml`**

Mirror `configs/training/msr_aller_gru_ppo_train.toml`:

- Replace the Gru layer block with `type = "lstm"`, `hidden_size = 32`.
- Keep `input_mask = [0..22]` (23-input PPO convention).
- Keep `[rl]` block with `algorithm = "ppo"`, `total_steps = 5_000_000`, `n_envs = 64`, `rollout_steps = 256`.
- Keep `[rl.ppo]` with `bptt_length = 32` (rollout_steps % bptt_length == 0, enforced by RLConfig.from_toml).
- `results_suffix = "lstm_ppo"`.

Param count: Dense(23->32) = 768, Lstm(32,32) = 8448, Dense(32->2) = 66. Total = **9282**.

- [ ] **Step 3: Register schemes in `compare_guidance.py`**

Find `SCHEMES` dict and `_NN_DEPLOY_SCHEMES` set. Add:

```python
"neural_network_lstm_pso": {
    "training_config": "configs/training/msr_aller_lstm_pso_train.toml",
    "deploy_via": "neural_network",
    # match fields from neural_network_gru_pso
},
"neural_network_lstm_ppo": {
    "training_config": "configs/training/msr_aller_lstm_ppo_train.toml",
    "deploy_via": "neural_network",
    # match fields from neural_network_gru_ppo
},
```

Add to `_NN_DEPLOY_SCHEMES`:
```python
_NN_DEPLOY_SCHEMES = {
    "neural_network",
    "neural_network_rl",
    "neural_network_gru_pso",
    "neural_network_gru_ppo",
    "neural_network_lstm_pso",
    "neural_network_lstm_ppo",
}
```

Match the exact style and field set of the GRU entries.

- [ ] **Step 4: Add `train_all.sh` aliases**

Open `train_all.sh` and extend the scheme-alias mapping. Find the case block (likely a switch on `$scheme`) and add:

```bash
lstm_pso|nn_lstm_pso)
    run_training "neural_network_lstm_pso" "configs/training/msr_aller_lstm_pso_train.toml"
    ;;
lstm_ppo|nn_lstm_ppo)
    run_training "neural_network_lstm_ppo" "configs/training/msr_aller_lstm_ppo_train.toml"
    ;;
```

Style-match the existing `gru_pso|nn_gru_pso)` and `gru_ppo|nn_gru_ppo)` blocks.

Also extend the "all schemes" default loop to include the two LSTM schemes (after `nn_gru_pso` and `nn_gru_ppo`).

- [ ] **Step 5: Validate configs parse**

```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run python -c "
from aerocapture.training.config import load_training_config
c1 = load_training_config('configs/training/msr_aller_lstm_pso_train.toml')
c2 = load_training_config('configs/training/msr_aller_lstm_ppo_train.toml')
print('PSO n_params =', sum(__import__('aerocapture.training.config', fromlist=['_layer_n_params'])._layer_n_params(l) for l in c1.network.architecture))
print('PPO n_params =', sum(__import__('aerocapture.training.config', fromlist=['_layer_n_params'])._layer_n_params(l) for l in c2.network.architecture))
"
```

Expected output:
```
PSO n_params = 9058
PPO n_params = 9282
```

(If `load_training_config` accepts Pydantic models for architecture entries, extract `.model_dump()` before calling `_layer_n_params`.)

- [ ] **Step 6: Verify `compare_guidance` sees both schemes**

```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run python -c "
from aerocapture.training.compare_guidance import SCHEMES, _NN_DEPLOY_SCHEMES
assert 'neural_network_lstm_pso' in SCHEMES
assert 'neural_network_lstm_ppo' in SCHEMES
assert 'neural_network_lstm_pso' in _NN_DEPLOY_SCHEMES
assert 'neural_network_lstm_ppo' in _NN_DEPLOY_SCHEMES
print('OK')
"
```

Expected: `OK`.

- [ ] **Step 7: Commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add \
  configs/training/msr_aller_lstm_pso_train.toml \
  configs/training/msr_aller_lstm_ppo_train.toml \
  src/python/aerocapture/training/compare_guidance.py \
  train_all.sh
git commit -m "$(cat <<'EOF'
feat(configs): Phase 2a LSTM PSO + PPO training configs + aliases

msr_aller_lstm_pso_train.toml: Dense(16->32, tanh) -> Lstm(32, 32) ->
Dense(32->2, linear), 9058 params, PSO n_pop=64 n_gen=1000 adaptive seed.

msr_aller_lstm_ppo_train.toml: Dense(23->32, tanh) -> Lstm(32, 32) ->
Dense(32->2, linear), 9282 params, PPO bptt_length=32 rollout_steps=256.

compare_guidance registers neural_network_lstm_pso +
neural_network_lstm_ppo; train_all.sh gets lstm_pso/lstm_ppo aliases.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: PSO-LSTM + PPO-LSTM smoke tests + BPTT chunk-invariant LSTM extension

**Files:**
- Create: `tests/test_lstm_pso_smoke.py`
- Create: `tests/test_lstm_ppo_smoke.py`
- Modify: `tests/test_ppo_bptt_chunk_invariant.py`

- [ ] **Step 1: Create `test_lstm_pso_smoke.py`**

Modeled on `tests/test_gru_pso_smoke.py`:

```python
"""Phase 2a @slow PSO-LSTM smoke test.

Runs 2 PSO gens on a minimal Dense(16->8) -> Lstm(8, 8) -> Dense(8->2)
architecture. Asserts (a) best_model.json is v2 with layer types
["dense", "lstm", "dense"], (b) nn_forward returns a finite 2-tuple on
a sample input, (c) per-gen cost is finite.

Wired into the python-pyo3 CI job.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

aerocapture_rs = pytest.importorskip("aerocapture_rs")


@pytest.mark.slow
def test_lstm_pso_smoke(tmp_path: Path) -> None:
    # Create a reduced training config
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "lstm_pso_smoke.toml"

    # Inherit from the real config and override to tiny values
    config_path.write_text(
        """
base = ["../../configs/training/msr_aller_lstm_pso_train.toml"]

[[network.architecture]]
type = "dense"
output_size = 8
activation = "tanh"

[[network.architecture]]
type = "lstm"
hidden_size = 8

[[network.architecture]]
type = "dense"
output_size = 2
activation = "linear"

[optimizer]
n_pop = 8
n_gen = 2
training_n_sims = 2
validation_n_sims = 2

[output]
dir = "smoke_output"
"""
    )

    output_dir = tmp_path / "smoke_output"

    # Run training
    result = subprocess.run(
        [
            "uv", "run", "python", "-m", "aerocapture.training.train",
            str(config_path),
            "--no-tui",
            "--skip-report",
            "--output-dir", str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
        timeout=300,
    )
    assert result.returncode == 0, f"training failed: {result.stderr}"

    # Verify best_model.json is v2 with LSTM
    model_path = output_dir / "best_model.json"
    assert model_path.exists()
    payload = json.loads(model_path.read_text())
    assert payload["format_version"] == 2
    types = [entry["type"] for entry in payload["architecture"]]
    assert types == ["dense", "lstm", "dense"]

    # Sanity: nn_forward returns finite output
    rs_handle = aerocapture_rs.load_policy_for_stateful_forward(str(model_path))
    # reduced arch: input_size = 16 (same as full model via input_mask)
    x = np.random.randn(16).astype(np.float64) * 0.1
    out = aerocapture_rs.stateful_forward_step(rs_handle, x)
    out = np.asarray(out)
    assert out.shape == (2,)
    assert np.all(np.isfinite(out))
```

Adapt the `base = [...]` path and CLI invocation to match the Phase 1 GRU smoke test structure; the key is: 2 gens on a tiny arch with training completes successfully and produces a loadable v2 JSON.

- [ ] **Step 2: Create `test_lstm_ppo_smoke.py`**

Modeled on `tests/test_gru_ppo_smoke.py`:

```python
"""Phase 2a @slow PPO-LSTM smoke test.

Runs 5 PPO updates on a minimal Dense(23->8) -> Lstm(8, 8) -> Dense(8->2)
architecture with bptt_length=8, rollout_steps=16, n_envs=4. Asserts
training completes without NaN loss and exported JSON loads through
aerocapture_rs.nn_forward.

Wired into the python-pyo3 CI job.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

aerocapture_rs = pytest.importorskip("aerocapture_rs")


@pytest.mark.slow
def test_lstm_ppo_smoke(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "lstm_ppo_smoke.toml"

    config_path.write_text(
        """
base = ["../../configs/training/msr_aller_lstm_ppo_train.toml"]

[[network.architecture]]
type = "dense"
output_size = 8
activation = "tanh"

[[network.architecture]]
type = "lstm"
hidden_size = 8

[[network.architecture]]
type = "dense"
output_size = 2
activation = "linear"

[rl]
total_steps = 320  # n_envs=4 * rollout_steps=16 * 5 updates
n_envs = 4
rollout_steps = 16

[rl.ppo]
bptt_length = 8
update_epochs = 2

[rl.validation]
n_sims = 2
interval_updates = 100  # don't trigger validation in smoke

[output]
dir = "smoke_output"
"""
    )

    output_dir = tmp_path / "smoke_output"

    result = subprocess.run(
        [
            "uv", "run", "python", "-m", "aerocapture.training.rl.train",
            str(config_path),
            "--no-tui",
            "--skip-report",
            "--output-dir", str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
        timeout=300,
    )
    assert result.returncode == 0, f"training failed: {result.stderr}"

    model_path = output_dir / "best_model.json"
    assert model_path.exists()
    payload = json.loads(model_path.read_text())
    assert payload["format_version"] == 2
    types = [entry["type"] for entry in payload["architecture"]]
    assert types == ["dense", "lstm", "dense"]

    rs_handle = aerocapture_rs.load_policy_for_stateful_forward(str(model_path))
    x = np.random.randn(23).astype(np.float64) * 0.1
    out = aerocapture_rs.stateful_forward_step(rs_handle, x)
    out = np.asarray(out)
    assert out.shape == (2,)
    assert np.all(np.isfinite(out))
```

Adapt the `base = [...]` path and CLI module invocation to match the Phase 1.5 PPO-GRU smoke test.

- [ ] **Step 3: Extend `test_ppo_bptt_chunk_invariant.py` with LSTM case**

Open `tests/test_ppo_bptt_chunk_invariant.py`. The existing GRU case proves that single-chunk vs multi-chunk BPTT forward values are bit-identical. Add an LSTM case after it:

```python
def test_ppo_bptt_lstm_chunk_invariant():
    """Dense -> Lstm -> Dense V2Policy: ppo_update_bptt forward values
    match bit-for-bit between single-chunk BPTT (bptt_length == rollout_steps)
    and multi-chunk BPTT (bptt_length == rollout_steps / 4). Gradients
    are expected to differ; forward values aren't.
    """
    # Body mirrors the GRU test exactly, with
    # architecture = [DenseSpec(..., "tanh"), LstmSpec(...), DenseSpec(..., "linear")]
    # rollout_steps = 32, tested against bptt_length in (32, 8)
    ...
```

Copy the GRU version verbatim and swap the middle layer for `LstmSpec(type="lstm", input_size=..., hidden_size=...)`. The `ppo_update_bptt` internals must already handle LSTM tuple state (Task 8's `_zero_state_where_done` extension); this test validates that.

- [ ] **Step 4: Run all three new tests**

```bash
cd /Users/govit/Git/Govit/Aerocapture
./build.sh
uv run pytest tests/test_lstm_pso_smoke.py tests/test_lstm_ppo_smoke.py -v -m slow 2>&1 | tail -15
uv run pytest tests/test_ppo_bptt_chunk_invariant.py -v 2>&1 | tail -10
```

Expected: all pass. Smoke tests take ~5-30s each; chunk-invariant is fast.

- [ ] **Step 5: Wire smoke tests into the python-pyo3 CI job**

Open `.github/workflows/ci.yml`. Find the `python-pyo3` job's test command (currently runs `pytest tests/test_pyo3.py tests/test_v2_rust_python_equivalence.py tests/test_gru_pso_smoke.py tests/test_gru_ppo_smoke.py`). Append:

```
tests/test_lstm_pso_smoke.py tests/test_lstm_ppo_smoke.py tests/test_flat_weights_to_json_lstm.py
```

- [ ] **Step 6: Run full Python test suite including slow markers**

```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run pytest tests -q -m "not slow" 2>&1 | tail -10
uv run pytest tests -q -m slow 2>&1 | tail -10
```

Expected: both invocations clean.

- [ ] **Step 7: Commit**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add \
  tests/test_lstm_pso_smoke.py \
  tests/test_lstm_ppo_smoke.py \
  tests/test_ppo_bptt_chunk_invariant.py \
  .github/workflows/ci.yml
git commit -m "$(cat <<'EOF'
test(nn): LSTM PSO + PPO smoke + BPTT chunk invariant extension

@slow tests: 2-gen PSO on tiny LSTM (~600 params), 5-update PPO-LSTM
with bptt_length=8 on tiny LSTM. Both assert (a) v2 JSON with
['dense','lstm','dense'] arch, (b) aerocapture_rs.nn_forward returns
finite 2-tuple. Wired into python-pyo3 CI job.

BPTT chunk-invariance test extended to LSTM: single-chunk vs 4-chunk
forward values bit-identical, validating that the tuple-state dispatch
composes correctly under ppo_update_bptt.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Final verification + docs sync + smart-commit

**Files:**
- Modify: `CLAUDE.md` (Phase 2a subsection)
- Modify: `TODO.md` (mark Phase 2a done, update carry-overs)
- Modify: `README.md` (if it has a phase list or stateful-NN section)

- [ ] **Step 1: Full Rust verification**

```bash
cd /Users/govit/Git/Govit/Aerocapture
./check_all.sh 2>&1 | tail -15
```

Expected: Rust tests pass, fmt clean, clippy clean, release build succeeds.

- [ ] **Step 2: Full Python verification**

```bash
cd /Users/govit/Git/Govit/Aerocapture
./lint_code.sh 2>&1 | tail -10
uv run pytest tests -q 2>&1 | tail -15
uv run pytest tests -q -m slow 2>&1 | tail -10
```

Expected: ruff clean, mypy clean, pytest clean (both non-slow and slow).

- [ ] **Step 3: PyO3 rebuild + equivalence gate**

```bash
cd /Users/govit/Git/Govit/Aerocapture
./build.sh
uv run pytest tests/test_v2_rust_python_equivalence.py -v 2>&1 | tail -15
```

Expected: all equivalence tests pass (dense, gru, lstm, input_mask, ppo_gru export roundtrip).

- [ ] **Step 4: Guidance golden regression**

```bash
cd /Users/govit/Git/Govit/Aerocapture
uv run pytest tests/test_rust_guidance_golden.py -v 2>&1 | tail -15
```

Expected: 6/6 golden files bit-identical (ftc, eqglide, energy_ctrl, pred_guid, fnpag, neural). Phase 2a touches NN infra, not physics — this must pass unchanged.

- [ ] **Step 5: Update `CLAUDE.md`**

Add a Phase 2a subsection after the existing "Phase 1.5 PPO-GRU + truncated BPTT" paragraph. Use the same structure as Phase 1 / 1.5:

```markdown
**Phase 2a LSTM MVP + activation-aware init (branch `feature/lstm-mvp`, 2026-04-18)** adds the second stateful layer type and its PSO + PPO-BPTT training configs. Closes the Phase 1 init carry-over as a dependency of LSTM forget-bias-1 init:
- **Rust**: `LstmLayer` struct (`weight_ih [4H, I]`, `weight_hh [4H, H]`, `bias_ih [4H]`, `bias_hh [4H]`; PyTorch `nn.LSTMCell` convention: i/f/g/o gates with `c_new = f*c + i*g`, `h_new = o*tanh(c_new)`, no peepholes), `Layer::Lstm` / `LayerSpec::Lstm { input_size, hidden_size }` / `LayerState::Lstm { h, c }` (first named struct variant with multi-tensor state), `LayerWeights for LstmLayer` (flat order: `weight_ih` row-major -> `weight_hh` row-major -> `bias_ih` -> `bias_hh`, scaled to 4H), `NeuralNetModel::save_json` / `from_v2_json` / `from_flat_weights_v2` Lstm arms, `TomlLayerSpec::Lstm` + `[[network.architecture]] type = "lstm"` parser, `aerocapture_rs.flat_weights_to_json` Lstm branch.
- **Python**: `LstmLayer` torch module (manual gate computation matching `nn.LSTMCell` bit-for-bit; tuple-state contract `forward(x, (h, c)) -> (h_new, (h_new, c_new))`), `LstmSpec` Pydantic schema in the discriminated union, `_zero_state_where_done` tuple dispatch (LSTM is the first multi-tensor state; recursion terminates at Tensor), `_lstm_specs` + `_layer_n_params` arms, `export_v2_policy_to_json` + `load_policy_from_json` Lstm branches (obs-norm bake-in rejects LSTM as first layer per the Phase 0 invariant).
- **Activation-aware init**: new `initialization_v2.py::init_v2_population(architecture, n_pop, bound_multiplier, rng)` with per-layer dispatch: dense uniform-in-Xavier-bound (unchanged), GRU tanh-Xavier on 3H gate matrices + `N(0, 0.01)` biases (retroactively applied), LSTM tanh-Xavier on 4H gate matrices + `N(0, 0.01)` i/g/o biases + `N(1.0, 0.01)` forget-bias on `bias_ih` only (Jozefowicz, Zaremba & Sutskever 2015). Closes the Phase 1 init carry-over.
- **Training configs**: `msr_aller_lstm_pso_train.toml` (Dense(16->32,tanh) -> Lstm(32, 32) -> Dense(32->2,linear), 9058 params, PSO `n_pop=64 n_gen=1000 seed_strategy="adaptive"`) and `msr_aller_lstm_ppo_train.toml` (Dense(23->32,tanh) -> Lstm(32, 32) -> Dense(32->2,linear), 9282 params, PPO `bptt_length=32 rollout_steps=256`). Registered as `neural_network_lstm_pso` + `neural_network_lstm_ppo` in `compare_guidance.SCHEMES` + `_NN_DEPLOY_SCHEMES`; `train_all.sh` aliases `lstm_pso` / `lstm_ppo` / `nn_lstm_pso` / `nn_lstm_ppo`.
- **Gates**: cross-language LSTM equivalence (100-step sequence with mid-sequence reset, max abs diff target 1e-10, actual machine epsilon), PSO-LSTM smoke (2 gens on reduced ~600-param arch), PPO-LSTM smoke (5 updates, `bptt_length=8`, tuple-state dispatch exercised), BPTT chunk-invariant extension for LSTM (single-chunk vs multi-chunk forward values bit-identical), feedforward PPO regression gate preserved, 6/6 guidance golden regressions bit-identical.

Full spec: `docs/superpowers/specs/2026-04-18-phase-2a-lstm-mvp-design.md`. Plan: `docs/superpowers/plans/2026-04-18-phase-2a-lstm-mvp-plan.md`.
```

- [ ] **Step 6: Update `TODO.md`**

Replace the `### Phase 2a [IN PROGRESS ...]` section with a `### Phase 2a [DONE 2026-04-18]` block summarizing what shipped (mirror the Phase 1 / 1.5 structure). Update the carry-over list at the end:

```markdown
**Out-of-Phase-2a carry-overs (deferred):**
- [ ] SAC-GRU / SAC-LSTM (Phase 1.6; SAC stays on GaussianPolicy).
- [ ] Recurrent critic (Phase 1.5 carry-over).
- [ ] Widen `load_policy_from_json` to accept v1 JSON (Phase 0 carry-over).
- [ ] Fix pre-existing clippy warnings in `src/rust/aerocapture-py/src/lib.rs`.

**Closed by Phase 2a:**
- [x] Per-layer activation-aware initialization for GRU and LSTM.
```

Also update the paper-grid checklist (line ~21-24) to check off LSTM × PSO and LSTM × BPTT cells.

- [ ] **Step 7: Update `README.md` if relevant**

```bash
cd /Users/govit/Git/Govit/Aerocapture
grep -n "Phase 1\|Phase 2\|GRU\|LSTM\|stateful" README.md | head
```

If the README has a phase list or a stateful-NN section, append a Phase 2a line mirroring the Phase 1 / 1.5 lines. Otherwise skip.

- [ ] **Step 8: Final commit of docs**

```bash
cd /Users/govit/Git/Govit/Aerocapture
git add CLAUDE.md TODO.md README.md 2>/dev/null || true
git commit -m "$(cat <<'EOF'
docs: Phase 2a LSTM MVP landed; sync CLAUDE.md / TODO.md / README.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(If README.md had no relevant section, omit it from the `git add`.)

- [ ] **Step 9: Invoke `smart-commit` skill targeting the whole `feature/lstm-mvp` branch**

Per user's global CLAUDE.md rule, the final step of any implementation plan is to invoke the `smart-commit` skill. Use the Skill tool:

```
Skill: smart-commit
Args: Take the whole git branch feature/lstm-mvp into account; sync any remaining doc drift (CLAUDE.md, README.md) with the final code state, then commit.
```

`smart-commit` will:
- Diff the branch against main.
- Refresh `CLAUDE.md` if any contract shifted during implementation.
- Refresh `README.md` if it lists phases.
- Commit any drift with a coherent message.
- Leave the branch ready for the user to merge at will (no push, per user's global rule).

---

## Self-Review

**Spec coverage:** every section of `docs/superpowers/specs/2026-04-18-phase-2a-lstm-mvp-design.md` maps to at least one task:

| Spec section | Task(s) |
|---|---|
| §3.1 Rust LstmLayer + forward | Task 1 |
| §3.2 Rust LayerState::Lstm { h, c } | Task 2 |
| §3.3 Rust TOML parser (TomlLayerSpec::Lstm) | Task 5 |
| §3.4 Python LstmLayer torch module | Task 7 |
| §3.5 `_zero_state_where_done` tuple dispatch | Task 8 |
| §3.6 Activation-aware init | Task 11 + Task 12 |
| §3.7 PSO flat-weight layout (`_lstm_specs`) | Task 9 |
| §3.8 JSON v2 format addition | Task 4 + Task 6 |
| §3.9 Compare-guidance + train_all.sh | Task 13 |
| §4.1 PSO training config | Task 13 |
| §4.2 PPO training config | Task 13 |
| §5.1 Unit tests (Rust forward, roundtrip, JSON; Python LSTM match, zero-state, init stats, n_params) | Tasks 1, 3, 4, 7, 8, 9, 11 |
| §5.2 Integration tests (cross-language equivalence, PSO smoke, PPO smoke, BPTT chunk invariant) | Tasks 10, 14 |
| §5.3 Regression gates | Task 15 |
| §5.4 CI wiring | Task 14 |
| §6 Compatibility | Implicit (dense-only v1 path untouched; verified by Task 15 regression gates) |
| §7 Risks / mitigations | Tasks 8 (tuple dispatch test), 10 (equivalence), 11 (init stats) |
| §10 Final step (smart-commit) | Task 15 Step 9 |

**Placeholder scan:** every code step has actual code. Every command has an expected outcome. No "TBD", "TODO", or "similar to Task N" without repeated code.

**Type consistency:** `LstmLayer` struct fields (`input_size`, `hidden_size`, `weight_ih`, `weight_hh`, `bias_ih`, `bias_hh`) identical across Rust (Task 1), PyTorch (Task 7), and JSON serialization (Task 4). `LayerState::Lstm { h, c }` named-struct variant consistent across Rust `neural.rs` forward dispatch (Task 2) and the cloning test (Task 2). `_lstm_specs` flat-weight order in Task 9 matches Rust `LayerWeights for LstmLayer` in Task 3 matches `init_v2_population` fill order in Task 11 matches JSON v2 weights dict keys in Task 4. Gate ordering (i, f, g, o) stated identically across Rust (§3.1), PyTorch (Task 7), `_lstm_specs` docstring (Task 9), and init dispatch (Task 11).

**Scope:** 15 tasks, single PR, one feature branch. Plan is focused on the approved Phase 2a spec with no scope creep. Phase 2b (Window-MLP) and Phase 1.6 (SAC-GRU) remain explicit non-goals.

---

**Plan complete.** Saved to `docs/superpowers/plans/2026-04-18-phase-2a-lstm-mvp-plan.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task with review between tasks.

**2. Inline Execution** — execute tasks in this session using `executing-plans`, batch execution with checkpoints.

Which approach?
