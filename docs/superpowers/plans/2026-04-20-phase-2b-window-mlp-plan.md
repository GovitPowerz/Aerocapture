# Phase 2b Window-MLP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land Window-MLP as the third stateful layer type on the Phase 0/1/1.5/2a stack. Window has zero trainable parameters and maintains a FIFO ring buffer of the last `n_steps` inputs; it widens the observation vector by history for the next Dense layer. Trained on PSO only; PPO paths fail loudly at `build_layer` / `load_policy_from_json` with a clear pointer to the spec.

**Architecture:** Rust `Layer` gains a `Window(WindowLayer)` variant; `LayerSpec::Window { input_size, n_steps }` and `LayerState::Window { buffer: VecDeque<Vec<f64>> }` extend the enums. `LayerWeights for WindowLayer` is the first zero-parameter impl (`n_params() == 0`, `to_flat() == Vec::new()`, no-op `from_flat`). Python `WindowLayer` torch module mirrors the Rust forward for cross-language equivalence only; PPO dispatch raises `NotImplementedError`. PSO uses the Rust runtime directly and is unaffected by V2Policy limitations.

**Tech Stack:** Rust 2024 edition, PyO3 for Python bindings, Python 3.14, PyTorch (manual window forward), Pydantic v2 discriminated unions, pymoo PSO, pytest.

**Spec:** `docs/superpowers/specs/2026-04-20-phase-2b-window-mlp-design.md`

---

## Task 0: TODO.md marker

**Files:**
- Modify: `TODO.md`

- [ ] **Step 1: Mark Phase 2b in progress**

Replace the "Phase 2b -- Window-MLP" stub with a "DOING 2026-04-20" marker:

```markdown
### Phase 2b -- Window-MLP (ring buffer, no new matmul) [DOING 2026-04-20 on feature/window-mlp]
- [ ] Rust `WindowLayer` + `Layer::Window` + `LayerSpec::Window { input_size, n_steps }` + `LayerState::Window { buffer: VecDeque<Vec<f64>> }` + `TomlLayerSpec::Window`
- [ ] `LayerWeights for WindowLayer` zero-param impl + JSON v2 + PyO3 test
- [ ] Python `WindowLayer` torch module + `WindowSpec` pydantic + `build_layer` PPO-rejection guard
- [ ] `_layer_param_specs` / `_layer_n_params` / `_layer_output_size` Window arms + `init_v2_population` no-op continue
- [ ] Training config `msr_aller_window_pso_train.toml` + `compare_guidance` + `train_all.sh` registration
- [ ] Cross-language equivalence test + PSO smoke test + PPO-rejection test (@slow python-pyo3 CI + @fast main CI)

Spec: `docs/superpowers/specs/2026-04-20-phase-2b-window-mlp-design.md`.
Plan: `docs/superpowers/plans/2026-04-20-phase-2b-window-mlp-plan.md`.
```

- [ ] **Step 2: Commit**

```bash
git add TODO.md
git commit -m "docs(todo): mark Phase 2b in progress on feature/window-mlp"
```

---

## Task 1: Rust LayerSpec::Window variant + WindowLayer struct

**Files:**
- Modify: `src/rust/src/data/neural.rs` (LayerSpec enum, Layer enum, add WindowLayer struct)
- Test: `src/rust/src/data/neural.rs` (#[cfg(test)] module)

- [ ] **Step 1: Write the failing test**

Append to the existing `#[cfg(test)] mod tests` block in `src/rust/src/data/neural.rs`:

```rust
#[test]
fn window_layer_struct_and_spec_variants_construct() {
    let spec = LayerSpec::Window { input_size: 4, n_steps: 3 };
    match spec {
        LayerSpec::Window { input_size, n_steps } => {
            assert_eq!(input_size, 4);
            assert_eq!(n_steps, 3);
        }
        _ => panic!("expected LayerSpec::Window"),
    }

    let layer = WindowLayer { input_size: 4, n_steps: 3 };
    assert_eq!(layer.input_size, 4);
    assert_eq!(layer.n_steps, 3);

    let enum_layer = Layer::Window(layer);
    match enum_layer {
        Layer::Window(w) => {
            assert_eq!(w.input_size, 4);
            assert_eq!(w.n_steps, 3);
        }
        _ => panic!("expected Layer::Window"),
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test --manifest-path src/rust/Cargo.toml --package aerocapture data::neural::tests::window_layer_struct_and_spec_variants_construct
```

Expected: compile error `variant Window not found on LayerSpec` (and `WindowLayer` not defined).

- [ ] **Step 3: Add LayerSpec::Window variant + WindowLayer struct + Layer::Window variant**

In `src/rust/src/data/neural.rs`, extend the `LayerSpec` enum (add variant after Lstm):

```rust
Window {
    input_size: usize,
    n_steps: usize,
},
```

Extend the `Layer` enum (add variant after Lstm):

```rust
Window(WindowLayer),
```

Add the `WindowLayer` struct near `LstmLayer` (no weights):

```rust
/// Window-MLP layer: FIFO ring buffer of the last `n_steps` inputs,
/// concatenated into a vector of length `n_steps * input_size`.
/// Zero trainable parameters; all trainable weight lives in the downstream Dense layer.
#[derive(Debug, Clone)]
pub struct WindowLayer {
    pub input_size: usize,
    pub n_steps: usize,
}
```

Update the comment in `neural.rs:205` and `neural.rs:436` from `// Phases 2b-4 add: Window, Attention, LayerNorm, Ssm` to `// Phases 3-4 add: Attention, LayerNorm, Ssm` (Window just landed).

- [ ] **Step 4: Run test to verify it passes**

```bash
cargo test --manifest-path src/rust/Cargo.toml --package aerocapture data::neural::tests::window_layer_struct_and_spec_variants_construct
```

Expected: PASS.

Note: other match arms on `LayerSpec` and `Layer` will now error until Tasks 2-5 land. This is expected — we proceed TDD-style and add the arms as we touch each file.

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/data/neural.rs
git commit -m "feat(nn): add WindowLayer + Layer::Window + LayerSpec::Window variants"
```

---

## Task 2: Rust WindowLayer forward + LayerState::Window + for_layer + reset

**Files:**
- Modify: `src/rust/src/data/nn_state.rs` (LayerState enum + for_layer + reset)
- Modify: `src/rust/src/data/neural.rs` (NeuralNetModel::forward match arm for Window, add WindowLayer::forward impl)
- Test: `src/rust/src/data/nn_state.rs` (#[cfg(test)] module)
- Test: `src/rust/src/data/neural.rs` (#[cfg(test)] module)

- [ ] **Step 1: Write the failing LayerState test**

Append to the `#[cfg(test)]` block in `src/rust/src/data/nn_state.rs`:

```rust
#[test]
fn layer_state_window_for_layer_prefills_buffer_with_zero_vectors() {
    use std::collections::VecDeque;
    let layer = Layer::Window(crate::data::neural::WindowLayer {
        input_size: 4,
        n_steps: 3,
    });
    let state = LayerState::for_layer(&layer);
    if let LayerState::Window { buffer } = state {
        assert_eq!(buffer.len(), 3);
        for slot in buffer.iter() {
            assert_eq!(slot.len(), 4);
            for &v in slot {
                assert_eq!(v, 0.0);
            }
        }
    } else {
        panic!("expected LayerState::Window");
    }
}

#[test]
fn layer_state_window_reset_clears_buffer_to_zeros() {
    use std::collections::VecDeque;
    let mut state = LayerState::Window {
        buffer: VecDeque::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]),
    };
    state.reset();
    if let LayerState::Window { buffer } = state {
        assert_eq!(buffer.len(), 2);
        for slot in buffer.iter() {
            assert!(slot.iter().all(|&v| v == 0.0));
        }
    } else {
        panic!("expected LayerState::Window after reset");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test --manifest-path src/rust/Cargo.toml --package aerocapture data::nn_state::tests::layer_state_window
```

Expected: FAIL — `variant Window not found` on LayerState.

- [ ] **Step 3: Extend LayerState + for_layer + reset**

In `src/rust/src/data/nn_state.rs`, add `VecDeque` import at the top and extend the enum + methods:

```rust
use std::collections::VecDeque;

// ... elsewhere in the file ...

pub enum LayerState {
    None,
    Gru { h: Vec<f64> },
    Lstm { h: Vec<f64>, c: Vec<f64> },
    Window { buffer: VecDeque<Vec<f64>> },
}
```

Update the enum comment header from `// Phase 2b+: Window { buffer: ... }, Ssm { ... }` to `// Phase 3+: Ssm { h: Vec<f64> }` (drop the Window line since it's landed).

In `for_layer`, add the Window arm (after the Lstm arm):

```rust
Layer::Window(w) => {
    let mut buffer = VecDeque::with_capacity(w.n_steps);
    for _ in 0..w.n_steps {
        buffer.push_back(vec![0.0; w.input_size]);
    }
    LayerState::Window { buffer }
}
```

In `reset`, add the Window arm:

```rust
LayerState::Window { buffer } => {
    for slot in buffer.iter_mut() {
        for v in slot.iter_mut() {
            *v = 0.0;
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test --manifest-path src/rust/Cargo.toml --package aerocapture data::nn_state::tests::layer_state_window
```

Expected: PASS.

- [ ] **Step 5: Write the failing WindowLayer::forward test**

Append to the `#[cfg(test)]` block in `src/rust/src/data/neural.rs`:

```rust
#[test]
fn window_layer_forward_push_pop_and_concat_zero_padded() {
    use crate::data::nn_state::LayerState;
    use std::collections::VecDeque;

    let layer = WindowLayer { input_size: 2, n_steps: 3 };
    let mut state = LayerState::for_layer(&Layer::Window(layer.clone()));

    // Tick 0: first real input [1.0, 2.0]. Buffer becomes [[0,0], [0,0], [1,2]].
    let out0 = layer.forward(&[1.0, 2.0], &mut state);
    assert_eq!(out0, vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0]);

    // Tick 1: [3.0, 4.0]. Buffer becomes [[0,0], [1,2], [3,4]].
    let out1 = layer.forward(&[3.0, 4.0], &mut state);
    assert_eq!(out1, vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0]);

    // Tick 2: [5.0, 6.0]. Buffer becomes [[1,2], [3,4], [5,6]].
    let out2 = layer.forward(&[5.0, 6.0], &mut state);
    assert_eq!(out2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Verify buffer is at steady-state capacity (always n_steps=3 entries).
    if let LayerState::Window { buffer } = state {
        assert_eq!(buffer.len(), 3);
    } else {
        panic!("expected Window state");
    }
}

#[test]
#[should_panic(expected = "WindowLayer::forward called with non-Window state")]
fn window_layer_forward_panics_on_wrong_state_variant() {
    let layer = WindowLayer { input_size: 2, n_steps: 3 };
    let mut wrong_state = crate::data::nn_state::LayerState::None;
    layer.forward(&[1.0, 2.0], &mut wrong_state);
}
```

- [ ] **Step 6: Run tests to verify they fail**

```bash
cargo test --manifest-path src/rust/Cargo.toml --package aerocapture data::neural::tests::window_layer_forward
```

Expected: FAIL — `WindowLayer::forward` does not exist.

- [ ] **Step 7: Implement WindowLayer::forward + NeuralNetModel::forward dispatch**

Add `WindowLayer::forward` near the other `*Layer::forward` impls in `src/rust/src/data/neural.rs`:

```rust
impl WindowLayer {
    pub fn forward(&self, input: &[f64], state: &mut crate::data::nn_state::LayerState) -> Vec<f64> {
        let buffer = match state {
            crate::data::nn_state::LayerState::Window { buffer } => buffer,
            _ => panic!("WindowLayer::forward called with non-Window state"),
        };
        assert_eq!(input.len(), self.input_size,
            "WindowLayer expected input_size={}, got {}", self.input_size, input.len());
        buffer.pop_front();
        buffer.push_back(input.to_vec());
        let mut out = Vec::with_capacity(self.n_steps * self.input_size);
        for slot in buffer.iter() {
            out.extend_from_slice(slot);
        }
        out
    }
}
```

In `NeuralNetModel::forward` (around `src/rust/src/data/neural.rs:1003`), add the Window match arm after the Lstm arm:

```rust
(Layer::Window(w), LayerState::Window { .. }) => {
    x = w.forward(&x, layer_state);
}
```

Leave the trailing `_ => panic!("layer/state variant mismatch ...")` arm as-is; update its literal error message to include Window in the enumeration:

```rust
"layer/state variant mismatch (construction invariant -- LayerState::for_layer maps Layer::Dense -> None, Layer::Gru -> Gru, Layer::Lstm -> Lstm, Layer::Window -> Window)"
```

- [ ] **Step 8: Run tests to verify they pass**

```bash
cargo test --manifest-path src/rust/Cargo.toml --package aerocapture data::neural::tests::window_layer_forward
cargo test --manifest-path src/rust/Cargo.toml --package aerocapture data::nn_state::tests
```

Expected: PASS (both).

- [ ] **Step 9: Commit**

```bash
git add src/rust/src/data/nn_state.rs src/rust/src/data/neural.rs
git commit -m "feat(nn): LayerState::Window + WindowLayer::forward dispatch"
```

---

## Task 3: Rust LayerWeights for WindowLayer (zero-param impl)

**Files:**
- Modify: `src/rust/src/data/neural.rs` (add `impl LayerWeights for WindowLayer`)
- Test: `src/rust/src/data/neural.rs` (#[cfg(test)] module)

- [ ] **Step 1: Write the failing test**

Append to the `#[cfg(test)]` block in `src/rust/src/data/neural.rs`:

```rust
#[test]
fn window_layer_weights_trait_zero_params() {
    let layer = WindowLayer { input_size: 4, n_steps: 8 };
    assert_eq!(layer.n_params(), 0);
    assert_eq!(layer.to_flat(), Vec::<f64>::new());

    let mut layer_mut = layer.clone();
    layer_mut.from_flat(&[]);  // no-op, must not panic
    assert_eq!(layer_mut.to_flat(), Vec::<f64>::new());
}

#[test]
#[should_panic(expected = "WindowLayer takes no weights")]
fn window_layer_from_flat_panics_on_nonempty_input() {
    let mut layer = WindowLayer { input_size: 4, n_steps: 8 };
    layer.from_flat(&[0.1, 0.2]);
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test --manifest-path src/rust/Cargo.toml --package aerocapture data::neural::tests::window_layer_weights
```

Expected: FAIL — `LayerWeights` trait not implemented for `WindowLayer`.

- [ ] **Step 3: Add `impl LayerWeights for WindowLayer`**

In `src/rust/src/data/neural.rs`, near the `LayerWeights for LstmLayer` impl:

```rust
impl LayerWeights for WindowLayer {
    fn n_params(&self) -> usize { 0 }
    fn to_flat(&self) -> Vec<f64> { Vec::new() }
    fn from_flat(&mut self, flat: &[f64]) {
        assert!(flat.is_empty(), "WindowLayer takes no weights, got {} values", flat.len());
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cargo test --manifest-path src/rust/Cargo.toml --package aerocapture data::neural::tests::window_layer_weights
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/data/neural.rs
git commit -m "feat(nn): LayerWeights for WindowLayer (zero-param impl)"
```

---

## Task 4: Rust JSON v2 save/load for Window + from_flat_weights_v2

**Files:**
- Modify: `src/rust/src/data/neural.rs` (save_json, from_v2_json, from_flat_weights_v2)
- Test: `src/rust/src/data/neural.rs` (#[cfg(test)] module)

- [ ] **Step 1: Write the failing roundtrip test**

Append to the `#[cfg(test)]` block:

```rust
#[test]
fn window_json_v2_roundtrip_spec_only() {
    let model = NeuralNetModel {
        architecture: vec![
            LayerSpec::Window { input_size: 4, n_steps: 3 },
            LayerSpec::Dense { input_size: 12, output_size: 2, activation: Activation::Linear },
        ],
        layers: vec![
            Layer::Window(WindowLayer { input_size: 4, n_steps: 3 }),
            Layer::Dense(DenseLayer {
                w: vec![vec![0.1; 12]; 2],
                b: vec![0.0; 2],
                activation: Activation::Linear,
            }),
        ],
        output_interpretation: "atan2".into(),
        input_mask: None,
        ablated_input: None,
    };

    let json = model.save_json().unwrap();

    // Window entry is spec-only (no weights dict under weights["layer_0"]).
    assert!(json.contains(r#""type":"window""#));
    assert!(json.contains(r#""input_size":4"#));
    assert!(json.contains(r#""n_steps":3"#));

    // Roundtrip: parse + compare key fields.
    let parsed = NeuralNetModel::from_json_str(&json).unwrap();
    match &parsed.architecture[0] {
        LayerSpec::Window { input_size, n_steps } => {
            assert_eq!(*input_size, 4);
            assert_eq!(*n_steps, 3);
        }
        _ => panic!("expected LayerSpec::Window"),
    }
    match &parsed.layers[0] {
        Layer::Window(w) => {
            assert_eq!(w.input_size, 4);
            assert_eq!(w.n_steps, 3);
        }
        _ => panic!("expected Layer::Window"),
    }
}

#[test]
fn window_from_flat_weights_v2_produces_zero_param_layer() {
    let arch = vec![
        LayerSpec::Window { input_size: 4, n_steps: 3 },
        LayerSpec::Dense { input_size: 12, output_size: 2, activation: Activation::Linear },
    ];
    // Total param count = 0 (window) + 12*2 + 2 = 26.
    let flat: Vec<f64> = (0..26).map(|i| i as f64 * 0.01).collect();
    let model = NeuralNetModel::from_flat_weights_v2(
        &arch, &flat, "atan2".into(), None, None,
    ).unwrap();

    match &model.layers[0] {
        Layer::Window(w) => {
            assert_eq!(w.input_size, 4);
            assert_eq!(w.n_steps, 3);
        }
        _ => panic!("expected Layer::Window"),
    }
    // Dense layer absorbed the full 26 params.
    match &model.layers[1] {
        Layer::Dense(d) => {
            assert_eq!(d.w.len(), 2);
            assert_eq!(d.w[0].len(), 12);
            assert_eq!(d.b.len(), 2);
        }
        _ => panic!("expected Layer::Dense"),
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test --manifest-path src/rust/Cargo.toml --package aerocapture data::neural::tests::window_json_v2_roundtrip
cargo test --manifest-path src/rust/Cargo.toml --package aerocapture data::neural::tests::window_from_flat_weights_v2
```

Expected: FAIL — `save_json` and `from_v2_json` and `from_flat_weights_v2` don't handle Window.

- [ ] **Step 3: Extend save_json + from_v2_json + from_flat_weights_v2**

In `NeuralNetModel::save_json` (the per-spec match), add the Window arm (returns spec-only JSON, no weights):

```rust
LayerSpec::Window { input_size, n_steps } => {
    architecture_json.push(serde_json::json!({
        "type": "window",
        "input_size": input_size,
        "n_steps": n_steps,
    }));
    // No entry in `weights` dict for this layer -- it is zero-param.
}
```

In `from_v2_json` (the match on `layer_type.as_str()`), add the `"window"` arm:

```rust
"window" => {
    let input_size = layer_entry.get("input_size")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| DataError::InvalidArchitecture(
            format!("(window) layer {i} missing input_size")
        ))? as usize;
    let n_steps = layer_entry.get("n_steps")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| DataError::InvalidArchitecture(
            format!("(window) layer {i} missing n_steps")
        ))? as usize;
    if input_size == 0 || n_steps == 0 {
        return Err(DataError::InvalidArchitecture(
            "(window) input_size and n_steps must be positive".into()
        ));
    }
    architecture.push(LayerSpec::Window { input_size, n_steps });
    layers.push(Layer::Window(WindowLayer { input_size, n_steps }));
}
```

In `from_flat_weights_v2` (the per-spec match that builds `layers`), add the Window arm (consumes zero flat weights, increments cursor by 0):

```rust
LayerSpec::Window { input_size, n_steps } => {
    layers.push(Layer::Window(WindowLayer {
        input_size: *input_size,
        n_steps: *n_steps,
    }));
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test --manifest-path src/rust/Cargo.toml --package aerocapture data::neural::tests::window_json
cargo test --manifest-path src/rust/Cargo.toml --package aerocapture data::neural::tests::window_from_flat
```

Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/data/neural.rs
git commit -m "feat(nn): JSON v2 read+write for Window + from_flat_weights_v2"
```

---

## Task 5: Rust TomlLayerSpec::Window + to_layer_spec arm

**Files:**
- Modify: `src/rust/src/config.rs` (TomlLayerSpec enum, to_layer_spec)
- Test: `src/rust/src/config.rs` (#[cfg(test)] module, near `toml_layer_spec_to_layer_spec_*` tests)

- [ ] **Step 1: Write the failing test**

Append to the `#[cfg(test)] mod tests` block in `src/rust/src/config.rs`:

```rust
#[test]
fn toml_layer_spec_to_layer_spec_window() {
    let toml_spec = TomlLayerSpec::Window { input_size: 4, n_steps: 8 };
    match toml_spec.to_layer_spec().unwrap() {
        crate::data::neural::LayerSpec::Window { input_size, n_steps } => {
            assert_eq!(input_size, 4);
            assert_eq!(n_steps, 8);
        }
        _ => panic!("expected LayerSpec::Window"),
    }
}

#[test]
fn toml_layer_spec_window_parses_from_toml_string() {
    let toml_str = r#"
[[network.architecture]]
type = "window"
input_size = 4
n_steps = 8
"#;
    #[derive(serde::Deserialize)]
    struct Wrapper {
        network: NetworkArch,
    }
    #[derive(serde::Deserialize)]
    struct NetworkArch {
        architecture: Vec<TomlLayerSpec>,
    }
    let parsed: Wrapper = toml::from_str(toml_str).unwrap();
    match &parsed.network.architecture[0] {
        TomlLayerSpec::Window { input_size, n_steps } => {
            assert_eq!(*input_size, 4);
            assert_eq!(*n_steps, 8);
        }
        _ => panic!("expected TomlLayerSpec::Window"),
    }
}

#[test]
fn toml_layer_spec_window_rejects_zero_fields() {
    let zero_input = TomlLayerSpec::Window { input_size: 0, n_steps: 8 };
    assert!(zero_input.to_layer_spec().is_err());
    let zero_n_steps = TomlLayerSpec::Window { input_size: 4, n_steps: 0 };
    assert!(zero_n_steps.to_layer_spec().is_err());
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test --manifest-path src/rust/Cargo.toml --package aerocapture config::tests::toml_layer_spec_window
```

Expected: FAIL — `TomlLayerSpec::Window` variant doesn't exist.

- [ ] **Step 3: Extend TomlLayerSpec + to_layer_spec**

In `src/rust/src/config.rs`, add the Window variant to `TomlLayerSpec` (after `Lstm`):

```rust
Window {
    input_size: usize,
    n_steps: usize,
},
```

In `TomlLayerSpec::to_layer_spec`, add the Window arm:

```rust
TomlLayerSpec::Window { input_size, n_steps } => {
    if *input_size == 0 || *n_steps == 0 {
        return Err(ParseError::Validation(
            format!("Window layer input_size and n_steps must be positive (got input_size={}, n_steps={})", input_size, n_steps)
        ));
    }
    Ok(LayerSpec::Window {
        input_size: *input_size,
        n_steps: *n_steps,
    })
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test --manifest-path src/rust/Cargo.toml --package aerocapture config::tests::toml_layer_spec_window
```

Expected: PASS.

- [ ] **Step 5: Full Rust check**

```bash
./check_all.sh
```

Expected: `cargo test` + `cargo fmt --check` + `cargo clippy` + release build all pass. All 10 guidance golden regression tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/config.rs
git commit -m "feat(config): TOML [[network.architecture]] parses type = \"window\""
```

---

## Task 6: PyO3 flat_weights_to_json test for Window architecture

**Files:**
- Create: `tests/test_flat_weights_to_json_window.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_flat_weights_to_json_window.py`:

```python
"""Verify flat_weights_to_json handles Window layers (zero-weight entries)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("aerocapture_rs")
import aerocapture_rs


def test_flat_weights_to_json_window_roundtrip(tmp_path: Path) -> None:
    # Architecture: Window(4, 3) -> Dense(12 -> 2, linear).
    # Window has zero params; Dense has 12*2 + 2 = 26 params.
    architecture = [
        {"type": "window", "input_size": 4, "n_steps": 3},
        {"type": "dense", "input_size": 12, "output_size": 2, "activation": "linear"},
    ]
    flat = np.arange(26, dtype=np.float64) * 0.01

    json_path = tmp_path / "window_model.json"
    aerocapture_rs.flat_weights_to_json(
        architecture_json=json.dumps(architecture),
        flat_weights=flat,
        output_interpretation="atan2",
        input_mask=None,
        ablated_input=None,
        path=str(json_path),
    )

    loaded = json.loads(json_path.read_text())
    assert loaded["format_version"] == 2
    assert len(loaded["architecture"]) == 2

    # Window entry is spec-only -- no weights dict under weights["layer_0"].
    window_entry = loaded["architecture"][0]
    assert window_entry == {"type": "window", "input_size": 4, "n_steps": 3}
    assert "layer_0" not in loaded.get("weights", {})

    # Dense entry has standard w/b keys.
    dense_entry = loaded["architecture"][1]
    assert dense_entry["type"] == "dense"
    assert "layer_1" in loaded["weights"]
    assert "w" in loaded["weights"]["layer_1"]
    assert "b" in loaded["weights"]["layer_1"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_flat_weights_to_json_window.py -v
```

Expected: FAIL — either the json format error surfaces, or the test fails at an assertion (Rust already supports Window at this point via Tasks 1-4).

If the test passes immediately: that's fine — it's a regression guard. Move on to Step 3.

- [ ] **Step 3: If test fails, confirm the failure is Window-specific (not infrastructure)**

If the Rust side is correctly dispatching Window (Task 4 landed), this test should pass. No Rust changes needed. If it fails for an infrastructure reason, diagnose in Task 4 and iterate.

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_flat_weights_to_json_window.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_flat_weights_to_json_window.py
git commit -m "test(pyo3): flat_weights_to_json handles Window architecture"
```

---

## Task 7: Python WindowSpec + LayerSpec union update

**Files:**
- Modify: `src/python/aerocapture/training/rl/schemas.py`
- Test: `tests/test_window_spec.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_window_spec.py`:

```python
"""WindowSpec pydantic validation + LayerSpec discriminated union dispatch."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from aerocapture.training.rl.schemas import LayerSpec, WindowSpec


def test_window_spec_constructs_with_required_fields() -> None:
    spec = WindowSpec(type="window", input_size=4, n_steps=8)
    assert spec.input_size == 4
    assert spec.n_steps == 8


def test_window_spec_rejects_zero_fields() -> None:
    with pytest.raises(ValidationError):
        WindowSpec(type="window", input_size=0, n_steps=8)
    with pytest.raises(ValidationError):
        WindowSpec(type="window", input_size=4, n_steps=0)


def test_window_spec_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        WindowSpec(type="window", input_size=4, n_steps=8, activation="tanh")  # type: ignore[call-arg]


def test_layer_spec_discriminator_dispatches_to_window() -> None:
    adapter = TypeAdapter(LayerSpec)
    parsed = adapter.validate_python({"type": "window", "input_size": 4, "n_steps": 8})
    assert isinstance(parsed, WindowSpec)
    assert parsed.input_size == 4
    assert parsed.n_steps == 8
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_window_spec.py -v
```

Expected: FAIL — `WindowSpec` does not exist.

- [ ] **Step 3: Add WindowSpec + extend the union**

In `src/python/aerocapture/training/rl/schemas.py`:

```python
class WindowSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["window"]
    input_size: int = Field(ge=1)
    n_steps: int = Field(ge=1)


LayerSpec = Annotated[DenseSpec | GruSpec | LstmSpec | WindowSpec, Discriminator("type")]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_window_spec.py -v
```

Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/rl/schemas.py tests/test_window_spec.py
git commit -m "feat(nn): WindowSpec pydantic + LayerSpec discriminated union"
```

---

## Task 8: Python WindowLayer torch module

**Files:**
- Create: `src/python/aerocapture/training/rl/layers/window.py`
- Test: `tests/test_window_layer_python.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_window_layer_python.py`:

```python
"""Python WindowLayer torch module -- forward contract + new_state dtype tracking."""

from __future__ import annotations

import torch

from aerocapture.training.rl.layers.window import WindowLayer


def test_window_forward_rolls_buffer_and_concatenates() -> None:
    layer = WindowLayer(input_size=2, n_steps=3).double()
    state = layer.new_state(batch_size=1)  # (1, 3, 2) zeros
    assert state.shape == (1, 3, 2)
    assert torch.all(state == 0.0)

    x0 = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    out0, state = layer.forward(x0, state)
    # Output shape: (batch, n_steps * input_size) = (1, 6).
    assert out0.shape == (1, 6)
    # Buffer after tick 0: [[0,0], [0,0], [1,2]].
    assert torch.equal(out0, torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 2.0]], dtype=torch.float64))

    x1 = torch.tensor([[3.0, 4.0]], dtype=torch.float64)
    out1, state = layer.forward(x1, state)
    assert torch.equal(out1, torch.tensor([[0.0, 0.0, 1.0, 2.0, 3.0, 4.0]], dtype=torch.float64))

    x2 = torch.tensor([[5.0, 6.0]], dtype=torch.float64)
    out2, state = layer.forward(x2, state)
    assert torch.equal(out2, torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=torch.float64))


def test_window_new_state_respects_module_dtype() -> None:
    layer = WindowLayer(input_size=4, n_steps=2)
    # Default f32.
    state_f32 = layer.new_state(batch_size=2)
    assert state_f32.dtype == torch.float32

    # After .double().
    layer.double()
    state_f64 = layer.new_state(batch_size=2)
    assert state_f64.dtype == torch.float64


def test_window_has_zero_trainable_parameters() -> None:
    layer = WindowLayer(input_size=4, n_steps=8)
    n_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    assert n_params == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_window_layer_python.py -v
```

Expected: FAIL — `WindowLayer` module does not exist.

- [ ] **Step 3: Implement WindowLayer torch module**

Create `src/python/aerocapture/training/rl/layers/window.py`:

```python
"""Window-MLP layer torch mirror.

Zero trainable parameters. Maintains a FIFO ring buffer of the last `n_steps`
inputs and concatenates them into a vector of length `n_steps * input_size`
for the next Dense layer.

Used by the cross-language equivalence test only. V2Policy does NOT construct
this layer: `build_layer(WindowSpec)` raises NotImplementedError because
Window-MLP is PSO-only in Phase 2b (PSO bypasses V2Policy and uses the Rust
runtime directly via aerocapture_rs.nn_forward).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class WindowLayer(nn.Module):
    def __init__(self, input_size: int, n_steps: int) -> None:
        super().__init__()
        if input_size <= 0 or n_steps <= 0:
            raise ValueError(
                f"WindowLayer input_size and n_steps must be positive "
                f"(got input_size={input_size}, n_steps={n_steps})"
            )
        self.input_size = input_size
        self.n_steps = n_steps
        # Zero-param layers still need a dtype/device anchor for new_state.
        # A non-persistent buffer is the idiomatic torch approach (doesn't
        # appear in state_dict, does participate in .double() / .to()).
        self.register_buffer("_dtype_anchor", torch.zeros(1), persistent=False)

    def forward(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        # x:     (batch, input_size)
        # state: (batch, n_steps, input_size)
        assert x.shape[-1] == self.input_size, \
            f"WindowLayer expected input_size={self.input_size}, got {x.shape[-1]}"
        assert state.shape[1:] == (self.n_steps, self.input_size), \
            f"WindowLayer expected state shape (_, {self.n_steps}, {self.input_size}), " \
            f"got {tuple(state.shape)}"
        # Roll: drop the oldest slot, append the new input.
        new_state = torch.cat([state[:, 1:], x.unsqueeze(1)], dim=1)
        out = new_state.reshape(x.shape[0], -1)
        return out, new_state

    def new_state(self, batch_size: int) -> Tensor:
        return torch.zeros(
            batch_size,
            self.n_steps,
            self.input_size,
            dtype=self._dtype_anchor.dtype,
            device=self._dtype_anchor.device,
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_window_layer_python.py -v
```

Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/rl/layers/window.py tests/test_window_layer_python.py
git commit -m "feat(nn): Python WindowLayer torch module (zero-param, FIFO buffer)"
```

---

## Task 9: build_layer PPO-rejection guard + layers/__init__ export

**Files:**
- Modify: `src/python/aerocapture/training/rl/layers/__init__.py`
- Test: `tests/test_window_build_layer_rejection.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_window_build_layer_rejection.py`:

```python
"""build_layer(WindowSpec) raises NotImplementedError (PPO-rejection guard)."""

from __future__ import annotations

import pytest

from aerocapture.training.rl.layers import WindowLayer, build_layer
from aerocapture.training.rl.schemas import WindowSpec


def test_window_layer_is_exported_from_layers_module() -> None:
    # WindowLayer itself is exported for the cross-language equivalence test.
    assert WindowLayer is not None


def test_build_layer_raises_on_window_spec() -> None:
    spec = WindowSpec(type="window", input_size=4, n_steps=8)
    with pytest.raises(NotImplementedError) as exc_info:
        build_layer(spec)
    assert "Window-MLP is PSO-only" in str(exc_info.value)
    assert "docs/superpowers/specs/2026-04-20-phase-2b-window-mlp-design.md" in str(exc_info.value)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_window_build_layer_rejection.py -v
```

Expected: FAIL — `WindowLayer` not importable from layers package; `build_layer` doesn't handle WindowSpec.

- [ ] **Step 3: Update layers/__init__.py**

Edit `src/python/aerocapture/training/rl/layers/__init__.py`:

```python
"""Torch mirrors of Rust layer types. One file per layer variant."""

from __future__ import annotations

from torch import nn

from aerocapture.training.rl.layers.dense import DenseLayer
from aerocapture.training.rl.layers.gru import GruLayer
from aerocapture.training.rl.layers.lstm import LstmLayer
from aerocapture.training.rl.layers.window import WindowLayer
from aerocapture.training.rl.schemas import (
    DenseSpec,
    GruSpec,
    LayerSpec,
    LstmSpec,
    WindowSpec,
)

__all__ = ["DenseLayer", "GruLayer", "LstmLayer", "WindowLayer", "build_layer"]


def build_layer(spec: LayerSpec) -> nn.Module:
    """Dispatch a LayerSpec to its torch module constructor."""
    if isinstance(spec, DenseSpec):
        return DenseLayer(spec.input_size, spec.output_size, spec.activation)
    if isinstance(spec, GruSpec):
        return GruLayer(spec.input_size, spec.hidden_size)
    if isinstance(spec, LstmSpec):
        return LstmLayer(spec.input_size, spec.hidden_size)
    if isinstance(spec, WindowSpec):
        raise NotImplementedError(
            "Window-MLP is PSO-only in Phase 2b; PPO use deferred. "
            "See docs/superpowers/specs/2026-04-20-phase-2b-window-mlp-design.md"
        )
    raise ValueError(f"Unknown layer spec: {spec!r}")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_window_build_layer_rejection.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/rl/layers/__init__.py tests/test_window_build_layer_rejection.py
git commit -m "feat(rl): build_layer PPO-rejection guard for WindowSpec"
```

---

## Task 10: config.py _layer_n_params + _layer_output_size + describe_architecture Window arms

**Files:**
- Modify: `src/python/aerocapture/training/config.py`
- Test: `tests/test_window_config_helpers.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_window_config_helpers.py`:

```python
"""config.py Window arms for _layer_n_params, _layer_output_size, describe_architecture."""

from __future__ import annotations

from aerocapture.training.config import (
    NetworkConfig,
    _layer_n_params,
    _layer_output_size,
    describe_architecture,
)


def test_layer_n_params_window_returns_zero() -> None:
    entry = {"type": "window", "input_size": 16, "n_steps": 8}
    assert _layer_n_params(entry) == 0


def test_layer_output_size_window_is_input_times_n_steps() -> None:
    entry = {"type": "window", "input_size": 16, "n_steps": 8}
    assert _layer_output_size(entry) == 128

    entry = {"type": "window", "input_size": 4, "n_steps": 3}
    assert _layer_output_size(entry) == 12


def test_describe_architecture_renders_window_layer() -> None:
    architecture = [
        {"type": "window", "input_size": 16, "n_steps": 8},
        {"type": "dense", "input_size": 128, "output_size": 32, "activation": "swish"},
    ]
    net = NetworkConfig(
        layer_sizes=[],
        activations=[],
        input_mask=None,
        architecture=architecture,
        n_base_coef=128 * 32 + 32,
        n_input=16,
        n_output=32,
    )
    s = describe_architecture(net, output_interpretation="atan2")
    assert "window" in s
    assert "16" in s  # input_size
    assert "128" in s  # output = n_steps * input_size
    assert "n_steps=8" in s
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_window_config_helpers.py -v
```

Expected: FAIL — `_layer_n_params`, `_layer_output_size`, and `describe_architecture` all raise `Unknown v2 layer type: 'window'`.

- [ ] **Step 3: Add Window arms to config.py helpers**

In `src/python/aerocapture/training/config.py`:

`_layer_n_params`:

```python
if ltype == "window":
    return 0  # zero trainable parameters
```

(Insert after the `"lstm"` arm, before `raise ValueError`.)

`_layer_output_size`:

```python
if ltype == "window":
    return int(entry["input_size"]) * int(entry["n_steps"])
```

(Insert after the `"lstm"` arm, before `raise ValueError`.)

In `describe_architecture`, extend the per-layer rendering block:

```python
for i, entry in enumerate(network.architecture):
    ltype = entry["type"]
    in_size = _layer_input_size(entry)
    out_size = _layer_output_size(entry)
    if ltype == "dense":
        tail = entry.get("activation", "?")
    elif ltype in ("gru", "lstm"):
        tail = f"hidden_size={entry['hidden_size']}"
    elif ltype == "window":
        tail = f"n_steps={entry['n_steps']}"
    else:
        tail = ltype
    lines.append(f"  layer {i}: {ltype:<6} {in_size:>4} -> {out_size:<4} {tail}")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_window_config_helpers.py -v
```

Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/config.py tests/test_window_config_helpers.py
git commit -m "feat(nn): config helpers _layer_n_params + _layer_output_size + describe_architecture Window arms"
```

---

## Task 11: encoding._layer_param_specs Window arm (returns [])

**Files:**
- Modify: `src/python/aerocapture/training/encoding.py`
- Test: `tests/test_window_param_specs.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_window_param_specs.py`:

```python
"""encoding._layer_param_specs returns [] for WindowSpec."""

from __future__ import annotations

from aerocapture.training.encoding import _layer_param_specs, nn_param_specs_from_v2
from aerocapture.training.rl.schemas import DenseSpec, WindowSpec


def test_layer_param_specs_window_returns_empty() -> None:
    spec = WindowSpec(type="window", input_size=16, n_steps=8)
    specs = _layer_param_specs(spec, layer_idx=0, bound_multiplier=2.0)
    assert specs == []


def test_nn_param_specs_from_v2_handles_mixed_window_dense() -> None:
    architecture = [
        WindowSpec(type="window", input_size=4, n_steps=3),
        DenseSpec(type="dense", input_size=12, output_size=2, activation="linear"),
    ]
    specs = nn_param_specs_from_v2(architecture, bound_multiplier=2.0)
    # Only the Dense layer contributes: 12*2 weights + 2 biases = 26 specs.
    assert len(specs) == 26
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_window_param_specs.py -v
```

Expected: FAIL — `_layer_param_specs` raises `Unknown layer type for PSO specs: WindowSpec(...)`.

- [ ] **Step 3: Add WindowSpec arm to _layer_param_specs**

In `src/python/aerocapture/training/encoding.py`, import `WindowSpec` and extend the dispatch:

```python
from aerocapture.training.rl.schemas import DenseSpec, GruSpec, LayerSpec, LstmSpec, WindowSpec


def _layer_param_specs(layer: LayerSpec, layer_idx: int, bound_multiplier: float) -> list[ParamSpec]:
    if isinstance(layer, DenseSpec):
        return _dense_specs(layer, layer_idx, bound_multiplier)
    if isinstance(layer, GruSpec):
        return _gru_specs(layer, layer_idx, bound_multiplier)
    if isinstance(layer, LstmSpec):
        return _lstm_specs(layer, layer_idx, bound_multiplier)
    if isinstance(layer, WindowSpec):
        return []  # zero trainable params
    msg = f"Unknown layer type for PSO specs: {layer!r}"
    raise ValueError(msg)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_window_param_specs.py -v
```

Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/encoding.py tests/test_window_param_specs.py
git commit -m "feat(nn): encoding._layer_param_specs Window arm returns []"
```

---

## Task 12: init_v2_population Window no-op branch

**Files:**
- Modify: `src/python/aerocapture/training/initialization_v2.py`
- Test: `tests/test_init_v2_population.py` (add Window case)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_init_v2_population.py`:

```python
def test_init_v2_population_skips_window_layer_with_zero_params() -> None:
    from aerocapture.training.initialization_v2 import init_v2_population

    architecture = [
        {"type": "window", "input_size": 4, "n_steps": 3},
        {"type": "dense", "input_size": 12, "output_size": 2, "activation": "linear"},
    ]
    rng = np.random.default_rng(42)
    pop = init_v2_population(architecture, n_pop=8, bound_multiplier=2.0, rng=rng)

    # Total params = 0 (window) + 12*2 + 2 (dense) = 26.
    assert pop.shape == (8, 26)
    # All values are finite and within the expected Dense bound.
    assert np.all(np.isfinite(pop))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_init_v2_population.py::test_init_v2_population_skips_window_layer_with_zero_params -v
```

Expected: FAIL — `init_v2_population: unknown layer type 'window'`.

- [ ] **Step 3: Add Window no-op arm to _fill_layer**

In `src/python/aerocapture/training/initialization_v2.py`, extend `_fill_layer`:

```python
def _fill_layer(entry: dict, slab: np.ndarray, bound_multiplier: float, rng: np.random.Generator) -> None:
    t = entry["type"]
    if t == "dense":
        _fill_dense(entry, slab, bound_multiplier, rng)
    elif t == "gru":
        _fill_gru(entry, slab, bound_multiplier, rng)
    elif t == "lstm":
        _fill_lstm(entry, slab, bound_multiplier, rng)
    elif t == "window":
        # Zero trainable params: slab has width 0, nothing to fill.
        # _layer_n_params(window) == 0 makes the outer cursor step by 0,
        # so this is a no-op branch that only exists to stay off the
        # `raise ValueError` path.
        assert slab.shape[1] == 0, f"window slab expected 0-width, got {slab.shape[1]}"
    else:
        raise ValueError(f"init_v2_population: unknown layer type {t!r}")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_init_v2_population.py::test_init_v2_population_skips_window_layer_with_zero_params -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/initialization_v2.py tests/test_init_v2_population.py
git commit -m "feat(init): init_v2_population Window no-op branch"
```

---

## Task 13: export_v2_policy_to_json Window branch + obs-norm guard

**Files:**
- Modify: `src/python/aerocapture/training/rl/export.py`
- Test: `tests/test_window_export.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_window_export.py`:

```python
"""export_v2_policy_to_json obs-norm guard rejects Window as layer 0."""

from __future__ import annotations

from pathlib import Path

import pytest

# This test only exercises the export guard; it does NOT build a V2Policy
# (that would go through build_layer, which raises for Window). We construct
# a minimal stand-in by mocking policy.architecture to bypass the build step.


def test_export_obs_norm_rejects_window_as_layer_0(tmp_path: Path) -> None:
    from aerocapture.training.rl.export import export_v2_policy_to_json
    from aerocapture.training.rl.schemas import DenseSpec, WindowSpec
    from unittest.mock import MagicMock

    # Fake policy with Window as first layer spec. The export guard should
    # fire before it inspects policy.layers.
    policy = MagicMock()
    policy.architecture = [
        WindowSpec(type="window", input_size=4, n_steps=3),
        DenseSpec(type="dense", input_size=12, output_size=2, activation="linear"),
    ]
    obs_normalizer = MagicMock()  # non-None triggers the guard

    with pytest.raises(NotImplementedError) as exc_info:
        export_v2_policy_to_json(
            policy, str(tmp_path / "out.json"), obs_normalizer=obs_normalizer
        )
    assert "WindowSpec" in str(exc_info.value) or "Window" in str(exc_info.value)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_window_export.py -v
```

Expected: FAIL — obs-norm guard currently only rejects GruSpec/LstmSpec.

- [ ] **Step 3: Extend obs-norm guard + Window export arm**

In `src/python/aerocapture/training/rl/export.py`:

First, extend the obs-norm bake-in guard to include `WindowSpec`:

```python
if obs_normalizer is not None and isinstance(
    policy.architecture[0], (GruSpec, LstmSpec, WindowSpec)
):
    raise NotImplementedError(
        f"Obs normalizer bake-in into layer 0 is only supported for DenseSpec, "
        f"got {type(policy.architecture[0]).__name__}. Export without the bake-in."
    )
```

Import `WindowSpec` at the top of the file.

Then, in the per-layer export loop, add a Window branch that writes a spec-only entry (no weights dict):

```python
for i, (spec, layer) in enumerate(zip(policy.architecture, policy.layers)):
    if isinstance(spec, WindowSpec):
        architecture_json.append({
            "type": "window",
            "input_size": spec.input_size,
            "n_steps": spec.n_steps,
        })
        # No weight entry for this layer -- it is zero-param.
        continue
    # ... existing Dense / GRU / LSTM arms ...
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_window_export.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/rl/export.py tests/test_window_export.py
git commit -m "feat(rl): export_v2_policy_to_json Window branch + obs-norm guard"
```

---

## Task 14: load_policy_from_json Window rejection

**Files:**
- Modify: `src/python/aerocapture/training/model_io.py`
- Test: `tests/test_window_ppo_rejection.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_window_ppo_rejection.py`:

```python
"""load_policy_from_json rejects v2 JSON with Window layers (PPO-rejection)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from aerocapture.training.model_io import load_policy_from_json


def test_load_policy_from_json_rejects_window_architecture(tmp_path: Path) -> None:
    # Minimal v2 JSON with a Window first layer.
    arch_json = {
        "format_version": 2,
        "architecture": [
            {"type": "window", "input_size": 4, "n_steps": 3},
            {"type": "dense", "input_size": 12, "output_size": 2, "activation": "linear"},
        ],
        "weights": {
            "layer_1": {
                "w": [[0.01] * 12, [0.01] * 12],
                "b": [0.0, 0.0],
            }
        },
        "output_interpretation": "atan2",
        "input_mask": None,
        "ablated_input": None,
    }
    json_path = tmp_path / "window.json"
    json_path.write_text(json.dumps(arch_json))

    with pytest.raises(NotImplementedError) as exc_info:
        load_policy_from_json(str(json_path), device=torch.device("cpu"))
    msg = str(exc_info.value)
    assert "Window" in msg
    assert "PSO-only" in msg
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_window_ppo_rejection.py -v
```

Expected: FAIL — `load_policy_from_json` does not short-circuit on Window.

- [ ] **Step 3: Add Window-rejection branch to load_policy_from_json**

In `src/python/aerocapture/training/model_io.py`:

Import `WindowSpec` at the top.

Near the top of `load_policy_from_json`, right after `architecture` is parsed via `TypeAdapter`, add:

```python
if any(isinstance(spec, WindowSpec) for spec in architecture):
    raise NotImplementedError(
        "Window-MLP is PSO-only in Phase 2b; load_policy_from_json is a PPO/SAC entry "
        "point that cannot construct V2Policy with Window layers. "
        "See docs/superpowers/specs/2026-04-20-phase-2b-window-mlp-design.md"
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_window_ppo_rejection.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/model_io.py tests/test_window_ppo_rejection.py
git commit -m "feat(rl): load_policy_from_json Window rejection guard"
```

---

## Task 15: Cross-language equivalence test (stateful)

**Files:**
- Create: `tests/test_rust_python_window_equivalence.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_rust_python_window_equivalence.py`:

```python
"""Rust <-> Python equivalence for Window + Dense architectures.

Machine-epsilon forward equality across a 100-step sequence using
aerocapture_rs.nn_forward_sequence (single-threaded NnState) vs explicit
Python forward with (batch=1, n_steps, input_size) state tensor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("aerocapture_rs")
import aerocapture_rs

from aerocapture.training.rl.layers.dense import DenseLayer
from aerocapture.training.rl.layers.window import WindowLayer


def test_rust_python_window_stateful_equivalence(tmp_path: Path) -> None:
    # Architecture: Window(4, 4) -> Dense(16, 4, tanh) -> Dense(4, 2, linear).
    input_size = 4
    n_steps = 4
    flat_size = input_size * n_steps  # 16

    # Build Python modules (f64).
    window = WindowLayer(input_size=input_size, n_steps=n_steps).double()
    dense1 = DenseLayer(flat_size, 4, activation="tanh").double()
    dense2 = DenseLayer(4, 2, activation="linear").double()

    # Randomize Dense weights.
    rng = np.random.default_rng(2026)
    with torch.no_grad():
        dense1.weight.copy_(torch.tensor(rng.normal(0.0, 0.3, (4, flat_size)), dtype=torch.float64))
        dense1.bias.copy_(torch.tensor(rng.normal(0.0, 0.1, (4,)), dtype=torch.float64))
        dense2.weight.copy_(torch.tensor(rng.normal(0.0, 0.3, (2, 4)), dtype=torch.float64))
        dense2.bias.copy_(torch.tensor(rng.normal(0.0, 0.1, (2,)), dtype=torch.float64))

    # Flat weights in the canonical order for Rust loading:
    # Window: [] (zero params)
    # Dense1: row-major (out, in) W, then b
    # Dense2: row-major (out, in) W, then b
    flat = np.concatenate([
        dense1.weight.detach().numpy().reshape(-1),
        dense1.bias.detach().numpy().reshape(-1),
        dense2.weight.detach().numpy().reshape(-1),
        dense2.bias.detach().numpy().reshape(-1),
    ]).astype(np.float64)

    architecture = [
        {"type": "window", "input_size": input_size, "n_steps": n_steps},
        {"type": "dense", "input_size": flat_size, "output_size": 4, "activation": "tanh"},
        {"type": "dense", "input_size": 4, "output_size": 2, "activation": "linear"},
    ]
    json_path = tmp_path / "window_model.json"
    aerocapture_rs.flat_weights_to_json(
        architecture_json=json.dumps(architecture),
        flat_weights=flat,
        output_interpretation="atan2",
        input_mask=None,
        ablated_input=None,
        path=str(json_path),
    )

    # 100 random f64 inputs.
    sequence = rng.normal(0.0, 1.0, (100, input_size)).astype(np.float64)

    # Rust: one NnState threaded across the full sequence.
    rust_out = aerocapture_rs.nn_forward_sequence(str(json_path), sequence)
    rust_out = np.asarray(rust_out, dtype=np.float64)
    assert rust_out.shape == (100, 2)

    # Python: explicit buffer state threaded across the sequence.
    state = window.new_state(batch_size=1)  # (1, n_steps, input_size)
    py_out = np.empty((100, 2), dtype=np.float64)
    for t in range(100):
        x = torch.tensor(sequence[t : t + 1], dtype=torch.float64)  # (1, input_size)
        w_out, state = window.forward(x, state)                      # (1, flat_size)
        y = dense1(w_out)
        y = dense2(y)
        py_out[t] = y.detach().numpy().reshape(-1)

    max_abs_diff = float(np.max(np.abs(rust_out - py_out)))
    assert max_abs_diff < 1e-10, f"max abs diff = {max_abs_diff:e} exceeds 1e-10"

    # Target: machine epsilon (~1e-15 or tighter, consistent with GRU 4.4e-16 and LSTM ~1e-16).
    print(f"Window equivalence max abs diff = {max_abs_diff:e}")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_rust_python_window_equivalence.py -v -s
```

Expected: FAIL initially — either `nn_forward_sequence` errors (Rust might not dispatch window correctly) or max_abs_diff is non-zero.

If the test fails on a Rust forward issue: diagnose in Task 2/4 and iterate.

- [ ] **Step 3: Fix any issues, re-run**

Typical issues to check:
- Weight flat order (Window has zero entries, so the next Dense's flat slice starts at offset 0).
- Pre-fill convention (both sides must initialize the buffer with zero vectors).
- Buffer roll direction (both must pop oldest / push newest).

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_rust_python_window_equivalence.py -v -s
```

Expected: PASS. Print line should show max abs diff ~1e-15 or smaller.

- [ ] **Step 5: Commit**

```bash
git add tests/test_rust_python_window_equivalence.py
git commit -m "test(nn): cross-language Window equivalence at machine epsilon"
```

---

## Task 16: PSO smoke test + python-pyo3 CI wiring

**Files:**
- Create: `tests/test_window_pso_smoke.py`
- Modify: `.github/workflows/ci.yml` (python-pyo3 job test list)

- [ ] **Step 1: Write the PSO smoke test**

Create `tests/test_window_pso_smoke.py`:

```python
"""PSO training smoke test for Window-MLP (2 gens, reduced arch, ~40 params)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("aerocapture_rs")
import aerocapture_rs


@pytest.mark.slow
def test_window_pso_two_gens(tmp_path: Path) -> None:
    from aerocapture.training.encoding import (
        decode_normalized_array,
        nn_param_specs_from_v2,
    )
    from aerocapture.training.initialization_v2 import init_v2_population
    from aerocapture.training.rl.schemas import DenseSpec, WindowSpec

    # Reduced arch: Window(4, 4) -> Dense(16, 4, swish) -> Dense(4, 2, linear).
    architecture_specs = [
        WindowSpec(type="window", input_size=4, n_steps=4),
        DenseSpec(type="dense", input_size=16, output_size=4, activation="swish"),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    architecture_dicts = [s.model_dump() for s in architecture_specs]

    # n_params = 0 + (16*4 + 4) + (4*2 + 2) = 0 + 68 + 10 = 78
    param_specs = nn_param_specs_from_v2(architecture_specs, bound_multiplier=2.0)
    assert len(param_specs) == 78

    # Initial population.
    rng = np.random.default_rng(1234)
    pop_physical = init_v2_population(
        architecture_dicts, n_pop=4, bound_multiplier=2.0, rng=rng
    )
    assert pop_physical.shape == (4, 78)
    assert np.all(np.isfinite(pop_physical))

    # Serialize the first individual to JSON v2 and verify nn_forward returns finite 2-tuple.
    best_flat = pop_physical[0].astype(np.float64)
    json_path = tmp_path / "window_pso_best.json"
    aerocapture_rs.flat_weights_to_json(
        architecture_json=json.dumps(architecture_dicts),
        flat_weights=best_flat,
        output_interpretation="atan2",
        input_mask=None,
        ablated_input=None,
        path=str(json_path),
    )

    # Verify JSON schema.
    loaded = json.loads(json_path.read_text())
    assert loaded["format_version"] == 2
    assert [e["type"] for e in loaded["architecture"]] == ["window", "dense", "dense"]

    # Run a single Rust forward to confirm the model is valid.
    obs = np.zeros(4, dtype=np.float64)
    out = aerocapture_rs.nn_forward(str(json_path), obs)
    out_arr = np.asarray(out, dtype=np.float64)
    assert out_arr.shape == (2,)
    assert np.all(np.isfinite(out_arr))
```

- [ ] **Step 2: Run the smoke test**

```bash
uv run pytest tests/test_window_pso_smoke.py -v -m slow
```

Expected: PASS in < 3 s wall-clock.

- [ ] **Step 3: Add the new test files to the CI python-pyo3 job**

Edit `.github/workflows/ci.yml`. Find the python-pyo3 job's pytest invocation (already listing `test_pyo3.py`, `test_v2_rust_python_equivalence.py`, `test_gru_pso_smoke.py`, `test_gru_ppo_smoke.py`, plus the LSTM tests). Append:

```yaml
            tests/test_flat_weights_to_json_window.py
            tests/test_rust_python_window_equivalence.py
            tests/test_window_pso_smoke.py
```

Also add a Window-rejection test to the main python job (it doesn't need PyO3):

```yaml
            tests/test_window_ppo_rejection.py
            tests/test_window_build_layer_rejection.py
            tests/test_window_spec.py
            tests/test_window_layer_python.py
            tests/test_window_config_helpers.py
            tests/test_window_param_specs.py
            tests/test_window_export.py
```

(Exact placement depends on the current ci.yml layout. If the main job uses `pytest tests/` broadly, no new entry is needed -- the tests auto-discover. Only the python-pyo3 job needs explicit listing.)

- [ ] **Step 4: Commit**

```bash
git add tests/test_window_pso_smoke.py .github/workflows/ci.yml
git commit -m "test(nn): Window PSO smoke + python-pyo3 CI wiring"
```

---

## Task 17: Training config + compare_guidance / train_all.sh registration

**Files:**
- Create: `configs/training/msr_aller_window_pso_train.toml`
- Modify: `src/python/aerocapture/training/compare_guidance.py`
- Modify: `train_all.sh`

- [ ] **Step 1: Create the training config**

Create `configs/training/msr_aller_window_pso_train.toml`:

```toml
# Phase 2b Window-MLP PSO training config.
# Arch: Window(16, 8) -> Dense(128 -> 32, swish) -> Dense(32 -> 8, swish) -> Dense(8 -> 2, linear).
# Trainable param count: 128*32 + 32 + 32*8 + 8 + 8*2 + 2 = 4410.

base = [
    "./common.toml",
    "../missions/mars.toml",
]

[simulation]
mission = "msr_aller"
guidance = "neural_network"

[data]
neural_network = "training_output/neural_network_window_pso/best_model.json"

[network]
# 16-input baseline (orbital + aero + thermal state), matching the MLP baseline config.
# Window provides history -- no reference trajectory or bounce-gated exit inputs.
input_mask = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
output_interpretation = "atan2"

[[network.architecture]]
type = "window"
input_size = 16
n_steps = 8

[[network.architecture]]
type = "dense"
input_size = 128
output_size = 32
activation = "swish"

[[network.architecture]]
type = "dense"
input_size = 32
output_size = 8
activation = "swish"

[[network.architecture]]
type = "dense"
input_size = 8
output_size = 2
activation = "linear"

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

[optimizer.pso]
# Defaults from common.toml; mirror gru_pso_train / lstm_pso_train choices.
w = 0.5
c1 = 1.5
c2 = 1.5
```

- [ ] **Step 2: Register neural_network_window_pso in compare_guidance**

Edit `src/python/aerocapture/training/compare_guidance.py`. Add `neural_network_window_pso` to:
- `SCHEMES` list
- `SCHEME_TRAINING_CONFIGS` dict (pointing to the new TOML)
- `_NN_DEPLOY_SCHEMES` set

Search the existing `neural_network_lstm_pso` entries and mirror them exactly for the Window case.

- [ ] **Step 3: Register window_pso / nn_window_pso / window aliases in train_all.sh**

Edit `train_all.sh`. Find the `case` block that maps aliases to schemes (around where `lstm_pso` is mapped). Append:

```bash
    window_pso|nn_window_pso|window)
        scheme="neural_network_window_pso"
        ;;
```

- [ ] **Step 4: Sanity-check the config loads and describes correctly**

```bash
uv run python -c "
from aerocapture.training.config import load_training_config, describe_architecture
cfg = load_training_config('configs/training/msr_aller_window_pso_train.toml')
print(describe_architecture(cfg.network))
"
```

Expected output starts with `Network architecture (4410 params):` followed by the Window + Dense layer list.

- [ ] **Step 5: Verify compare_guidance registration**

```bash
uv run python -c "
from aerocapture.training.compare_guidance import SCHEMES, SCHEME_TRAINING_CONFIGS, _NN_DEPLOY_SCHEMES
assert 'neural_network_window_pso' in SCHEMES
assert SCHEME_TRAINING_CONFIGS['neural_network_window_pso'].endswith('msr_aller_window_pso_train.toml')
assert 'neural_network_window_pso' in _NN_DEPLOY_SCHEMES
print('OK')
"
```

Expected: `OK`.

- [ ] **Step 6: Verify train_all.sh alias**

```bash
bash -c "source train_all.sh --dry-run window_pso 2>&1 | head" || true
# Or just grep:
grep -A 2 "window_pso" train_all.sh
```

Expected: the alias block is present.

- [ ] **Step 7: Commit**

```bash
git add configs/training/msr_aller_window_pso_train.toml \
        src/python/aerocapture/training/compare_guidance.py \
        train_all.sh
git commit -m "feat(configs): Phase 2b Window-MLP PSO training config + aliases"
```

---

## Task 18: Final verification + CI + TODO/CLAUDE.md sync

**Files:**
- Modify: `TODO.md`
- Modify: `CLAUDE.md` (Phase 2b sync in the Stateful NN Runtime section + extensibility contract)

- [ ] **Step 1: Run full Rust check**

```bash
./check_all.sh
```

Expected: all green -- `cargo test`, `cargo fmt --check`, `cargo clippy`, release build. All 10 guidance golden regressions bit-identical.

- [ ] **Step 2: Run full Python test suite (excluding @slow)**

```bash
uv run pytest tests/ -v --deselect tests/test_lstm_pso_smoke.py --deselect tests/test_lstm_ppo_smoke.py --deselect tests/test_gru_pso_smoke.py --deselect tests/test_gru_ppo_smoke.py --deselect tests/test_window_pso_smoke.py
```

(Or simply `uv run pytest tests/ -v -m "not slow"`.)

Expected: all pass. Ruff + mypy clean via `./lint_code.sh`.

- [ ] **Step 3: Run slow tests individually to confirm CI wiring**

```bash
uv run pytest tests/test_window_pso_smoke.py tests/test_rust_python_window_equivalence.py tests/test_flat_weights_to_json_window.py -v
```

Expected: all 3 pass.

- [ ] **Step 4: Update TODO.md (mark Phase 2b done)**

In `TODO.md`, replace the `[DOING 2026-04-20 on feature/window-mlp]` marker with `[DONE 2026-04-XX]` (use actual landing date), check off all the nested items, and add a shipping summary block mirroring Phase 2a's format:

```markdown
### Phase 2b -- Window-MLP (PSO only, ring buffer, no new matmul) [DONE 2026-04-XX]

Shipped on branch `feature/window-mlp` (N commits on top of Phase 2a).
Cross-language equivalence: Window forward matches at machine epsilon.
PSO-Window smoke + PPO-rejection guard tests wired into CI.

- [x] Rust `WindowLayer` + `Layer::Window` + `LayerSpec::Window` + `LayerState::Window { buffer: VecDeque<Vec<f64>> }` + `TomlLayerSpec::Window`
- [x] `LayerWeights for WindowLayer` zero-param impl + JSON v2 + PyO3 test
- [x] Python `WindowLayer` torch module + `WindowSpec` pydantic + `build_layer` PPO-rejection guard
- [x] `_layer_param_specs` / `_layer_n_params` / `_layer_output_size` Window arms + `init_v2_population` no-op branch
- [x] Training config `msr_aller_window_pso_train.toml` + `compare_guidance` + `train_all.sh` registration
- [x] Cross-language equivalence test + PSO smoke test + PPO-rejection test

Spec: `docs/superpowers/specs/2026-04-20-phase-2b-window-mlp-design.md`.
Plan: `docs/superpowers/plans/2026-04-20-phase-2b-window-mlp-plan.md`.

**Out-of-Phase-2b carry-overs (still deferred):**
- [ ] PPO-BPTT for Window (Phase 2b.5 if reviewers request; requires ndim dispatch for (B, n_steps, input_size) buffer state).
- [ ] SAC-GRU / SAC-LSTM / SAC-Window (Phase 1.6; SAC stays on GaussianPolicy).
- [ ] Recurrent critic (Phase 1.5 carry-over).
- [ ] Widen `load_policy_from_json` to accept v1 JSON (Phase 0 carry-over).
- [ ] Fix pre-existing clippy warnings in `src/rust/aerocapture-py/src/lib.rs`.

**Closed by Phase 2b:**
- [x] Zero-trainable-parameter scalar-state layer locked in as a supported case (`_layer_param_specs` empty-list arm, `LayerWeights::from_flat` no-op, `init_v2_population` continue branch).
```

- [ ] **Step 5: Update CLAUDE.md**

In `CLAUDE.md`, under "Stateful NN Runtime Infrastructure", append a "Phase 2b Window-MLP MVP" subsection mirroring the Phase 2a LSTM MVP block:

```markdown
**Phase 2b Window-MLP (branch `feature/window-mlp`, 2026-04-XX)** adds the third stateful layer type with zero trainable parameters:
- **Rust**: `WindowLayer` struct (fields `input_size`, `n_steps`; no weights), `Layer::Window` / `LayerSpec::Window` / `LayerState::Window { buffer: VecDeque<Vec<f64>> }` (pre-filled with `n_steps` zero vectors of length `input_size`), `LayerWeights for WindowLayer` zero-param impl (`n_params() == 0`, `to_flat() == Vec::new()`, no-op `from_flat`), `TomlLayerSpec::Window { input_size, n_steps }` with the GRU/LSTM input-size-explicit convention, `NeuralNetModel::save_json` / `from_v2_json` / `from_flat_weights_v2` Window arms (JSON v2 entry is spec-only: `{"type":"window","input_size","n_steps"}`, no weights dict).
- **Python**: `WindowSpec` pydantic schema in the discriminated union, `WindowLayer` torch module (used only by the cross-language equivalence test; `build_layer(WindowSpec)` raises `NotImplementedError` -- Window-MLP is PSO-only in Phase 2b, PPO use deferred), `_layer_param_specs(WindowSpec)` returns `[]`, `config.py::_layer_n_params(window) == 0`, `_layer_output_size(window) == n_steps * input_size`, `init_v2_population` Window no-op branch, `export_v2_policy_to_json` Window branch writes spec-only JSON entry and the obs-norm bake-in guard rejects Window as layer 0, `load_policy_from_json` rejects v2 JSON with Window layers (same PPO-rejection pattern).
- **Training**: `configs/training/msr_aller_window_pso_train.toml` (Window(16,8) -> Dense(128->32,swish) -> Dense(32->8,swish) -> Dense(8->2,linear), 4410 params, PSO `n_pop=64 n_gen=2000 seed_strategy="adaptive"`). Registered as `neural_network_window_pso` in `compare_guidance.SCHEMES` + `_NN_DEPLOY_SCHEMES`; `train_all.sh` aliases `window_pso` / `nn_window_pso` / `window`.
- **Gates**: cross-language Window equivalence (100-step stateful sequence via `nn_forward_sequence`, target machine epsilon), PSO-Window smoke (@slow: 2-gen reduced-arch), PPO-rejection tests (@fast: `build_layer` and `load_policy_from_json` raise `NotImplementedError`). 10/10 guidance golden regressions bit-identical.

Full spec: `docs/superpowers/specs/2026-04-20-phase-2b-window-mlp-design.md`. Plan: `docs/superpowers/plans/2026-04-20-phase-2b-window-mlp-plan.md`.
```

Also extend the "Extensibility (post-Phase-2a contract) -- scalar-state layers" bullet to note Phase 2b contributed the zero-param case:

```markdown
**Extensibility (post-Phase-2b contract) -- scalar-state layers**: (existing list unchanged). Phase 2b locked in **zero-trainable-parameter scalar-state layers** as a supported case: `_layer_param_specs` returns `[]`, `LayerWeights::from_flat` is a no-op, `init_v2_population` contributes a `continue` branch. No other file in the contract list changes for zero-param layers.
```

- [ ] **Step 6: Verify docs + run smart-commit**

Run the smart-commit skill with a branch-scoped message:

```
Use the smart-commit skill, considering the whole feature/window-mlp branch history.
```

Expected: smart-commit auto-syncs CLAUDE.md / README.md / TODO.md if any drift is detected, then produces a single commit per logical group.

- [ ] **Step 7: Final `check_all` pass**

```bash
./check_all.sh && ./lint_code.sh
```

Expected: all green.

---

## Self-review notes

The plan covers every Section-2 scope item in the spec:

- Rust LayerSpec/Layer/WindowLayer: Task 1
- WindowLayer::forward + LayerState::Window + for_layer + reset: Task 2
- LayerWeights for WindowLayer: Task 3
- JSON v2 save/load + from_flat_weights_v2: Task 4
- TomlLayerSpec::Window + to_layer_spec: Task 5
- PyO3 flat_weights_to_json window test: Task 6
- WindowSpec pydantic + LayerSpec union: Task 7
- Python WindowLayer torch module: Task 8
- build_layer PPO-rejection guard: Task 9
- config.py _layer_n_params / _layer_output_size / describe_architecture: Task 10
- encoding._layer_param_specs: Task 11
- init_v2_population Window continue: Task 12
- export_v2_policy_to_json + obs-norm guard: Task 13
- load_policy_from_json rejection: Task 14
- Cross-language equivalence test: Task 15
- PSO smoke + CI wiring: Task 16
- PPO-rejection test: Task 9 + Task 14 (build_layer + load_policy_from_json respectively)
- Training config + registration: Task 17
- TODO.md / CLAUDE.md sync + smart-commit: Task 18

Method / type consistency: `_layer_n_params` uses raw dicts (`entry["type"]`), `_layer_param_specs` uses pydantic specs (isinstance). Both patterns preserved. `from_flat_weights_v2` in Task 4 consumes zero flat values for Window (cursor += 0), matching the LayerWeights contract in Task 3.

No placeholders remain. Every step ships complete code or a concrete command with expected output.
