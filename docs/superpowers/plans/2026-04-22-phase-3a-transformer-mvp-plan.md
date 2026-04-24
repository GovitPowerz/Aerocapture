# Phase 3a Transformer MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land a 1-layer pre-norm Transformer with causal window attention (N=64 KV ring buffer) as the fifth stateful layer type on the Phase 0/1/1.5/2a/2b stack. Trained on PSO only; PPO paths fail loudly at `build_layer` / `load_policy_from_json` with a clear pointer to the spec.

**Architecture:** Rust `Layer` gains a `Transformer(TransformerLayer)` variant with 4 projection matrices (Q/K/V/O), 2-layer FFN (GELU-exact), and 2 LayerNorms (biased variance, eps=1e-5). `LayerSpec::Transformer { d_model, n_heads, d_ffn, n_seq }` and `LayerState::Transformer { k_cache: VecDeque<Vec<f64>>, v_cache: VecDeque<Vec<f64>> }` extend the enums. Cache grows organically from 0 to `n_seq`, then stays at `n_seq` (no zero padding). Sinusoidal PE is indexed relative to the current buffer slot; `k_pe_offsets = W_K @ PE` and `v_pe_offsets = W_V @ PE` are precomputed once at load via `rebuild_pe_offsets` (called in both `from_flat_weights_v2` and `from_v2_json`; NOT part of the flat chromosome). Python mirror lives in `rl/layers/transformer.py` with manual softmax / LN / GELU / multi-head split for bit-equivalence; `build_layer` and `load_policy_from_json` raise `NotImplementedError` to gate the PPO path.

**Tech Stack:** Rust 2024 edition (libm for `erf`), PyO3 for Python bindings, Python 3.14, PyTorch (manual attention), Pydantic v2 discriminated unions, pymoo PSO, pytest.

**Spec:** `docs/superpowers/specs/2026-04-22-phase-3a-transformer-mvp-design.md`

---

## Task 0: TODO.md marker

**Files:**
- Modify: `TODO.md`

- [ ] **Step 1: Mark Phase 3 in progress with 3a split**

Replace the current Phase 3 block with:

```markdown
### Phase 3a -- Transformer MVP (PSO only) [DOING 2026-04-22 on feature/transformer-mvp]
- [ ] Rust `TransformerLayer` + `Layer::Transformer` + `LayerSpec::Transformer { d_model, n_heads, d_ffn, n_seq }` + `LayerState::Transformer { k_cache, v_cache }` + `TomlLayerSpec::Transformer`
- [ ] `LayerWeights for TransformerLayer` + derived-at-load PE-offset pattern (`rebuild_pe_offsets` in both `from_flat_weights_v2` and `from_v2_json`)
- [ ] Python `TransformerLayer` torch module (manual LN/GELU/softmax/MHA) + `TransformerSpec` pydantic + `build_layer` PPO-rejection guard
- [ ] `_transformer_specs` (Xavier on projections + FFN, N(1,0.01) on LN gamma, near-zero on biases) + `_layer_n_params` / `_layer_output_size` / `init_v2_population` Transformer arms
- [ ] Training config `msr_aller_transformer_pso_train.toml` + `compare_guidance` + `train_all.sh` registration
- [ ] Cross-language equivalence + warm-up + PSO smoke + PPO-rejection tests (CI wiring)

Spec: `docs/superpowers/specs/2026-04-22-phase-3a-transformer-mvp-design.md`.
Plan: `docs/superpowers/plans/2026-04-22-phase-3a-transformer-mvp-plan.md`.

### Phase 3b -- Transformer PPO-BPTT (follow-up)
- [ ] Deferred from 3a. Requires `hidden_shapes` arm for stacked `(2, n_seq, d_model)` + per-env cache-length scalar, ndim-dispatch arm in `ppo_update_bptt`, PPO smoke + BPTT chunk-invariant tests, training TOML `msr_aller_transformer_ppo_train.toml`.
```

- [ ] **Step 2: Commit**

```bash
git add TODO.md
git commit -m "docs(todo): mark Phase 3a in progress on feature/transformer-mvp"
```

---

## Task 1: Rust sinusoidal PE + LayerNorm + GELU helpers

**Files:**
- Modify: `src/rust/src/data/neural.rs` (add `build_pe_table`, `layer_norm_biased`, `gelu_exact` free functions near the top of the module)
- Add dependency: `libm` (if not already present) in `src/rust/Cargo.toml`
- Test: `src/rust/src/data/neural.rs` (`#[cfg(test)] mod tests`)

- [ ] **Step 1: Verify `libm` is in `src/rust/Cargo.toml`**

```bash
grep -n "libm" src/rust/Cargo.toml
```

If absent, add `libm = "0.2"` to `[dependencies]`. `libm::erf` is the correctly-rounded IEEE-754 erf implementation used by the Python `torch.special.erf` cross-check.

- [ ] **Step 2: Write failing tests for all three helpers**

Append to the existing `#[cfg(test)] mod tests` block:

```rust
#[test]
fn gelu_exact_matches_spec_values() {
    // Hand-computed f64 values of 0.5 * x * (1 + erf(x / sqrt(2))).
    // Generated with Python: 0.5 * x * (1 + math.erf(x / math.sqrt(2)))
    assert!((gelu_exact(0.0) - 0.0).abs() < 1e-15);
    assert!((gelu_exact(1.0) - 0.8413447460685429).abs() < 1e-14);
    assert!((gelu_exact(-1.0) - (-0.15865525393145707)).abs() < 1e-14);
    assert!((gelu_exact(2.5) - 2.4849712868889297).abs() < 1e-13);
}

#[test]
fn layer_norm_biased_zero_mean_unit_var() {
    // Symmetric 4-element vector around 0 -> mean=0, biased var = (1+4+9+16)/4 = 7.5,
    // std = sqrt(7.5 + 1e-5), each output = x / std.
    let x = [1.0_f64, 2.0, 3.0, 4.0];
    let gamma = [1.0, 1.0, 1.0, 1.0];
    let beta = [0.0, 0.0, 0.0, 0.0];
    let out = layer_norm_biased(&x, &gamma, &beta, 1e-5);
    let mean: f64 = out.iter().sum::<f64>() / 4.0;
    assert!(mean.abs() < 1e-12);  // output should be zero-mean
    let var: f64 = out.iter().map(|v| v * v).sum::<f64>() / 4.0;
    assert!((var - 1.0).abs() < 1e-4);  // unit variance (up to eps floor)
}

#[test]
fn layer_norm_applies_gamma_beta() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let gamma = [2.0, 2.0, 2.0, 2.0];
    let beta = [1.0, 1.0, 1.0, 1.0];
    let out = layer_norm_biased(&x, &gamma, &beta, 1e-5);
    // Expected: 2 * normalized + 1
    let plain = layer_norm_biased(&x, &[1.0; 4], &[0.0; 4], 1e-5);
    for (i, v) in out.iter().enumerate() {
        assert!((v - (2.0 * plain[i] + 1.0)).abs() < 1e-12);
    }
}

#[test]
fn pe_table_shape_and_known_entries() {
    let pe = build_pe_table(4, 4);
    assert_eq!(pe.len(), 4);
    assert_eq!(pe[0].len(), 4);
    // PE[0, :] = [sin(0), cos(0), sin(0), cos(0)] = [0, 1, 0, 1]
    assert!((pe[0][0] - 0.0).abs() < 1e-15);
    assert!((pe[0][1] - 1.0).abs() < 1e-15);
    assert!((pe[0][2] - 0.0).abs() < 1e-15);
    assert!((pe[0][3] - 1.0).abs() < 1e-15);
    // PE[1, 0] = sin(1.0), PE[1, 1] = cos(1.0)
    assert!((pe[1][0] - 1.0_f64.sin()).abs() < 1e-15);
    assert!((pe[1][1] - 1.0_f64.cos()).abs() < 1e-15);
    // PE[1, 2] = sin(1.0 / 10000^(2/4)) = sin(1.0 / 100) = sin(0.01)
    assert!((pe[1][2] - 0.01_f64.sin()).abs() < 1e-14);
    assert!((pe[1][3] - 0.01_f64.cos()).abs() < 1e-14);
}
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
cd src/rust && cargo test --lib data::neural::tests::gelu_exact_matches_spec_values 2>&1 | tail -20
```
Expected: compile error `cannot find function gelu_exact`.

- [ ] **Step 4: Add the three helpers**

Insert at the top of `src/rust/src/data/neural.rs` (after `use` statements, before existing types):

```rust
#[inline]
pub(crate) fn gelu_exact(z: f64) -> f64 {
    // Exact GELU: 0.5 * z * (1 + erf(z / sqrt(2)))
    // Uses libm::erf for IEEE-754 correct rounding; matches torch.special.erf.
    const INV_SQRT2: f64 = 0.7071067811865475_f64;
    0.5 * z * (1.0 + libm::erf(z * INV_SQRT2))
}

pub(crate) fn layer_norm_biased(
    x: &[f64],
    gamma: &[f64],
    beta: &[f64],
    eps: f64,
) -> Vec<f64> {
    debug_assert_eq!(x.len(), gamma.len());
    debug_assert_eq!(x.len(), beta.len());
    let n = x.len() as f64;
    // Sequential FIFO reduction for cross-language bit-identity.
    let mut mean = 0.0;
    for v in x { mean += *v; }
    mean /= n;
    let mut var = 0.0;
    for v in x { let d = *v - mean; var += d * d; }
    var /= n;  // biased: 1/N, NOT Bessel 1/(N-1); matches torch nn.LayerNorm default.
    let inv_std = 1.0 / (var + eps).sqrt();
    x.iter().zip(gamma).zip(beta)
        .map(|((xi, g), b)| ((*xi - mean) * inv_std) * g + b)
        .collect()
}

pub(crate) fn build_pe_table(n_seq: usize, d_model: usize) -> Vec<Vec<f64>> {
    // Standard Vaswani et al. 2017 sinusoidal positional encoding.
    // PE[pos, 2k]   = sin(pos / 10000^(2k / d_model))
    // PE[pos, 2k+1] = cos(pos / 10000^(2k / d_model))
    // Iteration order: pos outer, i inner. Matches Python mirror for bit-identity.
    (0..n_seq).map(|pos| {
        (0..d_model).map(|i| {
            let k = i / 2;
            let div = 10000.0_f64.powf((2.0 * k as f64) / d_model as f64);
            let angle = pos as f64 / div;
            if i % 2 == 0 { angle.sin() } else { angle.cos() }
        }).collect()
    }).collect()
}
```

- [ ] **Step 5: Run tests to verify PASS**

```bash
cd src/rust && cargo test --lib data::neural::tests::gelu_exact_matches_spec_values \
    data::neural::tests::layer_norm_biased_zero_mean_unit_var \
    data::neural::tests::layer_norm_applies_gamma_beta \
    data::neural::tests::pe_table_shape_and_known_entries -- --nocapture
```
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/rust/Cargo.toml src/rust/src/data/neural.rs
git commit -m "feat(nn): sinusoidal PE + biased LayerNorm + GELU-exact helpers

Pure functions used by the forthcoming Transformer layer. GELU uses
libm::erf for IEEE-754 correct rounding, matching torch.special.erf.
LayerNorm uses biased (1/N) variance to match torch.nn.LayerNorm default.
PE iteration order is pos-outer, i-inner for cross-language bit-identity."
```

---

## Task 2: Rust TransformerLayer struct + rebuild_pe_offsets

**Files:**
- Modify: `src/rust/src/data/neural.rs` (add `TransformerLayer` struct and methods; do NOT wire into `Layer` enum yet -- that's Task 4)
- Test: `src/rust/src/data/neural.rs` (`#[cfg(test)] mod tests`)

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn transformer_layer_rebuild_pe_offsets_matches_matmul() {
    // Manually construct a TransformerLayer with known W_K, W_V.
    // k_pe_offsets[i] should equal W_K @ PE[i] (no bias), same for V.
    let d_model = 4;
    let n_seq = 3;
    // W_K = identity -> k_pe_offsets == PE table
    let w_k: Vec<Vec<f64>> = (0..d_model).map(|i| {
        (0..d_model).map(|j| if i == j { 1.0 } else { 0.0 }).collect()
    }).collect();
    let w_v: Vec<Vec<f64>> = w_k.clone();
    let layer = TransformerLayer {
        d_model, n_heads: 2, d_head: 2, d_ffn: 8, n_seq,
        w_q: vec![vec![0.0; d_model]; d_model], b_q: vec![0.0; d_model],
        w_k: w_k.clone(), b_k: vec![0.0; d_model],
        w_v: w_v.clone(), b_v: vec![0.0; d_model],
        w_o: vec![vec![0.0; d_model]; d_model], b_o: vec![0.0; d_model],
        w_ffn1: vec![vec![0.0; d_model]; 8], b_ffn1: vec![0.0; 8],
        w_ffn2: vec![vec![0.0; 8]; d_model], b_ffn2: vec![0.0; d_model],
        ln1_gamma: vec![1.0; d_model], ln1_beta: vec![0.0; d_model],
        ln2_gamma: vec![1.0; d_model], ln2_beta: vec![0.0; d_model],
        k_pe_offsets: Vec::new(),
        v_pe_offsets: Vec::new(),
    };
    let mut layer = layer;
    layer.rebuild_pe_offsets();
    let pe = build_pe_table(n_seq, d_model);
    // With W_K = I, k_pe_offsets should equal PE exactly.
    for i in 0..n_seq {
        for j in 0..d_model {
            assert!((layer.k_pe_offsets[i][j] - pe[i][j]).abs() < 1e-15);
            assert!((layer.v_pe_offsets[i][j] - pe[i][j]).abs() < 1e-15);
        }
    }
}
```

- [ ] **Step 2: Run test to confirm it fails** (struct doesn't exist).

- [ ] **Step 3: Add the struct + `new` + `rebuild_pe_offsets`**

Insert into `src/rust/src/data/neural.rs` after existing layer types:

```rust
#[derive(Debug, Clone)]
pub struct TransformerLayer {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_head: usize,   // d_model / n_heads; validated at construction
    pub d_ffn: usize,
    pub n_seq: usize,

    pub w_q: Vec<Vec<f64>>, pub b_q: Vec<f64>,
    pub w_k: Vec<Vec<f64>>, pub b_k: Vec<f64>,
    pub w_v: Vec<Vec<f64>>, pub b_v: Vec<f64>,
    pub w_o: Vec<Vec<f64>>, pub b_o: Vec<f64>,

    pub w_ffn1: Vec<Vec<f64>>, pub b_ffn1: Vec<f64>,
    pub w_ffn2: Vec<Vec<f64>>, pub b_ffn2: Vec<f64>,

    pub ln1_gamma: Vec<f64>, pub ln1_beta: Vec<f64>,
    pub ln2_gamma: Vec<f64>, pub ln2_beta: Vec<f64>,

    // Derived at load time; NOT part of the flat chromosome.
    pub k_pe_offsets: Vec<Vec<f64>>,
    pub v_pe_offsets: Vec<Vec<f64>>,
}

fn matvec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    // m shape: [rows][cols], v shape: [cols] -> out shape: [rows].
    // Sequential dot product for cross-language bit-identity.
    m.iter().map(|row| {
        let mut acc = 0.0;
        for (a, b) in row.iter().zip(v) { acc += a * b; }
        acc
    }).collect()
}

impl TransformerLayer {
    /// Recompute k_pe_offsets[i] = W_K @ PE[i] and v_pe_offsets[i] = W_V @ PE[i]
    /// for i in 0..n_seq. MUST be called after any mutation to w_k or w_v;
    /// called from `from_flat` and `from_v2_json` entry points.
    pub fn rebuild_pe_offsets(&mut self) {
        let pe = build_pe_table(self.n_seq, self.d_model);
        self.k_pe_offsets = pe.iter().map(|p| matvec(&self.w_k, p)).collect();
        self.v_pe_offsets = pe.iter().map(|p| matvec(&self.w_v, p)).collect();
    }
}
```

- [ ] **Step 4: Run test to verify PASS**

```bash
cd src/rust && cargo test --lib data::neural::tests::transformer_layer_rebuild_pe_offsets_matches_matmul -- --nocapture
```
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/data/neural.rs
git commit -m "feat(nn): TransformerLayer struct + rebuild_pe_offsets derived-state pattern"
```

---

## Task 3: Rust TransformerLayer::forward

**Files:**
- Modify: `src/rust/src/data/neural.rs` (add `TransformerLayer::forward` method)
- Test: `src/rust/src/data/neural.rs` (`#[cfg(test)] mod tests`)

- [ ] **Step 1: Write the failing test**

Golden values here are pre-computed in Python using the exact algorithm described in spec section 3.4 with hand-specified weights. For this task, we use a degenerate configuration that makes the expected output analytically tractable:

```rust
#[test]
fn transformer_forward_single_token_zero_weights_is_residual() {
    // All projections zero + LN gamma=1, beta=0 + FFN zero means:
    //   x_norm1 = LN(x)
    //   q = k = v = 0 (since W_Q/W_K/W_V = 0 and b_{q,k,v} = 0)
    //   attention output = 0 (all-zero scores, all-zero values)
    //   x1 = x + W_O @ 0 + b_o = x
    //   ffn_out = 0
    //   out = x1 + 0 = x
    let d_model = 4; let n_heads = 2; let d_ffn = 8; let n_seq = 3;
    let layer = make_zero_transformer(d_model, n_heads, d_ffn, n_seq);
    let mut k_cache = VecDeque::new();
    let mut v_cache = VecDeque::new();
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let out = layer.forward(&x, &mut k_cache, &mut v_cache);
    for i in 0..d_model {
        assert!((out[i] - x[i]).abs() < 1e-12, "out[{}]={} x[{}]={}", i, out[i], i, x[i]);
    }
    // Cache grew by 1
    assert_eq!(k_cache.len(), 1);
    assert_eq!(v_cache.len(), 1);
}

#[test]
fn transformer_forward_cache_grows_then_saturates() {
    let d_model = 4; let n_heads = 2; let d_ffn = 8; let n_seq = 3;
    let mut layer = make_zero_transformer(d_model, n_heads, d_ffn, n_seq);
    // Small nonzero w_k so k pushes are distinct; w_v stays zero so V-sum is zero.
    layer.w_k[0][0] = 1.0;
    layer.rebuild_pe_offsets();
    let mut k_cache = VecDeque::new();
    let mut v_cache = VecDeque::new();
    for step in 0..5 {
        let x = vec![step as f64, 0.0, 0.0, 0.0];
        let _ = layer.forward(&x, &mut k_cache, &mut v_cache);
        let expected_len = (step + 1).min(n_seq);
        assert_eq!(k_cache.len(), expected_len, "step {step}");
        assert_eq!(v_cache.len(), expected_len, "step {step}");
    }
    // After 5 steps with n_seq=3, cache should contain K from steps 2, 3, 4 (FIFO).
    // k[0] after step 0 was LN(x) applied via W_K where x=[0,0,0,0] -> all zero.
    // Simpler: just assert the invariant holds.
    assert_eq!(k_cache.len(), 3);
}

fn make_zero_transformer(d_model: usize, n_heads: usize, d_ffn: usize, n_seq: usize)
    -> TransformerLayer
{
    let mut layer = TransformerLayer {
        d_model, n_heads, d_head: d_model / n_heads, d_ffn, n_seq,
        w_q: vec![vec![0.0; d_model]; d_model], b_q: vec![0.0; d_model],
        w_k: vec![vec![0.0; d_model]; d_model], b_k: vec![0.0; d_model],
        w_v: vec![vec![0.0; d_model]; d_model], b_v: vec![0.0; d_model],
        w_o: vec![vec![0.0; d_model]; d_model], b_o: vec![0.0; d_model],
        w_ffn1: vec![vec![0.0; d_model]; d_ffn], b_ffn1: vec![0.0; d_ffn],
        w_ffn2: vec![vec![0.0; d_ffn]; d_model], b_ffn2: vec![0.0; d_model],
        ln1_gamma: vec![1.0; d_model], ln1_beta: vec![0.0; d_model],
        ln2_gamma: vec![1.0; d_model], ln2_beta: vec![0.0; d_model],
        k_pe_offsets: Vec::new(), v_pe_offsets: Vec::new(),
    };
    layer.rebuild_pe_offsets();
    layer
}
```

- [ ] **Step 2: Run tests to confirm they fail** (`forward` not defined).

- [ ] **Step 3: Implement `forward`**

Add to the `impl TransformerLayer` block:

```rust
impl TransformerLayer {
    /// Single-token forward for inference. Consumes x (length d_model), mutates
    /// k_cache and v_cache (pushing the current projected K/V, evicting oldest
    /// if len > n_seq). Returns output of length d_model.
    pub fn forward(
        &self,
        x: &[f64],
        k_cache: &mut VecDeque<Vec<f64>>,
        v_cache: &mut VecDeque<Vec<f64>>,
    ) -> Vec<f64> {
        debug_assert_eq!(x.len(), self.d_model);

        // 1. LN1
        let x_norm1 = layer_norm_biased(x, &self.ln1_gamma, &self.ln1_beta, 1e-5);

        // 2. Q, K, V projections (with bias)
        let mut q = matvec(&self.w_q, &x_norm1);
        for (qi, bi) in q.iter_mut().zip(&self.b_q) { *qi += bi; }
        let mut k = matvec(&self.w_k, &x_norm1);
        for (ki, bi) in k.iter_mut().zip(&self.b_k) { *ki += bi; }
        let mut v = matvec(&self.w_v, &x_norm1);
        for (vi, bi) in v.iter_mut().zip(&self.b_v) { *vi += bi; }

        // 3. Push into cache, evict if over capacity
        k_cache.push_back(k);
        v_cache.push_back(v);
        while k_cache.len() > self.n_seq {
            k_cache.pop_front();
            v_cache.pop_front();
        }
        let cache_len = k_cache.len();

        // 4. Multi-head attention. For each head, compute scores over the cache,
        //    softmax (max-subtraction FIFO), weighted sum of V.
        let inv_sqrt_d_head = 1.0 / (self.d_head as f64).sqrt();
        let mut attn_out = vec![0.0_f64; self.d_model];

        for h in 0..self.n_heads {
            let h_start = h * self.d_head;
            let h_end = h_start + self.d_head;
            let q_h = &q[h_start..h_end];

            // Scores over the cache
            let mut scores = Vec::with_capacity(cache_len);
            for i in 0..cache_len {
                let k_eff_h = slot_k_eff_head(
                    &k_cache[i], &self.k_pe_offsets[i], h_start, h_end,
                );
                let mut s = 0.0;
                for (a, b) in q_h.iter().zip(k_eff_h.iter()) { s += a * b; }
                scores.push(s * inv_sqrt_d_head);
            }

            // Max-subtraction softmax, sequential FIFO
            let mut max_score = scores[0];
            for s in &scores[1..] { if *s > max_score { max_score = *s; } }
            let mut exp_sum = 0.0;
            let mut exp_scores = Vec::with_capacity(cache_len);
            for s in &scores {
                let e = (*s - max_score).exp();
                exp_scores.push(e);
                exp_sum += e;
            }
            // Weighted sum of V_eff head slice
            for i in 0..cache_len {
                let w = exp_scores[i] / exp_sum;
                for j in h_start..h_end {
                    attn_out[j] += w * (v_cache[i][j] + self.v_pe_offsets[i][j]);
                }
            }
        }

        // 5. Output projection + residual
        let mut proj = matvec(&self.w_o, &attn_out);
        for (pi, bi) in proj.iter_mut().zip(&self.b_o) { *pi += bi; }
        let mut x1 = vec![0.0; self.d_model];
        for i in 0..self.d_model { x1[i] = x[i] + proj[i]; }

        // 6. LN2 + FFN + residual
        let x_norm2 = layer_norm_biased(&x1, &self.ln2_gamma, &self.ln2_beta, 1e-5);
        let mut hidden = matvec(&self.w_ffn1, &x_norm2);
        for (hi, bi) in hidden.iter_mut().zip(&self.b_ffn1) { *hi += bi; }
        for h in hidden.iter_mut() { *h = gelu_exact(*h); }
        let mut ffn_out = matvec(&self.w_ffn2, &hidden);
        for (fi, bi) in ffn_out.iter_mut().zip(&self.b_ffn2) { *fi += bi; }

        let mut out = vec![0.0; self.d_model];
        for i in 0..self.d_model { out[i] = x1[i] + ffn_out[i]; }
        out
    }
}

#[inline]
fn slot_k_eff_head(
    k_cached: &[f64], k_pe_offset: &[f64], h_start: usize, h_end: usize,
) -> Vec<f64> {
    // Returns k_cached[h_start..h_end] + k_pe_offset[h_start..h_end]
    let mut out = Vec::with_capacity(h_end - h_start);
    for j in h_start..h_end {
        out.push(k_cached[j] + k_pe_offset[j]);
    }
    out
}
```

- [ ] **Step 4: Run tests to verify PASS**

```bash
cd src/rust && cargo test --lib data::neural::tests::transformer_forward -- --nocapture
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/data/neural.rs
git commit -m "feat(nn): TransformerLayer::forward (single-token inference)

Pre-norm block with multi-head causal attention over a VecDeque KV cache
(organic growth 0 -> n_seq). Max-subtraction softmax with sequential FIFO
reduction for cross-language bit-identity. PE offsets added at attention
time from precomputed k_pe_offsets / v_pe_offsets."
```

---

## Task 4: Rust LayerSpec::Transformer variant + Layer dispatch + LayerState

**Files:**
- Modify: `src/rust/src/data/neural.rs` (extend `LayerSpec` and `Layer` enums)
- Modify: `src/rust/src/data/nn_state.rs` (extend `LayerState` enum)
- Test: same files

- [ ] **Step 1: Write the failing tests**

In `src/rust/src/data/neural.rs` tests:

```rust
#[test]
fn layer_spec_transformer_variant_serializes() {
    let spec = LayerSpec::Transformer {
        d_model: 32, n_heads: 4, d_ffn: 64, n_seq: 64,
    };
    let json = serde_json::to_string(&spec).unwrap();
    assert!(json.contains("\"type\":\"transformer\""));
    assert!(json.contains("\"d_model\":32"));
    let round: LayerSpec = serde_json::from_str(&json).unwrap();
    match round {
        LayerSpec::Transformer { d_model, n_heads, d_ffn, n_seq } => {
            assert_eq!((d_model, n_heads, d_ffn, n_seq), (32, 4, 64, 64));
        }
        _ => panic!("wrong variant"),
    }
}
```

In `src/rust/src/data/nn_state.rs` tests:

```rust
#[test]
fn layer_state_transformer_for_layer_starts_empty() {
    let spec = crate::data::neural::LayerSpec::Transformer {
        d_model: 32, n_heads: 4, d_ffn: 64, n_seq: 64,
    };
    let state = LayerState::for_layer(&spec);
    match state {
        LayerState::Transformer { k_cache, v_cache } => {
            assert_eq!(k_cache.len(), 0);
            assert_eq!(v_cache.len(), 0);
        }
        _ => panic!("wrong variant"),
    }
}

#[test]
fn layer_state_transformer_reset_clears_caches() {
    let spec = crate::data::neural::LayerSpec::Transformer {
        d_model: 4, n_heads: 2, d_ffn: 8, n_seq: 3,
    };
    let mut state = LayerState::for_layer(&spec);
    if let LayerState::Transformer { k_cache, v_cache } = &mut state {
        k_cache.push_back(vec![1.0, 2.0, 3.0, 4.0]);
        v_cache.push_back(vec![5.0, 6.0, 7.0, 8.0]);
    }
    state.reset();
    match state {
        LayerState::Transformer { k_cache, v_cache } => {
            assert_eq!(k_cache.len(), 0);
            assert_eq!(v_cache.len(), 0);
        }
        _ => panic!("wrong variant"),
    }
}
```

- [ ] **Step 2: Run tests to confirm they fail**.

- [ ] **Step 3: Extend `LayerSpec` and `Layer` enums**

In `src/rust/src/data/neural.rs`, extend `LayerSpec`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum LayerSpec {
    Dense       { input_size: usize, output_size: usize, activation: Activation },
    Gru         { input_size: usize, hidden_size: usize },
    Lstm        { input_size: usize, hidden_size: usize },
    Window      { input_size: usize, n_steps: usize },
    Transformer { d_model: usize, n_heads: usize, d_ffn: usize, n_seq: usize },
}
```

Extend `Layer` enum:

```rust
#[derive(Debug, Clone)]
pub enum Layer {
    Dense(DenseLayer),
    Gru(GruLayer),
    Lstm(LstmLayer),
    Window(WindowLayer),
    Transformer(TransformerLayer),
}
```

- [ ] **Step 4: Extend `LayerState` enum in `nn_state.rs`**

```rust
#[derive(Debug, Clone)]
pub enum LayerState {
    None,
    Gru(Vec<f64>),
    Lstm { h: Vec<f64>, c: Vec<f64> },
    Window { buffer: VecDeque<Vec<f64>> },
    Transformer {
        k_cache: VecDeque<Vec<f64>>,
        v_cache: VecDeque<Vec<f64>>,
    },
}
```

Add matching arms to `LayerState::for_layer`:

```rust
LayerSpec::Transformer { .. } => LayerState::Transformer {
    k_cache: VecDeque::new(),
    v_cache: VecDeque::new(),
},
```

And to `LayerState::reset`:

```rust
LayerState::Transformer { k_cache, v_cache } => {
    k_cache.clear();
    v_cache.clear();
}
```

- [ ] **Step 5: Run tests to verify PASS**

```bash
cd src/rust && cargo test --lib data::neural::tests::layer_spec_transformer_variant_serializes \
    data::nn_state::tests::layer_state_transformer_for_layer_starts_empty \
    data::nn_state::tests::layer_state_transformer_reset_clears_caches
```
Expected: 3 passed.

- [ ] **Step 6: Fix any exhaustiveness warnings**

`cargo check` may flag non-exhaustive matches in downstream code. Add `Layer::Transformer(_) => todo!("Task 5: from_flat / Task 7: forward dispatch")` or the specific implementation arm depending on which task owns it. Note: Tasks 5-7 will fill these in.

- [ ] **Step 7: Commit**

```bash
git add src/rust/src/data/neural.rs src/rust/src/data/nn_state.rs
git commit -m "feat(nn): LayerSpec::Transformer + Layer::Transformer + LayerState::Transformer variants"
```

---

## Task 5: Rust LayerWeights for TransformerLayer + from_flat_weights_v2 arm

**Files:**
- Modify: `src/rust/src/data/neural.rs` (`LayerWeights` impl for `TransformerLayer`, `from_flat_weights_v2` arm)
- Test: same file

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn transformer_layer_weights_flat_roundtrip() {
    // Build a layer with nonzero unique weights, flatten, rebuild, re-flatten,
    // assert equality (and that rebuild_pe_offsets was called so PE offsets match).
    let d_model = 4; let n_heads = 2; let d_ffn = 6; let n_seq = 3;
    let spec = LayerSpec::Transformer { d_model, n_heads, d_ffn, n_seq };

    // n_params formula: 4*d^2 + 2*ffn*d + ffn + 9*d
    //                 = 4*16 + 2*24 + 6 + 36 = 64 + 48 + 6 + 36 = 154
    let n_params = 4 * d_model * d_model + 2 * d_ffn * d_model + d_ffn + 9 * d_model;
    assert_eq!(n_params, 154);

    // Unique fingerprint values: f[i] = (i as f64) * 0.01 + 0.5
    let flat: Vec<f64> = (0..n_params).map(|i| (i as f64) * 0.01 + 0.5).collect();

    let mut cursor = 0;
    let layer = <TransformerLayer as LayerWeights>::from_flat(&flat, &mut cursor, &spec).unwrap();
    assert_eq!(cursor, n_params);
    assert_eq!(layer.k_pe_offsets.len(), n_seq);  // rebuild_pe_offsets ran
    assert_eq!(layer.v_pe_offsets.len(), n_seq);

    let round = layer.to_flat();
    assert_eq!(round.len(), n_params);
    for (a, b) in flat.iter().zip(round.iter()) {
        assert!((a - b).abs() < 1e-15);
    }
}
```

- [ ] **Step 2: Run test to confirm it fails**.

- [ ] **Step 3: Implement `LayerWeights for TransformerLayer`**

Flat order (from spec section 3.3): `w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o, w_ffn1, b_ffn1, w_ffn2, b_ffn2, ln1_gamma, ln1_beta, ln2_gamma, ln2_beta`. Matrices are row-major.

```rust
impl LayerWeights for TransformerLayer {
    fn n_params(&self) -> usize {
        4 * self.d_model * self.d_model
        + 2 * self.d_ffn * self.d_model
        + self.d_ffn
        + 9 * self.d_model
    }

    fn to_flat(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n_params());
        let push_mat = |out: &mut Vec<f64>, m: &[Vec<f64>]| {
            for row in m { out.extend_from_slice(row); }
        };
        push_mat(&mut out, &self.w_q);  out.extend_from_slice(&self.b_q);
        push_mat(&mut out, &self.w_k);  out.extend_from_slice(&self.b_k);
        push_mat(&mut out, &self.w_v);  out.extend_from_slice(&self.b_v);
        push_mat(&mut out, &self.w_o);  out.extend_from_slice(&self.b_o);
        push_mat(&mut out, &self.w_ffn1); out.extend_from_slice(&self.b_ffn1);
        push_mat(&mut out, &self.w_ffn2); out.extend_from_slice(&self.b_ffn2);
        out.extend_from_slice(&self.ln1_gamma);
        out.extend_from_slice(&self.ln1_beta);
        out.extend_from_slice(&self.ln2_gamma);
        out.extend_from_slice(&self.ln2_beta);
        out
    }

    fn from_flat(
        flat: &[f64],
        cursor: &mut usize,
        spec: &LayerSpec,
    ) -> Result<Self, DataError> {
        let (d_model, n_heads, d_ffn, n_seq) = match spec {
            LayerSpec::Transformer { d_model, n_heads, d_ffn, n_seq } =>
                (*d_model, *n_heads, *d_ffn, *n_seq),
            _ => return Err(DataError::InvalidArchitecture(
                "TransformerLayer::from_flat got non-Transformer spec".into()
            )),
        };
        if n_heads == 0 || d_model % n_heads != 0 {
            return Err(DataError::InvalidArchitecture(format!(
                "(transformer) d_model={d_model} not divisible by n_heads={n_heads}"
            )));
        }
        let d_head = d_model / n_heads;

        let read_mat = |flat: &[f64], cursor: &mut usize, rows: usize, cols: usize|
            -> Result<Vec<Vec<f64>>, DataError>
        {
            let mut m = Vec::with_capacity(rows);
            for _ in 0..rows {
                if *cursor + cols > flat.len() {
                    return Err(DataError::InvalidArchitecture(
                        "(transformer) flat slice too short".into()
                    ));
                }
                m.push(flat[*cursor..*cursor + cols].to_vec());
                *cursor += cols;
            }
            Ok(m)
        };
        let read_vec = |flat: &[f64], cursor: &mut usize, n: usize|
            -> Result<Vec<f64>, DataError>
        {
            if *cursor + n > flat.len() {
                return Err(DataError::InvalidArchitecture(
                    "(transformer) flat slice too short".into()
                ));
            }
            let v = flat[*cursor..*cursor + n].to_vec();
            *cursor += n;
            Ok(v)
        };

        let w_q = read_mat(flat, cursor, d_model, d_model)?;
        let b_q = read_vec(flat, cursor, d_model)?;
        let w_k = read_mat(flat, cursor, d_model, d_model)?;
        let b_k = read_vec(flat, cursor, d_model)?;
        let w_v = read_mat(flat, cursor, d_model, d_model)?;
        let b_v = read_vec(flat, cursor, d_model)?;
        let w_o = read_mat(flat, cursor, d_model, d_model)?;
        let b_o = read_vec(flat, cursor, d_model)?;
        let w_ffn1 = read_mat(flat, cursor, d_ffn, d_model)?;
        let b_ffn1 = read_vec(flat, cursor, d_ffn)?;
        let w_ffn2 = read_mat(flat, cursor, d_model, d_ffn)?;
        let b_ffn2 = read_vec(flat, cursor, d_model)?;
        let ln1_gamma = read_vec(flat, cursor, d_model)?;
        let ln1_beta  = read_vec(flat, cursor, d_model)?;
        let ln2_gamma = read_vec(flat, cursor, d_model)?;
        let ln2_beta  = read_vec(flat, cursor, d_model)?;

        let mut layer = TransformerLayer {
            d_model, n_heads, d_head, d_ffn, n_seq,
            w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o,
            w_ffn1, b_ffn1, w_ffn2, b_ffn2,
            ln1_gamma, ln1_beta, ln2_gamma, ln2_beta,
            k_pe_offsets: Vec::new(), v_pe_offsets: Vec::new(),
        };
        layer.rebuild_pe_offsets();
        Ok(layer)
    }
}
```

- [ ] **Step 4: Add the `Transformer` arm in `from_flat_weights_v2`**

Find the match arm in `NeuralNetModel::from_flat_weights_v2` that dispatches per `LayerSpec` and add:

```rust
LayerSpec::Transformer { .. } => {
    let layer = <TransformerLayer as LayerWeights>::from_flat(flat, cursor, spec)?;
    Layer::Transformer(layer)
}
```

- [ ] **Step 5: Run round-trip test**

```bash
cd src/rust && cargo test --lib data::neural::tests::transformer_layer_weights_flat_roundtrip -- --nocapture
```
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/data/neural.rs
git commit -m "feat(nn): LayerWeights for TransformerLayer + from_flat_weights_v2 arm

Flat order matches spec section 3.3 (w_q, b_q, w_k, b_k, w_v, b_v,
w_o, b_o, w_ffn1, b_ffn1, w_ffn2, b_ffn2, ln1_gamma, ln1_beta,
ln2_gamma, ln2_beta). from_flat calls rebuild_pe_offsets after
loading w_k / w_v so PE offsets stay consistent with the loaded weights."
```

---

## Task 6: Rust JSON v2 save/load for Transformer

**Files:**
- Modify: `src/rust/src/data/neural.rs` (`NeuralNetModel::save_json` and `from_v2_json` Transformer arms)
- Test: `src/rust/src/data/neural.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn transformer_json_v2_save_load_roundtrip() {
    // Build a small model via from_flat_weights_v2, save to JSON, load, compare.
    let architecture = vec![
        LayerSpec::Dense {
            input_size: 8, output_size: 4, activation: Activation::Linear
        },
        LayerSpec::Transformer {
            d_model: 4, n_heads: 2, d_ffn: 8, n_seq: 3,
        },
        LayerSpec::Dense {
            input_size: 4, output_size: 2, activation: Activation::Linear
        },
    ];
    let n_params = NeuralNetModel::n_params_for_architecture(&architecture);
    let flat: Vec<f64> = (0..n_params).map(|i| (i as f64) * 0.003 - 0.7).collect();
    let model = NeuralNetModel::from_flat_weights_v2(&architecture, None, &flat).unwrap();

    let tmp = tempfile::NamedTempFile::new().unwrap();
    model.save_json(tmp.path()).unwrap();

    let loaded = NeuralNetModel::from_json_file(tmp.path()).unwrap();
    assert_eq!(loaded.architecture().len(), 3);

    // Drive both forward on the same input, expect bit-identity.
    let x = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let mut s1 = NnState::for_model(&model);
    let mut s2 = NnState::for_model(&loaded);
    let y1 = model.forward(&mut s1, &x);
    let y2 = loaded.forward(&mut s2, &x);
    assert_eq!(y1.len(), y2.len());
    for (a, b) in y1.iter().zip(y2.iter()) {
        assert!((a - b).abs() < 1e-15, "a={a} b={b}");
    }
}
```

- [ ] **Step 2: Run test to confirm it fails**.

- [ ] **Step 3: Implement `save_json` Transformer arm**

Find the match arm in `NeuralNetModel::save_json` (the loop that builds the architecture JSON + weights dict) and add:

```rust
(LayerSpec::Transformer { d_model, n_heads, d_ffn, n_seq },
 Layer::Transformer(layer)) => {
    arch_json.push(json!({
        "type": "transformer",
        "d_model": d_model,
        "n_heads": n_heads,
        "d_ffn":   d_ffn,
        "n_seq":   n_seq,
    }));
    weights_json.insert(format!("layer_{i}"), json!({
        "w_q": layer.w_q, "b_q": layer.b_q,
        "w_k": layer.w_k, "b_k": layer.b_k,
        "w_v": layer.w_v, "b_v": layer.b_v,
        "w_o": layer.w_o, "b_o": layer.b_o,
        "w_ffn1": layer.w_ffn1, "b_ffn1": layer.b_ffn1,
        "w_ffn2": layer.w_ffn2, "b_ffn2": layer.b_ffn2,
        "ln1": { "gamma": layer.ln1_gamma, "beta": layer.ln1_beta },
        "ln2": { "gamma": layer.ln2_gamma, "beta": layer.ln2_beta },
    }));
}
```

- [ ] **Step 4: Implement `from_v2_json` Transformer arm**

Find the match arm in `NeuralNetModel::from_v2_json` that dispatches per layer `"type"`:

```rust
"transformer" => {
    let d_model = parse_usize("d_model")?;
    let n_heads = parse_usize("n_heads")?;
    let d_ffn   = parse_usize("d_ffn")?;
    let n_seq   = parse_usize("n_seq")?;
    if d_model == 0 || n_heads == 0 || d_ffn == 0 || n_seq == 0 {
        return Err(DataError::InvalidArchitecture(
            "(transformer) all shape fields must be positive".into()
        ));
    }
    if d_model % n_heads != 0 {
        return Err(DataError::InvalidArchitecture(format!(
            "(transformer) d_model={d_model} not divisible by n_heads={n_heads}"
        )));
    }
    let d_head = d_model / n_heads;

    let w = layer_weights.get(&format!("layer_{i}")).ok_or_else(|| DataError::InvalidArchitecture(
        format!("(transformer) missing weights for layer_{i}")
    ))?;
    let read_mat = |key: &str| -> Result<Vec<Vec<f64>>, DataError> {
        serde_json::from_value(w[key].clone())
            .map_err(|e| DataError::InvalidArchitecture(format!("(transformer) layer_{i}.{key}: {e}")))
    };
    let read_vec = |key: &str| -> Result<Vec<f64>, DataError> {
        serde_json::from_value(w[key].clone())
            .map_err(|e| DataError::InvalidArchitecture(format!("(transformer) layer_{i}.{key}: {e}")))
    };

    let mut layer = TransformerLayer {
        d_model, n_heads, d_head, d_ffn, n_seq,
        w_q: read_mat("w_q")?, b_q: read_vec("b_q")?,
        w_k: read_mat("w_k")?, b_k: read_vec("b_k")?,
        w_v: read_mat("w_v")?, b_v: read_vec("b_v")?,
        w_o: read_mat("w_o")?, b_o: read_vec("b_o")?,
        w_ffn1: read_mat("w_ffn1")?, b_ffn1: read_vec("b_ffn1")?,
        w_ffn2: read_mat("w_ffn2")?, b_ffn2: read_vec("b_ffn2")?,
        ln1_gamma: serde_json::from_value(w["ln1"]["gamma"].clone())
            .map_err(|e| DataError::InvalidArchitecture(format!("layer_{i}.ln1.gamma: {e}")))?,
        ln1_beta:  serde_json::from_value(w["ln1"]["beta"].clone())
            .map_err(|e| DataError::InvalidArchitecture(format!("layer_{i}.ln1.beta: {e}")))?,
        ln2_gamma: serde_json::from_value(w["ln2"]["gamma"].clone())
            .map_err(|e| DataError::InvalidArchitecture(format!("layer_{i}.ln2.gamma: {e}")))?,
        ln2_beta:  serde_json::from_value(w["ln2"]["beta"].clone())
            .map_err(|e| DataError::InvalidArchitecture(format!("layer_{i}.ln2.beta: {e}")))?,
        k_pe_offsets: Vec::new(), v_pe_offsets: Vec::new(),
    };
    layer.rebuild_pe_offsets();
    (LayerSpec::Transformer { d_model, n_heads, d_ffn, n_seq },
     Layer::Transformer(layer))
}
```

The `parse_usize`, `layer_weights`, `i`, and other closures/bindings follow the existing `from_v2_json` helper convention (see the Window arm for the clean pattern). Adjust names if the existing code uses different helper names.

- [ ] **Step 5: Run JSON round-trip test**

```bash
cd src/rust && cargo test --lib data::neural::tests::transformer_json_v2_save_load_roundtrip -- --nocapture
```
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/data/neural.rs
git commit -m "feat(nn): JSON v2 save/load for Transformer

save_json writes the 14-key weights dict (w_q..b_ffn2 + ln1/ln2 gamma/beta).
from_v2_json reads them back and calls rebuild_pe_offsets so PE offsets
match the loaded w_k / w_v (derived-at-load pattern, not part of the JSON)."
```

---

## Task 7: Rust NeuralNetModel::forward dispatch for Transformer

**Files:**
- Modify: `src/rust/src/data/neural.rs` (`NeuralNetModel::forward` match arm)
- Test: `src/rust/src/data/neural.rs` (integration-style)

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn neural_net_model_forward_transformer_threads_state() {
    let architecture = vec![
        LayerSpec::Dense {
            input_size: 4, output_size: 4, activation: Activation::Linear
        },
        LayerSpec::Transformer {
            d_model: 4, n_heads: 2, d_ffn: 8, n_seq: 3,
        },
        LayerSpec::Dense {
            input_size: 4, output_size: 2, activation: Activation::Linear
        },
    ];
    let n_params = NeuralNetModel::n_params_for_architecture(&architecture);
    let flat: Vec<f64> = (0..n_params).map(|i| ((i % 7) as f64) * 0.01).collect();
    let model = NeuralNetModel::from_flat_weights_v2(&architecture, None, &flat).unwrap();
    let mut state = NnState::for_model(&model);

    let x = vec![0.5, -0.3, 0.7, 0.1];
    // Drive the model for 5 steps; cache should saturate at n_seq=3.
    let mut outputs = Vec::new();
    for _ in 0..5 {
        outputs.push(model.forward(&mut state, &x));
    }
    // All outputs finite
    for o in &outputs {
        for v in o { assert!(v.is_finite()); }
        assert_eq!(o.len(), 2);
    }
    // Cache length should be 3 after 5 steps
    if let LayerState::Transformer { k_cache, .. } = &state.layer_states[1] {
        assert_eq!(k_cache.len(), 3);
    } else {
        panic!("expected Transformer state at layer 1");
    }
}
```

- [ ] **Step 2: Run test to confirm it fails**.

- [ ] **Step 3: Add the dispatch arm**

In `NeuralNetModel::forward`, find the per-layer match and add:

```rust
(Layer::Transformer(layer), LayerState::Transformer { k_cache, v_cache }) => {
    current = layer.forward(&current, k_cache, v_cache);
}
```

Also ensure any remaining `todo!()` placeholders from Task 4 are removed.

- [ ] **Step 4: Run tests**

```bash
cd src/rust && cargo test --lib data::neural::tests::neural_net_model_forward_transformer_threads_state -- --nocapture
```
Expected: 1 passed.

- [ ] **Step 5: Run the full Rust test suite to catch regressions**

```bash
cd src/rust && cargo test --lib 2>&1 | tail -30
```
Expected: all green. Any newly-failing test is a regression from incomplete match arms -- fix before moving on.

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/data/neural.rs
git commit -m "feat(nn): NeuralNetModel::forward dispatches to TransformerLayer"
```

---

## Task 8: Rust TomlLayerSpec::Transformer + config validator

**Files:**
- Modify: `src/rust/src/config.rs` (`TomlLayerSpec` enum + `to_layer_spec` match)
- Test: `src/rust/src/config.rs` (or `src/rust/tests/` if integration)

- [ ] **Step 1: Write the failing test**

Append to the existing `#[cfg(test)] mod tests` block in `config.rs`:

```rust
#[test]
fn toml_layer_spec_transformer_parses() {
    let toml_str = r#"
[[network.architecture]]
type = "transformer"
d_model = 32
n_heads = 4
d_ffn = 64
n_seq = 64
"#;
    #[derive(serde::Deserialize)]
    struct NetworkWrapper { network: Network }
    #[derive(serde::Deserialize)]
    struct Network { architecture: Vec<TomlLayerSpec> }
    let w: NetworkWrapper = toml::from_str(toml_str).unwrap();
    assert_eq!(w.network.architecture.len(), 1);
    let spec = w.network.architecture[0].to_layer_spec().unwrap();
    match spec {
        LayerSpec::Transformer { d_model, n_heads, d_ffn, n_seq } => {
            assert_eq!((d_model, n_heads, d_ffn, n_seq), (32, 4, 64, 64));
        }
        _ => panic!("wrong variant"),
    }
}

#[test]
fn toml_layer_spec_transformer_rejects_bad_heads() {
    let toml_str = r#"
[[network.architecture]]
type = "transformer"
d_model = 33
n_heads = 4
d_ffn = 64
n_seq = 64
"#;
    #[derive(serde::Deserialize)]
    struct NetworkWrapper { network: Network }
    #[derive(serde::Deserialize)]
    struct Network { architecture: Vec<TomlLayerSpec> }
    let w: NetworkWrapper = toml::from_str(toml_str).unwrap();
    let err = w.network.architecture[0].to_layer_spec().unwrap_err();
    assert!(format!("{err}").contains("not divisible"));
}
```

- [ ] **Step 2: Run tests to confirm they fail**.

- [ ] **Step 3: Extend `TomlLayerSpec`**

```rust
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum TomlLayerSpec {
    Dense       { input_size: usize, output_size: usize, activation: String },
    Gru         { input_size: usize, hidden_size: usize },
    Lstm        { input_size: usize, hidden_size: usize },
    Window      { input_size: usize, n_steps: usize },
    Transformer { d_model: usize, n_heads: usize, d_ffn: usize, n_seq: usize },
}

impl TomlLayerSpec {
    pub fn to_layer_spec(&self) -> Result<LayerSpec, ConfigError> {
        match self {
            // ...existing arms...
            TomlLayerSpec::Transformer { d_model, n_heads, d_ffn, n_seq } => {
                if *n_heads == 0 || *d_model % n_heads != 0 {
                    return Err(ConfigError::Invalid(format!(
                        "(transformer) d_model={d_model} not divisible by n_heads={n_heads}"
                    )));
                }
                if *d_model == 0 || *d_ffn == 0 || *n_seq == 0 {
                    return Err(ConfigError::Invalid(
                        "(transformer) all shape fields must be positive".into()
                    ));
                }
                Ok(LayerSpec::Transformer {
                    d_model: *d_model, n_heads: *n_heads, d_ffn: *d_ffn, n_seq: *n_seq,
                })
            }
        }
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cd src/rust && cargo test --lib config::tests::toml_layer_spec_transformer_parses \
    config::tests::toml_layer_spec_transformer_rejects_bad_heads
```
Expected: 2 passed.

- [ ] **Step 5: Full Rust test suite + clippy + fmt**

```bash
cd src/rust && cargo fmt --check && cargo clippy -p aerocapture -- -D warnings && cargo test --lib 2>&1 | tail -10
```
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/rust/src/config.rs
git commit -m "feat(config): TomlLayerSpec::Transformer parser + validator"
```

---

## Task 9: Rebuild PyO3 + verify flat_weights_to_json works

**Files:**
- Run: `maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml`
- Test: `tests/test_pyo3_transformer_flat_weights.py` (new file)

- [ ] **Step 1: Rebuild PyO3**

```bash
uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml
```
Expected: compiles without warnings; outputs `.so` into the venv.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_pyo3_transformer_flat_weights.py
"""Smoke test for aerocapture_rs.flat_weights_to_json with a Transformer layer.

Verifies the Rust path end-to-end: flat -> from_flat_weights_v2 -> save_json,
without touching Python V2Policy. This is the PSO hot path.
"""
import json
import aerocapture_rs
import numpy as np


def test_flat_weights_to_json_transformer_roundtrip(tmp_path):
    architecture = [
        {"type": "dense", "input_size": 8, "output_size": 4, "activation": "linear"},
        {"type": "transformer", "d_model": 4, "n_heads": 2, "d_ffn": 8, "n_seq": 3},
        {"type": "dense", "input_size": 4, "output_size": 2, "activation": "linear"},
    ]
    # Transformer params: 4*16 + 2*32 + 8 + 36 = 172
    # Dense 0: 8*4 + 4 = 36
    # Dense 2: 4*2 + 2 = 10
    # Total: 218
    rng = np.random.default_rng(0)
    flat = rng.standard_normal(218)

    out_path = tmp_path / "model.json"
    aerocapture_rs.flat_weights_to_json(architecture, flat.tolist(), str(out_path))

    with out_path.open() as f:
        obj = json.load(f)

    assert obj["format_version"] == 2
    assert len(obj["architecture"]) == 3
    assert obj["architecture"][1]["type"] == "transformer"
    assert obj["architecture"][1]["d_model"] == 4
    # weights dict has layer_0, layer_1, layer_2
    assert "layer_1" in obj["weights"]
    # Transformer layer has the 14-key weights schema
    layer_1 = obj["weights"]["layer_1"]
    for key in ["w_q", "b_q", "w_k", "b_k", "w_v", "b_v", "w_o", "b_o",
                "w_ffn1", "b_ffn1", "w_ffn2", "b_ffn2", "ln1", "ln2"]:
        assert key in layer_1, f"missing key {key}"
    assert "gamma" in layer_1["ln1"]
    assert "beta" in layer_1["ln1"]

    # Verify nn_forward loads it
    y = aerocapture_rs.nn_forward(str(out_path), [0.1] * 8)
    assert len(y) == 2
    assert all(np.isfinite(v) for v in y)
```

- [ ] **Step 3: Run test**

```bash
uv run pytest tests/test_pyo3_transformer_flat_weights.py -v
```
Expected: 1 passed.

- [ ] **Step 4: Commit**

```bash
git add tests/test_pyo3_transformer_flat_weights.py
git commit -m "test(pyo3): flat_weights_to_json + nn_forward work for Transformer"
```

---

## Task 10: Python TransformerSpec pydantic schema

**Files:**
- Modify: `src/python/aerocapture/training/rl/schemas.py`
- Test: `tests/test_rl_schemas.py` (or new `tests/test_transformer_spec.py` -- follow whichever pattern was used for Phase 2a/2b LSTM/Window spec tests)

- [ ] **Step 1: Check existing test file location**

```bash
grep -rn "class WindowSpec\|WindowSpec(" tests/ | head -5
```

- [ ] **Step 2: Write failing tests**

Add to the appropriate test file (following the Window/LSTM precedent):

```python
def test_transformer_spec_validates_shapes() -> None:
    spec = TransformerSpec(type="transformer", d_model=32, n_heads=4, d_ffn=64, n_seq=64)
    assert spec.d_model == 32
    assert spec.n_heads == 4


def test_transformer_spec_rejects_non_divisible_heads() -> None:
    with pytest.raises(ValidationError):
        TransformerSpec(type="transformer", d_model=33, n_heads=4, d_ffn=64, n_seq=64)


def test_transformer_spec_rejects_zero_fields() -> None:
    for kwargs in [
        dict(d_model=0, n_heads=1, d_ffn=1, n_seq=1),
        dict(d_model=4, n_heads=0, d_ffn=1, n_seq=1),
        dict(d_model=4, n_heads=2, d_ffn=0, n_seq=1),
        dict(d_model=4, n_heads=2, d_ffn=8, n_seq=0),
    ]:
        with pytest.raises(ValidationError):
            TransformerSpec(type="transformer", **kwargs)


def test_layerspec_discriminates_transformer() -> None:
    from aerocapture.training.rl.schemas import LayerSpec, TransformerSpec
    raw = {"type": "transformer", "d_model": 8, "n_heads": 2, "d_ffn": 16, "n_seq": 4}
    spec = TypeAdapter(LayerSpec).validate_python(raw)
    assert isinstance(spec, TransformerSpec)
```

- [ ] **Step 3: Run tests to confirm they fail** (`TransformerSpec` doesn't exist).

- [ ] **Step 4: Add `TransformerSpec` to `schemas.py`**

```python
class TransformerSpec(BaseModel):
    type: Literal["transformer"]
    d_model: int
    n_heads: int
    d_ffn: int
    n_seq: int

    @model_validator(mode="after")
    def _validate_shapes(self) -> "TransformerSpec":
        if self.d_model <= 0 or self.n_heads <= 0 or self.d_ffn <= 0 or self.n_seq <= 0:
            raise ValueError("all Transformer shape fields must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
            )
        return self


LayerSpec = Annotated[
    DenseSpec | GruSpec | LstmSpec | WindowSpec | TransformerSpec,
    Discriminator("type"),
]
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_rl_schemas.py -v -k transformer  # or the actual test file
```
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/rl/schemas.py tests/
git commit -m "feat(nn): Python TransformerSpec pydantic schema + discriminated union entry"
```

---

## Task 11: Python TransformerLayer torch module

**Files:**
- Create: `src/python/aerocapture/training/rl/layers/transformer.py`
- Test: `tests/test_python_transformer_layer.py`

- [ ] **Step 1: Write the failing unit test**

```python
# tests/test_python_transformer_layer.py
"""Unit tests for the Python TransformerLayer torch module.

Cross-language equivalence tests against Rust live in
test_rust_python_transformer_equivalence.py.
"""
import math
import torch
from aerocapture.training.rl.layers.transformer import TransformerLayer


def test_transformer_layer_output_shape() -> None:
    layer = TransformerLayer(d_model=16, n_heads=2, d_ffn=32, n_seq=8).double()
    state = layer.new_state(batch_size=1)
    x = torch.randn(1, 16, dtype=torch.float64)
    out, new_state = layer(x, state)
    assert out.shape == (1, 16)
    assert new_state[0].shape == (1, 1, 16)
    assert new_state[1].shape == (1, 1, 16)


def test_transformer_cache_grows_to_n_seq_then_saturates() -> None:
    layer = TransformerLayer(d_model=8, n_heads=2, d_ffn=16, n_seq=3).double()
    state = layer.new_state(batch_size=1)
    for step in range(6):
        x = torch.randn(1, 8, dtype=torch.float64)
        _, state = layer(x, state)
        expected = min(step + 1, 3)
        assert state[0].shape[1] == expected, f"step {step}: got {state[0].shape[1]}"


def test_transformer_residual_dominates_when_weights_zero() -> None:
    # Zero all projection + FFN weights, LN gamma=1 / beta=0 -> output == input.
    layer = TransformerLayer(d_model=4, n_heads=2, d_ffn=8, n_seq=3).double()
    with torch.no_grad():
        for lin in [layer.w_q, layer.w_k, layer.w_v, layer.w_o, layer.w_ffn1, layer.w_ffn2]:
            lin.weight.zero_()
            lin.bias.zero_()
        layer.ln1_gamma.fill_(1.0); layer.ln1_beta.zero_()
        layer.ln2_gamma.fill_(1.0); layer.ln2_beta.zero_()
    state = layer.new_state(batch_size=1)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    out, _ = layer(x, state)
    for i in range(4):
        assert abs(out[0, i].item() - x[0, i].item()) < 1e-12


def test_transformer_gelu_exact_vs_torch() -> None:
    # Spot-check that the layer's GELU is the exact form, not the tanh approx.
    layer = TransformerLayer(d_model=4, n_heads=2, d_ffn=8, n_seq=2).double()
    z = torch.tensor([1.0, -1.0, 2.5], dtype=torch.float64)
    ours = 0.5 * z * (1.0 + torch.special.erf(z * (1.0 / math.sqrt(2.0))))
    # torch.nn.functional.gelu defaults to exact
    theirs = torch.nn.functional.gelu(z)
    torch.testing.assert_close(ours, theirs, atol=1e-14, rtol=0)
```

- [ ] **Step 2: Run tests to confirm they fail**.

- [ ] **Step 3: Create `transformer.py`**

Create `src/python/aerocapture/training/rl/layers/transformer.py`:

```python
"""TransformerLayer (PyTorch mirror of the Rust implementation).

Cross-language contract (enforced by
tests/test_rust_python_transformer_equivalence.py):

- LayerNorm uses biased (1/N) variance with eps=1e-5 (torch.nn.LayerNorm default).
- GELU is the exact form: 0.5 * x * (1 + erf(x / sqrt(2))), via torch.special.erf.
- Softmax uses max-subtraction. Manual over the cache time axis for deterministic
  FIFO reduction matching the Rust VecDeque iteration order.
- Multi-head split is a contiguous slice along d_model: head h -> [h*d_head .. (h+1)*d_head].
- Positional encoding is relative-to-buffer: newest token at slot cache_len - 1.
- PE offsets for K/V are computed at forward time as (w_k.weight @ pe_table[:cache_len].T).T;
  no bias is included in the PE shift. Matches Rust's precomputed k_pe_offsets
  modulo iteration order (< 1e-10 tolerance).

Note: this module is consumed ONLY by the cross-language equivalence test. The
production PPO path raises NotImplementedError in build_layer; PSO bypasses this
module entirely and drives the Rust runtime via aerocapture_rs.nn_forward.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

_INV_SQRT2: float = 1.0 / math.sqrt(2.0)


def _build_sinusoidal_pe(n_seq: int, d_model: int) -> Tensor:
    """Match Rust build_pe_table iteration order: pos outer, i inner.

    Explicit f64 loop -- no broadcast / arange fusion -- so operand ordering
    matches the Rust sequential implementation.
    """
    pe = torch.zeros(n_seq, d_model, dtype=torch.float64)
    for pos in range(n_seq):
        for i in range(d_model):
            k = i // 2
            div = 10000.0 ** ((2.0 * k) / d_model)
            angle = pos / div
            pe[pos, i] = math.sin(angle) if i % 2 == 0 else math.cos(angle)
    return pe


def _manual_ln(x: Tensor, gamma: Tensor, beta: Tensor, eps: float) -> Tensor:
    # x: (batch, d_model)
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)  # biased (1/N)
    return (x - mean) / torch.sqrt(var + eps) * gamma + beta


def _manual_causal_attention(
    q: Tensor,           # (batch, d_model)
    k_eff: Tensor,       # (batch, cache_len, d_model)
    v_eff: Tensor,       # (batch, cache_len, d_model)
    n_heads: int,
    d_head: int,
) -> Tensor:
    batch, cache_len, d_model = k_eff.shape
    # Split heads: (batch, n_heads, d_head) for q, (batch, cache_len, n_heads, d_head) for k/v
    q_h = q.view(batch, n_heads, d_head)
    k_h = k_eff.view(batch, cache_len, n_heads, d_head)
    v_h = v_eff.view(batch, cache_len, n_heads, d_head)
    inv_sqrt_d = 1.0 / math.sqrt(d_head)

    # scores: (batch, n_heads, cache_len) = sum over d_head of q_h[b,h,:] * k_h[b,i,h,:]
    # Use einsum for clarity.
    scores = torch.einsum("bhd,bihd->bhi", q_h, k_h) * inv_sqrt_d
    # Max-subtraction softmax along cache_len dim
    max_scores, _ = scores.max(dim=-1, keepdim=True)
    exp_scores = torch.exp(scores - max_scores)
    weights = exp_scores / exp_scores.sum(dim=-1, keepdim=True)

    # Weighted sum: (batch, n_heads, d_head)
    head_out = torch.einsum("bhi,bihd->bhd", weights, v_h)
    return head_out.reshape(batch, n_heads * d_head)  # (batch, d_model)


class TransformerLayer(nn.Module):
    """Manual 1-layer Transformer block for 1-for-1 Rust equivalence."""

    def __init__(self, d_model: int, n_heads: int, d_ffn: int, n_seq: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_ffn = d_ffn
        self.n_seq = n_seq

        self.w_q = nn.Linear(d_model, d_model, bias=True)
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)
        self.w_o = nn.Linear(d_model, d_model, bias=True)

        self.w_ffn1 = nn.Linear(d_model, d_ffn, bias=True)
        self.w_ffn2 = nn.Linear(d_ffn, d_model, bias=True)

        self.ln1_gamma = nn.Parameter(torch.ones(d_model))
        self.ln1_beta = nn.Parameter(torch.zeros(d_model))
        self.ln2_gamma = nn.Parameter(torch.ones(d_model))
        self.ln2_beta = nn.Parameter(torch.zeros(d_model))

        self.register_buffer(
            "pe_table",
            _build_sinusoidal_pe(n_seq, d_model),
            persistent=False,
        )

    def forward(
        self,
        x: Tensor,                                # (batch, d_model)
        state: tuple[Tensor, Tensor],             # (k_cache, v_cache)
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        # 1. LN1
        x_norm1 = _manual_ln(x, self.ln1_gamma, self.ln1_beta, eps=1e-5)
        # 2. QKV
        q = self.w_q(x_norm1)
        k = self.w_k(x_norm1)
        v = self.w_v(x_norm1)
        # 3. Push into cache, evict oldest if over n_seq
        k_cache, v_cache = state
        k_cache = torch.cat([k_cache, k.unsqueeze(1)], dim=1)
        v_cache = torch.cat([v_cache, v.unsqueeze(1)], dim=1)
        if k_cache.shape[1] > self.n_seq:
            k_cache = k_cache[:, 1:]
            v_cache = v_cache[:, 1:]
        cache_len = k_cache.shape[1]
        # 4. PE offsets, relative-to-buffer
        pe_slice = self.pe_table[:cache_len].to(dtype=x.dtype, device=x.device)
        k_pe = (self.w_k.weight @ pe_slice.T).T  # (cache_len, d_model)
        v_pe = (self.w_v.weight @ pe_slice.T).T
        k_eff = k_cache + k_pe.unsqueeze(0)
        v_eff = v_cache + v_pe.unsqueeze(0)
        # 5. Attention + residual
        attn_out = _manual_causal_attention(q, k_eff, v_eff, self.n_heads, self.d_head)
        x1 = x + self.w_o(attn_out)
        # 6. LN2 + FFN + residual
        x_norm2 = _manual_ln(x1, self.ln2_gamma, self.ln2_beta, eps=1e-5)
        ffn_hidden = self.w_ffn1(x_norm2)
        ffn_hidden_act = 0.5 * ffn_hidden * (1.0 + torch.special.erf(ffn_hidden * _INV_SQRT2))
        ffn_out = self.w_ffn2(ffn_hidden_act)
        out = x1 + ffn_out
        return out, (k_cache, v_cache)

    def new_state(self, batch_size: int) -> tuple[Tensor, Tensor]:
        device = self.w_q.weight.device
        dtype = self.w_q.weight.dtype
        empty = torch.zeros(batch_size, 0, self.d_model, device=device, dtype=dtype)
        return (empty.clone(), empty.clone())
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_python_transformer_layer.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/rl/layers/transformer.py tests/test_python_transformer_layer.py
git commit -m "feat(nn): Python TransformerLayer torch module (manual LN/GELU/softmax/MHA)

Mirror of the Rust TransformerLayer for cross-language equivalence.
Not used on the PSO hot path; consumed only by the equivalence test
(PPO path raises NotImplementedError in build_layer)."
```

---

## Task 12: Python build_layer Transformer rejection

**Files:**
- Modify: `src/python/aerocapture/training/rl/layers/__init__.py`
- Test: `tests/test_transformer_ppo_rejection.py` (Task 20 lands this as a proper gate; this task adds a minimal build_layer-only test to fail fast)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_layer_transformer_rejection.py
"""Assert build_layer raises NotImplementedError for TransformerSpec.

The full PPO-rejection test (including load_policy_from_json + JSON v2 file
construction) lives in tests/test_transformer_ppo_rejection.py (Task 21).
"""
import pytest
from aerocapture.training.rl.layers import build_layer
from aerocapture.training.rl.schemas import TransformerSpec


def test_build_layer_rejects_transformer_spec() -> None:
    spec = TransformerSpec(type="transformer", d_model=4, n_heads=2, d_ffn=8, n_seq=4)
    with pytest.raises(NotImplementedError, match="Transformer is PSO-only in Phase 3a"):
        build_layer(spec)
```

- [ ] **Step 2: Run to confirm it fails**.

- [ ] **Step 3: Add the rejection branch**

In `src/python/aerocapture/training/rl/layers/__init__.py`, extend `build_layer`:

```python
def build_layer(spec: LayerSpec) -> nn.Module:
    if isinstance(spec, DenseSpec):       return DenseLayer(spec.input_size, spec.output_size, spec.activation)
    if isinstance(spec, GruSpec):         return GruLayer(spec.input_size, spec.hidden_size)
    if isinstance(spec, LstmSpec):        return LstmLayer(spec.input_size, spec.hidden_size)
    if isinstance(spec, WindowSpec):
        raise NotImplementedError(
            "Window-MLP is PSO-only in Phase 2b; PPO use deferred. "
            "See docs/superpowers/specs/2026-04-20-phase-2b-window-mlp-design.md"
        )
    if isinstance(spec, TransformerSpec):
        raise NotImplementedError(
            "Transformer is PSO-only in Phase 3a; PPO use deferred. "
            "See docs/superpowers/specs/2026-04-22-phase-3a-transformer-mvp-design.md"
        )
    raise TypeError(f"Unknown layer spec: {spec!r}")
```

- [ ] **Step 4: Run test**

```bash
uv run pytest tests/test_build_layer_transformer_rejection.py -v
```
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/rl/layers/__init__.py tests/test_build_layer_transformer_rejection.py
git commit -m "feat(nn): build_layer raises NotImplementedError for TransformerSpec"
```

---

## Task 13: Python _transformer_specs + config.py helper arms

**Files:**
- Modify: `src/python/aerocapture/training/encoding.py` (add `_transformer_specs` + `_layer_param_specs` arm)
- Modify: `src/python/aerocapture/training/config.py` (`_layer_n_params`, `_layer_output_size`, `describe_architecture` arms)
- Test: `tests/test_transformer_encoding.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_transformer_encoding.py
"""Encoding + config helper arms for TransformerSpec."""
from aerocapture.training.config import _layer_n_params, _layer_output_size
from aerocapture.training.encoding import _layer_param_specs
from aerocapture.training.rl.schemas import TransformerSpec


def test_transformer_n_params_formula() -> None:
    spec = TransformerSpec(type="transformer", d_model=32, n_heads=4, d_ffn=64, n_seq=64)
    # 4*d^2 + 2*ffn*d + ffn + 9*d
    expected = 4 * 32 * 32 + 2 * 64 * 32 + 64 + 9 * 32
    assert _layer_n_params(spec) == expected == 8544


def test_transformer_output_size_is_d_model() -> None:
    spec = TransformerSpec(type="transformer", d_model=32, n_heads=4, d_ffn=64, n_seq=64)
    assert _layer_output_size(spec) == 32


def test_transformer_param_specs_length_matches_n_params() -> None:
    spec = TransformerSpec(type="transformer", d_model=8, n_heads=2, d_ffn=16, n_seq=4)
    specs = _layer_param_specs(spec, bound_multiplier=1.0)
    assert len(specs) == _layer_n_params(spec)
```

- [ ] **Step 2: Run tests to confirm they fail**.

- [ ] **Step 3: Add `_transformer_specs` to `encoding.py`**

```python
def _transformer_specs(spec: TransformerSpec, bound_multiplier: float) -> list[ParamSpec]:
    """ParamSpec list in canonical flat order (spec section 3.3).

    INVARIANT: ordering MUST match Rust's TransformerLayer::to_flat / from_flat
    cursor advance order (w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o, w_ffn1, b_ffn1,
    w_ffn2, b_ffn2, ln1_gamma, ln1_beta, ln2_gamma, ln2_beta). Regression-covered
    by the flat-chromosome round-trip test in Rust.

    Bounds:
      - Projection matrices (Q/K/V/O): Xavier uniform, bound = sqrt(6/(2*d_model)) * mul
      - FFN1/FFN2:                     Xavier uniform, bound = sqrt(6/(d_model+d_ffn)) * mul
      - Biases:                        tight uniform [-0.1*mul, 0.1*mul]
      - LN gamma:                      uniform around 1.0: [1 - 0.01*mul, 1 + 0.01*mul]
      - LN beta:                       tight uniform [-0.01*mul, 0.01*mul]
    """
    from math import sqrt

    mul = bound_multiplier
    d_model = spec.d_model
    d_ffn = spec.d_ffn

    proj_bound = sqrt(6.0 / (2.0 * d_model)) * mul
    ffn_bound  = sqrt(6.0 / (d_model + d_ffn)) * mul
    bias_bound = 0.1 * mul
    gamma_lo, gamma_hi = 1.0 - 0.01 * mul, 1.0 + 0.01 * mul
    beta_bound = 0.01 * mul

    specs: list[ParamSpec] = []
    # 4 projection matrices
    for _ in range(4):
        specs.extend(ParamSpec(name="", lo=-proj_bound, hi=proj_bound)
                     for _ in range(d_model * d_model))
        specs.extend(ParamSpec(name="", lo=-bias_bound, hi=bias_bound)
                     for _ in range(d_model))
    # FFN1: (d_ffn, d_model) + (d_ffn,)
    specs.extend(ParamSpec(name="", lo=-ffn_bound, hi=ffn_bound)
                 for _ in range(d_ffn * d_model))
    specs.extend(ParamSpec(name="", lo=-bias_bound, hi=bias_bound)
                 for _ in range(d_ffn))
    # FFN2: (d_model, d_ffn) + (d_model,)
    specs.extend(ParamSpec(name="", lo=-ffn_bound, hi=ffn_bound)
                 for _ in range(d_model * d_ffn))
    specs.extend(ParamSpec(name="", lo=-bias_bound, hi=bias_bound)
                 for _ in range(d_model))
    # LN1 gamma + beta
    specs.extend(ParamSpec(name="", lo=gamma_lo, hi=gamma_hi) for _ in range(d_model))
    specs.extend(ParamSpec(name="", lo=-beta_bound, hi=beta_bound) for _ in range(d_model))
    # LN2 gamma + beta
    specs.extend(ParamSpec(name="", lo=gamma_lo, hi=gamma_hi) for _ in range(d_model))
    specs.extend(ParamSpec(name="", lo=-beta_bound, hi=beta_bound) for _ in range(d_model))
    return specs
```

Extend `_layer_param_specs`:

```python
def _layer_param_specs(spec: LayerSpec, bound_multiplier: float) -> list[ParamSpec]:
    if isinstance(spec, DenseSpec):       return _dense_specs(spec, bound_multiplier)
    if isinstance(spec, GruSpec):         return _gru_specs(spec, bound_multiplier)
    if isinstance(spec, LstmSpec):        return _lstm_specs(spec, bound_multiplier)
    if isinstance(spec, WindowSpec):      return []
    if isinstance(spec, TransformerSpec): return _transformer_specs(spec, bound_multiplier)
    raise TypeError(f"Unknown layer spec: {spec!r}")
```

- [ ] **Step 4: Extend `config.py`**

Add arms to `_layer_n_params`, `_layer_output_size`, and `describe_architecture` for `TransformerSpec`. The existing Window/GRU/LSTM patterns are the template.

```python
# config.py

def _layer_n_params(spec: LayerSpec) -> int:
    # ...existing arms...
    if isinstance(spec, TransformerSpec):
        return (4 * spec.d_model * spec.d_model
                + 2 * spec.d_ffn * spec.d_model
                + spec.d_ffn
                + 9 * spec.d_model)
    raise TypeError(...)


def _layer_output_size(spec: LayerSpec) -> int:
    # ...existing arms...
    if isinstance(spec, TransformerSpec):
        return spec.d_model
    raise TypeError(...)


def describe_architecture(architecture: list[LayerSpec]) -> str:
    parts = []
    for spec in architecture:
        # ...existing arms...
        if isinstance(spec, TransformerSpec):
            parts.append(
                f"Transformer(d_model={spec.d_model}, n_heads={spec.n_heads}, "
                f"d_ffn={spec.d_ffn}, n_seq={spec.n_seq})"
            )
        # ...
    return " -> ".join(parts)
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_transformer_encoding.py -v
```
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/python/aerocapture/training/encoding.py src/python/aerocapture/training/config.py tests/test_transformer_encoding.py
git commit -m "feat(nn): _transformer_specs + _layer_n_params / _layer_output_size / describe_architecture arms"
```

---

## Task 14: Python init_v2_population Transformer arm

**Files:**
- Modify: `src/python/aerocapture/training/initialization_v2.py`
- Test: `tests/test_init_v2_transformer.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_init_v2_transformer.py
import numpy as np
from aerocapture.training.initialization_v2 import init_v2_population
from aerocapture.training.rl.schemas import DenseSpec, TransformerSpec


def test_init_v2_population_transformer_slab_shape_and_bounds() -> None:
    architecture = [
        DenseSpec(type="dense", input_size=8, output_size=4, activation="linear"),
        TransformerSpec(type="transformer", d_model=4, n_heads=2, d_ffn=8, n_seq=4),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    n_pop = 16
    rng = np.random.default_rng(42)
    pop = init_v2_population(architecture, n_pop, bound_multiplier=1.0, rng=rng)
    # Dense 0: 8*4 + 4 = 36
    # Transformer: 4*16 + 2*32 + 8 + 36 = 64 + 64 + 8 + 36 = 172
    # Dense 2: 4*2 + 2 = 10
    # Total: 218
    assert pop.shape == (n_pop, 218)
    # All finite
    assert np.all(np.isfinite(pop))
```

- [ ] **Step 2: Run test to confirm it fails**.

- [ ] **Step 3: Add the Transformer branch**

Extend `init_v2_population` (find the dispatch block; follow the pattern used for GRU/LSTM/Window):

```python
elif isinstance(spec, TransformerSpec):
    # Use the ParamSpec-derived bounds directly: _transformer_specs already
    # encodes Xavier on projections/FFN, N(1, 0.01*mul) on gamma (via uniform
    # bound 0.01*mul around 1.0), and tight uniform on biases + beta.
    specs = _transformer_specs(spec, bound_multiplier)
    n = len(specs)
    slab = np.empty((n_pop, n))
    for j, ps in enumerate(specs):
        slab[:, j] = rng.uniform(ps.lo, ps.hi, size=n_pop)
    pop_slabs.append(slab)
```

The import `from aerocapture.training.encoding import _transformer_specs` is fine; it's already in the module's namespace via the existing `_dense_specs` / `_gru_specs` / `_lstm_specs` imports. Adjust if the module uses a different dispatch pattern.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_init_v2_transformer.py -v
```
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/initialization_v2.py tests/test_init_v2_transformer.py
git commit -m "feat(nn): init_v2_population Transformer arm (Xavier projections + near-identity LN)"
```

---

## Task 15: Python export_v2_policy_to_json Transformer branch + obs-norm guard

**Files:**
- Modify: `src/python/aerocapture/training/rl/export.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_export_v2_transformer.py
import json
import pytest
import torch
from aerocapture.training.rl.export import export_v2_policy_to_json
from aerocapture.training.rl.schemas import DenseSpec, TransformerSpec


def _make_policy(architecture, dtype=torch.float64):
    from aerocapture.training.rl.layers import DenseLayer
    from aerocapture.training.rl.layers.transformer import TransformerLayer
    from aerocapture.training.rl.policy import V2Policy  # adjust import path if needed

    # V2Policy's normal path goes through build_layer which would reject Transformer.
    # For the export test, construct layers directly (bypass build_layer).
    layers = []
    for spec in architecture:
        if isinstance(spec, DenseSpec):
            layers.append(DenseLayer(spec.input_size, spec.output_size, spec.activation))
        elif isinstance(spec, TransformerSpec):
            layers.append(TransformerLayer(spec.d_model, spec.n_heads, spec.d_ffn, spec.n_seq))
        else:
            raise TypeError(spec)
    policy = V2Policy.__new__(V2Policy)  # bypass __init__ if it goes through build_layer
    torch.nn.Module.__init__(policy)
    policy.architecture = architecture
    policy.layers = torch.nn.ModuleList(layers)
    policy.log_std = torch.nn.Parameter(torch.full((2,), -0.5, dtype=dtype))
    return policy.to(dtype=dtype)


def test_export_transformer_writes_14_key_weights(tmp_path):
    architecture = [
        DenseSpec(type="dense", input_size=8, output_size=4, activation="linear"),
        TransformerSpec(type="transformer", d_model=4, n_heads=2, d_ffn=8, n_seq=4),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    policy = _make_policy(architecture)
    out_path = tmp_path / "model.json"
    export_v2_policy_to_json(policy, str(out_path), obs_normalizer=None)

    obj = json.loads(out_path.read_text())
    assert obj["format_version"] == 2
    layer_1 = obj["weights"]["layer_1"]
    for key in ["w_q", "b_q", "w_k", "b_k", "w_v", "b_v", "w_o", "b_o",
                "w_ffn1", "b_ffn1", "w_ffn2", "b_ffn2", "ln1", "ln2"]:
        assert key in layer_1


def test_export_obs_normalizer_rejects_transformer_as_first_layer(tmp_path):
    # Transformer as layer 0 -> cannot bake in affine shift.
    architecture = [
        TransformerSpec(type="transformer", d_model=4, n_heads=2, d_ffn=8, n_seq=4),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    policy = _make_policy(architecture)
    # Minimal fake obs_normalizer with mean / std attrs
    class FakeNorm:
        mean = torch.zeros(4)
        std = torch.ones(4)
    with pytest.raises(NotImplementedError, match="DenseSpec"):
        export_v2_policy_to_json(policy, str(tmp_path / "model.json"),
                                  obs_normalizer=FakeNorm())
```

If `V2Policy` requires specific init args that aren't easily bypassed, adapt the helper to whatever the existing export tests use (check Phase 2b's `test_rust_python_window_equivalence.py` for the pattern).

- [ ] **Step 2: Run tests to confirm they fail**.

- [ ] **Step 3: Extend `export_v2_policy_to_json`**

In the per-layer loop (after existing Dense / GRU / LSTM / Window branches):

```python
if isinstance(spec, TransformerSpec):
    arch_entry = {
        "type": "transformer",
        "d_model": spec.d_model,
        "n_heads": spec.n_heads,
        "d_ffn":   spec.d_ffn,
        "n_seq":   spec.n_seq,
    }
    architecture_json.append(arch_entry)
    weights[f"layer_{i}"] = {
        "w_q": layer.w_q.weight.detach().cpu().tolist(),
        "b_q": layer.w_q.bias.detach().cpu().tolist(),
        "w_k": layer.w_k.weight.detach().cpu().tolist(),
        "b_k": layer.w_k.bias.detach().cpu().tolist(),
        "w_v": layer.w_v.weight.detach().cpu().tolist(),
        "b_v": layer.w_v.bias.detach().cpu().tolist(),
        "w_o": layer.w_o.weight.detach().cpu().tolist(),
        "b_o": layer.w_o.bias.detach().cpu().tolist(),
        "w_ffn1": layer.w_ffn1.weight.detach().cpu().tolist(),
        "b_ffn1": layer.w_ffn1.bias.detach().cpu().tolist(),
        "w_ffn2": layer.w_ffn2.weight.detach().cpu().tolist(),
        "b_ffn2": layer.w_ffn2.bias.detach().cpu().tolist(),
        "ln1": {"gamma": layer.ln1_gamma.detach().cpu().tolist(),
                "beta":  layer.ln1_beta.detach().cpu().tolist()},
        "ln2": {"gamma": layer.ln2_gamma.detach().cpu().tolist(),
                "beta":  layer.ln2_beta.detach().cpu().tolist()},
    }
    continue
```

Extend the obs-normalizer guard:

```python
if obs_normalizer is not None and isinstance(
    policy.architecture[0], (GruSpec, LstmSpec, WindowSpec, TransformerSpec)
):
    raise NotImplementedError(
        f"Obs normalizer bake-in into layer 0 only supports DenseSpec, "
        f"got {type(policy.architecture[0]).__name__}. Export without the bake-in."
    )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_export_v2_transformer.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/rl/export.py tests/test_export_v2_transformer.py
git commit -m "feat(nn): export_v2_policy_to_json Transformer branch + obs-norm guard extension"
```

---

## Task 16: Python load_policy_from_json Transformer rejection

**Files:**
- Modify: `src/python/aerocapture/training/model_io.py`

- [ ] **Step 1: Write failing test**

Extend `tests/test_build_layer_transformer_rejection.py` (or add new `tests/test_load_policy_transformer_rejection.py`):

```python
def test_load_policy_from_json_rejects_transformer(tmp_path):
    import json, pytest
    from aerocapture.training.model_io import load_policy_from_json

    model_json = {
        "format_version": 2,
        "architecture": [
            {"type": "dense", "input_size": 8, "output_size": 4, "activation": "linear"},
            {"type": "transformer", "d_model": 4, "n_heads": 2, "d_ffn": 8, "n_seq": 4},
            {"type": "dense", "input_size": 4, "output_size": 2, "activation": "linear"},
        ],
        "weights": {
            "layer_0": {"w": [[0.0]*8]*4, "b": [0.0]*4},
            "layer_1": {
                "w_q": [[0.0]*4]*4, "b_q": [0.0]*4,
                "w_k": [[0.0]*4]*4, "b_k": [0.0]*4,
                "w_v": [[0.0]*4]*4, "b_v": [0.0]*4,
                "w_o": [[0.0]*4]*4, "b_o": [0.0]*4,
                "w_ffn1": [[0.0]*4]*8, "b_ffn1": [0.0]*8,
                "w_ffn2": [[0.0]*8]*4, "b_ffn2": [0.0]*4,
                "ln1": {"gamma": [1.0]*4, "beta": [0.0]*4},
                "ln2": {"gamma": [1.0]*4, "beta": [0.0]*4},
            },
            "layer_2": {"w": [[0.0]*4]*2, "b": [0.0]*2},
        },
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps(model_json))
    with pytest.raises(NotImplementedError, match="Transformer"):
        load_policy_from_json(str(path))
```

- [ ] **Step 2: Run to confirm failure**.

- [ ] **Step 3: Extend the rejection in `model_io.py::load_policy_from_json`**

```python
if any(isinstance(spec, (WindowSpec, TransformerSpec)) for spec in architecture):
    raise NotImplementedError(
        "Transformer / Window-MLP are PSO-only phases; load_policy_from_json "
        "is a PPO/SAC entry point that cannot construct V2Policy with these "
        "layers. See docs/superpowers/specs/2026-04-22-phase-3a-transformer-mvp-design.md"
    )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_build_layer_transformer_rejection.py tests/test_load_policy_transformer_rejection.py -v
```
Expected: passed.

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/model_io.py tests/
git commit -m "feat(nn): load_policy_from_json raises NotImplementedError on TransformerSpec"
```

---

## Task 17: Cross-language equivalence test

**Files:**
- Create: `tests/test_rust_python_transformer_equivalence.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_rust_python_transformer_equivalence.py
"""Cross-language Transformer equivalence (Rust vs Python mirror).

Architecture: Dense(8 -> 16, linear) -> Transformer(d_model=16, n_heads=2,
d_ffn=32, n_seq=8) -> Dense(16 -> 2, linear). Runs 100 f64 inputs (more than
n_seq=8 -> exercises cache saturation and eviction). Asserts max abs diff
< 1e-10; target machine epsilon.
"""
import json
import math

import aerocapture_rs
import numpy as np
import pytest
import torch
from aerocapture.training.rl.layers import DenseLayer
from aerocapture.training.rl.layers.transformer import TransformerLayer


@pytest.mark.slow
def test_transformer_rust_python_equivalence(tmp_path):
    rng = np.random.default_rng(seed=42)
    d_model = 16
    n_heads = 2
    d_ffn = 32
    n_seq = 8

    # 1) Build a Python model in f64, with DenseLayer + TransformerLayer + DenseLayer
    dense_in = DenseLayer(input_size=8, output_size=16, activation="linear").double()
    transformer = TransformerLayer(d_model=d_model, n_heads=n_heads, d_ffn=d_ffn, n_seq=n_seq).double()
    dense_out = DenseLayer(input_size=16, output_size=2, activation="linear").double()

    # Randomly initialize weights (f64). Use torch.manual_seed for reproducibility.
    torch.manual_seed(0)
    with torch.no_grad():
        for lin in [dense_in.linear, transformer.w_q, transformer.w_k, transformer.w_v,
                     transformer.w_o, transformer.w_ffn1, transformer.w_ffn2, dense_out.linear]:
            torch.nn.init.uniform_(lin.weight, -0.1, 0.1)
            torch.nn.init.uniform_(lin.bias, -0.05, 0.05)
        torch.nn.init.uniform_(transformer.ln1_gamma, 0.9, 1.1)
        torch.nn.init.uniform_(transformer.ln1_beta, -0.05, 0.05)
        torch.nn.init.uniform_(transformer.ln2_gamma, 0.9, 1.1)
        torch.nn.init.uniform_(transformer.ln2_beta, -0.05, 0.05)

    # 2) Serialize to v2 JSON (manual -- export_v2_policy_to_json expects V2Policy,
    #    but we want to sidestep the build_layer rejection for the test).
    architecture = [
        {"type": "dense", "input_size": 8, "output_size": 16, "activation": "linear"},
        {"type": "transformer", "d_model": d_model, "n_heads": n_heads, "d_ffn": d_ffn, "n_seq": n_seq},
        {"type": "dense", "input_size": 16, "output_size": 2, "activation": "linear"},
    ]
    weights = {
        "layer_0": {
            "w": dense_in.linear.weight.detach().tolist(),
            "b": dense_in.linear.bias.detach().tolist(),
        },
        "layer_1": {
            "w_q": transformer.w_q.weight.detach().tolist(),
            "b_q": transformer.w_q.bias.detach().tolist(),
            "w_k": transformer.w_k.weight.detach().tolist(),
            "b_k": transformer.w_k.bias.detach().tolist(),
            "w_v": transformer.w_v.weight.detach().tolist(),
            "b_v": transformer.w_v.bias.detach().tolist(),
            "w_o": transformer.w_o.weight.detach().tolist(),
            "b_o": transformer.w_o.bias.detach().tolist(),
            "w_ffn1": transformer.w_ffn1.weight.detach().tolist(),
            "b_ffn1": transformer.w_ffn1.bias.detach().tolist(),
            "w_ffn2": transformer.w_ffn2.weight.detach().tolist(),
            "b_ffn2": transformer.w_ffn2.bias.detach().tolist(),
            "ln1": {"gamma": transformer.ln1_gamma.detach().tolist(),
                    "beta":  transformer.ln1_beta.detach().tolist()},
            "ln2": {"gamma": transformer.ln2_gamma.detach().tolist(),
                    "beta":  transformer.ln2_beta.detach().tolist()},
        },
        "layer_2": {
            "w": dense_out.linear.weight.detach().tolist(),
            "b": dense_out.linear.bias.detach().tolist(),
        },
    }
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps({
        "format_version": 2,
        "architecture": architecture,
        "weights": weights,
    }))

    # 3) Generate 100 random f64 inputs
    inputs = rng.standard_normal((100, 8))

    # 4) Rust: thread NnState through nn_forward_sequence
    rust_outs = aerocapture_rs.nn_forward_sequence(str(model_path), inputs.tolist())
    rust_outs = np.asarray(rust_outs, dtype=np.float64)  # (100, 2)

    # 5) Python: thread tuple state through the three layers
    py_outs = np.empty((100, 2), dtype=np.float64)
    state = transformer.new_state(batch_size=1)
    dense_in.eval(); transformer.eval(); dense_out.eval()
    with torch.no_grad():
        for t in range(100):
            x = torch.tensor(inputs[t:t+1], dtype=torch.float64)  # (1, 8)
            h = dense_in(x)
            h, state = transformer(h, state)
            y = dense_out(h)
            py_outs[t] = y.squeeze(0).numpy()

    # 6) Assert bit-equivalence
    diff = np.abs(rust_outs - py_outs)
    max_diff = diff.max()
    print(f"max abs diff: {max_diff:.2e}")
    assert max_diff < 1e-10, f"cross-language mismatch: max diff = {max_diff}"
```

If `DenseLayer.forward` doesn't exist or uses a different signature than `.linear`, check Phase 2b's equivalence test for the existing invocation pattern and adapt.

- [ ] **Step 2: Run test**

```bash
uv run pytest tests/test_rust_python_transformer_equivalence.py -v
```
Expected: 1 passed, `max abs diff` printed (~1e-14 to 1e-15).

If it fails with diffs > 1e-10, inspect: PE iteration order, softmax max-subtract order, LN biased vs Bessel variance, GELU exact vs approximation, matrix row-major vs column-major convention.

- [ ] **Step 3: Commit**

```bash
git add tests/test_rust_python_transformer_equivalence.py
git commit -m "test(nn): cross-language Transformer equivalence at machine epsilon"
```

---

## Task 18: Warm-up test (cache growth semantics)

**Files:**
- Create: `tests/test_transformer_warmup.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_transformer_warmup.py
"""Transformer cache grows organically from 0 to n_seq, then saturates.

Verifies (a) no zero-padding, (b) deterministic output, (c) output matches
a reference implementation that explicitly slices the cache.
"""
import aerocapture_rs
import json
import numpy as np
import pytest
import torch
from aerocapture.training.rl.layers.transformer import TransformerLayer


@pytest.mark.slow
def test_transformer_cache_warmup(tmp_path):
    d_model, n_heads, d_ffn, n_seq = 8, 2, 16, 4
    transformer = TransformerLayer(d_model=d_model, n_heads=n_heads, d_ffn=d_ffn, n_seq=n_seq).double()
    torch.manual_seed(1)
    with torch.no_grad():
        for lin in [transformer.w_q, transformer.w_k, transformer.w_v, transformer.w_o,
                     transformer.w_ffn1, transformer.w_ffn2]:
            torch.nn.init.uniform_(lin.weight, -0.1, 0.1)
            torch.nn.init.uniform_(lin.bias, -0.05, 0.05)

    architecture = [{"type": "transformer", "d_model": d_model, "n_heads": n_heads,
                     "d_ffn": d_ffn, "n_seq": n_seq}]
    weights = {
        "layer_0": {
            "w_q": transformer.w_q.weight.detach().tolist(),
            "b_q": transformer.w_q.bias.detach().tolist(),
            "w_k": transformer.w_k.weight.detach().tolist(),
            "b_k": transformer.w_k.bias.detach().tolist(),
            "w_v": transformer.w_v.weight.detach().tolist(),
            "b_v": transformer.w_v.bias.detach().tolist(),
            "w_o": transformer.w_o.weight.detach().tolist(),
            "b_o": transformer.w_o.bias.detach().tolist(),
            "w_ffn1": transformer.w_ffn1.weight.detach().tolist(),
            "b_ffn1": transformer.w_ffn1.bias.detach().tolist(),
            "w_ffn2": transformer.w_ffn2.weight.detach().tolist(),
            "b_ffn2": transformer.w_ffn2.bias.detach().tolist(),
            "ln1": {"gamma": transformer.ln1_gamma.detach().tolist(),
                    "beta":  transformer.ln1_beta.detach().tolist()},
            "ln2": {"gamma": transformer.ln2_gamma.detach().tolist(),
                    "beta":  transformer.ln2_beta.detach().tolist()},
        },
    }
    model_path = tmp_path / "m.json"
    model_path.write_text(json.dumps({"format_version": 2, "architecture": architecture, "weights": weights}))

    rng = np.random.default_rng(7)
    inputs = rng.standard_normal((3, d_model))  # 3 steps, fewer than n_seq=4
    rust_outs = np.asarray(aerocapture_rs.nn_forward_sequence(str(model_path), inputs.tolist()))

    # Reference Python path
    state = transformer.new_state(batch_size=1)
    py_outs = np.empty_like(rust_outs)
    for t in range(3):
        x = torch.tensor(inputs[t:t+1], dtype=torch.float64)
        out, state = transformer(x, state)
        py_outs[t] = out.squeeze(0).detach().numpy()
        # After step t, cache length = t + 1 (still growing)
        assert state[0].shape[1] == t + 1
        assert state[1].shape[1] == t + 1

    diff = np.abs(rust_outs - py_outs).max()
    assert diff < 1e-10, f"warmup mismatch: {diff}"

    # Determinism: run again, expect identical output
    rust_outs_2 = np.asarray(aerocapture_rs.nn_forward_sequence(str(model_path), inputs.tolist()))
    assert np.array_equal(rust_outs, rust_outs_2)
```

- [ ] **Step 2: Run**

```bash
uv run pytest tests/test_transformer_warmup.py -v
```
Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_transformer_warmup.py
git commit -m "test(nn): Transformer cache grows organically and is deterministic"
```

---

## Task 19: Training config + compare_guidance + train_all.sh registration

**Files:**
- Create: `configs/training/msr_aller_transformer_pso_train.toml`
- Modify: `src/python/aerocapture/training/compare_guidance.py`
- Modify: `train_all.sh`

- [ ] **Step 1: Create the training TOML**

```toml
# configs/training/msr_aller_transformer_pso_train.toml
base = [
    "./common.toml",
    "../missions/mars.toml",
]

[simulation]
mission = "msr_aller"
guidance = "neural_network"

[data]
neural_network = "training_output/neural_network_transformer_pso/best_model.json"

[network]
input_mask = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

[[network.architecture]]
type = "dense"
input_size = 23
output_size = 32
activation = "linear"

[[network.architecture]]
type = "transformer"
d_model = 32
n_heads = 4
d_ffn = 64
n_seq = 64

[[network.architecture]]
type = "dense"
input_size = 32
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
```

- [ ] **Step 2: Register in `compare_guidance.py`**

Extend:

```python
SCHEMES = [
    # ...,
    "neural_network_transformer_pso",
]

SCHEME_TRAINING_CONFIGS = {
    # ...,
    "neural_network_transformer_pso": "configs/training/msr_aller_transformer_pso_train.toml",
}

_NN_DEPLOY_SCHEMES = {
    # ...,
    "neural_network_transformer_pso",
}
```

- [ ] **Step 3: Add `train_all.sh` aliases**

```bash
# train_all.sh (case statement)
    transformer_pso|nn_transformer_pso|transformer)
        scheme="neural_network_transformer_pso"
        ;;
```

- [ ] **Step 4: Smoke test the TOML parses end-to-end**

```bash
uv run python -c "
from aerocapture.training.toml_utils import load_toml_with_bases
cfg = load_toml_with_bases('configs/training/msr_aller_transformer_pso_train.toml')
arch = cfg['network']['architecture']
assert len(arch) == 3
assert arch[1]['type'] == 'transformer'
print('OK', arch)
"
```
Expected: prints OK + architecture dump.

- [ ] **Step 5: Commit**

```bash
git add configs/training/msr_aller_transformer_pso_train.toml \
        src/python/aerocapture/training/compare_guidance.py \
        train_all.sh
git commit -m "feat(configs): Transformer PSO training config + compare_guidance + train_all.sh

Architecture: Dense(23->32) -> Transformer(d_model=32, n_heads=4, d_ffn=64,
n_seq=64) -> Dense(32->2). 9,378 trainable params. PSO n_pop=64 n_gen=2000
with adaptive seed strategy."
```

---

## Task 20: PSO smoke test + CI wiring

**Files:**
- Create: `tests/test_transformer_pso_smoke.py`
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Write the smoke test**

```python
# tests/test_transformer_pso_smoke.py
"""2-gen PSO smoke test on a reduced Transformer architecture.

Asserts best_model.json is v2 with ['dense', 'transformer', 'dense']
architecture and nn_forward returns a finite 2-tuple.
"""
import json
import pytest
import shutil
import tempfile
from pathlib import Path

import aerocapture_rs
import numpy as np


@pytest.mark.slow
def test_transformer_pso_2_gen_smoke():
    """Reduced Transformer trains for 2 PSO generations without crashing."""
    # Create a reduced training TOML in a temp dir with overrides.
    src_toml = Path("configs/training/msr_aller_transformer_pso_train.toml")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out_dir = tmp_path / "training_output" / "neural_network_transformer_pso"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Override via CLI using --n-gen 2 --n-pop 16 --training-n-sims 8
        import subprocess
        result = subprocess.run([
            "uv", "run", "python", "-m", "aerocapture.training.train",
            str(src_toml),
            "--n-gen", "2",
            "--n-pop", "16",
            "--no-tui",
            "--skip-report",
            "--training-n-sims", "8",
            "--output-dir", str(out_dir.parent),
        ], capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            print("STDOUT:", result.stdout[-2000:])
            print("STDERR:", result.stderr[-2000:])
        assert result.returncode == 0

        best_model = out_dir / "best_model.json"
        assert best_model.exists()
        obj = json.loads(best_model.read_text())
        assert obj["format_version"] == 2
        assert [a["type"] for a in obj["architecture"]] == ["dense", "transformer", "dense"]

        # nn_forward returns finite 2-tuple
        y = aerocapture_rs.nn_forward(str(best_model), [0.1] * 23)
        assert len(y) == 2
        assert all(np.isfinite(v) for v in y)
```

**Note**: the `--training-n-sims` CLI flag may not exist. If it doesn't, create a reduced TOML inline in the test (copy the full config, overwrite `[optimizer] training_n_sims`, write to a temp path, and train against that). Check existing Phase 2b Window smoke test for the pattern.

- [ ] **Step 2: Run locally**

```bash
uv run pytest tests/test_transformer_pso_smoke.py -v -s
```
Expected: 1 passed in < 30s.

- [ ] **Step 3: Extend `ci.yml`**

Find the `python-pyo3` job's pytest invocation and append:

```yaml
pytest tests/test_pyo3.py \
       tests/test_v2_rust_python_equivalence.py \
       tests/test_gru_pso_smoke.py \
       tests/test_gru_ppo_smoke.py \
       tests/test_rust_python_window_equivalence.py \
       tests/test_window_pso_smoke.py \
       tests/test_pyo3_transformer_flat_weights.py \
       tests/test_rust_python_transformer_equivalence.py \
       tests/test_transformer_warmup.py \
       tests/test_transformer_pso_smoke.py
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_transformer_pso_smoke.py .github/workflows/ci.yml
git commit -m "test(nn): Transformer PSO smoke (2-gen training + CI wiring)"
```

---

## Task 21: PPO-rejection test consolidation + CI wiring

**Files:**
- Create: `tests/test_transformer_ppo_rejection.py` (consolidates build_layer + load_policy_from_json rejection tests)
- Possibly remove: `tests/test_build_layer_transformer_rejection.py` and `tests/test_load_policy_transformer_rejection.py` (if they were separate in Tasks 12/16)
- Modify: `.github/workflows/ci.yml` (main python job)

- [ ] **Step 1: Write the consolidated rejection test**

```python
# tests/test_transformer_ppo_rejection.py
"""Transformer is PSO-only in Phase 3a; PPO entry points must reject cleanly."""
import json
import pytest
from aerocapture.training.rl.layers import build_layer
from aerocapture.training.rl.schemas import TransformerSpec


def test_build_layer_rejects_transformer():
    spec = TransformerSpec(type="transformer", d_model=4, n_heads=2, d_ffn=8, n_seq=4)
    with pytest.raises(NotImplementedError, match="Transformer is PSO-only in Phase 3a"):
        build_layer(spec)


def test_load_policy_from_json_rejects_transformer(tmp_path):
    from aerocapture.training.model_io import load_policy_from_json

    model_json = {
        "format_version": 2,
        "architecture": [
            {"type": "dense", "input_size": 8, "output_size": 4, "activation": "linear"},
            {"type": "transformer", "d_model": 4, "n_heads": 2, "d_ffn": 8, "n_seq": 4},
            {"type": "dense", "input_size": 4, "output_size": 2, "activation": "linear"},
        ],
        "weights": {
            "layer_0": {"w": [[0.0]*8]*4, "b": [0.0]*4},
            "layer_1": {
                "w_q": [[0.0]*4]*4, "b_q": [0.0]*4,
                "w_k": [[0.0]*4]*4, "b_k": [0.0]*4,
                "w_v": [[0.0]*4]*4, "b_v": [0.0]*4,
                "w_o": [[0.0]*4]*4, "b_o": [0.0]*4,
                "w_ffn1": [[0.0]*4]*8, "b_ffn1": [0.0]*8,
                "w_ffn2": [[0.0]*8]*4, "b_ffn2": [0.0]*4,
                "ln1": {"gamma": [1.0]*4, "beta": [0.0]*4},
                "ln2": {"gamma": [1.0]*4, "beta": [0.0]*4},
            },
            "layer_2": {"w": [[0.0]*4]*2, "b": [0.0]*2},
        },
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps(model_json))
    with pytest.raises(NotImplementedError, match="Transformer"):
        load_policy_from_json(str(path))
```

- [ ] **Step 2: Remove redundant per-task test files**

```bash
rm -f tests/test_build_layer_transformer_rejection.py \
       tests/test_load_policy_transformer_rejection.py
```

(Skip whichever file doesn't exist.)

- [ ] **Step 3: Run**

```bash
uv run pytest tests/test_transformer_ppo_rejection.py -v
```
Expected: 2 passed.

- [ ] **Step 4: CI wiring**

This test is `@fast` (no `pytest.mark.slow`), so it runs automatically in the main python job via the catch-all `pytest tests/ -k "not slow"`. Verify by searching `.github/workflows/ci.yml`:

```bash
grep -n "not slow" .github/workflows/ci.yml
```

If the main python job uses an explicit file list instead of `-k "not slow"`, add `tests/test_transformer_ppo_rejection.py` to it.

- [ ] **Step 5: Commit**

```bash
git add tests/test_transformer_ppo_rejection.py .github/workflows/ci.yml
git rm tests/test_build_layer_transformer_rejection.py tests/test_load_policy_transformer_rejection.py 2>/dev/null || true
git commit -m "test(nn): consolidated Transformer PPO-rejection gate"
```

---

## Task 22: Full verification (check_all + guidance golden regressions)

**Files:**
- Run: `./check_all.sh` (Rust side), `./lint_code.sh` (Python), full `pytest`
- Run: guidance golden regression tests

- [ ] **Step 1: Run full Rust check**

```bash
./check_all.sh
```
Expected: fmt, clippy (for `aerocapture` only, not `aerocapture-py`), test, release build all pass.

- [ ] **Step 2: Run Python lint + mypy**

```bash
./lint_code.sh
```
Expected: ruff + mypy clean.

- [ ] **Step 3: Rebuild PyO3 (in case Rust changes invalidated the `.so`)**

```bash
uv run maturin develop --release --manifest-path src/rust/aerocapture-py/Cargo.toml
```

- [ ] **Step 4: Full pytest (including @slow)**

```bash
uv run pytest tests/ -v 2>&1 | tail -40
```
Expected: all tests pass, including the new Transformer tests.

- [ ] **Step 5: Guidance golden regressions**

```bash
uv run pytest tests/test_rust_guidance_regression.py -v
```
Expected: all 10 existing schemes bit-identical. If any regressed, the Transformer patches are leaking into the dispatch for existing schemes -- investigate.

- [ ] **Step 6: Commit**

No code changes in this task (verification only). If Steps 1-5 all pass, the implementation is ready for docs sync. Otherwise, fix the issue and re-run.

---

## Task 23: CLAUDE.md + TODO.md updates

**Files:**
- Modify: `CLAUDE.md` (new Phase 3a section, extensibility contract update for PE-offset precompute pattern)
- Modify: `TODO.md` (mark 3a DONE, add Phase 3b follow-up)

- [ ] **Step 1: Update `CLAUDE.md`**

Add a new section modeled on the Phase 2b block (find "Phase 2b Window-MLP" heading and insert after it):

```markdown
**Phase 3a Transformer MVP (branch `feature/transformer-mvp`, 2026-04-22)** adds the fourth stateful layer type: 1-layer pre-norm Transformer with causal window attention over an N=64 KV ring buffer, sinusoidal PE relative to ring-buffer slot. PSO-only; PPO deferred to Phase 3b.
- **Rust**: `TransformerLayer` struct (QKV/O projections + 2-layer FFN + 2x LayerNorm + derived `k_pe_offsets` / `v_pe_offsets`), `Layer::Transformer` / `LayerSpec::Transformer { d_model, n_heads, d_ffn, n_seq }` / `LayerState::Transformer { k_cache, v_cache }` (VecDeque; grows 0 -> n_seq organically, NO zero-padding -- different from Window-MLP; softmax over variable-length FIFO via sequential reduction), `LayerWeights for TransformerLayer` with canonical flat order (spec section 3.3) and `rebuild_pe_offsets` called in `from_flat` (PSO path) AND `from_v2_json` (JSON load path). GELU-exact via `libm::erf` (not tanh approximation). LayerNorm is biased variance (1/N) with eps=1e-5 matching torch defaults. `TomlLayerSpec::Transformer` + parser + divisibility validator.
- **Python**: `TransformerLayer` torch module (manual LN/GELU/softmax/MHA for bit-equivalence), `TransformerSpec` pydantic discriminated-union entry, `build_layer` + `load_policy_from_json` raise `NotImplementedError` (PPO gate), `_transformer_specs` PSO ParamSpec generator with Xavier on projections/FFN + N(1, 0.01) on LN gamma + tight-near-zero on biases/beta, `_layer_n_params` + `_layer_output_size` + `describe_architecture` + `init_v2_population` arms, `export_v2_policy_to_json` writes 14-key weights dict.
- **Training**: `configs/training/msr_aller_transformer_pso_train.toml` -- Dense(23 -> 32, linear) -> Transformer(d_model=32, n_heads=4, d_ffn=64, n_seq=64) -> Dense(32 -> 2, linear), 9,378 trainable params, PSO n_pop=64 n_gen=2000 seed_strategy="adaptive". Registered as `neural_network_transformer_pso` in `compare_guidance.SCHEMES` + `_NN_DEPLOY_SCHEMES`; `train_all.sh` aliases `transformer_pso` / `nn_transformer_pso` / `transformer`.
- **Gates**: cross-language Transformer equivalence (100-step sequence through `nn_forward_sequence`, max abs diff < 1e-10, target machine epsilon), warm-up test (cache grows 0 -> n_seq with no zero-padding, deterministic), PSO smoke (2 gens on reduced arch), PPO-rejection test (build_layer + load_policy_from_json both raise). All 10 guidance golden regressions bit-identical.

Full spec: `docs/superpowers/specs/2026-04-22-phase-3a-transformer-mvp-design.md`. Plan: `docs/superpowers/plans/2026-04-22-phase-3a-transformer-mvp-plan.md`.
```

Find the "Extensibility" paragraph at the end of the Phase 2b section and extend it with the PE-offset precompute pattern:

```markdown
**Derived-at-load-time per-layer fields (PE-offset precompute pattern)**: Phase 3a introduced fields that are NOT in the flat chromosome but ARE reconstructed from other loaded weights. `TransformerLayer::k_pe_offsets` and `v_pe_offsets` are matrices `[n_seq][d_model]` precomputed as `W_K @ PE_table` and `W_V @ PE_table` once at layer construction. Reconstruction happens in BOTH entry points: `from_flat` (PSO chromosome -> layer) AND `from_v2_json` (JSON -> layer) via `rebuild_pe_offsets`. Future layer types with derived state (Mamba `A_bar` / `B_bar`, for instance) follow the same pattern: store only the trainable parameters in the flat chromosome; derive dependent matrices at load time in both paths.
```

- [ ] **Step 2: Update `TODO.md`**

Replace the `### Phase 3a -- Transformer MVP (PSO only) [DOING ...]` block with:

```markdown
### Phase 3a -- Transformer MVP (PSO only) [DONE 2026-04-22]

Shipped on branch `feature/transformer-mvp` (N commits on top of main).
Cross-language Transformer equivalence matches at machine epsilon (max abs diff < 1e-10, actual ~1e-14 to 1e-15).
PSO smoke + warm-up + equivalence + PPO-rejection tests wired into the python-pyo3 CI job.

- [x] Rust `TransformerLayer` + `Layer::Transformer` + `LayerSpec::Transformer { d_model, n_heads, d_ffn, n_seq }` + `LayerState::Transformer { k_cache, v_cache }` + `TomlLayerSpec::Transformer`
- [x] `LayerWeights for TransformerLayer` + derived-at-load PE-offset pattern (`rebuild_pe_offsets` in both `from_flat_weights_v2` and `from_v2_json`)
- [x] Python `TransformerLayer` torch module + `TransformerSpec` pydantic + `build_layer` PPO-rejection guard
- [x] `_transformer_specs` (Xavier on projections + FFN, N(1,0.01) on LN gamma) + `_layer_n_params` / `_layer_output_size` / `init_v2_population` arms
- [x] Training config `msr_aller_transformer_pso_train.toml` + `compare_guidance` + `train_all.sh` registration
- [x] Cross-language equivalence + warm-up + PSO smoke + PPO-rejection tests (CI wiring)

Spec: `docs/superpowers/specs/2026-04-22-phase-3a-transformer-mvp-design.md`.
Plan: `docs/superpowers/plans/2026-04-22-phase-3a-transformer-mvp-plan.md`.

**Out-of-Phase-3a carry-overs (still deferred):**
- [ ] PPO-BPTT for Transformer (Phase 3b; requires `hidden_shapes` arm for stacked `(2, n_seq, d_model)` + per-env cache-length scalar, ndim-dispatch arm in `ppo_update_bptt`, PPO smoke + BPTT chunk-invariant tests, training TOML with `bptt_length = 32`).
- [ ] SAC-Transformer (Phase 1.6).
- [ ] Recurrent critic (Phase 1.5 carry-over).
- [ ] Widen `load_policy_from_json` to accept v1 JSON (Phase 0 carry-over).
- [ ] Fix pre-existing clippy warnings in `src/rust/aerocapture-py/src/lib.rs`.
- [ ] Multi-layer Transformer stacks (TOML-level; not exercised by 3a paper baseline).

**Closed by Phase 3a:**
- [x] Derived-at-load-time per-layer fields (PE-offset precompute) supported end-to-end for future Mamba / SSM layers.

### Phase 3b -- Transformer PPO-BPTT (follow-up)
- [ ] Deferred from 3a. Paper grid row "Transformer under BPTT-PPO" not closed yet.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md TODO.md
git commit -m "docs: record Phase 3a Transformer MVP landed"
```

---

## Task 24: smart-commit final pass

**Files:** Full branch scope.

- [ ] **Step 1: Invoke the smart-commit skill with branch scope**

Run the user-level `smart-commit` skill, telling it to take the whole `feature/transformer-mvp` branch into account (docs + code + tests). This is the per-user CLAUDE.md convention for finishing an implementation plan.

- [ ] **Step 2: Review the final diff**

```bash
git log --oneline main..HEAD
git diff main..HEAD --stat
```
Expected: ~24 commits, coherent sequence.

- [ ] **Step 3: Handoff**

Phase 3a complete. Branch ready for merge consideration (user reviews; no auto-push per user CLAUDE.md).

---

## Self-Review Checklist

Before handing off this plan, verify:

- [x] **Spec coverage.** Every numbered section in the spec (scope, Rust layer, LayerState, LayerWeights, JSON v2, forward dispatch, TOML, Python schema / module / build_layer / encoding / config helpers / init / export / load, training config, 4 tests, CI, CLAUDE.md / TODO.md sync) maps to a task in this plan.
- [x] **Placeholder scan.** No "TBD" / "TODO: implement" / "similar to Task N" / "add appropriate error handling" hedges. Every code step has the actual code.
- [x] **Type consistency.** `TransformerLayer` / `TransformerSpec` / `LayerSpec::Transformer` / `LayerState::Transformer { k_cache, v_cache }` / `rebuild_pe_offsets` / `k_pe_offsets` / `v_pe_offsets` / `n_seq` used consistently across all tasks.
- [x] **PE-offset rebuild path.** `rebuild_pe_offsets` called in both `from_flat_weights_v2` (Task 5) and `from_v2_json` (Task 6).
- [x] **Flat-order invariant documented.** Task 5 (`_transformer_specs` Python) cross-references Rust flat order in Task 5 canonical order doc; regression-covered by the Rust round-trip test in Task 5.
- [x] **Cache-growth semantics locked in.** Tasks 3 (Rust forward), 11 (Python forward), 17 (equivalence), 18 (warm-up) all verify cache grows organically 0 -> n_seq with no zero-padding.
- [x] **PPO rejection gate.** `build_layer` (Task 12) + `load_policy_from_json` (Task 16) + consolidated test (Task 21).
