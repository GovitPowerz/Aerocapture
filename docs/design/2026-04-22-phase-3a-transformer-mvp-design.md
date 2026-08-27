# Phase 3a -- Transformer MVP (PSO-only)

**Date:** 2026-04-22
**Status:** Design approved, ready for implementation planning
**Parent effort:** LSTM / Transformer / Mamba architectures for neural guidance (see `TODO.md`).
**Predecessors:** Phase 0 (stateful NN runtime infrastructure), Phase 1 (PSO-GRU), Phase 1.5 (PPO-GRU + truncated BPTT), Phase 2a (LSTM MVP + activation-aware init), Phase 2b (Window-MLP, PSO-only).

## 1. Context

The paper experimental grid has six architectures across two training axes. Phases 0/1/1.5/2a/2b have shipped the runtime infrastructure, the extensibility contract for scalar-state, multi-tensor-state, zero-param, and PSO-only layer types, activation-aware init, and end-to-end training on four of the six architectures (MLP baseline, Window-MLP, GRU, LSTM).

Phase 3a adds **Transformer**: a single pre-norm Transformer block with causal window attention over a fixed N=64 KV ring buffer. Per-tick inference runs the block on one token (the current input, after input projection), caches the projected K and V in the ring buffer, and computes attention against the buffer. The positional signal is a sinusoidal PE indexed **relative to the current buffer slot** (newest token at slot `len - 1`), so the policy is translation-invariant in episode time.

The TODO grid schedules Transformer under both PSO and BPTT-PPO. Phase 3a scopes **PSO only**, deliberately matching the Phase 2b staging. Rationale:

1. **Cross-language parity is new ground.** Attention adds QKV projections, manual softmax, causal iteration over a ring buffer, sinusoidal PE, two LayerNorms, and a GELU FFN -- eight non-trivial numerical contracts that all have to match to machine epsilon. Pinning PSO first validates the runtime before stacking BPTT on top.
2. **Multi-tensor state, but bigger than LSTM.** The KV cache is 2 parallel `VecDeque<Vec<f64>>` of length `n_seq=64` with `d_model=32` each, i.e. ~4 kB per env per layer. PPO rollout buffers would need a `(T, B, 2, n_seq, d_model)` slab **plus** a per-env "cache length" counter (since the cache grows organically from 0 to N). That ndim==5 + scalar-counter dispatch is meaningful additional surface that deserves its own phase.
3. **Paper result already covered.** The MLP vs Transformer PSO comparison fully populates the "PSO row" of the paper's experimental grid.

The PPO path gets a **clean error at build time**: `build_layer(TransformerSpec)` raises `NotImplementedError` with a pointer to this spec. Same for `load_policy_from_json`. PSO bypasses V2Policy entirely (Rust forward via `aerocapture_rs.nn_forward`), so PSO training is unaffected.

## 2. Scope

**In scope:**

- Rust `TransformerLayer` struct with fields for QKV projections (bias=true), output projection, 2-layer FFN (hidden=d_ffn, GELU), 2x LayerNorm (learnable gamma/beta, eps=1e-5, biased population variance matching PyTorch `nn.LayerNorm` default), and **derived** K/V PE offset matrices (precomputed at load time, not stored as trainable weights).
- `Layer::Transformer`, `LayerSpec::Transformer { d_model, n_heads, d_ffn, n_seq }` (all `usize`; no activation enum -- FFN activation is GELU-exact, hard-coded), `LayerState::Transformer { k_cache: VecDeque<Vec<f64>>, v_cache: VecDeque<Vec<f64>> }`, `TomlLayerSpec::Transformer { d_model, n_heads, d_ffn, n_seq }`.
- `LayerWeights for TransformerLayer`: canonical flat ordering documented in section 3.3; `from_flat` reconstructs the layer and re-derives `k_pe_offsets` / `v_pe_offsets` from `w_k` / `w_v` and the sinusoidal PE table.
- `NeuralNetModel::save_json` / `from_v2_json` / `from_flat_weights_v2` Transformer arms. JSON v2 entry includes spec fields plus a `weights` dict with 14 arrays: `w_q`, `b_q`, `w_k`, `b_k`, `w_v`, `b_v`, `w_o`, `b_o`, `w_ffn1`, `b_ffn1`, `w_ffn2`, `b_ffn2`, `ln1`, `ln2`. `ln1` / `ln2` are `{gamma, beta}` pairs.
- `NeuralNetModel::forward` routes through `Layer::Transformer` with the single-token forward defined in section 3.4: LN1 -> QKV projections -> cache push (evict if `len > n_seq`) -> PE-offset attention over cache -> W_O projection + residual -> LN2 -> FFN + residual.
- PyTorch `TransformerLayer` module in `rl/layers/transformer.py` with forward `(x: Tensor, state: tuple[Tensor, Tensor]) -> (out: Tensor, new_state: tuple[Tensor, Tensor])`. Manual softmax, manual GELU (exact form `0.5 * x * (1 + erf(x / sqrt(2)))` via `torch.special.erf`), manual LayerNorm (not `nn.LayerNorm`), manual multi-head reshape. Pure `nn.Linear` for projections. Consumed only by the cross-language equivalence test; PPO rejects the layer.
- `TransformerSpec` pydantic class appended to the `LayerSpec` discriminated union on the `type` field. Validators reject `d_model == 0`, `n_heads == 0`, `d_model % n_heads != 0`, `d_ffn == 0`, `n_seq == 0`.
- `build_layer` in `rl/layers/__init__.py` dispatches `TransformerSpec` to **raise `NotImplementedError`** with a clear message ("Transformer is PSO-only in Phase 3a; PPO use deferred -- see docs/superpowers/specs/2026-04-22-phase-3a-transformer-mvp-design.md").
- `_layer_param_specs` Transformer dispatch returns the full concatenated PSO ParamSpec list documented in section 3.6 (Xavier bounds on projections and FFN, small-normal on biases, N(1, 0.01) on LN gamma, small-normal on LN beta).
- `nn_param_specs_from_v2` naturally picks up the Transformer arm through `_layer_param_specs`.
- `config.py::_layer_n_params` Transformer arm returns the formula `4*(d_model*d_model + d_model) + 2*d_ffn + d_model + d_ffn*d_model + d_model*d_ffn + 4*d_model` (see section 3.6); `_layer_output_size` returns `d_model`.
- `describe_architecture` Transformer arm alongside dense/gru/lstm/window.
- `export_v2_policy_to_json`: Transformer branch writes the 14-key `weights` dict in the canonical layout. Obs-norm bake-in guard extended to reject `TransformerSpec` as layer 0 (the affine-shift bake-in assumes the first layer is a `Dense` so `W_new = W/std`, `b_new = b - W @ (mean/std)` is well-defined; attention layers have QKV fan-outs and residual paths that cannot absorb the transform without breaking bit-identity).
- `load_policy_from_json`: raises `NotImplementedError` when any layer is `TransformerSpec`, consistent with `build_layer`.
- `init_v2_population`: Transformer dispatch applies Xavier uniform to the 8 projection / FFN matrices (`w_q`, `w_k`, `w_v`, `w_o`, `w_ffn1`, `w_ffn2`), small-normal `N(0, 0.01 * mul)` to the 8 biases, `N(1, 0.01 * mul)` to LN gamma (2 vectors), and `N(0, 0.01 * mul)` to LN beta (2 vectors). Same `bound_multiplier` semantics as existing arms.
- Training config `configs/training/msr_aller_transformer_pso_train.toml`: Dense(23 -> 32, linear) -> Transformer(d_model=32, n_heads=4, d_ffn=64, n_seq=64) -> Dense(32 -> 2, linear). PSO `n_pop=64 n_gen=2000 seed_strategy="adaptive"`, `training_n_sims=20`, `validation_n_sims=1000`.
- `compare_guidance.SCHEMES` / `SCHEME_TRAINING_CONFIGS` / `_NN_DEPLOY_SCHEMES` registration as `neural_network_transformer_pso` (goes through the Rust `neural_network` runtime, same as GRU-PSO / LSTM-PSO / Window-PSO).
- `train_all.sh` aliases: `transformer_pso`, `nn_transformer_pso`, `transformer`.
- Cross-language equivalence test `test_rust_python_transformer_equivalence.py`: architecture = Dense(8 -> 16, linear) -> Transformer(d_model=16, n_heads=2, d_ffn=32, n_seq=8) -> Dense(16 -> 2, linear). Uses `aerocapture_rs.nn_forward_sequence` to thread a single `NnState` across 100 random f64 inputs (sequence longer than `n_seq` to exercise eviction and PE wrap semantics); Python forward threads the tuple state through `V2Policy.forward_mean_logstd` explicitly. Asserts max abs diff < 1e-10, target machine epsilon.
- Warm-up test `test_transformer_warmup.py`: drives the same architecture for fewer than `n_seq` steps and asserts that (a) the buffer grows organically (`len` equals step number, not pre-filled), (b) attention output is deterministic, (c) output matches a reference implementation that explicitly slices the cache to `[0 : current_len]`.
- PSO smoke test `test_transformer_pso_smoke.py` (@slow, python-pyo3 CI job): 2-gen PSO on reduced arch Dense(8 -> 16, linear) -> Transformer(d_model=16, n_heads=2, d_ffn=32, n_seq=16) -> Dense(16 -> 2, linear) (~1.8k params), asserts `best_model.json` is v2 with `["dense", "transformer", "dense"]` and `nn_forward` returns a finite 2-tuple.
- PPO-rejection test `test_transformer_ppo_rejection.py` (@fast, main python job): constructs a minimal Transformer v2 JSON and asserts `load_policy_from_json` + `build_layer` both raise `NotImplementedError` with the expected message fragment.
- `ci.yml` workflow extension: four new tests added (`test_rust_python_transformer_equivalence.py`, `test_transformer_warmup.py`, `test_transformer_pso_smoke.py` in python-pyo3 job; `test_transformer_ppo_rejection.py` in the main python job).
- `CLAUDE.md` + `TODO.md` sync after landing: Phase 3a checkbox done, extensibility contract updated to document Transformer's PE-offset precompute pattern, remaining phases (Phase 3b PPO-Transformer, Phase 4 Mamba) confirmed open.

**Out of scope:**

- **PPO-BPTT for Transformer.** Deferred to Phase 3b. PPO path errors at `build_layer` time.
- **`_zero_state_where_done` tuple-of-Tensor extension for Transformer state.** The helper already handles `tuple` via Phase 2a LSTM. No change needed now because Transformer never reaches V2Policy in Phase 3a. Phase 3b will revisit.
- **Rollout buffer `hidden_shapes` / ndim dispatch for Transformer.** Same reasoning; Phase 3b.
- **Recurrent critic.** Carry-over from Phase 1.5, orthogonal.
- **SAC-Transformer.** Phase 1.6 umbrella, still deferred.
- **Widen `load_policy_from_json` to v1 JSON.** Phase 0 carry-over, still deferred.
- **Multi-layer Transformer stacks.** Phase 3a ships 1-layer configs only. Stacking is a TOML-level concern (repeat the `[[network.architecture]]` block); no runtime change needed, but the paper baseline is single-layer so stacked configs are not in the initial training set.
- **Alternative attention variants** (Rotary, ALiBi, flash attention, sliding-window mask with dilations). All deferred; sinusoidal + naive causal over the ring buffer is the paper baseline.
- **Supporting Transformer as a non-first or non-middle layer.** The Rust `TransformerLayer::forward` is position-agnostic (buffers whatever input it receives), but the initial training config places Transformer in the middle per the paper design.

## 3. Detailed design

### 3.1 Rust LayerSpec + Layer + TransformerLayer

```rust
// src/rust/src/data/neural.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum LayerSpec {
    Dense       { input_size: usize, output_size: usize, activation: Activation },
    Gru         { input_size: usize, hidden_size: usize },
    Lstm        { input_size: usize, hidden_size: usize },
    Window      { input_size: usize, n_steps: usize },
    Transformer { d_model: usize, n_heads: usize, d_ffn: usize, n_seq: usize },
}

#[derive(Debug, Clone)]
pub enum Layer {
    Dense(DenseLayer),
    Gru(GruLayer),
    Lstm(LstmLayer),
    Window(WindowLayer),
    Transformer(TransformerLayer),
}

#[derive(Debug, Clone)]
pub struct TransformerLayer {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_head: usize,   // d_model / n_heads (derived, validated == integer)
    pub d_ffn: usize,
    pub n_seq: usize,

    // Attention projections (all row-major, shape [d_model, d_model], bias [d_model])
    pub w_q: Vec<Vec<f64>>, pub b_q: Vec<f64>,
    pub w_k: Vec<Vec<f64>>, pub b_k: Vec<f64>,
    pub w_v: Vec<Vec<f64>>, pub b_v: Vec<f64>,
    pub w_o: Vec<Vec<f64>>, pub b_o: Vec<f64>,

    // FFN (W1: [d_ffn, d_model] + [d_ffn]; W2: [d_model, d_ffn] + [d_model])
    pub w_ffn1: Vec<Vec<f64>>, pub b_ffn1: Vec<f64>,
    pub w_ffn2: Vec<Vec<f64>>, pub b_ffn2: Vec<f64>,

    // LayerNorm params (both shape [d_model])
    pub ln1_gamma: Vec<f64>, pub ln1_beta: Vec<f64>,
    pub ln2_gamma: Vec<f64>, pub ln2_beta: Vec<f64>,

    // Derived at load time: k_pe_offsets[i] = W_K @ PE[i], v_pe_offsets[i] = W_V @ PE[i]
    // Shape [n_seq][d_model]. Regenerated in from_flat whenever w_k / w_v change.
    pub k_pe_offsets: Vec<Vec<f64>>,
    pub v_pe_offsets: Vec<Vec<f64>>,
}
```

**Sinusoidal PE table** (recomputed once per layer construction, stored per-layer for clarity; all Transformer layers at the same `d_model` / `n_seq` share the same table up to floating-point determinism):

```rust
fn build_pe_table(n_seq: usize, d_model: usize) -> Vec<Vec<f64>> {
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

Iteration order is `pos` outer, `i` inner; each scalar is deterministic (f64 mul / div / sin / cos). Python mirror replicates the same order explicitly (no `torch.arange` broadcast, which could fuse differently).

### 3.2 LayerState::Transformer

```rust
// src/rust/src/data/nn_state.rs

#[derive(Debug, Clone)]
pub enum LayerState {
    None,
    Gru(Vec<f64>),
    Lstm { h: Vec<f64>, c: Vec<f64> },
    Window { buffer: VecDeque<Vec<f64>> },
    Transformer {
        k_cache: VecDeque<Vec<f64>>,  // each Vec<f64> has length d_model, FIFO: front = oldest
        v_cache: VecDeque<Vec<f64>>,  // parallel to k_cache
    },
}

impl LayerState {
    pub fn for_layer(spec: &LayerSpec) -> Self {
        match spec {
            LayerSpec::Dense { .. } => LayerState::None,
            LayerSpec::Gru { hidden_size, .. } => LayerState::Gru(vec![0.0; *hidden_size]),
            LayerSpec::Lstm { hidden_size, .. } => LayerState::Lstm {
                h: vec![0.0; *hidden_size], c: vec![0.0; *hidden_size],
            },
            LayerSpec::Window { input_size, n_steps } => LayerState::Window {
                buffer: VecDeque::from(vec![vec![0.0; *input_size]; *n_steps]),
            },
            LayerSpec::Transformer { .. } => LayerState::Transformer {
                k_cache: VecDeque::new(),  // grows organically; NOT pre-filled
                v_cache: VecDeque::new(),
            },
        }
    }

    pub fn reset(&mut self) {
        match self {
            // ...other arms unchanged...
            LayerState::Transformer { k_cache, v_cache } => {
                k_cache.clear();
                v_cache.clear();
            }
        }
    }
}
```

**Design decision: cache grows organically from 0 to n_seq, then stays at n_seq.** This is *different* from Window-MLP which pre-fills with zeros. Rationale:

- Softmax over a variable-length FIFO is as natural as softmax over a fixed length. No masking is needed because no slots are "invalid" -- every token in the cache is real.
- Pre-filling with zero `k` / `v` tokens would mean early-tick softmax distributes weight across `(1 real + 63 zero)` tokens, diluting the attention signal. PSO (no gradients) won't learn to suppress zero-slot attention, so the policy would see systematically wrong outputs for the first 63 ticks of every episode.
- Memory-wise, growing costs `O(current_len)` which is at worst `n_seq = 64` allocations per episode, negligible.
- Cross-language equivalence requires both sides to iterate the cache in the same order and over the same length; both use FIFO (front = oldest) with `len() == min(current_step + 1, n_seq)`.

### 3.3 LayerWeights canonical flat ordering

```
Order (all matrices row-major):
  1.  w_q         [d_model * d_model]
  2.  b_q         [d_model]
  3.  w_k         [d_model * d_model]
  4.  b_k         [d_model]
  5.  w_v         [d_model * d_model]
  6.  b_v         [d_model]
  7.  w_o         [d_model * d_model]
  8.  b_o         [d_model]
  9.  w_ffn1      [d_ffn * d_model]
  10. b_ffn1      [d_ffn]
  11. w_ffn2      [d_model * d_ffn]
  12. b_ffn2      [d_model]
  13. ln1_gamma   [d_model]
  14. ln1_beta    [d_model]
  15. ln2_gamma   [d_model]
  16. ln2_beta    [d_model]
```

Total: `4 * d_model * d_model + 4 * d_model + d_ffn * d_model + d_ffn + d_model * d_ffn + d_model + 4 * d_model`
     = `4 * d_model^2 + 2 * d_ffn * d_model + d_ffn + 9 * d_model`.

For `d_model=32, d_ffn=64`: `4*1024 + 2*2048 + 64 + 9*32 = 4096 + 4096 + 64 + 288 = 8544` per Transformer layer.

```rust
// LayerWeights for TransformerLayer

impl LayerWeights for TransformerLayer {
    fn n_params(&self) -> usize {
        4 * self.d_model * self.d_model
        + 2 * self.d_ffn * self.d_model
        + self.d_ffn
        + 9 * self.d_model
    }

    fn to_flat(&self) -> Vec<f64> { /* concatenate in canonical order */ }

    fn from_flat(
        flat: &[f64],
        cursor: &mut usize,
        spec: &LayerSpec,
    ) -> Result<Self, DataError> {
        let LayerSpec::Transformer { d_model, n_heads, d_ffn, n_seq } = spec
            else { return Err(DataError::LayerSpecMismatch); };
        let d_head = d_model / n_heads;  // validated > 0 and divides evenly

        // read matrices + biases in canonical order, advancing cursor
        // ... (omitted for brevity; mirrors GruLayer / LstmLayer pattern)

        let mut layer = TransformerLayer {
            d_model: *d_model, n_heads: *n_heads, d_head,
            d_ffn: *d_ffn, n_seq: *n_seq,
            w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o,
            w_ffn1, b_ffn1, w_ffn2, b_ffn2,
            ln1_gamma, ln1_beta, ln2_gamma, ln2_beta,
            k_pe_offsets: Vec::new(),  // filled next
            v_pe_offsets: Vec::new(),
        };
        layer.rebuild_pe_offsets();
        Ok(layer)
    }
}

impl TransformerLayer {
    fn rebuild_pe_offsets(&mut self) {
        let pe = build_pe_table(self.n_seq, self.d_model);
        self.k_pe_offsets = pe.iter().map(|p| matvec(&self.w_k, p)).collect();
        self.v_pe_offsets = pe.iter().map(|p| matvec(&self.w_v, p)).collect();
    }
}
```

The PE offsets are NOT in the flat chromosome. They are derived state, refreshed via `rebuild_pe_offsets` in **both** entry points that construct a `TransformerLayer` from weights: `from_flat_weights_v2` (PSO chromosome -> layer) and `from_v2_json` (JSON load). Any future code path that mutates `w_k` or `w_v` MUST call `rebuild_pe_offsets` before the next forward, or attention output becomes silently wrong.

### 3.4 Forward pass (single tick)

Inputs: `x: &[f64]` of length `d_model` (output of the preceding layer, typically the input-projection Dense), `state: &mut LayerState::Transformer { k_cache, v_cache }`.

```
1. x_norm1 = LN(x, ln1_gamma, ln1_beta, eps=1e-5)

2. q = w_q @ x_norm1 + b_q     # [d_model]
   k = w_k @ x_norm1 + b_k     # [d_model]
   v = w_v @ x_norm1 + b_v     # [d_model]

3. k_cache.push_back(k)
   v_cache.push_back(v)
   if k_cache.len() > n_seq:
       k_cache.pop_front()
       v_cache.pop_front()
   cache_len = k_cache.len()   # in [1, n_seq]

4. # Multi-head attention. Reshape q, and each k_cache[i] + k_pe_offsets[i], into [n_heads, d_head]:
   q_heads[h]       = q[h*d_head .. (h+1)*d_head]                               # [d_head]
   k_eff_heads[i,h] = (k_cache[i] + k_pe_offsets[i])[h*d_head .. (h+1)*d_head]  # [d_head]
   v_eff_heads[i,h] = (v_cache[i] + v_pe_offsets[i])[h*d_head .. (h+1)*d_head]  # [d_head]
   # Note: PE offsets use slot index i directly (0 = oldest in current window,
   # cache_len - 1 = current token). This is the relative-to-buffer PE from Q3.

5. For each head h in 0..n_heads:
     scores[i] = dot(q_heads[h], k_eff_heads[i,h]) / sqrt(d_head)  for i in 0..cache_len
     # FIFO reduction order, front = oldest = i=0
     max_score = scores[0]; for i in 1..cache_len: max_score = max(max_score, scores[i])
     exp_scores[i] = exp(scores[i] - max_score)                    for i in 0..cache_len
     sum_exp = 0; for i in 0..cache_len: sum_exp += exp_scores[i]
     weights[i] = exp_scores[i] / sum_exp
     head_out[h] = sum_i weights[i] * v_eff_heads[i, h]             # [d_head]

6. attn_out = concat(head_out[0..n_heads])                          # [d_model]
   x1 = x + (w_o @ attn_out + b_o)                                  # residual

7. x_norm2 = LN(x1, ln2_gamma, ln2_beta, eps=1e-5)
   ffn_hidden = w_ffn1 @ x_norm2 + b_ffn1                           # [d_ffn]
   ffn_hidden_act[j] = gelu_exact(ffn_hidden[j])                    # [d_ffn]
   ffn_out = w_ffn2 @ ffn_hidden_act + b_ffn2                       # [d_model]
   out = x1 + ffn_out                                               # residual
```

**LayerNorm (biased variance matching PyTorch `nn.LayerNorm` default):**

```
mean = sum(x) / d_model                    # sequential FIFO reduction
var  = sum((x[j] - mean)^2) / d_model      # biased (1/N), not Bessel (1/(N-1))
std  = sqrt(var + eps)
LN(x) = (x - mean) / std * gamma + beta
```

**GELU (exact):**

```
gelu_exact(z) = 0.5 * z * (1.0 + erf(z / sqrt(2)))
```

Rust uses `libm::erf` (already in the dependency tree or added to `Cargo.toml`; single-file impact). Python mirror uses `torch.special.erf`. Both are IEEE-754 correctly-rounded up to the last ULP; bit-identity requires that both sides use the same `erf` entry point and the same sqrt argument (baked as a constant `1.4142135623730951`).

**Softmax ordering:** we iterate the `VecDeque` front-to-back via `.iter()` (deterministic FIFO), compute `max_score` via sequential fold, compute `exp_scores` in the same order, sum sequentially, divide. Python mirror uses explicit Python loops over the time axis for the equivalence-test path (not `F.softmax` which dispatches to a tree reduction). For production PPO in Phase 3b, we may allow `F.softmax` once the equivalence test has shown tolerance; for now, the manual variant is the default.

### 3.5 Python TransformerSpec + TransformerLayer module

```python
# src/python/aerocapture/training/rl/schemas.py

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
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}")
        return self

LayerSpec = Annotated[
    DenseSpec | GruSpec | LstmSpec | WindowSpec | TransformerSpec,
    Discriminator("type"),
]
```

```python
# src/python/aerocapture/training/rl/layers/transformer.py

class TransformerLayer(nn.Module):
    """Manual implementation for 1-for-1 Rust equivalence.

    Cross-language contract:
      - LN uses population variance (1/N), eps=1e-5
      - GELU is exact (erf-based), not the tanh approximation
      - Softmax is max-subtraction + sequential FIFO reduction
      - Multi-head split is contiguous slice of the d_model dimension
      - PE is relative-to-buffer (newest token at slot current_len - 1)
      - PE offsets for K/V are derived at __init__ time from w_k / w_v
        and the sinusoidal PE table.
    """

    def __init__(self, d_model: int, n_heads: int, d_ffn: int, n_seq: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.d_ffn, self.n_seq = d_ffn, n_seq

        self.w_q = nn.Linear(d_model, d_model, bias=True)
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)
        self.w_o = nn.Linear(d_model, d_model, bias=True)

        self.w_ffn1 = nn.Linear(d_model, d_ffn, bias=True)
        self.w_ffn2 = nn.Linear(d_ffn, d_model, bias=True)

        self.ln1_gamma = nn.Parameter(torch.ones(d_model))
        self.ln1_beta  = nn.Parameter(torch.zeros(d_model))
        self.ln2_gamma = nn.Parameter(torch.ones(d_model))
        self.ln2_beta  = nn.Parameter(torch.zeros(d_model))

        # PE table [n_seq, d_model] -- fixed, non-trainable
        pe = _build_sinusoidal_pe(n_seq, d_model)  # returns Tensor
        self.register_buffer("pe_table", pe, persistent=False)

    def forward(
        self,
        x: Tensor,                               # (batch, d_model)
        state: tuple[Tensor, Tensor],            # (k_cache, v_cache), each (batch, current_len, d_model)
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        # 1. LN1
        x_norm1 = _manual_ln(x, self.ln1_gamma, self.ln1_beta, eps=1e-5)
        # 2. QKV
        q = self.w_q(x_norm1); k = self.w_k(x_norm1); v = self.w_v(x_norm1)
        # 3. Push k, v into state, evict if > n_seq
        k_cache, v_cache = state
        k_cache = torch.cat([k_cache, k.unsqueeze(1)], dim=1)
        v_cache = torch.cat([v_cache, v.unsqueeze(1)], dim=1)
        if k_cache.shape[1] > self.n_seq:
            k_cache = k_cache[:, 1:]
            v_cache = v_cache[:, 1:]
        cache_len = k_cache.shape[1]
        # 4. PE offsets per slot: [cache_len, d_model]
        k_pe_eff = self.w_k.weight @ self.pe_table[:cache_len].T  # NO bias for the PE shift
        k_pe_eff = k_pe_eff.T  # (cache_len, d_model)
        v_pe_eff = (self.w_v.weight @ self.pe_table[:cache_len].T).T
        k_eff = k_cache + k_pe_eff.unsqueeze(0)  # (batch, cache_len, d_model)
        v_eff = v_cache + v_pe_eff.unsqueeze(0)
        # 5. Multi-head attention with manual softmax
        attn_out = _manual_causal_attention(q, k_eff, v_eff, self.n_heads, self.d_head)
        # 6. Output projection + residual
        x1 = x + self.w_o(attn_out)
        # 7. LN2 + FFN + residual
        x_norm2 = _manual_ln(x1, self.ln2_gamma, self.ln2_beta, eps=1e-5)
        ffn_hidden = self.w_ffn1(x_norm2)
        ffn_hidden_act = 0.5 * ffn_hidden * (1.0 + torch.special.erf(ffn_hidden * _INV_SQRT2))
        ffn_out = self.w_ffn2(ffn_hidden_act)
        out = x1 + ffn_out
        return out, (k_cache, v_cache)

    def new_state(self, batch_size: int) -> tuple[Tensor, Tensor]:
        # Empty cache -- organic growth. dtype/device tracked via a parameter.
        device = self.w_q.weight.device
        dtype  = self.w_q.weight.dtype
        empty  = torch.zeros(batch_size, 0, self.d_model, device=device, dtype=dtype)
        return (empty.clone(), empty.clone())
```

`_manual_causal_attention`, `_manual_ln`, and `_build_sinusoidal_pe` are small private helpers in `transformer.py` that match the Rust iteration order exactly.

```python
# src/python/aerocapture/training/rl/layers/__init__.py

def build_layer(spec: LayerSpec) -> nn.Module:
    if isinstance(spec, DenseSpec):       return DenseLayer(spec.input_size, spec.output_size, spec.activation)
    if isinstance(spec, GruSpec):         return GruLayer(spec.input_size, spec.hidden_size)
    if isinstance(spec, LstmSpec):        return LstmLayer(spec.input_size, spec.hidden_size)
    if isinstance(spec, WindowSpec):      raise NotImplementedError(...)  # Phase 2b
    if isinstance(spec, TransformerSpec): raise NotImplementedError(
        "Transformer is PSO-only in Phase 3a; PPO use deferred. "
        "See docs/superpowers/specs/2026-04-22-phase-3a-transformer-mvp-design.md"
    )
    raise TypeError(f"Unknown layer spec: {spec!r}")
```

Note: `build_layer(TransformerSpec)` raising is for the **PPO / SAC production path**. The cross-language equivalence test does NOT go through `build_layer`; it instantiates `TransformerLayer` directly. So the test and the PPO rejection can coexist cleanly.

### 3.6 Config helpers, encoding, export/load

```python
# src/python/aerocapture/training/config.py

def _layer_n_params(spec: LayerSpec) -> int:
    # ...existing arms...
    if isinstance(spec, TransformerSpec):
        return (4 * spec.d_model * spec.d_model
              + 2 * spec.d_ffn   * spec.d_model
              + spec.d_ffn
              + 9 * spec.d_model)
    raise TypeError(...)

def _layer_output_size(spec: LayerSpec) -> int:
    # ...existing arms...
    if isinstance(spec, TransformerSpec): return spec.d_model
    raise TypeError(...)
```

```python
# src/python/aerocapture/training/encoding.py

def _transformer_specs(spec: TransformerSpec, bound_multiplier: float) -> list[ParamSpec]:
    """Activation-aware bounds for Transformer.

    Xavier uniform on all 8 projection / FFN matrices:
      - W_{Q,K,V,O}: fan_in = d_model, fan_out = d_model -> bound = sqrt(6 / (2 * d_model))
      - W_ffn1: fan_in = d_model, fan_out = d_ffn -> bound = sqrt(6 / (d_model + d_ffn))
      - W_ffn2: fan_in = d_ffn, fan_out = d_model -> same bound as w_ffn1 (symmetric fan sum)
    Biases: tight uniform [-0.1*mul, 0.1*mul]
    LN gamma: uniform around 1.0, [1 - 0.01*mul, 1 + 0.01*mul]
    LN beta:  uniform around 0.0, [-0.01*mul, 0.01*mul]
    """
    # returns list[ParamSpec] in the canonical flat order (section 3.3).
    # INVARIANT: this ordering MUST match Rust's `to_flat` / `from_flat` cursor
    # advance order exactly, or PSO chromosomes decode into scrambled weights.
    # Regression-covered by the flat-chromosome round-trip test (section 4, gate 8).

def _layer_param_specs(spec: LayerSpec, bound_multiplier: float) -> list[ParamSpec]:
    # ...existing arms...
    if isinstance(spec, TransformerSpec): return _transformer_specs(spec, bound_multiplier)
    raise TypeError(...)
```

The `init_v2_population` Transformer branch decodes its allocated chromosome slab through the same ParamSpec list, producing a fresh population with Xavier-initialized projections + near-identity LayerNorms + tight-near-zero biases.

```python
# src/python/aerocapture/training/rl/export.py::export_v2_policy_to_json

for i, (spec, layer) in enumerate(zip(policy.architecture, policy.layers)):
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
    # ...existing Dense/GRU/LSTM arms...

# Obs-norm bake-in guard:
if obs_normalizer is not None and isinstance(
    policy.architecture[0], (GruSpec, LstmSpec, WindowSpec, TransformerSpec)
):
    raise NotImplementedError(
        f"Obs normalizer bake-in into layer 0 only supports DenseSpec, "
        f"got {type(policy.architecture[0]).__name__}. Export without the bake-in."
    )
```

```python
# src/python/aerocapture/training/model_io.py::load_policy_from_json

if any(isinstance(spec, (WindowSpec, TransformerSpec)) for spec in architecture):
    raise NotImplementedError(
        "Transformer / Window-MLP are PSO-only phases; load_policy_from_json is a "
        "PPO/SAC entry point that cannot construct V2Policy with these layers. "
        "See docs/superpowers/specs/2026-04-22-phase-3a-transformer-mvp-design.md"
    )
```

PSO training writes the v2 JSON directly via Rust's `aerocapture_rs.flat_weights_to_json` (unchanged from Phase 1+). No Python-side Transformer serialization is needed on the PSO hot path; `export_v2_policy_to_json` is only used by the equivalence test and by Phase 3b PPO code.

### 3.7 TOML parser + training config

```rust
// src/rust/src/config.rs

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
                if d_model % n_heads != 0 {
                    return Err(ConfigError::Invalid(format!(
                        "(transformer) d_model={d_model} not divisible by n_heads={n_heads}"
                    )));
                }
                Ok(LayerSpec::Transformer {
                    d_model: *d_model, n_heads: *n_heads, d_ffn: *d_ffn, n_seq: *n_seq,
                })
            }
        }
    }
}
```

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
# Full 23-input candidate vector (baseline MLP uses the same).

[[network.architecture]]
type = "dense"
input_size = 23
output_size = 32
activation = "linear"

[[network.architecture]]
type = "transformer"
d_model = 32
n_heads = 4
d_ffn   = 64
n_seq   = 64

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

Parameter count: `23*32 + 32 + 8544 + 32*2 + 2 = 736 + 32 + 8544 + 64 + 2 = 9378`. Matches TODO ("~10k params").

```python
# src/python/aerocapture/training/compare_guidance.py

SCHEMES = [..., "neural_network_transformer_pso"]
SCHEME_TRAINING_CONFIGS = {
    ...,
    "neural_network_transformer_pso": "configs/training/msr_aller_transformer_pso_train.toml",
}
_NN_DEPLOY_SCHEMES = {..., "neural_network_transformer_pso"}
```

```bash
# train_all.sh
    transformer_pso|nn_transformer_pso|transformer)
        scheme="neural_network_transformer_pso"
        ;;
```

### 3.8 Tests

**`tests/test_rust_python_transformer_equivalence.py`**

- Architecture: Dense(8 -> 16, linear) -> Transformer(d_model=16, n_heads=2, d_ffn=32, n_seq=8) -> Dense(16 -> 2, linear)
- Build `V2Policy` via direct `TransformerLayer` construction (bypassing `build_layer` which rejects). Convert to f64 (`policy.double()`). Export to temp v2 JSON via a hand-rolled serializer that matches `export_v2_policy_to_json`'s Transformer branch (the guard in `export_v2_policy_to_json` is scoped to `obs_normalizer is not None`, so `obs_normalizer=None` goes through cleanly).
- Generate 100 random f64 inputs via `np.random.default_rng(seed=42).standard_normal((100, 8))`.
- Run both paths: `aerocapture_rs.nn_forward_sequence` (Rust, threads `NnState` internally) vs Python (maintains `(k_cache, v_cache)` tuple in a loop, calls `TransformerLayer.forward` each step).
- Because we run 100 steps but `n_seq = 8`, the cache fills up around step 8 and evicts from step 9 onward -- this exercises both the growth phase (PE indices 0..7) and the steady-state phase (PE indices 0..7 again, with the relative semantics shifting each step).
- Assert `max(|rust - python|) < 1e-10`. Target machine epsilon (expected ~1e-14 to 1e-15 based on the number of floating-point ops in one forward).

**`tests/test_transformer_warmup.py`**

- Same architecture at smaller `n_seq = 4`, drive 3 input steps (fewer than `n_seq`).
- After each step, read the cache length via a debug accessor or through `nn_forward_sequence` with `n_steps_requested = step + 1` (whichever is cleaner).
- Assert cache length after step k equals `k + 1`.
- Assert attention output is deterministic across repeated runs with the same input (catches any unseeded randomness).

**`tests/test_transformer_pso_smoke.py`** (@pytest.mark.slow, python-pyo3 CI job)

- Reduced TOML: Dense(8 -> 16, linear) -> Transformer(d_model=16, n_heads=2, d_ffn=32, n_seq=16) -> Dense(16 -> 2, linear) (~1800 params).
- `n_sims = 16`, `n_pop = 16`, `n_gen = 2`, `seed_strategy = "fixed"`, `max_time = 200` (short sim to bound wall-clock).
- Run `aerocapture.training.train` against the reduced config, assert:
  - `training_output/<scheme>/best_model.json` exists.
  - Parsed JSON has `format_version = 2` and architecture `["dense", "transformer", "dense"]`.
  - A subsequent `aerocapture_rs.nn_forward` call on the best model returns a finite 2-tuple.

**`tests/test_transformer_ppo_rejection.py`** (@fast, main python job)

- Construct a minimal Transformer v2 JSON file in a temp dir (Dense -> Transformer -> Dense, 2 heads, d_model=4, etc.).
- Assert `aerocapture.training.model_io.load_policy_from_json(path)` raises `NotImplementedError` containing "Transformer is PSO-only in Phase 3a".
- Assert `build_layer(TransformerSpec(type="transformer", d_model=4, n_heads=2, d_ffn=8, n_seq=4))` raises `NotImplementedError` with the same message fragment.

**`ci.yml`** extension:

```yaml
# python-pyo3 job (@slow tests allowed):
pytest tests/test_pyo3.py \
       tests/test_v2_rust_python_equivalence.py \
       tests/test_gru_pso_smoke.py \
       tests/test_gru_ppo_smoke.py \
       tests/test_rust_python_window_equivalence.py \
       tests/test_window_pso_smoke.py \
       tests/test_rust_python_transformer_equivalence.py \
       tests/test_transformer_warmup.py \
       tests/test_transformer_pso_smoke.py

# main python job (@fast only):
pytest tests/ -k "not slow"
# (picks up test_transformer_ppo_rejection.py automatically)
```

### 3.9 CLAUDE.md + TODO.md updates

At merge time, `CLAUDE.md` gains a Phase 3a section mirroring the Phase 2a/2b blocks (Rust / Python / Training / Gates / Carry-overs), and the scalar-state extensibility contract paragraph is updated to document Transformer's **PE-offset precompute** as a new pattern (derived-at-load-time fields that are NOT part of the flat chromosome but ARE reconstructed in `from_flat`). `TODO.md` Phase 3 becomes Phase 3a, with Phase 3b added as a follow-up for PPO-BPTT.

## 4. Verification gates

1. **Cross-language equivalence.** `test_rust_python_transformer_equivalence.py` passes with max abs diff < 1e-10 over a 100-step sequence. Target machine epsilon.
2. **Warm-up correctness.** `test_transformer_warmup.py` passes: cache grows organically, no zero padding, deterministic output.
3. **PSO smoke.** `test_transformer_pso_smoke.py` passes (~3-5 s wall clock per the reduced arch).
4. **PPO rejection.** `test_transformer_ppo_rejection.py` asserts both rejection paths raise with the expected message.
5. **Guidance golden regressions.** All 10 existing schemes stay bit-identical (Transformer is a new scheme, no golden yet).
6. **Rust check_all.** `fmt`, `clippy`, `test`, release build all pass.
7. **Python lint + mypy + tests.** `ruff`, `mypy --strict`, `pytest`, all green.
8. **Flat chromosome round-trip.** `to_flat(from_flat(flat)) == flat` for a random f64 slab of the right length. Existing framework-level test in `test_nn_weights_roundtrip.py` is extended with a Transformer case.

## 5. Out-of-Phase-3a carry-overs

- **PPO-BPTT for Transformer** (Phase 3b). Requires `_zero_state_where_done` tuple branch confirmed (Phase 2a already handles `tuple[Tensor, Tensor]`); `hidden_shapes` arm for stacked `(2, n_seq, d_model)` + per-env cache-length scalar tracking (the cache grows organically 0 -> n_seq, so different envs can have different valid prefixes at any given timestep -- materially trickier than LSTM where all hidden states are same-shape); matching ndim-dispatch arm in `_np_state_to_torch` / `_torch_state_to_np` / `ppo_update_bptt`; PPO smoke + BPTT chunk-invariant tests; new training TOML `msr_aller_transformer_ppo_train.toml` with `[rl.ppo] bptt_length = 32`.
- **SAC-Transformer** (Phase 1.6). SAC stays on GaussianPolicy until the umbrella migration lands.
- **Recurrent critic** (Phase 1.5 carry-over). Feedforward critic mirroring policy trunk widths is fine for 1-layer Transformer at d_model=32.
- **Widen `load_policy_from_json` to v1 JSON** (Phase 0 carry-over, still deferred).
- **Fix pre-existing clippy warnings in `src/rust/aerocapture-py/src/lib.rs`** (Phase 0 carry-over, 3 warnings -- one-line each).
- **Multi-layer Transformer stacks.** TOML supports it via repeat blocks; paper baseline is single-layer so not exercised by this phase.
- **Alternative attention variants** (Rotary / ALiBi / flash / sliding-window / dilated). Not needed for the paper grid.

## 6. Closed by Phase 3a

- **PE-offset precompute pattern** locked in as a supported extensibility case: derived-at-load-time per-layer fields that are NOT part of the flat chromosome but ARE reconstructed from `from_flat` once the source weights (`w_k`, `w_v`) have been loaded. Documented in the extensibility-contract section of `CLAUDE.md` so future phases (Mamba with its derived A_bar / B_bar matrices, etc.) can follow the same pattern without re-deriving the convention.
