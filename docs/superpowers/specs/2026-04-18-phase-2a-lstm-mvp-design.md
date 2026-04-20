# Phase 2a -- LSTM MVP (PSO + PPO-BPTT) with Activation-Aware Init

**Date:** 2026-04-18
**Status:** Design approved, ready for implementation planning
**Parent effort:** LSTM / Transformer / Mamba architectures for neural guidance (see `TODO.md`).
**Predecessors:** Phase 0 (stateful NN runtime infrastructure, PR #37), Phase 1 (PSO-GRU MVP, `feature/gru-mvp`), Phase 1.5 (PPO-GRU + truncated BPTT, PR #38).

## 1. Context

Phase 0 shipped the stateful NN runtime (JSON v2, `NnState`, `LayerState` enum, `V2Policy` PyTorch mirror, cross-language equivalence at machine epsilon) and the extensibility contract. Phase 1 validated the stack end-to-end on GRU under PSO. Phase 1.5 lifted PPO to recurrent policies with chunked truncated BPTT. The second architecture now needs to land.

Phase 2a adds LSTM as the second stateful layer type and trains it on both axes (PSO + PPO-BPTT) in one PR. LSTM is the first layer with a **multi-tensor hidden state** (`(h, c)` tuple), which is the real exercise of the Phase 0 extensibility contract: `_zero_state_where_done` in `policy.py` currently raises `TypeError` on non-Tensor state precisely so that the first multi-tensor addition must make the contract explicit rather than silently matmul-erroring. Landing LSTM under both PSO and PPO in one PR means the tuple-state contract is exercised end-to-end (Rust forward, PyTorch forward, BPTT over a chunk with per-env zero-on-done) before the PR merges.

The phase also folds in **activation-aware initialization**, which has been deferred twice (Phase 1, Phase 1.5 carry-over). LSTM is the natural forcing function: pure uniform-in-ParamSpec-bounds initialization gives a forget-gate bias near zero, and the empirical RNN literature is unambiguous that LSTM trains noticeably slower without forget-bias-1 init (Jozefowicz, Zaremba & Sutskever 2015, "An Empirical Exploration of Recurrent Network Architectures"). Publishing LSTM results from uniform init would invite reviewer pushback, so the init refactor is scheduled for this phase. GRU gets tanh-Xavier init retroactively as a side-effect.

The scientific purpose of the phase is to fill the second and third cells of the paper grid (LSTM × {PSO, PPO}), so that the comparison across architectures is apples-to-apples on both training axes.

## 2. Scope

**In scope:**

- Rust `LstmLayer` with PyTorch `nn.LSTMCell` gate convention (i, f, g, o), vanilla (no peepholes), two biases (`bias_ih`, `bias_hh`, both kept for PyTorch match).
- `Layer::Lstm`, `LayerSpec::Lstm { input_size, hidden_size }`, `LayerState::Lstm { h, c }` (named struct variant), `TomlLayerSpec::Lstm { hidden_size }`.
- `LayerWeights for LstmLayer` with flat order `weight_ih` row-major -> `weight_hh` row-major -> `bias_ih` -> `bias_hh` (matches GRU pattern, 4H rows instead of 3H).
- `NeuralNetModel::save_json` / `from_v2_json` / `from_flat_weights_v2` Lstm arms.
- PyTorch `LstmLayer` module in `rl/layers/lstm.py` with manual gate computation matching `nn.LSTMCell` bit-for-bit, uniform `forward(x, state: tuple[Tensor, Tensor]) -> (h_new, (h_new, c_new))` contract, `new_state(batch, device)` returning `(zeros, zeros)` with the module's parameter dtype.
- `LstmSpec` pydantic class appended to the `LayerSpec` discriminated union; `build_layer` dispatch.
- `_zero_state_where_done` extended to dispatch on `isinstance(state, tuple)`: zero each tuple element via the existing Tensor branch; keep the `TypeError` fall-through for future non-Tensor non-tuple states.
- `nn_param_specs_from_v2` + `_lstm_specs` ParamSpec generator: tanh-Xavier bounds on each 4H gate block (ih, hh), small Gaussian bias bounds; dispatch by `spec.type == "lstm"`.
- `export_v2_policy_to_json` / `load_policy_from_json` Lstm branches. Obs-norm bake-in raises `NotImplementedError` when layer 0 is LSTM, matching the Phase 0 invariant enforced for GRU.
- `config.py::_layer_n_params` lstm arm: `4*H*I + 4*H*H + 8*H`.
- **Activation-aware init** (`training/initialization.py`): new `init_v2_population(architecture, n_pop, bound_multiplier, rng) -> ndarray[n_pop, n_params]` with per-layer-type dispatch:
  - Dense: existing Xavier/He/LeCun per activation (lifted from the dense-only path).
  - GRU: tanh-Xavier on each of the 3H gate blocks, σ=0.01 Gaussian for biases.
  - LSTM: tanh-Xavier on each of the 4H gate blocks, σ=0.01 Gaussian for i/g/o biases, **forget-gate bias slice initialized to 1.0 + σ=0.01 Gaussian noise** (Jozefowicz et al 2015).
- `train.py` routes PSO initial population through `init_v2_population` when `cfg.network.architecture` is set. GRU retroactively benefits.
- Training configs: `msr_aller_lstm_pso_train.toml` (Dense(16->32, tanh) -> Lstm(32, 32) -> Dense(32->2, linear), PSO, 9058 params) and `msr_aller_lstm_ppo_train.toml` (Dense(23->32, tanh) -> Lstm(32, 32) -> Dense(32->2, linear), PPO `bptt_length=32`).
- `compare_guidance.SCHEMES` + `_NN_DEPLOY_SCHEMES` register `neural_network_lstm_pso` and `neural_network_lstm_ppo`, dispatching through the Rust `neural_network` runtime.
- `train_all.sh` aliases: `lstm_pso`, `lstm_ppo`, `nn_lstm_pso`, `nn_lstm_ppo`.
- Test coverage: cross-language LSTM equivalence (target machine epsilon), PSO smoke (2 gens, ~600-param arch), PPO smoke (5 updates, ~600-param arch), init statistics gate (Xavier std ± tolerance, LSTM forget-bias slice in [0.9, 1.1], GRU retroactively matches Xavier), feedforward PPO regression gate preserved, 6/6 guidance golden regressions bit-identical.

**Out of scope for Phase 2a:**

- Window-MLP (Phase 2b, next PR on this branch or a sibling branch).
- SAC-LSTM (Phase 1.6 carry-over; SAC stays on `GaussianPolicy`).
- Recurrent critic (Phase 1.5 carry-over; feedforward critic stays).
- Widening `load_policy_from_json` to accept v1 JSON (Phase 0 carry-over).
- Clippy warnings in `src/rust/aerocapture-py/src/lib.rs` (Phase 1 carry-over; out of scope, addressed in a separate cleanup PR).
- Peephole LSTM, bidirectional LSTM, layer-norm LSTM, ConvLSTM -- all out of paper scope.
- Transformer, Mamba, Window-MLP (Phases 2b / 3 / 4).

## 3. Architecture

### 3.1 Rust LSTM layer (`src/rust/src/data/neural.rs`)

```rust
#[derive(Debug, Clone)]
pub struct LstmLayer {
    pub input_size: usize,
    pub hidden_size: usize,
    pub weight_ih: Vec<Vec<f64>>, // [4H rows, I cols]  row-major, gate order (i, f, g, o)
    pub weight_hh: Vec<Vec<f64>>, // [4H rows, H cols]
    pub bias_ih:   Vec<f64>,      // [4H]
    pub bias_hh:   Vec<f64>,      // [4H]
}

pub enum Layer {
    Dense(DenseLayer),
    Gru(GruLayer),
    Lstm(LstmLayer),
}

pub enum LayerSpec {
    Dense { input_size: usize, output_size: usize, activation: Activation },
    Gru   { input_size: usize, hidden_size: usize },
    Lstm  { input_size: usize, hidden_size: usize },
}
```

**Gate convention** (PyTorch `nn.LSTMCell`, gate ordering i, f, g, o; the 4H gate axis is concatenated in that order):

Let `ih = weight_ih @ x + bias_ih` and `hh = weight_hh @ h_prev + bias_hh`, both of length `4H`. Slice each into four `H`-wide blocks `(ih_i, ih_f, ih_g, ih_o)` and likewise for `hh`. Then:

```
i      = sigmoid(ih_i + hh_i)
f      = sigmoid(ih_f + hh_f)
g      = tanh(   ih_g + hh_g)
o      = sigmoid(ih_o + hh_o)
c_new  = f * c_prev + i * g
h_new  = o * tanh(c_new)
```

No peepholes. The two biases are mathematically redundant (`nn.LSTMCell` keeps them for CuDNN compatibility); we keep them for bit-for-bit PyTorch parity.

**Forward signature** (`Layer::forward` arm):

```rust
Layer::Lstm(layer) => {
    let state = match state {
        LayerState::Lstm { h, c } => (h, c),
        _ => panic!("LSTM layer requires LstmState"),
    };
    let (h_new, c_new) = layer.step(x, state.0, state.1);
    *state.0 = h_new.clone();
    *state.1 = c_new;
    h_new
}
```

**`LayerWeights for LstmLayer`** (flat order matches GRU's pattern scaled to 4H):

```
to_flat: weight_ih row-major (4H*I)
       + weight_hh row-major (4H*H)
       + bias_ih  (4H)
       + bias_hh  (4H)
total = 4*H*I + 4*H*H + 8*H
```

`n_params` returns the same. `from_flat` splits the flat vector in the same order.

### 3.2 Rust LSTM state (`src/rust/src/data/nn_state.rs`)

```rust
pub enum LayerState {
    None,
    Gru(Vec<f64>),
    Lstm { h: Vec<f64>, c: Vec<f64> },
}

impl LayerState {
    pub fn for_layer(spec: &LayerSpec) -> Self {
        match spec {
            LayerSpec::Dense { .. } => LayerState::None,
            LayerSpec::Gru   { hidden_size, .. } => LayerState::Gru(vec![0.0; *hidden_size]),
            LayerSpec::Lstm  { hidden_size, .. } => LayerState::Lstm {
                h: vec![0.0; *hidden_size],
                c: vec![0.0; *hidden_size],
            },
        }
    }

    pub fn reset(&mut self) {
        match self {
            LayerState::None => {}
            LayerState::Gru(h) => h.iter_mut().for_each(|v| *v = 0.0),
            LayerState::Lstm { h, c } => {
                h.iter_mut().for_each(|v| *v = 0.0);
                c.iter_mut().for_each(|v| *v = 0.0);
            }
        }
    }
}
```

Named struct variant (not positional `LayerState::Lstm(Vec<f64>, Vec<f64>)`) because the `.0` / `.1` indexing convention is footgun-prone in a multi-tensor future (consider a layer with five state tensors); grep-ability of `LayerState::Lstm { h, ... }` also beats `LayerState::Lstm(h_something, _)`. `Clone` derives via the existing enum derive; RL rollout snapshots continue to work.

### 3.3 Rust TOML parser (`src/rust/src/config.rs`)

```rust
pub enum TomlLayerSpec {
    Dense { input_size: usize, output_size: usize, activation: String },
    Gru   { hidden_size: usize },
    Lstm  { hidden_size: usize },
}

impl TomlLayerSpec {
    pub fn to_layer_spec(&self, prev_output: usize) -> LayerSpec {
        match self {
            TomlLayerSpec::Dense { output_size, activation, .. } => LayerSpec::Dense {
                input_size: prev_output,
                output_size: *output_size,
                activation: Activation::from_str(activation),
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

`input_size` is inferred from the previous layer (consistent with Phase 1 GRU), not specified in TOML. Example:

```toml
[[network.architecture]]
type = "dense"
output_size = 32
activation = "tanh"

[[network.architecture]]
type = "lstm"
hidden_size = 32

[[network.architecture]]
type = "dense"
output_size = 2
activation = "linear"
```

### 3.4 Python LSTM layer (`src/python/aerocapture/training/rl/layers/lstm.py`)

```python
class LstmLayer(nn.Module):
    """Manual nn.LSTMCell reproduction. Matches PyTorch bit-for-bit on f64."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        H = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(4 * H, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * H, H))
        self.bias_ih   = nn.Parameter(torch.empty(4 * H))
        self.bias_hh   = nn.Parameter(torch.empty(4 * H))

    def forward(
        self, x: Tensor, state: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        h_prev, c_prev = state
        ih = F.linear(x,      self.weight_ih, self.bias_ih)   # (B, 4H)
        hh = F.linear(h_prev, self.weight_hh, self.bias_hh)   # (B, 4H)
        H = self.hidden_size
        gates = ih + hh
        i = torch.sigmoid(gates[..., 0*H : 1*H])
        f = torch.sigmoid(gates[..., 1*H : 2*H])
        g = torch.tanh(   gates[..., 2*H : 3*H])
        o = torch.sigmoid(gates[..., 3*H : 4*H])
        c_new = f * c_prev + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, (h_new, c_new)

    def new_state(self, batch_size: int, device, dtype=None):
        dtype = dtype or self.weight_ih.dtype
        H = self.hidden_size
        zeros = torch.zeros(batch_size, H, device=device, dtype=dtype)
        return (zeros, zeros.clone())
```

The uniform `forward(x, state) -> (y, new_state)` return contract lets `V2Policy.evaluate` iterate the layer list without type-sniffing; LSTM's state happens to be a tuple, GRU's is a Tensor, Dense's is `None`. The downstream `_zero_state_where_done` helper dispatches by type.

### 3.5 `_zero_state_where_done` tuple extension (`rl/policy.py`)

Current helper (Phase 1.5):

```python
def _zero_state_where_done(state, done_mask):
    if state is None:
        return state
    if isinstance(state, Tensor):
        return state * (~done_mask).to(state.dtype).unsqueeze(-1)
    raise TypeError(f"Unsupported state type: {type(state)}")
```

Phase 2a extension:

```python
def _zero_state_where_done(state, done_mask):
    if state is None:
        return state
    if isinstance(state, Tensor):
        return state * (~done_mask).to(state.dtype).unsqueeze(-1)
    if isinstance(state, tuple):
        return tuple(_zero_state_where_done(s, done_mask) for s in state)
    raise TypeError(f"Unsupported state type: {type(state)}")
```

Tuple branch recurses so that a future LSTM-in-LSTM or tuple-of-tuples composition (e.g. stacked LSTMs inside a single logical block) keeps working. The `TypeError` fall-through stays to force the next multi-tensor layer type (Mamba's SSM state, Transformer's KV cache) to come with its own explicit branch rather than silently pass through as a tuple and cause a wrong-shape downstream error.

### 3.6 Activation-aware init (`training/initialization.py`)

**Motivation:** Phase 1 deferred per-layer init and the v2 path falls back to uniform-in-ParamSpec-bounds. For dense-only archs that's fine (the dense ParamSpec bounds *are* Xavier/He/LeCun per activation, lifted from `initialization.py` via `compute_layer_bound`). For GRU and LSTM, the per-gate bounds are Xavier-tanh in `_gru_specs` / `_lstm_specs`, but the initial population is drawn uniformly within those bounds, which produces a *triangular-ish* weight distribution instead of Gaussian and a forget-bias of `0 ± bound` instead of `1 ± small_noise`. Empirically the difference matters most for LSTM forget-bias; fixing init is the least-hand-wavy way to publish the architecture comparison.

**New API:**

```python
def init_v2_population(
    architecture: list[LayerSpec],
    n_pop: int,
    bound_multiplier: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return (n_pop, n_params) initial chromosomes for the PSO path.
    Per-layer dispatch: dense -> existing Xavier/He/LeCun, gru -> tanh-Xavier on
    gate matrices + small bias, lstm -> tanh-Xavier on gate matrices + forget
    bias init to 1.0 + small noise on other biases.
    """
```

Implementation detail (per-layer slice indexing):

- Walk `architecture` layer-by-layer. For each layer, compute the flat-weight slice it owns (using the same `_layer_n_params` helper that `param_spaces.py` already uses).
- Dense: drop-in reuse of existing `compute_layer_bound(activation, fan_in) * uniform(-1, 1)` or the Gaussian variant. (Preserve whichever the dense-only path uses today to minimize diff against Phase 1 golden tests.)
- GRU: for each of the 3H gate rows, draw `weight_ih[gate_row, :]` as tanh-Xavier with fan_in=I, `weight_hh[gate_row, :]` as tanh-Xavier with fan_in=H, `bias_ih[gate_row] ~ N(0, 0.01)`, `bias_hh[gate_row] ~ N(0, 0.01)`.
- LSTM: same as GRU but 4H blocks. The forget-gate block is rows `[H:2*H]` of the 4H axis (per the (i, f, g, o) ordering). Override the forget-bias slice:
  ```
  bias_ih[H:2*H] = 1.0 + N(0, 0.01)
  bias_hh[H:2*H] = 0.0 + N(0, 0.01)   # keep bias_hh ~ 0 since bias_ih carries the +1
  ```
  The Jozefowicz recommendation is "forget-bias init to 1"; splitting the +1 across both biases would double-apply (sum is what enters the gate sigmoid). Put the +1 on `bias_ih` only.

**Integration:**

- `train.py`: when `cfg.network.architecture is not None`, call `init_v2_population(...)` instead of the existing `create_nn_initial_population` / uniform-via-ParamSpec paths. Dense-only v2 archs still route through this to preserve the Xavier behavior they already get under v1.
- `create_nn_initial_population` (v1 dense-only) is untouched. v1 training path is unchanged bit-for-bit.
- GRU retroactively benefits: re-running the existing Phase 1 PSO-GRU smoke test after this phase should not be required to match golden exactly (PSO is stochastic), but training convergence should be no worse.

**Bound multiplier:** `bound_multiplier` scales the Xavier std (PSO chromosome bounds in `_gru_specs` / `_lstm_specs` already apply this multiplier as a ceiling; the sampler respects the Xavier Gaussian std, not the bound). Clip draws to the ParamSpec bounds to stay inside PSO's search space.

### 3.7 Flat weight layout for LSTM PSO encoding

`encoding.py::_lstm_specs`:

```python
def _lstm_specs(input_size: int, hidden_size: int, bound_multiplier: float) -> list[ParamSpec]:
    """4H gate ordering (i, f, g, o). Bounds: tanh-Xavier on W_ih rows and W_hh
    rows per-gate; small bounds on biases; forget-bias init handled in
    init_v2_population, not in ParamSpec bounds (bounds are symmetric around 0).
    """
    # W_ih: (4H, I) row-major
    specs = [...]  # ParamSpec(lower=-bound_ih, upper=+bound_ih) per element
    # W_hh: (4H, H) row-major
    specs += [...]
    # bias_ih: (4H,)
    specs += [...]
    # bias_hh: (4H,)
    specs += [...]
    return specs
```

Match GRU's approach: the bounds are symmetric around 0 at the Xavier-std level. Forget-bias initialization is handled at population creation in `init_v2_population`; PSO chromosomes at later generations can drift the forget bias freely (bounds don't pin it). This matches how the dense Phase 1 path handles Xavier init: init-time only, bounds stay symmetric.

### 3.8 JSON v2 format addition

```json
{
  "format_version": 2,
  "architecture": [
    {"type": "dense", "input_size": 16, "output_size": 32, "activation": "tanh"},
    {"type": "lstm",  "input_size": 32, "hidden_size": 32},
    {"type": "dense", "input_size": 32, "output_size": 2,  "activation": "linear"}
  ],
  "input_mask": [...],
  "layers": [
    {"type": "dense", "w": [...], "b": [...]},
    {"type": "lstm",
     "weight_ih": [...],   "weight_hh": [...],
     "bias_ih":   [...],   "bias_hh":   [...]},
    {"type": "dense", "w": [...], "b": [...]}
  ]
}
```

Matrix elements serialize row-major (same as Dense `w` and GRU). `aerocapture_rs.flat_weights_to_json` (the Rust-side PSO chromosome serializer) gets an `Lstm` branch so Python continues to delegate JSON serialization to Rust.

### 3.9 Compare-guidance and training orchestration

- `compare_guidance.py`:
  - `SCHEMES = {..., "neural_network_lstm_pso": {...}, "neural_network_lstm_ppo": {...}}`. Both entries deploy through Rust `neural_network` runtime (they point `data.neural_network` at the training output's `best_model.json`).
  - `_NN_DEPLOY_SCHEMES |= {"neural_network_lstm_pso", "neural_network_lstm_ppo"}`.
- `train_all.sh`:
  - New aliases `lstm_pso` / `lstm_ppo` / `nn_lstm_pso` / `nn_lstm_ppo`, mapped to the corresponding training configs.
  - Dependency order: after `piecewise_constant` (ref trajectory + corridor) -- same as other NN schemes.

## 4. Training configs

### 4.1 `configs/training/msr_aller_lstm_pso_train.toml`

Mirror of `msr_aller_gru_pso_train.toml` with the gate layer swapped:

```toml
base = "common.toml"

[guidance]
scheme = "neural_network"

[network]
input_mask = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

[[network.architecture]]
type = "dense"
output_size = 32
activation = "tanh"

[[network.architecture]]
type = "lstm"
hidden_size = 32

[[network.architecture]]
type = "dense"
output_size = 2
activation = "linear"

[optimizer]
algorithm = "pso"
n_pop = 64
n_gen = 1000
seed_strategy = "adaptive"
training_n_sims = 20
validation_n_sims = 1000

[simulation]
results_suffix = "lstm_pso"
```

**Param count:** Dense(16->32) = 544, Lstm(32,32) = 4*(32*32 + 32*32 + 32 + 32) = 8448, Dense(32->2) = 66. Total = **9058**.

### 4.2 `configs/training/msr_aller_lstm_ppo_train.toml`

Mirror of `msr_aller_gru_ppo_train.toml`:

```toml
base = "common.toml"

[guidance]
scheme = "neural_network"

[network]
# Full 23-input candidate vector for PPO (matches the existing RL convention)
input_mask = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

[[network.architecture]]
type = "dense"
output_size = 32
activation = "tanh"

[[network.architecture]]
type = "lstm"
hidden_size = 32

[[network.architecture]]
type = "dense"
output_size = 2
activation = "linear"

[rl]
algorithm = "ppo"
total_steps = 5_000_000
n_envs = 64
rollout_steps = 256

[rl.ppo]
bptt_length = 32
learning_rate = 3e-4
clip_range = 0.2
entropy_coef = 0.0
update_epochs = 10
target_kl = 0.015

[rl.reward]
# (inherit from common.toml or RL defaults)

[simulation]
results_suffix = "lstm_ppo"
```

## 5. Test plan

### 5.1 Unit tests

- **Rust `test_lstm_forward_shape`**: build a minimal LSTM (I=3, H=2), feed 10 random f64 inputs sequentially, assert output finite and state updates non-trivially.
- **Rust `test_lstm_layer_weights_roundtrip`**: random `LstmLayer` -> `to_flat` -> `from_flat` -> compare element-wise (machine epsilon).
- **Rust `test_lstm_json_v2_roundtrip`**: random `LstmLayer` -> `save_json` -> `from_v2_json` -> forward on 10 inputs -> compare to original forward (machine epsilon).
- **Python `test_lstm_layer_matches_nn_lstmcell`**: build `LstmLayer(I, H)` and a matching `nn.LSTMCell(I, H)`, copy parameters, feed 10 random f64 inputs, assert max abs diff < 1e-10 on both h and c.
- **Python `test_zero_state_where_done_tuple`**: construct a `(Tensor, Tensor)` state and a `done_mask` with some True entries, call `_zero_state_where_done`, assert both tensors zeroed on those rows and preserved elsewhere.
- **Python `test_init_v2_population_forget_bias`**: build a Dense -> Lstm -> Dense architecture, call `init_v2_population(n_pop=1024)`, extract the `bias_ih[H:2*H]` forget-gate slice of each chromosome, assert mean in [0.9, 1.1] and std in [0.005, 0.02]. Also assert `bias_hh[H:2*H]` mean in [-0.01, 0.01] (forget contribution is on `bias_ih` only, not double-applied).
- **Python `test_init_v2_population_xavier_std`**: for the dense and GRU cases, assert per-layer weight std matches theoretical Xavier (tanh) within 10% relative tolerance.
- **Python `test_lstm_n_params`**: `_layer_n_params({'type':'lstm', 'input_size':32, 'hidden_size':32}) == 4*32*32 + 4*32*32 + 8*32 == 8448`.

### 5.2 Integration tests

- **`tests/test_v2_rust_python_equivalence.py` LSTM case**: build a Dense(5->4, tanh) -> Lstm(4, 4) -> Dense(4->2, linear) `V2Policy` in f64, export to v2 JSON, load via `aerocapture_rs.nn_forward`, feed 100 random f64 inputs sequentially (with reset at step 50 to exercise both the h and c zeroing path), assert max abs diff < 1e-10. Target: machine epsilon (< 1e-14) like GRU's 4.4e-16.
- **`tests/test_lstm_pso_smoke.py`** (`@pytest.mark.slow`, python-pyo3 CI): 2 PSO gens on Dense(16->8, tanh) -> Lstm(8, 8) -> Dense(8->2, linear), 16 real sims, assert (a) `best_model.json` is v2 with `["dense", "lstm", "dense"]`, (b) `nn_forward` returns finite 2-tuple on a sample input, (c) per-gen cost finite.
- **`tests/test_lstm_ppo_smoke.py`** (`@pytest.mark.slow`, python-pyo3 CI): 5 PPO updates on Dense(23->8, tanh) -> Lstm(8, 8) -> Dense(8->2, linear), `bptt_length=8`, `rollout_steps=16`, `n_envs=4`, assert (a) training completes without NaN loss, (b) exported JSON loads through `aerocapture_rs.nn_forward`.
- **`tests/test_ppo_bptt_lstm_chunk_invariant.py`**: extend the existing Phase 1.5 chunk-invariant test to LSTM. Fix seeds, run `ppo_update_bptt` with `bptt_length=rollout_steps` (one chunk) and `bptt_length=rollout_steps/4` (four chunks), assert forward values bit-identical (gradients differ, that's expected).

### 5.3 Regression gates

- Feedforward PPO regression gate (Phase 1.5) continues to pass: V2Policy + `bptt_length=rollout_steps` on dense-only `msr_aller_rl_train.toml` produces a loadable finite model after 5 updates.
- PSO-GRU smoke test (Phase 1) continues to pass. The init refactor changes the initial population distribution (from uniform-in-bounds to tanh-Xavier Gaussian), but the smoke test asserts shape + finiteness + JSON structure only, not specific cost values.
- 6/6 guidance golden regressions in `tests/reference_data/rust_golden/` bit-identical. LSTM doesn't touch physics; this should be automatic, but check.

### 5.4 CI wiring

- Rust unit + integration tests run in the existing `cargo test` job.
- Python tests run in the existing `pytest` job.
- PSO-LSTM + PPO-LSTM + Rust<>Python LSTM equivalence smoke tests run in the `python-pyo3` job (mirrors GRU's wiring).

## 6. Compatibility

- **JSON v1**: unchanged. All Phase 1 / 1.5 artifacts load bit-for-bit.
- **JSON v2 without LSTM**: unchanged. Existing v2 Dense-only and Dense->GRU->Dense artifacts load identically (parser falls through the same arms it already does).
- **`create_nn_initial_population`** (v1 dense-only path): untouched. v1 training convergence is unchanged.
- **GRU retroactive init change**: training convergence may improve (Xavier Gaussian is closer to theoretical recommendation than uniform-in-Xavier-bounds). This is the intended effect. PSO-GRU smoke-test assertions are shape + finiteness, not cost, so the CI gate stays green.
- **PPO-GRU `bptt_length` default of 32**: unchanged. LSTM reuses the same knob.
- **SAC**: untouched (stays on `GaussianPolicy`).

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `_zero_state_where_done` tuple branch has a subtle bug on deeply-nested tuples. | Unit test in §5.1. Recursion is strict (tuple of tuple of tuple); `isinstance(state, tuple)` keeps matching until it hits a Tensor. |
| LSTM gate-ordering disagreement between Rust and PyTorch. | Cross-language equivalence test at machine epsilon catches any off-by-one gate-slice error. This is the same gate as GRU's success story. |
| Forget-bias init mutates PSO initial population so that a previously-passing Phase 1 checkpoint becomes non-resumable. | Initial population is only drawn at gen 0. Resumes read the chromosome from checkpoint, not from `init_v2_population`. |
| GRU init change silently regresses PSO-GRU convergence. | PSO-GRU smoke test continues to pass (shape + finiteness). A full GRU retraining (`./train_all.sh nn_gru_pso`) after the PR lands verifies convergence empirically; if it regresses, the init change is reverted for GRU via a one-line dispatch (keep GRU on uniform, apply the Xavier + forget-bias dispatch to LSTM only). |
| LSTM flat-weight ordering in `LayerWeights::from_flat` desyncs from Python encoding. | `test_lstm_layer_weights_roundtrip` in Rust + `test_v2_rust_python_equivalence.py` LSTM case catches this jointly. |
| Param-count increase (9058 vs 6946) regresses PSO wall-clock. | PSO's per-eval cost is dominated by the Rust simulator, not the NN forward. A 30% param increase in the NN block is <1% of per-sim cost. Unlikely to bite. |

## 8. Implementation order (rough sketch for the plan skill)

1. Rust `LstmLayer` + `Layer::Lstm` + `LayerSpec::Lstm` + `LayerState::Lstm { h, c }` + `TomlLayerSpec::Lstm`. Unit tests §5.1 Rust rows.
2. `LayerWeights for LstmLayer` flat-weight round-trip. Unit test.
3. `NeuralNetModel::save_json` / `from_v2_json` / `from_flat_weights_v2` LSTM arms. JSON round-trip test.
4. `aerocapture_rs.flat_weights_to_json` Lstm branch.
5. Python `LstmLayer`, `LstmSpec`, `build_layer` dispatch. `test_lstm_layer_matches_nn_lstmcell`.
6. `_zero_state_where_done` tuple branch. `test_zero_state_where_done_tuple`.
7. `_lstm_specs`, `nn_param_specs_from_v2` dispatch, `config.py::_layer_n_params` Lstm arm.
8. `export_v2_policy_to_json` + `load_policy_from_json` Lstm branches.
9. Cross-language equivalence test `tests/test_v2_rust_python_equivalence.py` LSTM case.
10. `training/initialization.py::init_v2_population` -- implement dense + gru + lstm dispatch (with forget-bias-1 for LSTM). Unit tests §5.1 Python init rows.
11. `train.py` routes v2 PSO initial population through `init_v2_population`. Manual re-run of Phase 1 PSO-GRU smoke to confirm no shape regressions.
12. Training configs `msr_aller_lstm_pso_train.toml` + `msr_aller_lstm_ppo_train.toml`.
13. `compare_guidance` registration + `train_all.sh` aliases.
14. Smoke tests `test_lstm_pso_smoke.py` + `test_lstm_ppo_smoke.py`. Wire into `python-pyo3` CI job.
15. Extend `test_ppo_bptt_chunk_invariant.py` to LSTM.
16. Full verification: `./check_all.sh`, `uv run pytest tests`, `./lint_code.sh`, 6/6 guidance golden regressions.
17. Invoke `smart-commit` skill, targeting the whole `feature/lstm-mvp` branch.

## 9. Deliverables

- `src/rust/src/data/neural.rs`: `LstmLayer`, `Layer::Lstm`, `LayerSpec::Lstm`, `LayerWeights for LstmLayer`, LSTM arms in `save_json` / `from_v2_json` / `from_flat_weights_v2`.
- `src/rust/src/data/nn_state.rs`: `LayerState::Lstm { h, c }`, `for_layer` / `reset` arms.
- `src/rust/src/config.rs`: `TomlLayerSpec::Lstm`, `to_layer_spec` arm.
- `src/rust/aerocapture-py/src/lib.rs`: `flat_weights_to_json` Lstm branch.
- `src/python/aerocapture/training/rl/layers/lstm.py`: new.
- `src/python/aerocapture/training/rl/layers/__init__.py`: dispatch line.
- `src/python/aerocapture/training/rl/schemas.py`: `LstmSpec` + union entry.
- `src/python/aerocapture/training/rl/policy.py`: `_zero_state_where_done` tuple branch.
- `src/python/aerocapture/training/rl/export.py`: Lstm branch in `export_v2_policy_to_json`.
- `src/python/aerocapture/training/model_io.py`: Lstm branch in `load_policy_from_json`.
- `src/python/aerocapture/training/encoding.py`: `_lstm_specs`, `nn_param_specs_from_v2` dispatch.
- `src/python/aerocapture/training/config.py`: `_layer_n_params` Lstm arm.
- `src/python/aerocapture/training/initialization.py`: `init_v2_population` with dense/gru/lstm dispatch.
- `src/python/aerocapture/training/train.py`: route v2 initial population through `init_v2_population`.
- `src/python/aerocapture/training/compare_guidance.py`: register `neural_network_lstm_pso`, `neural_network_lstm_ppo`.
- `configs/training/msr_aller_lstm_pso_train.toml`: new.
- `configs/training/msr_aller_lstm_ppo_train.toml`: new.
- `train_all.sh`: `lstm_pso` / `lstm_ppo` / `nn_lstm_pso` / `nn_lstm_ppo` aliases.
- Tests per §5.
- `CLAUDE.md`: Phase 2a subsection (paralleling Phase 1 / 1.5 structure) summarizing the contract and the LSTM-specific details.
- `TODO.md`: Phase 2a checkbox list marked done; carry-overs updated (Phase 1 activation-aware init closed; v1 JSON / clippy / SAC-GRU / recurrent critic remain).

## 10. Final step

Invoke `smart-commit` skill against the whole `feature/lstm-mvp` branch (per user's global CLAUDE.md rule).
