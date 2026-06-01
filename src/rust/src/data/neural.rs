//! Neural network model with modular architecture.
//!
//! Supports arbitrary layer configurations (e.g. [6, 12, 2] or [6, 24, 12, 2])
//! with per-layer activation function choice. Loads from JSON format.

use super::DataError;
use crate::data::nn_state::{LayerState, NnState};
use serde::{Deserialize, Serialize};

#[inline]
pub(crate) fn gelu_exact(z: f64) -> f64 {
    // Exact GELU: 0.5 * z * (1 + erf(z / sqrt(2)))
    // Uses libm::erf for IEEE-754 correct rounding; matches torch.special.erf.
    const INV_SQRT2: f64 = 0.7071067811865475_f64;
    0.5 * z * (1.0 + libm::erf(z * INV_SQRT2))
}

pub(crate) fn layer_norm_biased(x: &[f64], gamma: &[f64], beta: &[f64], eps: f64) -> Vec<f64> {
    debug_assert_eq!(x.len(), gamma.len());
    debug_assert_eq!(x.len(), beta.len());
    let n = x.len() as f64;
    // Sequential reduction for cross-language bit-identity.
    let mut mean = 0.0;
    for v in x {
        mean += *v;
    }
    mean /= n;
    let mut var = 0.0;
    for v in x {
        let d = *v - mean;
        var += d * d;
    }
    var /= n; // biased: 1/N, NOT Bessel 1/(N-1); matches torch nn.LayerNorm default.
    let inv_std = 1.0 / (var + eps).sqrt();
    x.iter()
        .zip(gamma)
        .zip(beta)
        .map(|((xi, g), b)| ((*xi - mean) * inv_std) * g + b)
        .collect()
}

pub(crate) fn build_pe_table(n_seq: usize, d_model: usize) -> Vec<Vec<f64>> {
    // Standard Vaswani et al. 2017 sinusoidal positional encoding.
    // PE[pos, 2k]   = sin(pos / 10000^(2k / d_model))
    // PE[pos, 2k+1] = cos(pos / 10000^(2k / d_model))
    // Iteration order: pos outer, i inner. Matches Python mirror for bit-identity.
    (0..n_seq)
        .map(|pos| {
            (0..d_model)
                .map(|i| {
                    let k = i / 2;
                    let div = 10000.0_f64.powf((2.0 * k as f64) / d_model as f64);
                    let angle = pos as f64 / div;
                    if i % 2 == 0 { angle.sin() } else { angle.cos() }
                })
                .collect()
        })
        .collect()
}

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

/// Activation function for a layer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Tanh,
    Relu,
    Sigmoid,
    Asinh,
    Linear,
    Swish,
    Mish,
}

/// Output parameterization for the NN's bank-angle decoder.
///
/// `Atan2Signed` (default, backward-compatible): emits 2 outputs and
/// `bank = atan2(out[0], out[1]) ∈ (-π, π]`.
///
/// `AcosTanh`: emits 1 output through `tanh` and `bank = acos(out[0]) ∈ [0, π]`.
/// Only legal in `magnitude_only` mode (architecture validates last layer
/// `output_size = 1` with activation `tanh`).
///
/// `ScaledPi`: emits 1 tanh output; `bank = scaled_pi_n * π * out[0] ∈ [-n·π, n·π]`.
///
/// `Delta`: emits 1 tanh output; `bank = prev_realized + delta_max * out[0]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputParam {
    #[default]
    Atan2Signed,
    AcosTanh,
    ScaledPi,
    Delta,
}

impl Activation {
    fn apply(self, x: f64) -> f64 {
        match self {
            Activation::Tanh => x.tanh(),
            Activation::Relu => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Asinh => x.asinh(),
            Activation::Linear => x,
            Activation::Swish => x / (1.0 + (-x).exp()),
            Activation::Mish => x * (1.0_f64 + x.exp()).ln().tanh(),
        }
    }
}

/// Parse an activation name string into the Activation enum.
/// Uses serde's Activation deserialize so the canonical set of names
/// matches Activation's #[serde(rename_all = "snake_case")] derive.
pub fn parse_activation(s: &str) -> Result<Activation, DataError> {
    serde_json::from_str::<Activation>(&format!("\"{}\"", s))
        .map_err(|e| DataError(format!("parse_activation({:?}): {}", s, e)))
}

/// A dense (fully-connected) layer: affine transform + activation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseLayer {
    /// Weights [n_out × n_in], row-major: w[j][i] = weight from input i to output j.
    pub w: Vec<Vec<f64>>,
    /// Biases [n_out].
    pub b: Vec<f64>,
    /// Activation function applied after affine transform.
    pub activation: Activation,
}

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

/// Dot product `row . vec + bias`. Helper for per-gate pre-activation sums.
#[inline]
fn dot_plus_bias(row: &[f64], vec: &[f64], bias: f64) -> f64 {
    bias + row.iter().zip(vec).map(|(w, v)| w * v).sum::<f64>()
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
            let r = Activation::Sigmoid.apply(
                dot_plus_bias(&self.weight_ih[i], x, self.bias_ih[i])
                    + dot_plus_bias(&self.weight_hh[i], h_prev, self.bias_hh[i]),
            );
            // z gate: row i + H
            let z = Activation::Sigmoid.apply(
                dot_plus_bias(&self.weight_ih[i + h_size], x, self.bias_ih[i + h_size])
                    + dot_plus_bias(
                        &self.weight_hh[i + h_size],
                        h_prev,
                        self.bias_hh[i + h_size],
                    ),
            );
            // n gate: row i + 2H. The r-gate is applied to the hidden-side aggregate
            // (PyTorch nn.GRUCell convention, differs from Cho-2014's W_hn @ (r * h)).
            let s_ih_n = dot_plus_bias(
                &self.weight_ih[i + 2 * h_size],
                x,
                self.bias_ih[i + 2 * h_size],
            );
            let s_hh_n = dot_plus_bias(
                &self.weight_hh[i + 2 * h_size],
                h_prev,
                self.bias_hh[i + 2 * h_size],
            );
            let n = (s_ih_n + r * s_hh_n).tanh();

            h_new[i] = (1.0 - z) * n + z * h_prev[i];
        }
        h_new
    }
}

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
                + dot_plus_bias(
                    &self.weight_hh[idx + 2 * h],
                    h_prev,
                    self.bias_hh[idx + 2 * h],
                ))
            .tanh();
            // o gate: row idx + 3H
            let o = Activation::Sigmoid.apply(
                dot_plus_bias(&self.weight_ih[idx + 3 * h], x, self.bias_ih[idx + 3 * h])
                    + dot_plus_bias(
                        &self.weight_hh[idx + 3 * h],
                        h_prev,
                        self.bias_hh[idx + 3 * h],
                    ),
            );

            c_new[idx] = f * c_prev[idx] + i * g;
            h_new[idx] = o * c_new[idx].tanh();
        }
        (h_new, c_new)
    }
}

/// Window-MLP layer: FIFO ring buffer of the last `n_steps` inputs,
/// concatenated into a vector of length `n_steps * input_size`.
///
/// Zero trainable parameters -- all trainable weight lives in the downstream
/// Dense layer. Phase 2b MVP ships PSO-only; PPO use raises
/// NotImplementedError at Python-side `build_layer(WindowSpec)`.
#[derive(Debug, Clone)]
pub struct WindowLayer {
    pub input_size: usize,
    pub n_steps: usize,
}

impl WindowLayer {
    /// Push `input` onto the tail of the ring buffer, drop the oldest slot,
    /// and return the flattened buffer (length = `n_steps * input_size`).
    ///
    /// Buffer is pre-filled with zero vectors at episode start (see
    /// `LayerState::for_layer`) so every tick is branchless: one pop_front,
    /// one push_back, one flatten. Takes the `VecDeque` directly (rather than
    /// a `&mut LayerState`) so the caller can hold the match-destructured
    /// buffer reference across the call without a double-borrow.
    pub fn forward(
        &self,
        input: &[f64],
        buffer: &mut std::collections::VecDeque<Vec<f64>>,
    ) -> Vec<f64> {
        assert_eq!(
            input.len(),
            self.input_size,
            "WindowLayer expected input_size={}, got {}",
            self.input_size,
            input.len()
        );
        buffer.pop_front();
        buffer.push_back(input.to_vec());
        let mut out = Vec::with_capacity(self.n_steps * self.input_size);
        for slot in buffer.iter() {
            out.extend_from_slice(slot);
        }
        out
    }
}

/// 1-layer pre-norm Transformer block with causal window attention.
///
/// K/V cache lives in `LayerState::Transformer { k_cache, v_cache }`; this
/// struct holds only the weights. `k_pe_offsets` and `v_pe_offsets` are
/// derived at load time via `rebuild_pe_offsets`; they are NOT in the flat
/// chromosome. Any mutation to `w_k` or `w_v` MUST be followed by a call
/// to `rebuild_pe_offsets` before the next forward.
#[derive(Debug, Clone)]
pub struct TransformerLayer {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_head: usize, // d_model / n_heads; validated at construction
    pub d_ffn: usize,
    pub n_seq: usize,

    pub w_q: Vec<Vec<f64>>,
    pub b_q: Vec<f64>,
    pub w_k: Vec<Vec<f64>>,
    pub b_k: Vec<f64>,
    pub w_v: Vec<Vec<f64>>,
    pub b_v: Vec<f64>,
    pub w_o: Vec<Vec<f64>>,
    pub b_o: Vec<f64>,

    pub w_ffn1: Vec<Vec<f64>>,
    pub b_ffn1: Vec<f64>,
    pub w_ffn2: Vec<Vec<f64>>,
    pub b_ffn2: Vec<f64>,

    pub ln1_gamma: Vec<f64>,
    pub ln1_beta: Vec<f64>,
    pub ln2_gamma: Vec<f64>,
    pub ln2_beta: Vec<f64>,

    // Derived at load time; NOT part of the flat chromosome.
    pub k_pe_offsets: Vec<Vec<f64>>,
    pub v_pe_offsets: Vec<Vec<f64>>,
}

/// Sequential matrix-vector product: m is [rows][cols] (row-major), v is [cols].
/// Deterministic FIFO reduction for cross-language bit-identity.
pub(crate) fn matvec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter()
        .map(|row| {
            debug_assert_eq!(row.len(), v.len());
            let mut acc = 0.0_f64;
            for (a, b) in row.iter().zip(v) {
                acc += a * b;
            }
            acc
        })
        .collect()
}

/// Returns `(k_cached + k_pe_offset)[h_start..h_end]` element-wise.
/// Used in `TransformerLayer::forward` to add the position-encoding offset
/// to a cached K vector for a single head slice without allocating the full
/// d_model vector.
#[inline]
fn slot_k_eff_head(
    k_cached: &[f64],
    k_pe_offset: &[f64],
    h_start: usize,
    h_end: usize,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(h_end - h_start);
    for j in h_start..h_end {
        out.push(k_cached[j] + k_pe_offset[j]);
    }
    out
}

impl TransformerLayer {
    /// Recompute `k_pe_offsets[i] = W_K @ PE[i]` and `v_pe_offsets[i] = W_V @ PE[i]`
    /// for i in 0..n_seq. Biases are NOT included in the PE offset (they are added
    /// once per forward to the raw query/key projections, not through PE).
    ///
    /// Call this after any mutation to `w_k` or `w_v` -- specifically from both
    /// `from_flat` and `from_v2_json` entry points.
    pub fn rebuild_pe_offsets(&mut self) {
        let pe = build_pe_table(self.n_seq, self.d_model);
        self.k_pe_offsets = pe.iter().map(|p| matvec(&self.w_k, p)).collect();
        self.v_pe_offsets = pe.iter().map(|p| matvec(&self.w_v, p)).collect();
    }

    /// Single-token forward for inference.
    ///
    /// - `x`: input of length `d_model`
    /// - `k_cache`, `v_cache`: K/V cache (mutated: push new, evict oldest if `len > n_seq`)
    ///
    /// Returns output of length `d_model`.
    pub fn forward(
        &self,
        x: &[f64],
        k_cache: &mut std::collections::VecDeque<Vec<f64>>,
        v_cache: &mut std::collections::VecDeque<Vec<f64>>,
    ) -> Vec<f64> {
        debug_assert_eq!(x.len(), self.d_model);

        // 1. LN1
        let x_norm1 = layer_norm_biased(x, &self.ln1_gamma, &self.ln1_beta, 1e-5);

        // 2. Q, K, V projections (with bias)
        let mut q = matvec(&self.w_q, &x_norm1);
        for (qi, bi) in q.iter_mut().zip(&self.b_q) {
            *qi += bi;
        }
        let mut k = matvec(&self.w_k, &x_norm1);
        for (ki, bi) in k.iter_mut().zip(&self.b_k) {
            *ki += bi;
        }
        let mut v = matvec(&self.w_v, &x_norm1);
        for (vi, bi) in v.iter_mut().zip(&self.b_v) {
            *vi += bi;
        }

        // 3. Push into cache, evict oldest if over capacity
        k_cache.push_back(k);
        v_cache.push_back(v);
        while k_cache.len() > self.n_seq {
            k_cache.pop_front();
            v_cache.pop_front();
        }
        let cache_len = k_cache.len();

        // 4. Multi-head attention: for each head, scores over cache, stabilized
        //    softmax (max-subtraction, sequential FIFO), weighted sum of V.
        let inv_sqrt_d_head = 1.0 / (self.d_head as f64).sqrt();
        let mut attn_out = vec![0.0_f64; self.d_model];

        for h in 0..self.n_heads {
            let h_start = h * self.d_head;
            let h_end = h_start + self.d_head;
            let q_h = &q[h_start..h_end];

            // Scores over the cache
            let mut scores = Vec::with_capacity(cache_len);
            for (k_slot, k_pe) in k_cache.iter().zip(self.k_pe_offsets.iter()) {
                let k_eff_h = slot_k_eff_head(k_slot, k_pe, h_start, h_end);
                let mut s = 0.0;
                for (a, b) in q_h.iter().zip(k_eff_h.iter()) {
                    s += a * b;
                }
                scores.push(s * inv_sqrt_d_head);
            }

            // Max-subtraction softmax, sequential FIFO
            let mut max_score = scores[0];
            for s in &scores[1..] {
                if *s > max_score {
                    max_score = *s;
                }
            }
            let mut exp_scores = Vec::with_capacity(cache_len);
            let mut exp_sum = 0.0;
            for s in &scores {
                let e = (*s - max_score).exp();
                exp_scores.push(e);
                exp_sum += e;
            }

            // Weighted sum of V_eff (head slice)
            for i in 0..cache_len {
                let w = exp_scores[i] / exp_sum;
                for j in h_start..h_end {
                    attn_out[j] += w * (v_cache[i][j] + self.v_pe_offsets[i][j]);
                }
            }
        }

        // 5. Output projection + residual
        let mut proj = matvec(&self.w_o, &attn_out);
        for (pi, bi) in proj.iter_mut().zip(&self.b_o) {
            *pi += bi;
        }
        let mut x1 = vec![0.0; self.d_model];
        for i in 0..self.d_model {
            x1[i] = x[i] + proj[i];
        }

        // 6. LN2 + FFN + residual
        let x_norm2 = layer_norm_biased(&x1, &self.ln2_gamma, &self.ln2_beta, 1e-5);
        let mut hidden = matvec(&self.w_ffn1, &x_norm2);
        for (hi, bi) in hidden.iter_mut().zip(&self.b_ffn1) {
            *hi += bi;
        }
        for h in hidden.iter_mut() {
            *h = gelu_exact(*h);
        }
        let mut ffn_out = matvec(&self.w_ffn2, &hidden);
        for (fi, bi) in ffn_out.iter_mut().zip(&self.b_ffn2) {
            *fi += bi;
        }

        let mut out = vec![0.0; self.d_model];
        for i in 0..self.d_model {
            out[i] = x1[i] + ffn_out[i];
        }
        out
    }
}

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

/// Layer variant. Phase 1 ships Dense and Gru; Phase 2a adds Lstm; Phase 2b adds Window; Phase 3a adds Transformer; Phase 4a adds Mamba.
#[derive(Debug, Clone)]
pub enum Layer {
    Dense(DenseLayer),
    Gru(GruLayer),
    Lstm(LstmLayer),
    Window(WindowLayer),
    // Boxed: TransformerLayer is 472 bytes vs 112 for GruLayer; boxing keeps enum size uniform.
    Transformer(Box<TransformerLayer>),
    // Boxed: MambaLayer's stack footprint is ~200 bytes (3 DMatrix + 2 DVector
    // headers); weight data lives on the heap behind those pointers regardless
    // of boxing. The box is purely for enum-variant size uniformity against
    // Transformer (472 bytes) -- same `large_enum_variant` clippy motivation.
    Mamba(Box<MambaLayer>),
}

impl Layer {
    /// Input size of this layer (for forward-pass shape checks).
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

/// Trait for flattening and reconstructing a layer's parameters.
///
/// Each layer type implements its own canonical flat ordering:
/// dense = W (row-major) then b; gru/lstm/attention/ssm defined per variant
/// (see Phase 1+ for those). Order MUST match the PyTorch mirror in
/// src/python/aerocapture/training/rl/layers/<type>.py for PSO chromosome
/// compatibility.
///
/// Callers MUST ensure `flat.len() >= self.n_params()` before invoking
/// `from_flat`; it may panic otherwise. Length validation lives at the
/// caller (see `NeuralNetModel::from_flat_weights`) so the trait method
/// stays infallible and later impls don't invent per-layer error dialects.
pub trait LayerWeights {
    fn to_flat(&self) -> Vec<f64>;
    // `from_flat` takes `&mut self` by design: it overwrites this layer's
    // weights in place from a flat slice and returns elements consumed.
    #[allow(clippy::wrong_self_convention)]
    fn from_flat(&mut self, flat: &[f64]) -> usize;
    fn n_params(&self) -> usize;
}

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

    #[allow(clippy::wrong_self_convention)]
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

impl LayerWeights for WindowLayer {
    fn to_flat(&self) -> Vec<f64> {
        Vec::new()
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_flat(&mut self, flat: &[f64]) -> usize {
        assert!(
            flat.is_empty() || {
                // We may be handed a tail that still has remaining params for the
                // next layer; only the prefix we consume (0 bytes) matters.
                true
            },
            "WindowLayer takes no weights"
        );
        0
    }

    fn n_params(&self) -> usize {
        0
    }
}

impl LayerWeights for TransformerLayer {
    fn n_params(&self) -> usize {
        4 * self.d_model * self.d_model
            + 2 * self.d_ffn * self.d_model
            + self.d_ffn
            + 9 * self.d_model
    }

    fn to_flat(&self) -> Vec<f64> {
        fn push_mat(out: &mut Vec<f64>, m: &[Vec<f64>]) {
            for row in m {
                out.extend_from_slice(row);
            }
        }
        let mut out = Vec::with_capacity(self.n_params());
        push_mat(&mut out, &self.w_q);
        out.extend_from_slice(&self.b_q);
        push_mat(&mut out, &self.w_k);
        out.extend_from_slice(&self.b_k);
        push_mat(&mut out, &self.w_v);
        out.extend_from_slice(&self.b_v);
        push_mat(&mut out, &self.w_o);
        out.extend_from_slice(&self.b_o);
        push_mat(&mut out, &self.w_ffn1);
        out.extend_from_slice(&self.b_ffn1);
        push_mat(&mut out, &self.w_ffn2);
        out.extend_from_slice(&self.b_ffn2);
        out.extend_from_slice(&self.ln1_gamma);
        out.extend_from_slice(&self.ln1_beta);
        out.extend_from_slice(&self.ln2_gamma);
        out.extend_from_slice(&self.ln2_beta);
        out
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_flat(&mut self, flat: &[f64]) -> usize {
        fn read_mat(flat: &[f64], idx: &mut usize, rows: usize, cols: usize) -> Vec<Vec<f64>> {
            let mut m = Vec::with_capacity(rows);
            for _ in 0..rows {
                m.push(flat[*idx..*idx + cols].to_vec());
                *idx += cols;
            }
            m
        }
        fn read_vec(flat: &[f64], idx: &mut usize, n: usize) -> Vec<f64> {
            let v = flat[*idx..*idx + n].to_vec();
            *idx += n;
            v
        }

        let d = self.d_model;
        let f = self.d_ffn;
        let mut idx = 0;

        self.w_q = read_mat(flat, &mut idx, d, d);
        self.b_q = read_vec(flat, &mut idx, d);
        self.w_k = read_mat(flat, &mut idx, d, d);
        self.b_k = read_vec(flat, &mut idx, d);
        self.w_v = read_mat(flat, &mut idx, d, d);
        self.b_v = read_vec(flat, &mut idx, d);
        self.w_o = read_mat(flat, &mut idx, d, d);
        self.b_o = read_vec(flat, &mut idx, d);
        self.w_ffn1 = read_mat(flat, &mut idx, f, d);
        self.b_ffn1 = read_vec(flat, &mut idx, f);
        self.w_ffn2 = read_mat(flat, &mut idx, d, f);
        self.b_ffn2 = read_vec(flat, &mut idx, d);
        self.ln1_gamma = read_vec(flat, &mut idx, d);
        self.ln1_beta = read_vec(flat, &mut idx, d);
        self.ln2_gamma = read_vec(flat, &mut idx, d);
        self.ln2_beta = read_vec(flat, &mut idx, d);

        self.rebuild_pe_offsets();
        idx
    }
}

impl LayerWeights for MambaLayer {
    fn n_params(&self) -> usize {
        self.input_size * (3 * self.d_state + 2 * self.dt_rank + 2)
    }

    fn to_flat(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n_params());
        // 1. x_proj_w row-major: (dt_rank + 2*d_state, input_size)
        for i in 0..self.x_proj_w.nrows() {
            for j in 0..self.x_proj_w.ncols() {
                out.push(self.x_proj_w[(i, j)]);
            }
        }
        // 2. dt_proj_w row-major: (input_size, dt_rank)
        for i in 0..self.dt_proj_w.nrows() {
            for j in 0..self.dt_proj_w.ncols() {
                out.push(self.dt_proj_w[(i, j)]);
            }
        }
        // 3. dt_proj_b: (input_size,)
        for i in 0..self.dt_proj_b.len() {
            out.push(self.dt_proj_b[i]);
        }
        // 4. a_log row-major: (input_size, d_state)
        for i in 0..self.a_log.nrows() {
            for j in 0..self.a_log.ncols() {
                out.push(self.a_log[(i, j)]);
            }
        }
        // 5. d_skip: (input_size,)
        for i in 0..self.d_skip.len() {
            out.push(self.d_skip[i]);
        }
        out
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_flat(&mut self, flat: &[f64]) -> usize {
        let mut cursor = 0;
        // 1. x_proj_w: (dt_rank + 2*d_state, input_size) row-major
        let rows = self.dt_rank + 2 * self.d_state;
        let cols = self.input_size;
        self.x_proj_w =
            nalgebra::DMatrix::from_row_slice(rows, cols, &flat[cursor..cursor + rows * cols]);
        cursor += rows * cols;
        // 2. dt_proj_w: (input_size, dt_rank) row-major
        self.dt_proj_w = nalgebra::DMatrix::from_row_slice(
            self.input_size,
            self.dt_rank,
            &flat[cursor..cursor + self.input_size * self.dt_rank],
        );
        cursor += self.input_size * self.dt_rank;
        // 3. dt_proj_b: (input_size,)
        self.dt_proj_b = nalgebra::DVector::from_row_slice(&flat[cursor..cursor + self.input_size]);
        cursor += self.input_size;
        // 4. a_log: (input_size, d_state) row-major
        self.a_log = nalgebra::DMatrix::from_row_slice(
            self.input_size,
            self.d_state,
            &flat[cursor..cursor + self.input_size * self.d_state],
        );
        cursor += self.input_size * self.d_state;
        // 5. d_skip: (input_size,)
        self.d_skip = nalgebra::DVector::from_row_slice(&flat[cursor..cursor + self.input_size]);
        cursor += self.input_size;
        cursor
    }
}

impl MambaLayer {
    /// Single-tick forward. Mutates `h` in place (state update), returns `y`.
    ///
    /// Shapes: `x: [f64; input_size]`, `h: DMatrix<f64> (input_size, d_state)`,
    /// returns `Vec<f64>` length `input_size`.
    ///
    /// Numerical contract: Python mirror (`rl/layers/mamba.py`) must produce
    /// bit-identical f64 output. Uses `softplus` and `expm1_over_x` helpers
    /// (free functions in this module).
    pub fn forward(&self, x: &[f64], h: &mut nalgebra::DMatrix<f64>) -> Vec<f64> {
        debug_assert_eq!(x.len(), self.input_size);
        debug_assert_eq!(h.nrows(), self.input_size);
        debug_assert_eq!(h.ncols(), self.d_state);

        let x_vec = nalgebra::DVector::from_row_slice(x);

        // 1. Fused x_proj: produces (dt_rank + 2*d_state,)
        let proj = &self.x_proj_w * &x_vec;
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

        // 3. ZOH discretization + state update, per (d, n).
        //    A = -exp(a_log), Ā = exp(Δ·A), B̄ = Δ·B · expm1_over_x(Δ·A)
        //    h_new = Ā*h + B̄*x[d]
        //    y[d]   = Σ_n h_new[d, n] * C[n] + D[d] * x[d]
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

impl LayerWeights for Layer {
    fn to_flat(&self) -> Vec<f64> {
        match self {
            Layer::Dense(d) => d.to_flat(),
            Layer::Gru(g) => g.to_flat(),
            Layer::Lstm(l) => l.to_flat(),
            Layer::Window(w) => w.to_flat(),
            Layer::Transformer(t) => t.to_flat(),
            Layer::Mamba(m) => m.to_flat(),
        }
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_flat(&mut self, flat: &[f64]) -> usize {
        match self {
            Layer::Dense(d) => d.from_flat(flat),
            Layer::Gru(g) => g.from_flat(flat),
            Layer::Lstm(l) => l.from_flat(flat),
            Layer::Window(w) => w.from_flat(flat),
            Layer::Transformer(t) => t.from_flat(flat),
            Layer::Mamba(m) => m.from_flat(flat),
        }
    }

    fn n_params(&self) -> usize {
        match self {
            Layer::Dense(d) => d.n_params(),
            Layer::Gru(g) => g.n_params(),
            Layer::Lstm(l) => l.n_params(),
            Layer::Window(w) => w.n_params(),
            Layer::Transformer(t) => t.n_params(),
            Layer::Mamba(m) => m.n_params(),
        }
    }
}

/// JSON file structure for neural network models (v1 schema).
/// v1 always loads with `OutputParam::Atan2Signed` (the bank-decoder
/// parameterization is a v2 feature; v1 files predate it). The legacy
/// `output_interpretation` field is silently ignored. Output_size is
/// validated to match the parameterization at load time.
#[derive(Debug, Clone, Deserialize)]
struct NnJsonFile {
    #[allow(dead_code)]
    format_version: u32,
    architecture: NnArchitecture,
    weights: std::collections::BTreeMap<String, NnLayerWeights>,
    #[serde(default)]
    input_mask: Option<Vec<usize>>,
    #[serde(default)]
    ablated_input: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct NnArchitecture {
    layers: Vec<usize>,
    activations: Vec<Activation>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct NnLayerWeights {
    // Dense fields
    #[serde(skip_serializing_if = "Option::is_none", default)]
    w: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    b: Option<Vec<f64>>,
    // GRU / LSTM fields
    #[serde(skip_serializing_if = "Option::is_none", default)]
    weight_ih: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    weight_hh: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    bias_ih: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    bias_hh: Option<Vec<f64>>,
    // Transformer attention projection fields
    #[serde(skip_serializing_if = "Option::is_none", default)]
    w_q: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    b_q: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    w_k: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    b_k: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    w_v: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    b_v: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    w_o: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    b_o: Option<Vec<f64>>,
    // Transformer FFN fields
    #[serde(skip_serializing_if = "Option::is_none", default)]
    w_ffn1: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    b_ffn1: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    w_ffn2: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    b_ffn2: Option<Vec<f64>>,
    // Transformer LayerNorm fields (ln1 / ln2)
    #[serde(skip_serializing_if = "Option::is_none", default)]
    ln1_gamma: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    ln1_beta: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    ln2_gamma: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    ln2_beta: Option<Vec<f64>>,
    // Mamba SSM fields (Phase 4a)
    #[serde(skip_serializing_if = "Option::is_none", default)]
    x_proj_w: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    dt_proj_w: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    dt_proj_b: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    a_log: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    d_skip: Option<Vec<f64>>,
}

/// v2 layer spec: tagged-union over the layer type.
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

fn default_scaled_pi_n() -> f64 {
    1.0
}
fn default_delta_max() -> f64 {
    0.35
}

/// JSON file structure for neural network models (v2 schema).
/// `output_param` selects the bank-angle decoder: `Atan2Signed` (default,
/// 2-output `atan2`) or `AcosTanh` (1-output `acos(tanh(x))`, magnitude_only
/// mode only). When absent in older v2 files, defaults to `Atan2Signed`
/// for backward compat. The legacy `output_interpretation` field is silently
/// ignored.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct NnJsonFileV2 {
    format_version: u32,
    architecture: Vec<LayerSpec>,
    weights: std::collections::BTreeMap<String, NnLayerWeights>,
    #[serde(default)]
    input_mask: Option<Vec<usize>>,
    #[serde(default)]
    ablated_input: Option<usize>,
    #[serde(default)]
    ablated_value: f64,
    #[serde(default)]
    output_param: OutputParam,
    #[serde(default = "default_scaled_pi_n")]
    scaled_pi_n: f64,
    #[serde(default = "default_delta_max")]
    delta_max: f64,
}

/// Total number of candidate NN inputs (16 baseline + 4 reference trajectory + 1 exit-bank teacher + 4 lateral-state telemetry
/// + 6 (sin,cos) bank-history pairs for exit teacher / prev commanded / prev realized + 1 periapsis_alt
/// + 3 live correction-DV components).
pub const NN_FULL_INPUT_SIZE: usize = 35;

/// Modular neural network model.
///
/// Replaces the fixed-size `NeuralNetParams`. Supports arbitrary depth and width.
#[derive(Debug, Clone)]
pub struct NeuralNetModel {
    /// Canonical v2-shaped architecture spec (one entry per layer).
    pub architecture: Vec<LayerSpec>,
    /// Layer sizes: [input_size, hidden1, ..., output_size].
    pub layer_sizes: Vec<usize>,
    /// Network layers (len = layer_sizes.len() - 1).
    pub layers: Vec<Layer>,
    /// Optional input selection mask: indices into the full 35-input vector.
    /// Length must equal layer_sizes[0]. None means use inputs as-is.
    pub input_mask: Option<Vec<usize>>,
    /// Optional index of a single input to freeze (ablation analysis).
    /// Must be in [0, NN_FULL_INPUT_SIZE). None means no ablation.
    /// When set, `build_nn_input` overwrites `full_input[ablated_input]` with
    /// `ablated_value` (default 0.0 => classic zero-ablation).
    pub ablated_input: Option<usize>,
    /// Value to freeze the ablated input to. Default 0.0 (zero-ablation).
    /// Used for flip-ablation: freeze a binary ±1 flag to -1 / +1 instead of
    /// an out-of-distribution 0.
    pub ablated_value: f64,
    /// Output parameterization for the bank-angle decoder.
    /// Default: `Atan2Signed` (2-output atan2, backward-compatible).
    pub output_param: OutputParam,
    /// Half-range multiplier for `ScaledPi`: `bank = scaled_pi_n * π * out[0]`.
    pub scaled_pi_n: f64,
    /// Per-step increment bound for `Delta`: `bank = prev_realized + delta_max * out[0]`.
    pub delta_max: f64,
}

impl NeuralNetModel {
    /// Validate that the input mask is consistent with the expected layer-0 size and NN_FULL_INPUT_SIZE.
    pub fn validate_mask(mask: &Option<Vec<usize>>, expected_len: usize) -> Result<(), DataError> {
        if let Some(m) = mask {
            if m.len() != expected_len {
                return Err(DataError(format!(
                    "input_mask length ({}) does not match layer_sizes[0] ({})",
                    m.len(),
                    expected_len
                )));
            }
            for &idx in m {
                if idx >= NN_FULL_INPUT_SIZE {
                    return Err(DataError(format!(
                        "input_mask index {} out of range [0, {})",
                        idx, NN_FULL_INPUT_SIZE
                    )));
                }
            }
            let mut seen = std::collections::HashSet::new();
            for &idx in m {
                if !seen.insert(idx) {
                    return Err(DataError(format!(
                        "input_mask contains duplicate index {}",
                        idx
                    )));
                }
            }
        }
        Ok(())
    }

    /// Validate that the network's final layer produces the right number of outputs
    /// for the given `output_param`:
    /// - `Atan2Signed`: requires output_size == 2 (bank = atan2(out[0], out[1]))
    /// - `AcosTanh`:    requires output_size == 1 (bank = acos(tanh(out[0])))
    /// - `ScaledPi`:    requires output_size == 1 (bank = scaled_pi_n * π * tanh(out[0]))
    /// - `Delta`:       requires output_size == 1 (bank = prev_realized + delta_max * tanh(out[0]))
    pub fn validate_output_size(
        output_size: usize,
        output_param: OutputParam,
        path: &str,
    ) -> Result<(), DataError> {
        let expected = match output_param {
            OutputParam::Atan2Signed => 2,
            OutputParam::AcosTanh | OutputParam::ScaledPi | OutputParam::Delta => 1,
        };
        if output_size != expected {
            return Err(DataError(format!(
                "network output_size must be {} for output_param {:?}, got {} in {}",
                expected, output_param, output_size, path
            )));
        }
        Ok(())
    }

    /// Validate that the last layer's activation matches the output_param
    /// constraint. `AcosTanh`, `ScaledPi`, and `Delta` require `Tanh` so that
    /// `output[0] ∈ [-1, 1]`. `Atan2Signed` has no constraint.
    /// Without this guard a hand-crafted (or trainer-bug-produced) v2 JSON with
    /// `output_param: "acos_tanh"` plus `linear`/`asinh`/`swish` last activation
    /// loads silently and emits NaN at runtime when |out[0]| > 1.
    pub fn validate_output_activation(
        last_activation: Activation,
        output_param: OutputParam,
        path: &str,
    ) -> Result<(), DataError> {
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
    }

    /// Validate that ablated_input is within [0, NN_FULL_INPUT_SIZE).
    pub fn validate_ablated_input(ablated: &Option<usize>) -> Result<(), DataError> {
        if let Some(idx) = ablated
            && *idx >= NN_FULL_INPUT_SIZE
        {
            return Err(DataError(format!(
                "ablated_input index {} out of range [0, {})",
                idx, NN_FULL_INPUT_SIZE
            )));
        }
        Ok(())
    }

    /// Load NN model from a JSON file.
    pub fn load(path: &str) -> Result<Self, DataError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| DataError(format!("Cannot read {}: {}", path, e)))?;
        Self::from_json_str(&content, path)
    }

    /// Load from a JSON string. Dispatches by `format_version` (1 or 2).
    pub fn from_json_str(content: &str, path: &str) -> Result<Self, DataError> {
        let v: serde_json::Value = serde_json::from_str(content)
            .map_err(|e| DataError(format!("JSON parse error in {}: {}", path, e)))?;
        let fmt = v
            .get("format_version")
            .and_then(|x| x.as_u64())
            .unwrap_or(0);
        match fmt {
            1 => Self::from_v1_json(content, path),
            2 => Self::from_v2_json(content, path),
            other => Err(DataError(format!(
                "Unsupported format_version {} in {} (expected 1 or 2)",
                other, path
            ))),
        }
    }

    /// Load v1 JSON schema (architecture object with layers + activations).
    fn from_v1_json(content: &str, path: &str) -> Result<Self, DataError> {
        let file: NnJsonFile = serde_json::from_str(content)
            .map_err(|e| DataError(format!("JSON parse error in {}: {}", path, e)))?;

        let n_layers = file.architecture.layers.len() - 1;
        if file.architecture.activations.len() != n_layers {
            return Err(DataError(format!(
                "Activation count ({}) != layer count ({}) in {}",
                file.architecture.activations.len(),
                n_layers,
                path
            )));
        }

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let key = format!("layer_{}", i);
            let lw = file
                .weights
                .get(&key)
                .ok_or_else(|| DataError(format!("Missing {} in weights in {}", key, path)))?;

            let n_out = file.architecture.layers[i + 1];
            let n_in = file.architecture.layers[i];

            let w =
                lw.w.as_ref()
                    .ok_or_else(|| DataError(format!("Layer {} missing w in {}", i, path)))?;
            let b =
                lw.b.as_ref()
                    .ok_or_else(|| DataError(format!("Layer {} missing b in {}", i, path)))?;

            if w.len() != n_out || b.len() != n_out {
                return Err(DataError(format!(
                    "Layer {} size mismatch: expected {}x{}, got w={}x?, b={} in {}",
                    i,
                    n_out,
                    n_in,
                    w.len(),
                    b.len(),
                    path
                )));
            }

            layers.push(Layer::Dense(DenseLayer {
                w: w.clone(),
                b: b.clone(),
                activation: file.architecture.activations[i],
            }));
        }

        Self::validate_mask(&file.input_mask, file.architecture.layers[0])?;
        Self::validate_ablated_input(&file.ablated_input)?;

        let output_size = *file.architecture.layers.last().unwrap_or(&0);
        Self::validate_output_size(output_size, OutputParam::default(), path)?;

        let activations = file.architecture.activations;
        let layer_sizes = file.architecture.layers;
        let architecture: Vec<LayerSpec> = (0..layers.len())
            .map(|i| LayerSpec::Dense {
                input_size: layer_sizes[i],
                output_size: layer_sizes[i + 1],
                activation: activations[i],
            })
            .collect();

        Ok(NeuralNetModel {
            architecture,
            layer_sizes,
            layers,
            input_mask: file.input_mask,
            ablated_input: file.ablated_input,
            // v1 schema has no ablated_value; classic zero-ablation.
            ablated_value: 0.0,
            output_param: OutputParam::default(),
            scaled_pi_n: default_scaled_pi_n(),
            delta_max: default_delta_max(),
        })
    }

    /// Load v2 JSON schema (architecture is a tagged-layer list).
    fn from_v2_json(content: &str, path: &str) -> Result<Self, DataError> {
        let file: NnJsonFileV2 = serde_json::from_str(content)
            .map_err(|e| DataError(format!("JSON parse error in {}: {}", path, e)))?;

        // Chain consistency: layer i's output must feed layer i+1's input.
        // Dense: output_size -> next.input_size; Gru/Lstm: hidden_size -> next.input_size;
        // Window: n_steps * input_size -> next.input_size (zero-param buffer flatten).
        for i in 0..file.architecture.len().saturating_sub(1) {
            let prev_out = match &file.architecture[i] {
                LayerSpec::Dense { output_size, .. } => *output_size,
                LayerSpec::Gru { hidden_size, .. } => *hidden_size,
                LayerSpec::Lstm { hidden_size, .. } => *hidden_size,
                LayerSpec::Window {
                    input_size,
                    n_steps,
                } => *input_size * *n_steps,
                LayerSpec::Transformer { d_model, .. } => *d_model,
                LayerSpec::Mamba { input_size, .. } => *input_size,
            };
            let (next_in, next_label) = match &file.architecture[i + 1] {
                LayerSpec::Dense { input_size, .. } => (*input_size, "dense"),
                LayerSpec::Gru { input_size, .. } => (*input_size, "gru"),
                LayerSpec::Lstm { input_size, .. } => (*input_size, "lstm"),
                LayerSpec::Window { input_size, .. } => (*input_size, "window"),
                LayerSpec::Transformer { d_model, .. } => (*d_model, "transformer"),
                LayerSpec::Mamba { input_size, .. } => (*input_size, "mamba"),
            };
            let prev_label = match &file.architecture[i] {
                LayerSpec::Dense { .. } => "dense",
                LayerSpec::Gru { .. } => "gru",
                LayerSpec::Lstm { .. } => "lstm",
                LayerSpec::Window { .. } => "window",
                LayerSpec::Transformer { .. } => "transformer",
                LayerSpec::Mamba { .. } => "mamba",
            };
            if prev_out != next_in {
                return Err(DataError(format!(
                    "architecture chain mismatch at layer {}->{} in {}: layer {} ({}) produces output={}, but layer {} ({}) expects input={}",
                    i,
                    i + 1,
                    path,
                    i,
                    prev_label,
                    prev_out,
                    i + 1,
                    next_label,
                    next_in
                )));
            }
        }

        let mut layers = Vec::with_capacity(file.architecture.len());
        let mut layer_sizes = Vec::with_capacity(file.architecture.len() + 1);

        for (i, spec) in file.architecture.iter().enumerate() {
            match spec {
                LayerSpec::Dense {
                    input_size,
                    output_size,
                    activation,
                } => {
                    if i == 0 {
                        layer_sizes.push(*input_size);
                    }
                    layer_sizes.push(*output_size);

                    let key = format!("layer_{}", i);
                    let lw = file.weights.get(&key).ok_or_else(|| {
                        DataError(format!("Missing {} in weights in {}", key, path))
                    })?;

                    let w = lw
                        .w
                        .as_ref()
                        .ok_or_else(|| DataError(format!("Layer {} missing w in {}", i, path)))?;
                    let b = lw
                        .b
                        .as_ref()
                        .ok_or_else(|| DataError(format!("Layer {} missing b in {}", i, path)))?;

                    if w.len() != *output_size || b.len() != *output_size {
                        return Err(DataError(format!(
                            "Layer {} size mismatch: expected {}x{}, got w={}x?, b={} in {}",
                            i,
                            output_size,
                            input_size,
                            w.len(),
                            b.len(),
                            path
                        )));
                    }
                    for (row_idx, row) in w.iter().enumerate() {
                        if row.len() != *input_size {
                            return Err(DataError(format!(
                                "Layer {} weight row {} length mismatch: expected {}, got {} in {}",
                                i,
                                row_idx,
                                input_size,
                                row.len(),
                                path
                            )));
                        }
                    }

                    layers.push(Layer::Dense(DenseLayer {
                        w: w.clone(),
                        b: b.clone(),
                        activation: *activation,
                    }));
                }
                LayerSpec::Gru {
                    input_size,
                    hidden_size,
                } => {
                    if i == 0 {
                        layer_sizes.push(*input_size);
                    }
                    layer_sizes.push(*hidden_size);
                    let three_h = 3 * hidden_size;

                    let key = format!("layer_{}", i);
                    let lw = file.weights.get(&key).ok_or_else(|| {
                        DataError(format!("Missing {} in weights in {}", key, path))
                    })?;

                    let w_ih = lw.weight_ih.as_ref().ok_or_else(|| {
                        DataError(format!("Layer {} (gru) missing weight_ih in {}", i, path))
                    })?;
                    let w_hh = lw.weight_hh.as_ref().ok_or_else(|| {
                        DataError(format!("Layer {} (gru) missing weight_hh in {}", i, path))
                    })?;
                    let b_ih = lw.bias_ih.as_ref().ok_or_else(|| {
                        DataError(format!("Layer {} (gru) missing bias_ih in {}", i, path))
                    })?;
                    let b_hh = lw.bias_hh.as_ref().ok_or_else(|| {
                        DataError(format!("Layer {} (gru) missing bias_hh in {}", i, path))
                    })?;

                    if w_ih.len() != three_h {
                        return Err(DataError(format!(
                            "Layer {} (gru) weight_ih must have {} rows, got {} in {}",
                            i,
                            three_h,
                            w_ih.len(),
                            path
                        )));
                    }
                    if w_hh.len() != three_h {
                        return Err(DataError(format!(
                            "Layer {} (gru) weight_hh must have {} rows, got {} in {}",
                            i,
                            three_h,
                            w_hh.len(),
                            path
                        )));
                    }
                    if b_ih.len() != three_h || b_hh.len() != three_h {
                        return Err(DataError(format!(
                            "Layer {} (gru) biases must each have {} elements in {} (got bias_ih={}, bias_hh={})",
                            i,
                            three_h,
                            path,
                            b_ih.len(),
                            b_hh.len()
                        )));
                    }
                    for (r, row) in w_ih.iter().enumerate() {
                        if row.len() != *input_size {
                            return Err(DataError(format!(
                                "Layer {} (gru) weight_ih row {} length: expected {}, got {} in {}",
                                i,
                                r,
                                input_size,
                                row.len(),
                                path
                            )));
                        }
                    }
                    for (r, row) in w_hh.iter().enumerate() {
                        if row.len() != *hidden_size {
                            return Err(DataError(format!(
                                "Layer {} (gru) weight_hh row {} length: expected {}, got {} in {}",
                                i,
                                r,
                                hidden_size,
                                row.len(),
                                path
                            )));
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
                LayerSpec::Lstm {
                    input_size,
                    hidden_size,
                } => {
                    if i == 0 {
                        layer_sizes.push(*input_size);
                    }
                    layer_sizes.push(*hidden_size);
                    let four_h = 4 * hidden_size;

                    let key = format!("layer_{}", i);
                    let lw = file.weights.get(&key).ok_or_else(|| {
                        DataError(format!("Missing {} in weights in {}", key, path))
                    })?;

                    let w_ih = lw.weight_ih.as_ref().ok_or_else(|| {
                        DataError(format!("Layer {} (lstm) missing weight_ih in {}", i, path))
                    })?;
                    let w_hh = lw.weight_hh.as_ref().ok_or_else(|| {
                        DataError(format!("Layer {} (lstm) missing weight_hh in {}", i, path))
                    })?;
                    let b_ih = lw.bias_ih.as_ref().ok_or_else(|| {
                        DataError(format!("Layer {} (lstm) missing bias_ih in {}", i, path))
                    })?;
                    let b_hh = lw.bias_hh.as_ref().ok_or_else(|| {
                        DataError(format!("Layer {} (lstm) missing bias_hh in {}", i, path))
                    })?;

                    if w_ih.len() != four_h {
                        return Err(DataError(format!(
                            "Layer {} (lstm) weight_ih must have {} rows, got {} in {}",
                            i,
                            four_h,
                            w_ih.len(),
                            path
                        )));
                    }
                    if w_hh.len() != four_h {
                        return Err(DataError(format!(
                            "Layer {} (lstm) weight_hh must have {} rows, got {} in {}",
                            i,
                            four_h,
                            w_hh.len(),
                            path
                        )));
                    }
                    if b_ih.len() != four_h || b_hh.len() != four_h {
                        return Err(DataError(format!(
                            "Layer {} (lstm) biases must each have {} elements in {} (got bias_ih={}, bias_hh={})",
                            i,
                            four_h,
                            path,
                            b_ih.len(),
                            b_hh.len()
                        )));
                    }
                    for (r, row) in w_ih.iter().enumerate() {
                        if row.len() != *input_size {
                            return Err(DataError(format!(
                                "Layer {} (lstm) weight_ih row {} length: expected {}, got {} in {}",
                                i,
                                r,
                                input_size,
                                row.len(),
                                path
                            )));
                        }
                    }
                    for (r, row) in w_hh.iter().enumerate() {
                        if row.len() != *hidden_size {
                            return Err(DataError(format!(
                                "Layer {} (lstm) weight_hh row {} length: expected {}, got {} in {}",
                                i,
                                r,
                                hidden_size,
                                row.len(),
                                path
                            )));
                        }
                    }

                    layers.push(Layer::Lstm(LstmLayer {
                        input_size: *input_size,
                        hidden_size: *hidden_size,
                        weight_ih: w_ih.clone(),
                        weight_hh: w_hh.clone(),
                        bias_ih: b_ih.clone(),
                        bias_hh: b_hh.clone(),
                    }));
                }
                LayerSpec::Window {
                    input_size,
                    n_steps,
                } => {
                    if *input_size == 0 || *n_steps == 0 {
                        return Err(DataError(format!(
                            "Layer {} (window) input_size and n_steps must be positive in {}",
                            i, path
                        )));
                    }
                    if i == 0 {
                        layer_sizes.push(*input_size);
                    }
                    // Window's output is n_steps * input_size (flattened buffer).
                    layer_sizes.push(*input_size * *n_steps);
                    // Window has zero trainable parameters, so we don't look up
                    // weights["layer_i"] here -- save_json skips the entry and
                    // any present one (from a hand-crafted JSON) is ignored.
                    layers.push(Layer::Window(WindowLayer {
                        input_size: *input_size,
                        n_steps: *n_steps,
                    }));
                }
                LayerSpec::Transformer {
                    d_model,
                    n_heads,
                    d_ffn,
                    n_seq,
                } => {
                    if *d_model == 0 || *n_heads == 0 || *d_ffn == 0 || *n_seq == 0 {
                        return Err(DataError(format!(
                            "Layer {} (transformer) all shape fields must be positive in {}",
                            i, path
                        )));
                    }
                    if d_model % n_heads != 0 {
                        return Err(DataError(format!(
                            "Layer {} (transformer) d_model={} not divisible by n_heads={} in {}",
                            i, d_model, n_heads, path
                        )));
                    }
                    let d_head = d_model / n_heads;

                    let key = format!("layer_{}", i);
                    let lw = file.weights.get(&key).ok_or_else(|| {
                        DataError(format!("Missing {} in weights in {}", key, path))
                    })?;

                    macro_rules! req_mat {
                        ($field:ident) => {
                            lw.$field.as_ref().ok_or_else(|| {
                                DataError(format!(
                                    "Layer {} (transformer) missing {} in {}",
                                    i,
                                    stringify!($field),
                                    path
                                ))
                            })?
                        };
                    }
                    macro_rules! req_vec {
                        ($field:ident) => {
                            lw.$field.as_ref().ok_or_else(|| {
                                DataError(format!(
                                    "Layer {} (transformer) missing {} in {}",
                                    i,
                                    stringify!($field),
                                    path
                                ))
                            })?
                        };
                    }

                    if i == 0 {
                        layer_sizes.push(*d_model);
                    }
                    layer_sizes.push(*d_model);

                    // Read all weight tensors before shape validation (macros borrow lw).
                    let w_q = req_mat!(w_q);
                    let b_q = req_vec!(b_q);
                    let w_k = req_mat!(w_k);
                    let b_k = req_vec!(b_k);
                    let w_v = req_mat!(w_v);
                    let b_v = req_vec!(b_v);
                    let w_o = req_mat!(w_o);
                    let b_o = req_vec!(b_o);
                    let w_ffn1 = req_mat!(w_ffn1);
                    let b_ffn1 = req_vec!(b_ffn1);
                    let w_ffn2 = req_mat!(w_ffn2);
                    let b_ffn2 = req_vec!(b_ffn2);
                    let ln1_gamma = req_vec!(ln1_gamma);
                    let ln1_beta = req_vec!(ln1_beta);
                    let ln2_gamma = req_vec!(ln2_gamma);
                    let ln2_beta = req_vec!(ln2_beta);

                    // Validate matrix shapes: (name, matrix, expected_rows, expected_cols).
                    for (name, m, exp_rows, exp_cols) in [
                        ("w_q", w_q, *d_model, *d_model),
                        ("w_k", w_k, *d_model, *d_model),
                        ("w_v", w_v, *d_model, *d_model),
                        ("w_o", w_o, *d_model, *d_model),
                        ("w_ffn1", w_ffn1, *d_ffn, *d_model),
                        ("w_ffn2", w_ffn2, *d_model, *d_ffn),
                    ] {
                        if m.len() != exp_rows {
                            return Err(DataError(format!(
                                "Layer {} (transformer) {} must have {} rows, got {} in {}",
                                i,
                                name,
                                exp_rows,
                                m.len(),
                                path
                            )));
                        }
                        for (r, row) in m.iter().enumerate() {
                            if row.len() != exp_cols {
                                return Err(DataError(format!(
                                    "Layer {} (transformer) {} row {} length: expected {}, got {} in {}",
                                    i,
                                    name,
                                    r,
                                    exp_cols,
                                    row.len(),
                                    path
                                )));
                            }
                        }
                    }
                    // Validate vector lengths: (name, vector, expected_length).
                    for (name, v, expected) in [
                        ("b_q", b_q, *d_model),
                        ("b_k", b_k, *d_model),
                        ("b_v", b_v, *d_model),
                        ("b_o", b_o, *d_model),
                        ("b_ffn1", b_ffn1, *d_ffn),
                        ("b_ffn2", b_ffn2, *d_model),
                        ("ln1_gamma", ln1_gamma, *d_model),
                        ("ln1_beta", ln1_beta, *d_model),
                        ("ln2_gamma", ln2_gamma, *d_model),
                        ("ln2_beta", ln2_beta, *d_model),
                    ] {
                        if v.len() != expected {
                            return Err(DataError(format!(
                                "Layer {} (transformer) {} length: expected {}, got {} in {}",
                                i,
                                name,
                                expected,
                                v.len(),
                                path
                            )));
                        }
                    }

                    let mut layer = TransformerLayer {
                        d_model: *d_model,
                        n_heads: *n_heads,
                        d_head,
                        d_ffn: *d_ffn,
                        n_seq: *n_seq,
                        w_q: w_q.clone(),
                        b_q: b_q.clone(),
                        w_k: w_k.clone(),
                        b_k: b_k.clone(),
                        w_v: w_v.clone(),
                        b_v: b_v.clone(),
                        w_o: w_o.clone(),
                        b_o: b_o.clone(),
                        w_ffn1: w_ffn1.clone(),
                        b_ffn1: b_ffn1.clone(),
                        w_ffn2: w_ffn2.clone(),
                        b_ffn2: b_ffn2.clone(),
                        ln1_gamma: ln1_gamma.clone(),
                        ln1_beta: ln1_beta.clone(),
                        ln2_gamma: ln2_gamma.clone(),
                        ln2_beta: ln2_beta.clone(),
                        k_pe_offsets: Vec::new(),
                        v_pe_offsets: Vec::new(),
                    };
                    layer.rebuild_pe_offsets();
                    layers.push(Layer::Transformer(Box::new(layer)));
                }
                LayerSpec::Mamba {
                    input_size,
                    d_state,
                    dt_rank,
                } => {
                    if *input_size == 0 || *d_state == 0 || *dt_rank == 0 {
                        return Err(DataError(format!(
                            "Layer {} (mamba) input_size, d_state, and dt_rank must be positive in {}",
                            i, path
                        )));
                    }
                    if *dt_rank > *input_size {
                        return Err(DataError(format!(
                            "Layer {} (mamba) dt_rank={} must not exceed input_size={} in {}",
                            i, dt_rank, input_size, path
                        )));
                    }

                    if i == 0 {
                        layer_sizes.push(*input_size);
                    }
                    layer_sizes.push(*input_size);

                    let key = format!("layer_{}", i);
                    let lw = file.weights.get(&key).ok_or_else(|| {
                        DataError(format!("Missing {} in weights in {}", key, path))
                    })?;

                    macro_rules! req_mamba_mat {
                        ($field:ident) => {
                            lw.$field.as_ref().ok_or_else(|| {
                                DataError(format!(
                                    "Layer {} (mamba) missing {} in {}",
                                    i,
                                    stringify!($field),
                                    path
                                ))
                            })?
                        };
                    }
                    macro_rules! req_mamba_vec {
                        ($field:ident) => {
                            lw.$field.as_ref().ok_or_else(|| {
                                DataError(format!(
                                    "Layer {} (mamba) missing {} in {}",
                                    i,
                                    stringify!($field),
                                    path
                                ))
                            })?
                        };
                    }

                    let x_proj_w = req_mamba_mat!(x_proj_w);
                    let dt_proj_w = req_mamba_mat!(dt_proj_w);
                    let dt_proj_b = req_mamba_vec!(dt_proj_b);
                    let a_log = req_mamba_mat!(a_log);
                    let d_skip = req_mamba_vec!(d_skip);

                    let rows_x = dt_rank + 2 * d_state;
                    // Shape validation for matrices.
                    for (name, m, exp_rows, exp_cols) in [
                        ("x_proj_w", x_proj_w, rows_x, *input_size),
                        ("dt_proj_w", dt_proj_w, *input_size, *dt_rank),
                        ("a_log", a_log, *input_size, *d_state),
                    ] {
                        if m.len() != exp_rows {
                            return Err(DataError(format!(
                                "Layer {} (mamba) {} must have {} rows, got {} in {}",
                                i,
                                name,
                                exp_rows,
                                m.len(),
                                path
                            )));
                        }
                        for (r, row) in m.iter().enumerate() {
                            if row.len() != exp_cols {
                                return Err(DataError(format!(
                                    "Layer {} (mamba) {} row {} length: expected {}, got {} in {}",
                                    i,
                                    name,
                                    r,
                                    exp_cols,
                                    row.len(),
                                    path
                                )));
                            }
                        }
                    }
                    // Shape validation for vectors.
                    for (name, v, expected) in [
                        ("dt_proj_b", dt_proj_b, *input_size),
                        ("d_skip", d_skip, *input_size),
                    ] {
                        if v.len() != expected {
                            return Err(DataError(format!(
                                "Layer {} (mamba) {} length: expected {}, got {} in {}",
                                i,
                                name,
                                expected,
                                v.len(),
                                path
                            )));
                        }
                    }

                    // Convert Vec<Vec<f64>> -> DMatrix (row-major).
                    let to_dmatrix = |rows_data: &Vec<Vec<f64>>,
                                      nr: usize,
                                      nc: usize|
                     -> nalgebra::DMatrix<f64> {
                        let flat: Vec<f64> =
                            rows_data.iter().flat_map(|r| r.iter().copied()).collect();
                        nalgebra::DMatrix::from_row_slice(nr, nc, &flat)
                    };

                    layers.push(Layer::Mamba(Box::new(MambaLayer {
                        input_size: *input_size,
                        d_state: *d_state,
                        dt_rank: *dt_rank,
                        x_proj_w: to_dmatrix(x_proj_w, rows_x, *input_size),
                        dt_proj_w: to_dmatrix(dt_proj_w, *input_size, *dt_rank),
                        dt_proj_b: nalgebra::DVector::from_vec(dt_proj_b.clone()),
                        a_log: to_dmatrix(a_log, *input_size, *d_state),
                        d_skip: nalgebra::DVector::from_vec(d_skip.clone()),
                    })));
                }
            }
        }

        Self::validate_mask(&file.input_mask, layer_sizes[0])?;
        Self::validate_ablated_input(&file.ablated_input)?;

        let output_size = *layer_sizes.last().unwrap_or(&0);
        Self::validate_output_size(output_size, file.output_param, path)?;
        let last_activation = match file.architecture.last() {
            Some(LayerSpec::Dense { activation, .. }) => *activation,
            // Non-dense final layer with AcosTanh would have failed
            // validate_output_size when output_param=AcosTanh expects
            // output_size=1 (only Dense exposes a configurable output_size+activation
            // pair); for Atan2Signed the activation is irrelevant so default is fine.
            _ => Activation::Tanh,
        };
        Self::validate_output_activation(last_activation, file.output_param, path)?;

        Ok(NeuralNetModel {
            architecture: file.architecture,
            layer_sizes,
            layers,
            input_mask: file.input_mask,
            ablated_input: file.ablated_input,
            ablated_value: file.ablated_value,
            output_param: file.output_param,
            scaled_pi_n: file.scaled_pi_n,
            delta_max: file.delta_max,
        })
    }

    /// Save to JSON format (v2 schema: tagged-layer list).
    pub fn save_json(&self, path: &str) -> Result<(), DataError> {
        let mut weights = std::collections::BTreeMap::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let entry = match layer {
                Layer::Dense(d) => NnLayerWeights {
                    w: Some(d.w.clone()),
                    b: Some(d.b.clone()),
                    ..NnLayerWeights::default()
                },
                Layer::Gru(g) => NnLayerWeights {
                    weight_ih: Some(g.weight_ih.clone()),
                    weight_hh: Some(g.weight_hh.clone()),
                    bias_ih: Some(g.bias_ih.clone()),
                    bias_hh: Some(g.bias_hh.clone()),
                    ..NnLayerWeights::default()
                },
                Layer::Lstm(l) => NnLayerWeights {
                    weight_ih: Some(l.weight_ih.clone()),
                    weight_hh: Some(l.weight_hh.clone()),
                    bias_ih: Some(l.bias_ih.clone()),
                    bias_hh: Some(l.bias_hh.clone()),
                    ..NnLayerWeights::default()
                },
                // Window is zero-param; skip the weights entry entirely.
                Layer::Window(_) => continue,
                Layer::Transformer(t) => NnLayerWeights {
                    w_q: Some(t.w_q.clone()),
                    b_q: Some(t.b_q.clone()),
                    w_k: Some(t.w_k.clone()),
                    b_k: Some(t.b_k.clone()),
                    w_v: Some(t.w_v.clone()),
                    b_v: Some(t.b_v.clone()),
                    w_o: Some(t.w_o.clone()),
                    b_o: Some(t.b_o.clone()),
                    w_ffn1: Some(t.w_ffn1.clone()),
                    b_ffn1: Some(t.b_ffn1.clone()),
                    w_ffn2: Some(t.w_ffn2.clone()),
                    b_ffn2: Some(t.b_ffn2.clone()),
                    ln1_gamma: Some(t.ln1_gamma.clone()),
                    ln1_beta: Some(t.ln1_beta.clone()),
                    ln2_gamma: Some(t.ln2_gamma.clone()),
                    ln2_beta: Some(t.ln2_beta.clone()),
                    ..NnLayerWeights::default()
                },
                Layer::Mamba(m) => {
                    let dmatrix_rows = |mat: &nalgebra::DMatrix<f64>| -> Vec<Vec<f64>> {
                        (0..mat.nrows())
                            .map(|r| (0..mat.ncols()).map(|c| mat[(r, c)]).collect())
                            .collect()
                    };
                    NnLayerWeights {
                        x_proj_w: Some(dmatrix_rows(&m.x_proj_w)),
                        dt_proj_w: Some(dmatrix_rows(&m.dt_proj_w)),
                        dt_proj_b: Some(m.dt_proj_b.iter().copied().collect()),
                        a_log: Some(dmatrix_rows(&m.a_log)),
                        d_skip: Some(m.d_skip.iter().copied().collect()),
                        ..NnLayerWeights::default()
                    }
                }
            };
            weights.insert(format!("layer_{}", i), entry);
        }

        let file = NnJsonFileV2 {
            format_version: 2,
            architecture: self.architecture.clone(),
            weights,
            input_mask: self.input_mask.clone(),
            ablated_input: self.ablated_input,
            ablated_value: self.ablated_value,
            output_param: self.output_param,
            scaled_pi_n: self.scaled_pi_n,
            delta_max: self.delta_max,
        };

        let json = serde_json::to_string_pretty(&file)
            .map_err(|e| DataError(format!("JSON serialize error: {}", e)))?;
        std::fs::write(path, json)
            .map_err(|e| DataError(format!("Cannot write {}: {}", path, e)))?;

        Ok(())
    }

    /// Generic forward pass through all layers.
    ///
    /// Takes `&mut NnState` so stateful layers (Phase 1+: GRU/LSTM/Window/SSM) can mutate
    /// their per-sim hidden state. Phase 0 dense layers ignore the state slot.
    pub fn forward(&self, state: &mut NnState, input: &[f64]) -> Vec<f64> {
        assert_eq!(
            input.len(),
            self.layer_sizes[0],
            "NN input length ({}) does not match expected input size ({})",
            input.len(),
            self.layer_sizes[0],
        );
        assert_eq!(
            state.layer_states.len(),
            self.layers.len(),
            "NnState layer count ({}) does not match model layer count ({})",
            state.layer_states.len(),
            self.layers.len(),
        );
        let mut current = input.to_vec();
        for (layer, layer_state) in self.layers.iter().zip(state.layer_states.iter_mut()) {
            // Matches (Layer, LayerState) pairs. Construction invariant from
            // NnState::for_model: Dense pairs with None, Gru pairs with Gru{h}.
            // The catch-all below catches mismatches caused by future refactors
            // that accidentally break the invariant.
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
                (Layer::Lstm(l), LayerState::Lstm { h, c }) => {
                    let (h_new, c_new) = l.forward(h, c, &current);
                    *h = h_new.clone();
                    *c = c_new;
                    current = h_new;
                }
                (Layer::Window(w), LayerState::Window { buffer }) => {
                    current = w.forward(&current, buffer);
                }
                (Layer::Transformer(t), LayerState::Transformer { k_cache, v_cache }) => {
                    current = t.forward(&current, k_cache, v_cache);
                }
                (Layer::Mamba(m), LayerState::Mamba { h }) => {
                    current = m.forward(&current, h);
                }
                _ => unreachable!(
                    "layer/state variant mismatch (construction invariant -- LayerState::for_layer maps Layer::Dense -> None, Layer::Gru -> Gru, Layer::Lstm -> Lstm, Layer::Window -> Window, Layer::Transformer -> Transformer)"
                ),
            }
        }
        current
    }

    /// Total number of parameters (weights + biases).
    pub fn n_params(&self) -> usize {
        self.layers.iter().map(|l| l.n_params()).sum()
    }

    /// Flatten all weights and biases into a single vector.
    ///
    /// Order: for each layer, all weights (row-major) then all biases.
    pub fn to_flat_weights(&self) -> Vec<f64> {
        let mut flat = Vec::with_capacity(self.n_params());
        for layer in &self.layers {
            flat.extend(layer.to_flat());
        }
        flat
    }

    /// Reconstruct a model from a flat weight vector and architecture spec.
    pub fn from_flat_weights(
        weights: &[f64],
        layer_sizes: &[usize],
        activations: &[Activation],
    ) -> Result<Self, DataError> {
        if activations.len() != layer_sizes.len() - 1 {
            return Err(DataError("Activation count != layer count - 1".to_string()));
        }
        let mut architecture = Vec::with_capacity(activations.len());
        let mut layers = Vec::with_capacity(activations.len());
        let mut offset = 0;
        for i in 0..activations.len() {
            let n_in = layer_sizes[i];
            let n_out = layer_sizes[i + 1];
            architecture.push(LayerSpec::Dense {
                input_size: n_in,
                output_size: n_out,
                activation: activations[i],
            });
            let mut layer = Layer::Dense(DenseLayer {
                w: vec![vec![0.0; n_in]; n_out],
                b: vec![0.0; n_out],
                activation: activations[i],
            });
            let needed = layer.n_params();
            if offset + needed > weights.len() {
                return Err(DataError(format!(
                    "Weight vector length mismatch: consumed {} of {}",
                    offset + needed,
                    weights.len()
                )));
            }
            let consumed = layer.from_flat(&weights[offset..]);
            offset += consumed;
            layers.push(layer);
        }
        if offset != weights.len() {
            return Err(DataError(format!(
                "Weight vector length mismatch: consumed {} of {}",
                offset,
                weights.len()
            )));
        }
        Ok(NeuralNetModel {
            architecture,
            layer_sizes: layer_sizes.to_vec(),
            layers,
            input_mask: None,
            ablated_input: None,
            ablated_value: 0.0,
            output_param: OutputParam::default(),
            scaled_pi_n: default_scaled_pi_n(),
            delta_max: default_delta_max(),
        })
    }

    /// Construct a NeuralNetModel from a flat weight vector and v2 architecture spec.
    /// Used by the PyO3 flat_weights_to_json helper (Task 7) that routes PSO output
    /// through Rust. Unlike `from_flat_weights` (the v1 wrapper), this accepts
    /// heterogeneous architectures via `LayerSpec`.
    pub fn from_flat_weights_v2(
        flat: &[f64],
        architecture: &[LayerSpec],
        input_mask: Option<Vec<usize>>,
        output_param: OutputParam,
        scaled_pi_n: f64,
        delta_max: f64,
    ) -> Result<Self, DataError> {
        if architecture.is_empty() {
            return Err(DataError(
                "from_flat_weights_v2: empty architecture".to_string(),
            ));
        }
        let mut layers: Vec<Layer> = Vec::with_capacity(architecture.len());
        let mut layer_sizes: Vec<usize> = Vec::with_capacity(architecture.len() + 1);
        let mut offset: usize = 0;

        for (i, spec) in architecture.iter().enumerate() {
            let mut layer = match spec {
                LayerSpec::Dense {
                    input_size,
                    output_size,
                    activation,
                } => {
                    if i == 0 {
                        layer_sizes.push(*input_size);
                    }
                    layer_sizes.push(*output_size);
                    Layer::Dense(DenseLayer {
                        w: vec![vec![0.0; *input_size]; *output_size],
                        b: vec![0.0; *output_size],
                        activation: *activation,
                    })
                }
                LayerSpec::Gru {
                    input_size,
                    hidden_size,
                } => {
                    if i == 0 {
                        layer_sizes.push(*input_size);
                    }
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
                LayerSpec::Lstm {
                    input_size,
                    hidden_size,
                } => {
                    if i == 0 {
                        layer_sizes.push(*input_size);
                    }
                    layer_sizes.push(*hidden_size);
                    let four_h = 4 * hidden_size;
                    Layer::Lstm(LstmLayer {
                        input_size: *input_size,
                        hidden_size: *hidden_size,
                        weight_ih: vec![vec![0.0; *input_size]; four_h],
                        weight_hh: vec![vec![0.0; *hidden_size]; four_h],
                        bias_ih: vec![0.0; four_h],
                        bias_hh: vec![0.0; four_h],
                    })
                }
                LayerSpec::Window {
                    input_size,
                    n_steps,
                } => {
                    if *input_size == 0 || *n_steps == 0 {
                        return Err(DataError(format!(
                            "from_flat_weights_v2: Window layer {} input_size and n_steps must be positive",
                            i
                        )));
                    }
                    if i == 0 {
                        layer_sizes.push(*input_size);
                    }
                    layer_sizes.push(*input_size * *n_steps);
                    Layer::Window(WindowLayer {
                        input_size: *input_size,
                        n_steps: *n_steps,
                    })
                }
                LayerSpec::Transformer {
                    d_model,
                    n_heads,
                    d_ffn,
                    n_seq,
                } => {
                    if *n_heads == 0 || *d_model % *n_heads != 0 {
                        return Err(DataError(format!(
                            "from_flat_weights_v2: Transformer layer {} d_model={} not divisible by n_heads={}",
                            i, d_model, n_heads
                        )));
                    }
                    let d_head = d_model / n_heads;
                    let f = *d_ffn;
                    let d = *d_model;
                    if i == 0 {
                        layer_sizes.push(d);
                    }
                    layer_sizes.push(d);
                    Layer::Transformer(Box::new(TransformerLayer {
                        d_model: d,
                        n_heads: *n_heads,
                        d_head,
                        d_ffn: f,
                        n_seq: *n_seq,
                        w_q: vec![vec![0.0; d]; d],
                        b_q: vec![0.0; d],
                        w_k: vec![vec![0.0; d]; d],
                        b_k: vec![0.0; d],
                        w_v: vec![vec![0.0; d]; d],
                        b_v: vec![0.0; d],
                        w_o: vec![vec![0.0; d]; d],
                        b_o: vec![0.0; d],
                        w_ffn1: vec![vec![0.0; d]; f],
                        b_ffn1: vec![0.0; f],
                        w_ffn2: vec![vec![0.0; f]; d],
                        b_ffn2: vec![0.0; d],
                        ln1_gamma: vec![1.0; d],
                        ln1_beta: vec![0.0; d],
                        ln2_gamma: vec![1.0; d],
                        ln2_beta: vec![0.0; d],
                        k_pe_offsets: Vec::new(),
                        v_pe_offsets: Vec::new(),
                    }))
                }
                LayerSpec::Mamba {
                    input_size,
                    d_state,
                    dt_rank,
                } => {
                    if *dt_rank == 0 || *dt_rank > *input_size {
                        return Err(DataError(format!(
                            "from_flat_weights_v2: Mamba layer {} dt_rank={} invalid for input_size={}",
                            i, dt_rank, input_size
                        )));
                    }
                    if *d_state == 0 || *input_size == 0 {
                        return Err(DataError(format!(
                            "from_flat_weights_v2: Mamba layer {} input_size and d_state must be positive",
                            i
                        )));
                    }
                    if i == 0 {
                        layer_sizes.push(*input_size);
                    }
                    layer_sizes.push(*input_size);
                    let rows_x = dt_rank + 2 * d_state;
                    Layer::Mamba(Box::new(MambaLayer {
                        input_size: *input_size,
                        d_state: *d_state,
                        dt_rank: *dt_rank,
                        x_proj_w: nalgebra::DMatrix::<f64>::zeros(rows_x, *input_size),
                        dt_proj_w: nalgebra::DMatrix::<f64>::zeros(*input_size, *dt_rank),
                        dt_proj_b: nalgebra::DVector::<f64>::zeros(*input_size),
                        a_log: nalgebra::DMatrix::<f64>::zeros(*input_size, *d_state),
                        d_skip: nalgebra::DVector::<f64>::zeros(*input_size),
                    }))
                }
            };
            let needed = layer.n_params();
            if offset + needed > flat.len() {
                return Err(DataError(format!(
                    "from_flat_weights_v2: layer {} needs {} params but only {} remaining (total flat len {})",
                    i,
                    needed,
                    flat.len() - offset,
                    flat.len()
                )));
            }
            let consumed = layer.from_flat(&flat[offset..]);
            offset += consumed;
            layers.push(layer);
        }

        if offset != flat.len() {
            return Err(DataError(format!(
                "from_flat_weights_v2: weight vector length mismatch, consumed {} of {}",
                offset,
                flat.len()
            )));
        }

        Self::validate_mask(&input_mask, layer_sizes[0])?;

        let output_size = *layer_sizes.last().unwrap();
        Self::validate_output_size(output_size, output_param, "<flat_weights_v2>")?;
        let last_activation = match architecture.last() {
            Some(LayerSpec::Dense { activation, .. }) => *activation,
            _ => Activation::Tanh,
        };
        Self::validate_output_activation(last_activation, output_param, "<flat_weights_v2>")?;

        Ok(NeuralNetModel {
            architecture: architecture.to_vec(),
            layer_sizes,
            layers,
            input_mask,
            ablated_input: None,
            ablated_value: 0.0,
            output_param,
            scaled_pi_n,
            delta_max,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid NeuralNetModel with a given input size.
    fn make_model(input_size: usize) -> NeuralNetModel {
        NeuralNetModel {
            architecture: vec![
                LayerSpec::Dense {
                    input_size,
                    output_size: 4,
                    activation: Activation::Tanh,
                },
                LayerSpec::Dense {
                    input_size: 4,
                    output_size: 2,
                    activation: Activation::Linear,
                },
            ],
            layer_sizes: vec![input_size, 4, 2],
            layers: vec![
                Layer::Dense(DenseLayer {
                    w: vec![vec![0.1; input_size]; 4],
                    b: vec![0.0; 4],
                    activation: Activation::Tanh,
                }),
                Layer::Dense(DenseLayer {
                    w: vec![vec![0.1; 4]; 2],
                    b: vec![0.0; 2],
                    activation: Activation::Linear,
                }),
            ],
            input_mask: None,
            ablated_input: None,
            ablated_value: 0.0,
            output_param: OutputParam::default(),
            scaled_pi_n: default_scaled_pi_n(),
            delta_max: default_delta_max(),
        }
    }

    #[test]
    fn input_mask_stored_on_model() {
        let mask = Some(vec![0usize, 1, 2]);
        let model = NeuralNetModel {
            input_mask: mask.clone(),
            ..make_model(3)
        };
        assert_eq!(model.input_mask, mask);
    }

    #[test]
    fn input_mask_none_by_default() {
        let model = make_model(3);
        assert!(model.input_mask.is_none());
    }

    #[test]
    fn validate_mask_length_mismatch() {
        // mask has 2 entries but expected_len is 3
        let mask = Some(vec![0usize, 1]);
        let result = NeuralNetModel::validate_mask(&mask, 3);
        assert!(result.is_err());
        assert!(result.unwrap_err().0.contains("length"));
    }

    #[test]
    fn validate_mask_out_of_range() {
        // index == NN_FULL_INPUT_SIZE is out of range
        let mask = Some(vec![0usize, NN_FULL_INPUT_SIZE]);
        let result = NeuralNetModel::validate_mask(&mask, 2);
        assert!(result.is_err());
        assert!(result.unwrap_err().0.contains("out of range"));
    }

    #[test]
    fn validate_mask_duplicates() {
        let mask = Some(vec![0usize, 1, 0]);
        let result = NeuralNetModel::validate_mask(&mask, 3);
        assert!(result.is_err());
        assert!(result.unwrap_err().0.contains("duplicate"));
    }

    #[test]
    fn validate_mask_valid() {
        let mask = Some(vec![0usize, 5, 10]);
        let result = NeuralNetModel::validate_mask(&mask, 3);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_mask_none_is_ok() {
        let result = NeuralNetModel::validate_mask(&None, 16);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_ablated_input_out_of_range() {
        let result = NeuralNetModel::validate_ablated_input(&Some(NN_FULL_INPUT_SIZE));
        assert!(result.is_err());
        assert!(result.unwrap_err().0.contains("out of range"));
    }

    #[test]
    fn validate_ablated_input_valid() {
        // index 34 is the last valid index (NN_FULL_INPUT_SIZE - 1)
        let result = NeuralNetModel::validate_ablated_input(&Some(34));
        assert!(result.is_ok());
    }

    #[test]
    fn flat_weights_roundtrip_dense() {
        use crate::data::nn_state::NnState;

        let original = NeuralNetModel {
            architecture: vec![
                LayerSpec::Dense {
                    input_size: 4,
                    output_size: 3,
                    activation: Activation::Tanh,
                },
                LayerSpec::Dense {
                    input_size: 3,
                    output_size: 2,
                    activation: Activation::Linear,
                },
            ],
            layer_sizes: vec![4, 3, 2],
            layers: vec![
                Layer::Dense(DenseLayer {
                    w: vec![
                        vec![0.1, 0.2, 0.3, 0.4],
                        vec![0.5, 0.6, 0.7, 0.8],
                        vec![-0.1, -0.2, -0.3, -0.4],
                    ],
                    b: vec![0.01, 0.02, 0.03],
                    activation: Activation::Tanh,
                }),
                Layer::Dense(DenseLayer {
                    w: vec![vec![0.1, 0.2, 0.3], vec![-0.1, -0.2, -0.3]],
                    b: vec![0.1, -0.1],
                    activation: Activation::Linear,
                }),
            ],
            input_mask: None,
            ablated_input: None,
            ablated_value: 0.0,
            output_param: OutputParam::default(),
            scaled_pi_n: default_scaled_pi_n(),
            delta_max: default_delta_max(),
        };

        let flat = original.to_flat_weights();
        assert_eq!(flat.len(), original.n_params());
        let layer_sizes: Vec<usize> = original.layer_sizes.clone();
        let activations = vec![Activation::Tanh, Activation::Linear];
        let reconstructed =
            NeuralNetModel::from_flat_weights(&flat, &layer_sizes, &activations).unwrap();
        assert_eq!(reconstructed.n_params(), original.n_params());

        let input = vec![0.5, -0.3, 0.1, 0.7];
        let mut s0 = NnState::for_model(&original);
        let mut s1 = NnState::for_model(&reconstructed);
        let o0 = original.forward(&mut s0, &input);
        let o1 = reconstructed.forward(&mut s1, &input);
        assert_eq!(o0, o1);
    }

    #[test]
    fn gru_forward_known_output() {
        // Minimal 2-input, 2-hidden GRU with all-zero weights + biases.
        // r = sigmoid(0) = 0.5, z = sigmoid(0) = 0.5, n = tanh(0 + 0.5 * 0) = 0.
        // h_new[i] = (1 - 0.5) * 0 + 0.5 * h_prev[i] = 0.5 * h_prev[i].
        let gru = GruLayer {
            input_size: 2,
            hidden_size: 2,
            weight_ih: vec![vec![0.0, 0.0]; 6], // 3H=6 rows, 2 cols
            weight_hh: vec![vec![0.0, 0.0]; 6], // 3H=6 rows, 2 cols
            bias_ih: vec![0.0; 6],
            bias_hh: vec![0.0; 6],
        };
        let h_prev = vec![1.0, 2.0];
        let x = vec![0.5, -0.5];
        let h_new = gru.forward(&h_prev, &x);
        assert!((h_new[0] - 0.5).abs() < 1e-12);
        assert!((h_new[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn v2_json_parses_to_same_layers_as_v1() {
        let v1 = r#"{
          "format_version": 1,
          "architecture": { "layers": [3, 2], "activations": ["linear"] },
          "weights": { "layer_0": { "w": [[0.1,0.2,0.3],[0.4,0.5,0.6]], "b": [0.01,0.02] } },
          "output_interpretation": "atan2"
        }"#;
        let v2 = r#"{
          "format_version": 2,
          "architecture": [
            { "type": "dense", "input_size": 3, "output_size": 2, "activation": "linear" }
          ],
          "weights": { "layer_0": { "w": [[0.1,0.2,0.3],[0.4,0.5,0.6]], "b": [0.01,0.02] } },
          "output_interpretation": "atan2"
        }"#;
        let m1 = NeuralNetModel::from_json_str(v1, "v1").unwrap();
        let m2 = NeuralNetModel::from_json_str(v2, "v2").unwrap();
        assert_eq!(m1.layer_sizes, m2.layer_sizes);
        assert_eq!(m1.n_params(), m2.n_params());
        let input = vec![1.0, 2.0, 3.0];
        let mut s1 = NnState::for_model(&m1);
        let mut s2 = NnState::for_model(&m2);
        let o1 = m1.forward(&mut s1, &input);
        let o2 = m2.forward(&mut s2, &input);
        assert_eq!(o1, o2);
    }

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
            w_ih.push(
                (0..input_size)
                    .map(|k| (i * 10 + k) as f64 * 0.01)
                    .collect(),
            );
            w_hh.push(
                (0..hidden_size)
                    .map(|k| (i * 10 + k) as f64 * 0.001)
                    .collect(),
            );
        }
        let b_ih: Vec<f64> = (0..three_h).map(|i| 0.1 + i as f64 * 0.01).collect();
        let b_hh: Vec<f64> = (0..three_h).map(|i| 0.2 + i as f64 * 0.01).collect();

        let original = GruLayer {
            input_size,
            hidden_size,
            weight_ih: w_ih,
            weight_hh: w_hh,
            bias_ih: b_ih,
            bias_hh: b_hh,
        };

        let flat = original.to_flat();
        assert_eq!(flat.len(), original.n_params());

        // Reconstruct an empty-shaped GruLayer and fill via from_flat.
        let mut twin = GruLayer {
            input_size,
            hidden_size,
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

    #[test]
    fn lstm_flat_weights_roundtrip() {
        let original = LstmLayer {
            input_size: 3,
            hidden_size: 2,
            weight_ih: (0..8)
                .map(|i| (0..3).map(|j| (i * 3 + j) as f64 * 0.01).collect())
                .collect(),
            weight_hh: (0..8)
                .map(|i| (0..2).map(|j| 100.0 + (i * 2 + j) as f64 * 0.01).collect())
                .collect(),
            bias_ih: (0..8).map(|i| 200.0 + i as f64).collect(),
            bias_hh: (0..8).map(|i| 300.0 + i as f64).collect(),
        };

        let flat = original.to_flat();
        assert_eq!(flat.len(), 56); // 4H*I + 4H*H + 2*4H = 24 + 16 + 16
        assert_eq!(flat.len(), original.n_params());

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

        assert_eq!(reconstructed.weight_ih, original.weight_ih);
        assert_eq!(reconstructed.weight_hh, original.weight_hh);
        assert_eq!(reconstructed.bias_ih, original.bias_ih);
        assert_eq!(reconstructed.bias_hh, original.bias_hh);
    }

    #[test]
    fn v2_gru_json_roundtrip() {
        // Use hidden_size=2 so the GRU's output is a valid network output
        // (atan2 requires the final layer to produce exactly 2 values).
        let input_size = 2;
        let hidden_size = 2;
        let three_h = 6;
        let gru = GruLayer {
            input_size,
            hidden_size,
            weight_ih: (0..three_h)
                .map(|i| (0..input_size).map(|k| (i + k) as f64 * 0.01).collect())
                .collect(),
            weight_hh: (0..three_h)
                .map(|i| (0..hidden_size).map(|k| (i + k) as f64 * 0.02).collect())
                .collect(),
            bias_ih: (0..three_h).map(|i| 0.1 + i as f64 * 0.01).collect(),
            bias_hh: (0..three_h).map(|i| 0.2 + i as f64 * 0.01).collect(),
        };
        let original = NeuralNetModel {
            architecture: vec![LayerSpec::Gru {
                input_size,
                hidden_size,
            }],
            layer_sizes: vec![input_size, hidden_size],
            layers: vec![Layer::Gru(gru)],
            input_mask: None,
            ablated_input: None,
            ablated_value: 0.0,
            output_param: OutputParam::default(),
            scaled_pi_n: default_scaled_pi_n(),
            delta_max: default_delta_max(),
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

    #[test]
    fn from_flat_weights_v2_mixed_arch() {
        use crate::data::nn_state::NnState;

        // Dense(3->4,tanh) + Gru(4->4) + Dense(4->2,linear)
        let architecture = vec![
            LayerSpec::Dense {
                input_size: 3,
                output_size: 4,
                activation: Activation::Tanh,
            },
            LayerSpec::Gru {
                input_size: 4,
                hidden_size: 4,
            },
            LayerSpec::Dense {
                input_size: 4,
                output_size: 2,
                activation: Activation::Linear,
            },
        ];
        // Per-layer n_params:
        //   Dense 3->4: 3*4 + 4 = 16
        //   Gru H=4, I=4: 3*4*4 + 3*4*4 + 2*3*4 = 48 + 48 + 24 = 120
        //   Dense 4->2: 4*2 + 2 = 10
        // Total: 146
        let flat: Vec<f64> = (0..146).map(|i| 0.001 * i as f64).collect();
        let model = NeuralNetModel::from_flat_weights_v2(
            &flat,
            &architecture,
            None,
            OutputParam::default(),
            default_scaled_pi_n(),
            default_delta_max(),
        )
        .unwrap();
        assert_eq!(model.layers.len(), 3);
        assert_eq!(model.layer_sizes, vec![3, 4, 4, 2]);

        // Forward pass produces finite output.
        let mut state = NnState::for_model(&model);
        let out = model.forward(&mut state, &[0.1, 0.2, 0.3]);
        assert_eq!(out.len(), 2);
        for v in out.iter() {
            assert!(v.is_finite());
        }

        // JSON save/load roundtrip for the mixed Dense+Gru+Dense arch.
        // Catches copy-paste swaps between Dense and Gru serialization arms
        // that would survive the forward-is-finite check but diverge here.
        let tmpdir = std::env::temp_dir();
        let path = tmpdir.join("v2_mixed_arch_roundtrip.json");
        model.save_json(path.to_str().unwrap()).unwrap();
        let reloaded = NeuralNetModel::load(path.to_str().unwrap()).unwrap();
        let mut state2 = NnState::for_model(&reloaded);
        let out_reloaded = reloaded.forward(&mut state2, &[0.1, 0.2, 0.3]);
        for (a, b) in out.iter().zip(out_reloaded.iter()) {
            assert!(
                (a - b).abs() < 1e-15,
                "mixed-arch JSON roundtrip: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn from_v2_json_chain_mismatch_raises() {
        // Dense(23->32) -> Dense(16->2) -- second layer expects 16, first produces 32.
        let bad = r#"{
            "format_version": 2,
            "architecture": [
                {"type": "dense", "input_size": 23, "output_size": 32, "activation": "tanh"},
                {"type": "dense", "input_size": 16, "output_size": 2, "activation": "linear"}
            ],
            "weights": {
                "layer_0": {"w": [], "b": []},
                "layer_1": {"w": [], "b": []}
            },
            "output_interpretation": "atan2"
        }"#;
        let err = NeuralNetModel::from_v2_json(bad, "<test>");
        assert!(err.is_err(), "expected chain-mismatch error");
        let msg = err.err().unwrap().0;
        assert!(
            msg.contains("chain mismatch"),
            "error message should mention chain mismatch, got: {}",
            msg
        );
        assert!(
            msg.contains("output=32") && msg.contains("input=16"),
            "error message should quote the mismatched sizes, got: {}",
            msg
        );
    }

    #[test]
    fn from_flat_weights_v2_length_mismatch() {
        let architecture = vec![LayerSpec::Dense {
            input_size: 3,
            output_size: 4,
            activation: Activation::Tanh,
        }];
        // Dense 3->4 needs 16 params. Too short should Err.
        let flat = vec![0.0; 10];
        let err = NeuralNetModel::from_flat_weights_v2(
            &flat,
            &architecture,
            None,
            OutputParam::default(),
            default_scaled_pi_n(),
            default_delta_max(),
        );
        assert!(err.is_err());
        // Too long should also Err.
        let flat = vec![0.0; 20];
        let err = NeuralNetModel::from_flat_weights_v2(
            &flat,
            &architecture,
            None,
            OutputParam::default(),
            default_scaled_pi_n(),
            default_delta_max(),
        );
        assert!(err.is_err());
    }

    #[test]
    fn from_flat_weights_v2_carries_scaled_pi_knobs() {
        // minimal 3->1 tanh dense arch: 3*1 + 1 = 4 params
        let arch = vec![LayerSpec::Dense {
            input_size: 3,
            output_size: 1,
            activation: Activation::Tanh,
        }];
        let flat = vec![0.0_f64; 4];
        let m = NeuralNetModel::from_flat_weights_v2(
            &flat,
            &arch,
            None,
            OutputParam::ScaledPi,
            2.0,
            0.7,
        )
        .unwrap();
        assert_eq!(m.output_param, OutputParam::ScaledPi);
        assert!((m.scaled_pi_n - 2.0).abs() < 1e-15);
        assert!((m.delta_max - 0.7).abs() < 1e-15);
    }

    #[test]
    fn lstm_json_v2_roundtrip() {
        use crate::data::nn_state::NnState;

        let input_size = 3;
        let hidden_size = 4;
        let four_h = 16;
        let lstm = LstmLayer {
            input_size,
            hidden_size,
            weight_ih: (0..four_h)
                .map(|i| {
                    (0..input_size)
                        .map(|j| (i * input_size + j) as f64 * 0.001)
                        .collect()
                })
                .collect(),
            weight_hh: (0..four_h)
                .map(|i| {
                    (0..hidden_size)
                        .map(|j| 1.0 + (i * hidden_size + j) as f64 * 0.001)
                        .collect()
                })
                .collect(),
            bias_ih: (0..four_h).map(|i| 2.0 + i as f64 * 0.01).collect(),
            bias_hh: (0..four_h).map(|i| 3.0 + i as f64 * 0.01).collect(),
        };
        let dense_out = DenseLayer {
            w: vec![vec![0.5, -0.5, 0.25, 0.1]; 2],
            b: vec![0.0, 0.1],
            activation: Activation::Linear,
        };
        let original = NeuralNetModel {
            architecture: vec![
                LayerSpec::Lstm {
                    input_size,
                    hidden_size,
                },
                LayerSpec::Dense {
                    input_size: hidden_size,
                    output_size: 2,
                    activation: Activation::Linear,
                },
            ],
            layer_sizes: vec![input_size, hidden_size, 2],
            layers: vec![Layer::Lstm(lstm), Layer::Dense(dense_out)],
            input_mask: None,
            ablated_input: None,
            ablated_value: 0.0,
            output_param: OutputParam::default(),
            scaled_pi_n: default_scaled_pi_n(),
            delta_max: default_delta_max(),
        };

        let tmpdir = std::env::temp_dir();
        let path = tmpdir.join("lstm_v2_roundtrip.json");
        original.save_json(path.to_str().unwrap()).unwrap();

        let loaded = NeuralNetModel::load(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.layers.len(), 2);
        match &loaded.layers[0] {
            Layer::Lstm(l) => {
                assert_eq!(l.input_size, input_size);
                assert_eq!(l.hidden_size, hidden_size);
            }
            _ => panic!("expected Lstm layer at index 0"),
        }

        // Forward parity over multiple steps (stateful)
        let mut s0 = NnState::for_model(&original);
        let mut s1 = NnState::for_model(&loaded);
        let x = vec![0.1, -0.2, 0.3];
        for _ in 0..5 {
            let o0 = original.forward(&mut s0, &x);
            let o1 = loaded.forward(&mut s1, &x);
            for (a, b) in o0.iter().zip(o1.iter()) {
                assert!((a - b).abs() < 1e-14, "{} vs {}", a, b);
            }
        }
    }

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

    // ── WindowLayer tests ─────────────────────────────────────────────

    #[test]
    fn window_layer_struct_and_spec_variants_construct() {
        let spec = LayerSpec::Window {
            input_size: 4,
            n_steps: 3,
        };
        match spec {
            LayerSpec::Window {
                input_size,
                n_steps,
            } => {
                assert_eq!(input_size, 4);
                assert_eq!(n_steps, 3);
            }
            _ => panic!("expected LayerSpec::Window"),
        }

        let layer = WindowLayer {
            input_size: 4,
            n_steps: 3,
        };
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

    #[test]
    fn window_layer_weights_trait_zero_params() {
        let layer = WindowLayer {
            input_size: 4,
            n_steps: 8,
        };
        assert_eq!(layer.n_params(), 0);
        assert_eq!(layer.to_flat(), Vec::<f64>::new());

        let mut layer_mut = layer.clone();
        // from_flat on Window consumes 0 params regardless of remaining slice length;
        // this is load-bearing for from_flat_weights_v2's per-layer offset accounting.
        let consumed = layer_mut.from_flat(&[]);
        assert_eq!(consumed, 0);
        let consumed_with_tail = layer_mut.from_flat(&[0.1, 0.2, 0.3]);
        assert_eq!(consumed_with_tail, 0);
    }

    #[test]
    fn window_layer_forward_push_pop_and_concat_zero_padded() {
        use crate::data::nn_state::LayerState;

        let layer = WindowLayer {
            input_size: 2,
            n_steps: 3,
        };
        let mut state = LayerState::for_layer(&Layer::Window(layer.clone()));

        // Tick 0: first real input [1.0, 2.0]. Buffer becomes [[0,0], [0,0], [1,2]].
        let buffer = match &mut state {
            LayerState::Window { buffer } => buffer,
            _ => panic!("expected Window state"),
        };
        let out0 = layer.forward(&[1.0, 2.0], buffer);
        assert_eq!(out0, vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0]);

        // Tick 1: [3.0, 4.0]. Buffer becomes [[0,0], [1,2], [3,4]].
        let buffer = match &mut state {
            LayerState::Window { buffer } => buffer,
            _ => panic!("expected Window state"),
        };
        let out1 = layer.forward(&[3.0, 4.0], buffer);
        assert_eq!(out1, vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0]);

        // Tick 2: [5.0, 6.0]. Buffer becomes [[1,2], [3,4], [5,6]].
        let buffer = match &mut state {
            LayerState::Window { buffer } => buffer,
            _ => panic!("expected Window state"),
        };
        let out2 = layer.forward(&[5.0, 6.0], buffer);
        assert_eq!(out2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Buffer stays at steady-state capacity (always n_steps=3 entries).
        if let LayerState::Window { buffer } = state {
            assert_eq!(buffer.len(), 3);
        } else {
            panic!("expected Window state");
        }
    }

    #[test]
    fn window_layer_end_to_end_forward_through_neural_net_model() {
        use crate::data::nn_state::NnState;

        // Window(2, 3) -> Dense(6 -> 2, linear). Dense weights are the identity
        // on the first two flat-buffer slots so we can verify the whole chain.
        let arch = vec![
            LayerSpec::Window {
                input_size: 2,
                n_steps: 3,
            },
            LayerSpec::Dense {
                input_size: 6,
                output_size: 2,
                activation: Activation::Linear,
            },
        ];
        // Dense weights: row 0 picks buffer[0][0], row 1 picks buffer[0][1].
        let model = NeuralNetModel {
            architecture: arch,
            layer_sizes: vec![2, 6, 2],
            layers: vec![
                Layer::Window(WindowLayer {
                    input_size: 2,
                    n_steps: 3,
                }),
                Layer::Dense(DenseLayer {
                    w: vec![
                        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    b: vec![0.0, 0.0],
                    activation: Activation::Linear,
                }),
            ],
            input_mask: None,
            ablated_input: None,
            ablated_value: 0.0,
            output_param: OutputParam::default(),
            scaled_pi_n: default_scaled_pi_n(),
            delta_max: default_delta_max(),
        };
        let mut state = NnState::for_model(&model);

        // Tick 0: input [1.0, 2.0]. Buffer[0] is the oldest slot = zeros.
        let out = model.forward(&mut state, &[1.0, 2.0]);
        assert_eq!(out, vec![0.0, 0.0]);

        // Tick 1: input [3.0, 4.0]. Buffer[0] is still zeros (popped).
        let out = model.forward(&mut state, &[3.0, 4.0]);
        assert_eq!(out, vec![0.0, 0.0]);

        // Tick 2: input [5.0, 6.0]. Buffer[0] is now [1.0, 2.0].
        let out = model.forward(&mut state, &[5.0, 6.0]);
        assert_eq!(out, vec![1.0, 2.0]);

        // Tick 3: input [7.0, 8.0]. Buffer[0] is now [3.0, 4.0].
        let out = model.forward(&mut state, &[7.0, 8.0]);
        assert_eq!(out, vec![3.0, 4.0]);
    }

    #[test]
    fn window_json_v2_roundtrip_spec_only() {
        let model = NeuralNetModel {
            architecture: vec![
                LayerSpec::Window {
                    input_size: 4,
                    n_steps: 3,
                },
                LayerSpec::Dense {
                    input_size: 12,
                    output_size: 2,
                    activation: Activation::Linear,
                },
            ],
            layer_sizes: vec![4, 12, 2],
            layers: vec![
                Layer::Window(WindowLayer {
                    input_size: 4,
                    n_steps: 3,
                }),
                Layer::Dense(DenseLayer {
                    w: vec![vec![0.1; 12]; 2],
                    b: vec![0.0; 2],
                    activation: Activation::Linear,
                }),
            ],
            input_mask: None,
            ablated_input: None,
            ablated_value: 0.0,
            output_param: OutputParam::default(),
            scaled_pi_n: default_scaled_pi_n(),
            delta_max: default_delta_max(),
        };

        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap();
        model.save_json(path).unwrap();
        let content = std::fs::read_to_string(path).unwrap();

        assert!(content.contains("\"type\": \"window\""));
        assert!(content.contains("\"input_size\": 4"));
        assert!(content.contains("\"n_steps\": 3"));
        // Window has no weights entry in the weights dict (only Dense at index 1).
        assert!(content.contains("\"layer_1\""));
        assert!(!content.contains("\"layer_0\""));

        let parsed = NeuralNetModel::from_json_str(&content, path).unwrap();
        match &parsed.architecture[0] {
            LayerSpec::Window {
                input_size,
                n_steps,
            } => {
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
            LayerSpec::Window {
                input_size: 4,
                n_steps: 3,
            },
            LayerSpec::Dense {
                input_size: 12,
                output_size: 2,
                activation: Activation::Linear,
            },
        ];
        // Total param count = 0 (window) + 12*2 + 2 = 26.
        let flat: Vec<f64> = (0..26).map(|i| i as f64 * 0.01).collect();
        let model = NeuralNetModel::from_flat_weights_v2(
            &flat,
            &arch,
            None,
            OutputParam::default(),
            default_scaled_pi_n(),
            default_delta_max(),
        )
        .unwrap();

        match &model.layers[0] {
            Layer::Window(w) => {
                assert_eq!(w.input_size, 4);
                assert_eq!(w.n_steps, 3);
            }
            _ => panic!("expected Layer::Window"),
        }
        match &model.layers[1] {
            Layer::Dense(d) => {
                assert_eq!(d.w.len(), 2);
                assert_eq!(d.w[0].len(), 12);
                assert_eq!(d.b.len(), 2);
            }
            _ => panic!("expected Layer::Dense"),
        }
        assert_eq!(model.layer_sizes, vec![4, 12, 2]);
    }

    #[test]
    fn window_from_flat_weights_v2_rejects_zero_fields() {
        let arch = vec![LayerSpec::Window {
            input_size: 0,
            n_steps: 3,
        }];
        let flat: Vec<f64> = Vec::new();
        let err = NeuralNetModel::from_flat_weights_v2(
            &flat,
            &arch,
            None,
            OutputParam::default(),
            default_scaled_pi_n(),
            default_delta_max(),
        );
        assert!(err.is_err());
    }

    #[test]
    fn gelu_exact_matches_spec_values() {
        // Hand-computed f64 values of 0.5 * x * (1 + erf(x / sqrt(2))).
        // Generated with Python: 0.5 * x * (1 + math.erf(x / math.sqrt(2)))
        // Both sides use IEEE-754 correctly-rounded erf, so results are bit-identical.
        assert!((gelu_exact(0.0) - 0.0).abs() < 1e-15);
        assert!((gelu_exact(1.0) - 0.8413447460685429).abs() < 1e-15);
        assert!((gelu_exact(-1.0) - (-0.15865525393145707)).abs() < 1e-15);
        assert!((gelu_exact(2.5) - 2.4844758366855597).abs() < 1e-15);
    }

    #[test]
    fn layer_norm_biased_zero_mean_unit_var() {
        // Input [1,2,3,4]: mean=2.5, biased var=((-1.5)^2+(-0.5)^2+(0.5)^2+(1.5)^2)/4 = 1.25.
        // After normalization the output should be zero-mean with unit variance (up to eps).
        let x = [1.0_f64, 2.0, 3.0, 4.0];
        let gamma = [1.0, 1.0, 1.0, 1.0];
        let beta = [0.0, 0.0, 0.0, 0.0];
        let out = layer_norm_biased(&x, &gamma, &beta, 1e-5);
        let mean: f64 = out.iter().sum::<f64>() / 4.0;
        assert!(mean.abs() < 1e-12); // output should be zero-mean
        let var: f64 = out.iter().map(|v| v * v).sum::<f64>() / 4.0;
        assert!((var - 1.0).abs() < 1e-4); // unit variance (up to eps floor)
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

    #[test]
    fn transformer_layer_rebuild_pe_offsets_matches_matmul() {
        // With W_K = W_V = identity, k_pe_offsets and v_pe_offsets should equal the raw PE table.
        let d_model = 4;
        let n_seq = 3;
        let w_k: Vec<Vec<f64>> = (0..d_model)
            .map(|i| {
                (0..d_model)
                    .map(|j| if i == j { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect();
        let w_v: Vec<Vec<f64>> = w_k.clone();
        let mut layer = TransformerLayer {
            d_model,
            n_heads: 2,
            d_head: 2,
            d_ffn: 8,
            n_seq,
            w_q: vec![vec![0.0; d_model]; d_model],
            b_q: vec![0.0; d_model],
            w_k: w_k.clone(),
            b_k: vec![0.0; d_model],
            w_v: w_v.clone(),
            b_v: vec![0.0; d_model],
            w_o: vec![vec![0.0; d_model]; d_model],
            b_o: vec![0.0; d_model],
            w_ffn1: vec![vec![0.0; d_model]; 8],
            b_ffn1: vec![0.0; 8],
            w_ffn2: vec![vec![0.0; 8]; d_model],
            b_ffn2: vec![0.0; d_model],
            ln1_gamma: vec![1.0; d_model],
            ln1_beta: vec![0.0; d_model],
            ln2_gamma: vec![1.0; d_model],
            ln2_beta: vec![0.0; d_model],
            k_pe_offsets: Vec::new(),
            v_pe_offsets: Vec::new(),
        };
        layer.rebuild_pe_offsets();
        let pe = build_pe_table(n_seq, d_model);
        for (i, pe_row) in pe.iter().enumerate() {
            for (j, &pe_val) in pe_row.iter().enumerate() {
                assert!((layer.k_pe_offsets[i][j] - pe_val).abs() < 1e-15);
                assert!((layer.v_pe_offsets[i][j] - pe_val).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn transformer_forward_single_token_zero_weights_is_residual() {
        // All projections zero + LN gamma=1, beta=0 + FFN zero means:
        //   x_norm1 = LN(x)
        //   q = k = v = 0
        //   attention output = 0
        //   x1 = x + W_O @ 0 + b_o = x
        //   ffn_out = 0
        //   out = x1 + 0 = x
        let d_model = 4;
        let n_heads = 2;
        let d_ffn = 8;
        let n_seq = 3;
        let layer = make_zero_transformer(d_model, n_heads, d_ffn, n_seq);
        let mut k_cache = std::collections::VecDeque::new();
        let mut v_cache = std::collections::VecDeque::new();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let out = layer.forward(&x, &mut k_cache, &mut v_cache);
        for i in 0..d_model {
            assert!(
                (out[i] - x[i]).abs() < 1e-12,
                "out[{}]={} x[{}]={}",
                i,
                out[i],
                i,
                x[i]
            );
        }
        assert_eq!(k_cache.len(), 1);
        assert_eq!(v_cache.len(), 1);
    }

    #[test]
    fn transformer_forward_cache_grows_then_saturates() {
        let d_model = 4;
        let n_heads = 2;
        let d_ffn = 8;
        let n_seq = 3;
        let mut layer = make_zero_transformer(d_model, n_heads, d_ffn, n_seq);
        layer.w_k[0][0] = 1.0;
        layer.rebuild_pe_offsets();
        let mut k_cache = std::collections::VecDeque::new();
        let mut v_cache = std::collections::VecDeque::new();
        for step in 0..5 {
            let x = vec![step as f64, 0.0, 0.0, 0.0];
            let _ = layer.forward(&x, &mut k_cache, &mut v_cache);
            let expected_len = (step + 1).min(n_seq);
            assert_eq!(k_cache.len(), expected_len, "step {step}");
            assert_eq!(v_cache.len(), expected_len, "step {step}");
        }
        assert_eq!(k_cache.len(), 3);
    }

    fn make_zero_transformer(
        d_model: usize,
        n_heads: usize,
        d_ffn: usize,
        n_seq: usize,
    ) -> TransformerLayer {
        let mut layer = TransformerLayer {
            d_model,
            n_heads,
            d_head: d_model / n_heads,
            d_ffn,
            n_seq,
            w_q: vec![vec![0.0; d_model]; d_model],
            b_q: vec![0.0; d_model],
            w_k: vec![vec![0.0; d_model]; d_model],
            b_k: vec![0.0; d_model],
            w_v: vec![vec![0.0; d_model]; d_model],
            b_v: vec![0.0; d_model],
            w_o: vec![vec![0.0; d_model]; d_model],
            b_o: vec![0.0; d_model],
            w_ffn1: vec![vec![0.0; d_model]; d_ffn],
            b_ffn1: vec![0.0; d_ffn],
            w_ffn2: vec![vec![0.0; d_ffn]; d_model],
            b_ffn2: vec![0.0; d_model],
            ln1_gamma: vec![1.0; d_model],
            ln1_beta: vec![0.0; d_model],
            ln2_gamma: vec![1.0; d_model],
            ln2_beta: vec![0.0; d_model],
            k_pe_offsets: Vec::new(),
            v_pe_offsets: Vec::new(),
        };
        layer.rebuild_pe_offsets();
        layer
    }

    #[test]
    fn layer_spec_transformer_variant_serializes() {
        let spec = LayerSpec::Transformer {
            d_model: 32,
            n_heads: 4,
            d_ffn: 64,
            n_seq: 64,
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains("\"type\":\"transformer\""));
        assert!(json.contains("\"d_model\":32"));
        let round: LayerSpec = serde_json::from_str(&json).unwrap();
        match round {
            LayerSpec::Transformer {
                d_model,
                n_heads,
                d_ffn,
                n_seq,
            } => {
                assert_eq!((d_model, n_heads, d_ffn, n_seq), (32, 4, 64, 64));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn transformer_layer_weights_flat_roundtrip() {
        let d_model = 4usize;
        let n_heads = 2;
        let d_ffn = 6;
        let n_seq = 3;
        // n_params = 4*d_model^2 + 2*d_ffn*d_model + d_ffn + 9*d_model
        //          = 4*16 + 2*24 + 6 + 36 = 64 + 48 + 6 + 36 = 154
        let n_params = 4 * d_model * d_model + 2 * d_ffn * d_model + d_ffn + 9 * d_model;
        assert_eq!(n_params, 154);

        let flat: Vec<f64> = (0..n_params).map(|i| (i as f64) * 0.01 + 0.5).collect();

        let mut layer = TransformerLayer {
            d_model,
            n_heads,
            d_head: d_model / n_heads,
            d_ffn,
            n_seq,
            w_q: vec![vec![0.0; d_model]; d_model],
            b_q: vec![0.0; d_model],
            w_k: vec![vec![0.0; d_model]; d_model],
            b_k: vec![0.0; d_model],
            w_v: vec![vec![0.0; d_model]; d_model],
            b_v: vec![0.0; d_model],
            w_o: vec![vec![0.0; d_model]; d_model],
            b_o: vec![0.0; d_model],
            w_ffn1: vec![vec![0.0; d_model]; d_ffn],
            b_ffn1: vec![0.0; d_ffn],
            w_ffn2: vec![vec![0.0; d_ffn]; d_model],
            b_ffn2: vec![0.0; d_model],
            ln1_gamma: vec![1.0; d_model],
            ln1_beta: vec![0.0; d_model],
            ln2_gamma: vec![1.0; d_model],
            ln2_beta: vec![0.0; d_model],
            k_pe_offsets: Vec::new(),
            v_pe_offsets: Vec::new(),
        };
        let consumed = layer.from_flat(&flat);
        assert_eq!(consumed, n_params);
        assert_eq!(layer.k_pe_offsets.len(), n_seq); // rebuild_pe_offsets ran
        assert_eq!(layer.v_pe_offsets.len(), n_seq);

        let round = layer.to_flat();
        assert_eq!(round.len(), n_params);
        for (i, (a, b)) in flat.iter().zip(round.iter()).enumerate() {
            assert!((a - b).abs() < 1e-15, "mismatch at index {i}: {a} vs {b}");
        }
    }

    #[test]
    fn transformer_layer_weights_n_params_formula() {
        let layer = make_zero_transformer(4, 2, 6, 3);
        // 4*4*4 + 2*6*4 + 6 + 9*4 = 64 + 48 + 6 + 36 = 154
        assert_eq!(layer.n_params(), 154);
    }

    #[test]
    fn transformer_from_flat_weights_v2_roundtrip() {
        let d_model = 4;
        let n_heads = 2;
        let d_ffn = 6;
        let n_seq = 3;
        let arch = vec![
            LayerSpec::Transformer {
                d_model,
                n_heads,
                d_ffn,
                n_seq,
            },
            LayerSpec::Dense {
                input_size: d_model,
                output_size: 2,
                activation: Activation::Linear,
            },
        ];
        // Transformer: 154 params; Dense(4->2): 4*2 + 2 = 10 params
        let total = 154 + 10;
        let flat: Vec<f64> = (0..total).map(|i| (i as f64) * 0.01 + 0.5).collect();
        let model = NeuralNetModel::from_flat_weights_v2(
            &flat,
            &arch,
            None,
            OutputParam::default(),
            default_scaled_pi_n(),
            default_delta_max(),
        )
        .unwrap();
        let round = model.to_flat_weights();
        assert_eq!(round.len(), total);
        for (i, (a, b)) in flat.iter().zip(round.iter()).enumerate() {
            assert!((a - b).abs() < 1e-15, "mismatch at index {i}: {a} vs {b}");
        }
    }

    #[test]
    fn transformer_json_v2_save_load_roundtrip() {
        // Dense(8->4,linear) -> Transformer(d_model=4, n_heads=2, d_ffn=8, n_seq=3) -> Dense(4->2,linear)
        let d_model = 4usize;
        let n_heads = 2usize;
        let d_ffn = 8usize;
        let n_seq = 3usize;

        let architecture = vec![
            LayerSpec::Dense {
                input_size: 8,
                output_size: d_model,
                activation: Activation::Linear,
            },
            LayerSpec::Transformer {
                d_model,
                n_heads,
                d_ffn,
                n_seq,
            },
            LayerSpec::Dense {
                input_size: d_model,
                output_size: 2,
                activation: Activation::Linear,
            },
        ];

        // Dense(8->4): 8*4 + 4 = 36 params
        // Transformer(d=4, f=8): 4*4*4 (QKV each d*d) + 4*4 (w_o, d*d) + 8*4 (w_ffn1, f*d)
        //   + 4*8 (w_ffn2, d*f) + 4 biases each for b_q/b_k/b_v/b_o/b_ffn1/b_ffn2 + 2*4 ln params * 2
        //   = LayerWeights::n_params = 4*4*4 + 2*8*4 + 8 + 9*4 = ...
        // Use n_params() directly from the model.
        let dummy_flat_len = {
            // Build a zero model to get n_params without needing the exact formula.
            let mut sizes = vec![0usize];
            for spec in &architecture {
                let out = match spec {
                    LayerSpec::Dense { output_size, .. } => *output_size,
                    LayerSpec::Transformer { d_model, .. } => *d_model,
                    _ => 0,
                };
                sizes.push(out);
            }
            // Calculate: Dense 8->4 = 36, Transformer = n_params() by formula, Dense 4->2 = 10
            // TransformerLayer::n_params: 4*d*d + 2*f*d + f + 9*d
            //   = 4*16 + 2*8*4 + 8 + 9*4 = 64 + 64 + 8 + 36 = 172
            36 + 172 + 10
        };
        let flat: Vec<f64> = (0..dummy_flat_len)
            .map(|i| (i as f64) * 0.003 - 0.7)
            .collect();
        let model = NeuralNetModel::from_flat_weights_v2(
            &flat,
            &architecture,
            None,
            OutputParam::default(),
            default_scaled_pi_n(),
            default_delta_max(),
        )
        .unwrap();
        assert_eq!(model.n_params(), dummy_flat_len);

        let tmpdir = std::env::temp_dir();
        let path = tmpdir.join("transformer_v2_roundtrip.json");
        model.save_json(path.to_str().unwrap()).unwrap();

        let loaded = NeuralNetModel::load(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.architecture.len(), 3);
        assert_eq!(loaded.n_params(), model.n_params());

        // Flat-weight round-trip must be bit-identical.
        let orig_flat = model.to_flat_weights();
        let loaded_flat = loaded.to_flat_weights();
        assert_eq!(orig_flat.len(), loaded_flat.len());
        for (i, (a, b)) in orig_flat.iter().zip(loaded_flat.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-15,
                "roundtrip mismatch at {i}: {a} vs {b}"
            );
        }

        // Architecture spec must be identical.
        assert_eq!(
            format!("{:?}", model.architecture),
            format!("{:?}", loaded.architecture),
        );

        // Spot-check: middle layer is Transformer with correct shape.
        match &loaded.layers[1] {
            Layer::Transformer(t) => {
                assert_eq!(t.d_model, d_model);
                assert_eq!(t.n_heads, n_heads);
                assert_eq!(t.d_ffn, d_ffn);
                assert_eq!(t.n_seq, n_seq);
                // PE offsets must be rebuilt (non-empty after load).
                assert_eq!(t.k_pe_offsets.len(), n_seq);
                assert_eq!(t.v_pe_offsets.len(), n_seq);
            }
            _ => panic!("expected Transformer at layer 1"),
        }
    }

    #[test]
    fn neural_net_model_forward_transformer_threads_state() {
        // Dense(4->4) -> Transformer(d_model=4, n_heads=2, d_ffn=8, n_seq=3) -> Dense(4->2)
        // n_params: Dense=20, Transformer=4*4*4 + 2*8*4 + 8 + 9*4 = 172, Dense=10, total=202
        let architecture = vec![
            LayerSpec::Dense {
                input_size: 4,
                output_size: 4,
                activation: Activation::Linear,
            },
            LayerSpec::Transformer {
                d_model: 4,
                n_heads: 2,
                d_ffn: 8,
                n_seq: 3,
            },
            LayerSpec::Dense {
                input_size: 4,
                output_size: 2,
                activation: Activation::Linear,
            },
        ];
        let flat: Vec<f64> = (0..202).map(|i| ((i % 7) as f64) * 0.01).collect();
        let model = NeuralNetModel::from_flat_weights_v2(
            &flat,
            &architecture,
            None,
            OutputParam::default(),
            default_scaled_pi_n(),
            default_delta_max(),
        )
        .unwrap();
        assert_eq!(model.n_params(), 202);

        let mut state = NnState::for_model(&model);
        let x = vec![0.5, -0.3, 0.7, 0.1];

        // Drive for 5 steps; cache should saturate at n_seq=3.
        let mut outputs = Vec::new();
        for _ in 0..5 {
            outputs.push(model.forward(&mut state, &x));
        }

        // All finite, correct output shape.
        for o in &outputs {
            assert_eq!(o.len(), 2);
            for v in o {
                assert!(v.is_finite(), "output contains non-finite value: {v}");
            }
        }

        // Cache saturated at n_seq=3 after 5 steps.
        match &state.layer_states[1] {
            LayerState::Transformer { k_cache, .. } => assert_eq!(k_cache.len(), 3),
            _ => panic!("expected Transformer state at layer 1"),
        }
    }

    #[test]
    fn transformer_from_v2_json_rejects_wrong_w_q_shape() {
        let d_model = 4usize;
        let d_ffn = 8usize;
        let n_seq = 3usize;
        let n_heads = 2usize;

        let row_dm = vec![0.0_f64; d_model];
        let row_ffn_in = vec![0.0_f64; d_model];
        let row_ffn_out = vec![0.0_f64; d_ffn];
        let zero_bias_dm = vec![0.0_f64; d_model];
        let zero_bias_ffn = vec![0.0_f64; d_ffn];
        let gamma = vec![1.0_f64; d_model];
        let beta = vec![0.0_f64; d_model];

        // CORRUPT: w_q has d_model+1 rows instead of d_model.
        let bad_w_q: Vec<Vec<f64>> = (0..d_model + 1).map(|_| row_dm.clone()).collect();

        let json = serde_json::json!({
            "format_version": 2,
            "architecture": [{"type": "transformer", "d_model": d_model, "n_heads": n_heads, "d_ffn": d_ffn, "n_seq": n_seq}],
            "weights": {
                "layer_0": {
                    "w_q": bad_w_q,
                    "b_q": zero_bias_dm.clone(),
                    "w_k": (0..d_model).map(|_| row_dm.clone()).collect::<Vec<_>>(),
                    "b_k": zero_bias_dm.clone(),
                    "w_v": (0..d_model).map(|_| row_dm.clone()).collect::<Vec<_>>(),
                    "b_v": zero_bias_dm.clone(),
                    "w_o": (0..d_model).map(|_| row_dm.clone()).collect::<Vec<_>>(),
                    "b_o": zero_bias_dm.clone(),
                    "w_ffn1": (0..d_ffn).map(|_| row_ffn_in.clone()).collect::<Vec<_>>(),
                    "b_ffn1": zero_bias_ffn.clone(),
                    "w_ffn2": (0..d_model).map(|_| row_ffn_out.clone()).collect::<Vec<_>>(),
                    "b_ffn2": zero_bias_dm.clone(),
                    "ln1_gamma": gamma.clone(), "ln1_beta": beta.clone(),
                    "ln2_gamma": gamma.clone(), "ln2_beta": beta.clone(),
                }
            }
        });
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), serde_json::to_string(&json).unwrap()).unwrap();

        let result = NeuralNetModel::load(tmp.path().to_str().unwrap());
        assert!(
            result.is_err(),
            "expected error for wrong w_q row count, got Ok"
        );
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("w_q") || err.contains("transformer"),
            "expected error to mention w_q or transformer, got: {err}"
        );
    }

    #[test]
    fn transformer_from_v2_json_rejects_wrong_b_q_length() {
        let d_model = 4usize;
        let d_ffn = 8usize;
        let n_seq = 3usize;
        let n_heads = 2usize;

        let row_dm = vec![0.0_f64; d_model];
        let row_ffn_in = vec![0.0_f64; d_model];
        let row_ffn_out = vec![0.0_f64; d_ffn];
        let zero_bias_dm = vec![0.0_f64; d_model];
        let zero_bias_ffn = vec![0.0_f64; d_ffn];
        let gamma = vec![1.0_f64; d_model];
        let beta = vec![0.0_f64; d_model];

        // CORRUPT: b_q is one element too long.
        let bad_b_q = vec![0.0_f64; d_model + 1];

        let json = serde_json::json!({
            "format_version": 2,
            "architecture": [{"type": "transformer", "d_model": d_model, "n_heads": n_heads, "d_ffn": d_ffn, "n_seq": n_seq}],
            "weights": {
                "layer_0": {
                    "w_q": (0..d_model).map(|_| row_dm.clone()).collect::<Vec<_>>(),
                    "b_q": bad_b_q,
                    "w_k": (0..d_model).map(|_| row_dm.clone()).collect::<Vec<_>>(),
                    "b_k": zero_bias_dm.clone(),
                    "w_v": (0..d_model).map(|_| row_dm.clone()).collect::<Vec<_>>(),
                    "b_v": zero_bias_dm.clone(),
                    "w_o": (0..d_model).map(|_| row_dm.clone()).collect::<Vec<_>>(),
                    "b_o": zero_bias_dm.clone(),
                    "w_ffn1": (0..d_ffn).map(|_| row_ffn_in.clone()).collect::<Vec<_>>(),
                    "b_ffn1": zero_bias_ffn.clone(),
                    "w_ffn2": (0..d_model).map(|_| row_ffn_out.clone()).collect::<Vec<_>>(),
                    "b_ffn2": zero_bias_dm.clone(),
                    "ln1_gamma": gamma.clone(), "ln1_beta": beta.clone(),
                    "ln2_gamma": gamma.clone(), "ln2_beta": beta.clone(),
                }
            }
        });
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), serde_json::to_string(&json).unwrap()).unwrap();

        let result = NeuralNetModel::load(tmp.path().to_str().unwrap());
        assert!(
            result.is_err(),
            "expected error for wrong b_q length, got Ok"
        );
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("b_q") || err.contains("transformer"),
            "expected error to mention b_q or transformer, got: {err}"
        );
    }

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
        for &z in &[0.5_f64, -0.5, 1.0, -1.0, 5.0, -5.0, 0.01, -0.01] {
            let expected = z.exp_m1() / z;
            let got = expm1_over_x(z);
            assert!(
                (got - expected).abs() < 1e-15,
                "z={z}: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn expm1_over_x_taylor_branch_at_tiny_z() {
        // Taylor: 1 + z/2 + z^2/6 (error ~ z^3/24)
        // At z = 1e-10, Taylor and exact should agree to machine epsilon.
        let z = 1e-10;
        let taylor = 1.0 + z * 0.5 + z * z / 6.0;
        let got = expm1_over_x(z);
        assert!(
            (got - taylor).abs() < 1e-16,
            "z=1e-10: got {got}, taylor {taylor}"
        );
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
        // The two branches evaluate different formulas at z values ~1e-8 apart, so the
        // maximum expected delta is O(z) ≈ O(1e-8). 1e-9 is well within that bound.
        assert!((y1 - y2).abs() < 1e-9, "crossover jump: y1={y1}, y2={y2}");
    }

    #[test]
    fn mamba_to_flat_from_flat_roundtrip() {
        use rand::{RngExt, SeedableRng};

        let (input_size, d_state, dt_rank) = (8usize, 4usize, 2usize);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut rand_vec =
            |n: usize| -> Vec<f64> { (0..n).map(|_| rng.random_range(-1.0..1.0)).collect() };

        let x_proj_rows = dt_rank + 2 * d_state;
        let original = MambaLayer {
            input_size,
            d_state,
            dt_rank,
            x_proj_w: nalgebra::DMatrix::from_row_slice(
                x_proj_rows,
                input_size,
                &rand_vec(x_proj_rows * input_size),
            ),
            dt_proj_w: nalgebra::DMatrix::from_row_slice(
                input_size,
                dt_rank,
                &rand_vec(input_size * dt_rank),
            ),
            dt_proj_b: nalgebra::DVector::from_row_slice(&rand_vec(input_size)),
            a_log: nalgebra::DMatrix::from_row_slice(
                input_size,
                d_state,
                &rand_vec(input_size * d_state),
            ),
            d_skip: nalgebra::DVector::from_row_slice(&rand_vec(input_size)),
        };

        let expected_n = input_size * (3 * d_state + 2 * dt_rank + 2);
        assert_eq!(original.n_params(), expected_n);

        let flat = original.to_flat();
        assert_eq!(flat.len(), expected_n);

        // Build a zero-initialized MambaLayer with same shape, then from_flat in place.
        let mut reconstructed = MambaLayer {
            input_size,
            d_state,
            dt_rank,
            x_proj_w: nalgebra::DMatrix::zeros(x_proj_rows, input_size),
            dt_proj_w: nalgebra::DMatrix::zeros(input_size, dt_rank),
            dt_proj_b: nalgebra::DVector::zeros(input_size),
            a_log: nalgebra::DMatrix::zeros(input_size, d_state),
            d_skip: nalgebra::DVector::zeros(input_size),
        };
        let cursor = reconstructed.from_flat(&flat);
        assert_eq!(cursor, expected_n);

        assert_eq!(reconstructed.input_size, original.input_size);
        assert_eq!(reconstructed.d_state, original.d_state);
        assert_eq!(reconstructed.dt_rank, original.dt_rank);
        for i in 0..x_proj_rows {
            for j in 0..input_size {
                assert_eq!(reconstructed.x_proj_w[(i, j)], original.x_proj_w[(i, j)]);
            }
        }
        for i in 0..input_size {
            for j in 0..dt_rank {
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
    #[should_panic]
    fn mamba_from_flat_panics_on_short_slice() {
        // 4 * (3*2 + 2*1 + 2) == 4 * 10 == 40; one less = 39 should panic
        let (input_size, d_state, dt_rank) = (4usize, 2usize, 1usize);
        let x_proj_rows = dt_rank + 2 * d_state;
        let mut layer = MambaLayer {
            input_size,
            d_state,
            dt_rank,
            x_proj_w: nalgebra::DMatrix::zeros(x_proj_rows, input_size),
            dt_proj_w: nalgebra::DMatrix::zeros(input_size, dt_rank),
            dt_proj_b: nalgebra::DVector::zeros(input_size),
            a_log: nalgebra::DMatrix::zeros(input_size, d_state),
            d_skip: nalgebra::DVector::zeros(input_size),
        };
        let too_short = vec![0.0_f64; 39]; // one less than 40
        layer.from_flat(&too_short); // must panic
    }

    #[test]
    fn mamba_json_v2_save_load_roundtrip() {
        // Dense(8 -> 4, linear) -> Mamba(4, 2, 1) -> Dense(4 -> 2, linear)
        let architecture = vec![
            LayerSpec::Dense {
                input_size: 8,
                output_size: 4,
                activation: Activation::Linear,
            },
            LayerSpec::Mamba {
                input_size: 4,
                d_state: 2,
                dt_rank: 1,
            },
            LayerSpec::Dense {
                input_size: 4,
                output_size: 2,
                activation: Activation::Linear,
            },
        ];
        // Dense(8->4) = 36, Mamba(4, 2, 1) = 4*(6+2+2) = 40, Dense(4->2) = 10; total 86.
        let flat: Vec<f64> = (0..86).map(|i| (i as f64) * 0.017 - 0.9).collect();
        let model = NeuralNetModel::from_flat_weights_v2(
            &flat,
            &architecture,
            None,
            OutputParam::default(),
            default_scaled_pi_n(),
            default_delta_max(),
        )
        .unwrap();
        assert_eq!(model.n_params(), 86);

        let tmpdir = std::env::temp_dir();
        let path = tmpdir.join("mamba_v2_roundtrip.json");
        model.save_json(path.to_str().unwrap()).unwrap();

        // Sanity-check: JSON always includes dt_rank for Mamba layers
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(
            raw.contains("\"dt_rank\""),
            "save_json output must contain dt_rank field; got: {raw}"
        );

        let loaded = NeuralNetModel::load(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.architecture.len(), 3);
        assert_eq!(loaded.n_params(), model.n_params());

        // Flat-weight round-trip must be bit-identical.
        let orig_flat = model.to_flat_weights();
        let loaded_flat = loaded.to_flat_weights();
        assert_eq!(orig_flat.len(), loaded_flat.len());
        for (i, (a, b)) in orig_flat.iter().zip(loaded_flat.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-15,
                "roundtrip mismatch at {i}: {a} vs {b}"
            );
        }

        // Architecture spec must be identical.
        assert_eq!(
            format!("{:?}", model.architecture),
            format!("{:?}", loaded.architecture),
        );

        // Spot-check: middle layer is Mamba with correct shape.
        match &loaded.layers[1] {
            Layer::Mamba(m) => {
                assert_eq!(m.input_size, 4);
                assert_eq!(m.d_state, 2);
                assert_eq!(m.dt_rank, 1);
                assert_eq!(m.x_proj_w.nrows(), 1 + 2 * 2); // dt_rank + 2*d_state = 5
                assert_eq!(m.x_proj_w.ncols(), 4); // input_size
                assert_eq!(m.dt_proj_w.shape(), (4, 1));
                assert_eq!(m.a_log.shape(), (4, 2));
                assert_eq!(m.dt_proj_b.len(), 4);
                assert_eq!(m.d_skip.len(), 4);
            }
            _ => panic!("expected Mamba at layer 1"),
        }
    }

    #[test]
    fn mamba_from_v2_json_rejects_zero_dt_rank() {
        // Build a minimal v2 JSON by hand with dt_rank = 0 in the Mamba spec.
        // The constructor validators in from_v2_json must reject this.
        let dir = std::env::temp_dir();
        let path = dir.join("mamba_zero_dt_rank.json");
        let bad_json = r#"{
            "format_version": 2,
            "architecture": [
                {"type": "dense", "input_size": 4, "output_size": 4, "activation": "linear"},
                {"type": "mamba", "input_size": 4, "d_state": 2, "dt_rank": 0},
                {"type": "dense", "input_size": 4, "output_size": 2, "activation": "linear"}
            ],
            "weights": {
                "layer_0": {"w": [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], "b": [0.0, 0.0, 0.0, 0.0]},
                "layer_1": {
                    "x_proj_w": [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                    "dt_proj_w": [[], [], [], []],
                    "dt_proj_b": [0.0, 0.0, 0.0, 0.0],
                    "a_log": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    "d_skip": [0.0, 0.0, 0.0, 0.0]
                },
                "layer_2": {"w": [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], "b": [0.0, 0.0]}
            }
        }"#;
        std::fs::write(&path, bad_json).unwrap();
        let result = NeuralNetModel::load(path.to_str().unwrap());
        assert!(
            result.is_err(),
            "from_v2_json must reject Mamba with dt_rank = 0"
        );
    }

    #[test]
    fn mamba_forward_two_step_hand_verified() {
        use nalgebra::{DMatrix, DVector};

        // Minimal layer: d_inner=2, d_state=2, dt_rank=1
        //
        // x_proj: (5, 2) -- rows [dt_pre; B_0; B_1; C_0; C_1]
        // For x = [1, 0]: proj = [0, 1, 0, 1, 0] -> dt_pre=0, B=[1, 0], C=[1, 0]
        // For x = [0, 1]: proj = [0, 0, 1, 0, 1] -> dt_pre=0, B=[0, 1], C=[0, 1]
        //
        // dt_proj_w zero, bias such that softplus(b) = 0.5 -> b = log(exp(0.5) - 1)
        // Δ = 0.5 (per channel, constant)
        //
        // a_log = 0 -> A = -exp(0) = -1 (per (d, n))
        // Ā[d, n] = exp(Δ * A) = exp(-0.5) ≈ 0.6065306597126334
        // expm1_over_x(Δ * A) = (exp(-0.5) - 1) / (-0.5) ≈ 0.7869386805747332
        // B̄[d, n] = Δ * B[n] * expm1_over_x(Δ * A) = 0.5 * B[n] * 0.7869
        //
        // d_skip = 0 (no skip, isolate SSM)

        let x_proj_w = DMatrix::from_row_slice(
            5,
            2,
            &[
                0.0, 0.0, // dt_pre row
                1.0, 0.0, // B_0
                0.0, 1.0, // B_1
                1.0, 0.0, // C_0
                0.0, 1.0, // C_1
            ],
        );
        let dt_proj_w = DMatrix::from_row_slice(2, 1, &[0.0, 0.0]);
        let b_val = (0.5_f64.exp() - 1.0).ln(); // inv_softplus(0.5)
        let dt_proj_b = DVector::from_row_slice(&[b_val, b_val]);
        let a_log = DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 0.0]);
        let d_skip = DVector::from_row_slice(&[0.0, 0.0]);

        let layer = MambaLayer {
            input_size: 2,
            d_state: 2,
            dt_rank: 1,
            x_proj_w,
            dt_proj_w,
            dt_proj_b,
            a_log,
            d_skip,
        };

        let mut h = DMatrix::<f64>::zeros(2, 2);

        // Step 1: x = [1, 0]
        let x1 = [1.0, 0.0];
        let y1 = layer.forward(&x1, &mut h);
        assert!(
            (y1[0] - 0.3934693402873666).abs() < 1e-12,
            "step 1 y[0] = {}",
            y1[0]
        );
        assert!((y1[1] - 0.0).abs() < 1e-15, "step 1 y[1] = {}", y1[1]);

        // Step 2: x = [0, 1], h = [[0.39347, 0], [0, 0]]
        let x2 = [0.0, 1.0];
        let y2 = layer.forward(&x2, &mut h);
        assert!((y2[0] - 0.0).abs() < 1e-15, "step 2 y[0] = {}", y2[0]);
        assert!(
            (y2[1] - 0.3934693402873666).abs() < 1e-12,
            "step 2 y[1] = {}",
            y2[1]
        );
        // State h[0, 0] should now be ~0.23865 (exp(-0.5) * prev value)
        // Exact: exp(-0.5) * B_bar_step1 = 0.6065306597126334 * 0.3934693402873666
        assert!(
            (h[(0, 0)] - 0.2386512185411911).abs() < 1e-12,
            "h[0, 0] = {}",
            h[(0, 0)]
        );
    }

    mod mamba_proptest {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn mamba_flat_roundtrip_proptest(
                input_size in 1usize..=8,
                d_state in 1usize..=8,
                dt_rank in 1usize..=4,
                seed in 0u64..200,
            ) {
                use rand::{RngExt, SeedableRng};
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                let n = input_size * (3 * d_state + 2 * dt_rank + 2);
                let flat: Vec<f64> = (0..n).map(|_| rng.random_range(-5.0..5.0)).collect();

                let x_proj_rows = dt_rank + 2 * d_state;
                let mut layer = MambaLayer {
                    input_size,
                    d_state,
                    dt_rank,
                    x_proj_w: nalgebra::DMatrix::zeros(x_proj_rows, input_size),
                    dt_proj_w: nalgebra::DMatrix::zeros(input_size, dt_rank),
                    dt_proj_b: nalgebra::DVector::zeros(input_size),
                    a_log: nalgebra::DMatrix::zeros(input_size, d_state),
                    d_skip: nalgebra::DVector::zeros(input_size),
                };
                let cursor = layer.from_flat(&flat);
                prop_assert_eq!(cursor, n);
                prop_assert_eq!(layer.n_params(), n);

                let back = layer.to_flat();
                prop_assert_eq!(back.len(), n);
                for i in 0..n {
                    prop_assert_eq!(back[i], flat[i]);
                }
            }
        }

        proptest! {
            #[test]
            fn mamba_forward_finite_on_finite_inputs(
                d_inner in 1usize..=4,
                d_state in 1usize..=4,
                dt_rank in 1usize..=3,
                seed in 0u64..1000,
            ) {
                use rand::{RngExt, SeedableRng};
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                let rand_vec = |n: usize, rng: &mut rand::rngs::StdRng| -> Vec<f64> {
                    (0..n).map(|_| rng.random_range(-1.0..1.0)).collect()
                };
                let x_proj_w = nalgebra::DMatrix::from_row_slice(dt_rank + 2 * d_state, d_inner,
                    &rand_vec((dt_rank + 2 * d_state) * d_inner, &mut rng));
                let dt_proj_w = nalgebra::DMatrix::from_row_slice(d_inner, dt_rank,
                    &rand_vec(d_inner * dt_rank, &mut rng));
                let dt_proj_b = nalgebra::DVector::from_row_slice(&rand_vec(d_inner, &mut rng));
                let a_log = nalgebra::DMatrix::from_row_slice(d_inner, d_state, &rand_vec(d_inner * d_state, &mut rng));
                let d_skip = nalgebra::DVector::from_row_slice(&rand_vec(d_inner, &mut rng));

                let layer = MambaLayer {
                    input_size: d_inner, d_state, dt_rank,
                    x_proj_w, dt_proj_w, dt_proj_b, a_log, d_skip,
                };
                let x: Vec<f64> = rand_vec(d_inner, &mut rng);
                let mut h = nalgebra::DMatrix::<f64>::zeros(d_inner, d_state);

                for _ in 0..50 {
                    let y = layer.forward(&x, &mut h);
                    for v in &y {
                        prop_assert!(v.is_finite(), "y not finite: {v}");
                    }
                    for i in 0..d_inner {
                        for j in 0..d_state {
                            prop_assert!(h[(i, j)].is_finite(), "h[{i}, {j}] not finite");
                        }
                    }
                }
            }
        }
    }

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
            assert!(
                NeuralNetModel::validate_output_activation(Activation::Linear, p, "<t>").is_err()
            );
        }
    }

    #[test]
    fn output_param_default_is_atan2_signed() {
        let p: OutputParam = OutputParam::default();
        assert_eq!(p, OutputParam::Atan2Signed);
    }

    #[test]
    fn output_param_serde_round_trip() {
        let p = OutputParam::AcosTanh;
        let s = serde_json::to_string(&p).unwrap();
        assert_eq!(s, "\"acos_tanh\"");
        let back: OutputParam = serde_json::from_str(&s).unwrap();
        assert_eq!(back, p);

        let p2 = OutputParam::Atan2Signed;
        let s2 = serde_json::to_string(&p2).unwrap();
        assert_eq!(s2, "\"atan2_signed\"");

        let p3 = OutputParam::ScaledPi;
        let s3 = serde_json::to_string(&p3).unwrap();
        assert_eq!(s3, "\"scaled_pi\"");
        let back3: OutputParam = serde_json::from_str(&s3).unwrap();
        assert_eq!(back3, p3);

        let p4 = OutputParam::Delta;
        let s4 = serde_json::to_string(&p4).unwrap();
        assert_eq!(s4, "\"delta\"");
        let back4: OutputParam = serde_json::from_str(&s4).unwrap();
        assert_eq!(back4, p4);
    }

    #[test]
    fn output_param_persists_through_v2_json_round_trip() {
        let arch = vec![LayerSpec::Dense {
            input_size: 3,
            output_size: 1,
            activation: Activation::Tanh,
        }];
        let layers = vec![Layer::Dense(DenseLayer {
            w: vec![vec![0.1, 0.2, 0.3]],
            b: vec![0.4],
            activation: Activation::Tanh,
        })];
        let original = NeuralNetModel {
            architecture: arch,
            layer_sizes: vec![3, 1],
            layers,
            input_mask: None,
            ablated_input: None,
            ablated_value: 0.0,
            output_param: OutputParam::AcosTanh,
            scaled_pi_n: default_scaled_pi_n(),
            delta_max: default_delta_max(),
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.json");
        original.save_json(path.to_str().unwrap()).unwrap();
        let loaded = NeuralNetModel::load(path.to_str().unwrap()).unwrap();

        assert_eq!(loaded.output_param, OutputParam::AcosTanh);
    }

    #[test]
    fn scaled_pi_knobs_persist_through_v2_json_round_trip() {
        let arch = vec![LayerSpec::Dense {
            input_size: 3,
            output_size: 1,
            activation: Activation::Tanh,
        }];
        let layers = vec![Layer::Dense(DenseLayer {
            w: vec![vec![0.1, 0.2, 0.3]],
            b: vec![0.4],
            activation: Activation::Tanh,
        })];
        let original = NeuralNetModel {
            architecture: arch,
            layer_sizes: vec![3, 1],
            layers,
            input_mask: None,
            ablated_input: None,
            ablated_value: 0.0,
            output_param: OutputParam::ScaledPi,
            scaled_pi_n: 2.0,
            delta_max: 0.7,
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.json");
        original.save_json(path.to_str().unwrap()).unwrap();
        let loaded = NeuralNetModel::load(path.to_str().unwrap()).unwrap();

        assert_eq!(loaded.output_param, OutputParam::ScaledPi);
        assert!(
            (loaded.scaled_pi_n - 2.0).abs() < 1e-15,
            "scaled_pi_n: {}",
            loaded.scaled_pi_n
        );
        assert!(
            (loaded.delta_max - 0.7).abs() < 1e-15,
            "delta_max: {}",
            loaded.delta_max
        );
    }

    #[test]
    fn output_param_absent_in_json_loads_as_atan2_signed() {
        let json = r#"{
            "format_version": 2,
            "architecture": [{"type": "dense", "input_size": 2, "output_size": 2, "activation": "linear"}],
            "weights": {
                "layer_0": {"w": [[0.1, 0.2], [0.3, 0.4]], "b": [0.0, 0.0]}
            }
        }"#;
        let m = NeuralNetModel::from_json_str(json, "<test>").unwrap();
        assert_eq!(m.output_param, OutputParam::Atan2Signed);
    }

    #[test]
    fn acos_tanh_with_non_tanh_activation_rejected_at_v2_json_load() {
        let json = r#"{
            "format_version": 2,
            "output_param": "acos_tanh",
            "architecture": [{"type": "dense", "input_size": 2, "output_size": 1, "activation": "linear"}],
            "weights": {
                "layer_0": {"w": [[0.1, 0.2]], "b": [0.0]}
            }
        }"#;
        let result = NeuralNetModel::from_json_str(json, "<test>");
        assert!(result.is_err());
        let msg = format!("{:?}", result.unwrap_err());
        assert!(
            msg.contains("AcosTanh"),
            "expected AcosTanh in error, got: {}",
            msg
        );
        assert!(msg.contains("Tanh"), "expected Tanh in error, got: {}", msg);
    }

    #[test]
    fn acos_tanh_with_asinh_activation_rejected_at_v2_json_load() {
        let json = r#"{
            "format_version": 2,
            "output_param": "acos_tanh",
            "architecture": [{"type": "dense", "input_size": 2, "output_size": 1, "activation": "asinh"}],
            "weights": {
                "layer_0": {"w": [[0.1, 0.2]], "b": [0.0]}
            }
        }"#;
        let result = NeuralNetModel::from_json_str(json, "<test>");
        assert!(result.is_err());
    }

    #[test]
    fn acos_tanh_with_tanh_activation_accepted_at_v2_json_load() {
        let json = r#"{
            "format_version": 2,
            "output_param": "acos_tanh",
            "architecture": [{"type": "dense", "input_size": 2, "output_size": 1, "activation": "tanh"}],
            "weights": {
                "layer_0": {"w": [[0.1, 0.2]], "b": [0.0]}
            }
        }"#;
        let m = NeuralNetModel::from_json_str(json, "<test>").unwrap();
        assert_eq!(m.output_param, OutputParam::AcosTanh);
    }
}
