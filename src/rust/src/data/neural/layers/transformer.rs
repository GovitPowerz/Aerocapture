//! 1-layer pre-norm Transformer block with causal window attention.

use super::helpers::{build_pe_table, gelu_exact, layer_norm_biased, matvec};
use super::tensor::tensor_table;

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
    /// Zero-weight layer (LN gammas at 1.0); `n_heads` must divide `d_model`.
    /// PE offsets are derived on the first load (`post_load`).
    pub fn zeros(d_model: usize, n_heads: usize, d_ffn: usize, n_seq: usize) -> Self {
        let (d, f) = (d_model, d_ffn);
        Self {
            d_model,
            n_heads,
            d_head: d_model / n_heads,
            d_ffn,
            n_seq,
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
        }
    }

    /// Recompute `k_pe_offsets[i] = W_K @ PE[i]` and `v_pe_offsets[i] = W_V @ PE[i]`
    /// for i in 0..n_seq. Biases are NOT included in the PE offset (they are added
    /// once per forward to the raw query/key projections, not through PE).
    ///
    /// Call this after any mutation to `w_k` or `w_v`. Both load paths (JSON and
    /// PSO chromosome) reach it through the table's `post_load` hook.
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

// Flat order: the four attention projections (matrix, bias), the two FFN
// layers (matrix, bias), then the two LayerNorms (gamma, beta). `k_pe_offsets`
// / `v_pe_offsets` are derived, not stored: `post_load` rebuilds them.
tensor_table!(
    TransformerLayer {
        w_q,
        b_q,
        w_k,
        b_k,
        w_v,
        b_v,
        w_o,
        b_o,
        w_ffn1,
        b_ffn1,
        w_ffn2,
        b_ffn2,
        ln1_gamma,
        ln1_beta,
        ln2_gamma,
        ln2_beta
    },
    post_load = rebuild_pe_offsets
);
