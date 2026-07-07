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
