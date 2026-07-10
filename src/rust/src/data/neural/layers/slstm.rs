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
        // Multi-array indexing (pre/c/n/m/h_new, some offset by hs/2hs/3hs) is the
        // idiomatic exception to needless_range_loop -- an iterator-based rewrite
        // would need 5 separate iterator chains zipped, which is less clear.
        #[allow(clippy::needless_range_loop)]
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
        assert!(
            n.iter().all(|&v| v > 0.0),
            "n must be strictly positive after step 1"
        );
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
            assert!(
                out.iter().all(|v| v.is_finite()),
                "stabilizer failed: {out:?}"
            );
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
                last = l.forward(
                    &[(t as f64) * 0.1, 0.5, -0.5],
                    &mut h,
                    &mut c,
                    &mut n,
                    &mut m,
                );
            }
            last
        };
        assert_eq!(run(), run());
    }
}
