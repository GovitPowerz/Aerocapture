//! CfC (closed-form continuous-time) cell -- ncps "default" mode, cell-only.
//!
//! Forward (dt fixed at one guidance tick, absorbed into the learned time heads):
//!   cat = [x, h]
//!   xb  = lecun_tanh(W_bb @ cat + b_bb)
//!   g   = sigmoid(-(W_ta @ xb + b_ta) * CFC_DT + (W_tb @ xb + b_tb))
//!   h'  = (1 - g) * tanh(W_ff1 @ xb + b_ff1) + g * tanh(W_ff2 @ xb + b_ff2)
//! Output = h', bounded in (-1, 1) by construction.
//!
//! Canonical flat order (LayerWeights + PSO chromosome + torch mirror):
//!   w_bb, b_bb, w_ff1, b_ff1, w_ff2, b_ff2, w_ta, b_ta, w_tb, b_tb
//! (matrices row-major, interleaved matrix/bias pairs).

use super::super::Activation;
use super::helpers::{dot_plus_bias, lecun_tanh};
use super::tensor::tensor_table;

/// Fixed per-tick dt: guidance cadence is constant, so dt is absorbed into
/// the learned time heads t_a / t_b (spec: deliberate simplification).
const CFC_DT: f64 = 1.0;

#[derive(Debug, Clone)]
pub struct CfcLayer {
    pub input_size: usize,
    pub hidden_size: usize,
    pub backbone_units: usize,
    pub w_bb: Vec<Vec<f64>>,  // [B, I+H]
    pub b_bb: Vec<f64>,       // [B]
    pub w_ff1: Vec<Vec<f64>>, // [H, B]
    pub b_ff1: Vec<f64>,      // [H]
    pub w_ff2: Vec<Vec<f64>>, // [H, B]
    pub b_ff2: Vec<f64>,      // [H]
    pub w_ta: Vec<Vec<f64>>,  // [H, B]
    pub b_ta: Vec<f64>,       // [H]
    pub w_tb: Vec<Vec<f64>>,  // [H, B]
    pub b_tb: Vec<f64>,       // [H]
}

impl CfcLayer {
    pub fn zeros(input_size: usize, hidden_size: usize, backbone_units: usize) -> Self {
        let cat = input_size + hidden_size;
        Self {
            input_size,
            hidden_size,
            backbone_units,
            w_bb: vec![vec![0.0; cat]; backbone_units],
            b_bb: vec![0.0; backbone_units],
            w_ff1: vec![vec![0.0; backbone_units]; hidden_size],
            b_ff1: vec![0.0; hidden_size],
            w_ff2: vec![vec![0.0; backbone_units]; hidden_size],
            b_ff2: vec![0.0; hidden_size],
            w_ta: vec![vec![0.0; backbone_units]; hidden_size],
            b_ta: vec![0.0; hidden_size],
            w_tb: vec![vec![0.0; backbone_units]; hidden_size],
            b_tb: vec![0.0; hidden_size],
        }
    }

    /// One step: reads x + h, overwrites h with h_new, returns h_new as output.
    pub fn forward(&self, x: &[f64], h: &mut [f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.input_size);
        assert_eq!(h.len(), self.hidden_size);
        let mut cat = Vec::with_capacity(self.input_size + self.hidden_size);
        cat.extend_from_slice(x);
        cat.extend_from_slice(h);
        let xb: Vec<f64> = (0..self.backbone_units)
            .map(|j| lecun_tanh(dot_plus_bias(&self.w_bb[j], &cat, self.b_bb[j])))
            .collect();
        let mut h_new = vec![0.0; self.hidden_size];
        for (i, out) in h_new.iter_mut().enumerate() {
            let ff1 = dot_plus_bias(&self.w_ff1[i], &xb, self.b_ff1[i]).tanh();
            let ff2 = dot_plus_bias(&self.w_ff2[i], &xb, self.b_ff2[i]).tanh();
            let t_a = dot_plus_bias(&self.w_ta[i], &xb, self.b_ta[i]);
            let t_b = dot_plus_bias(&self.w_tb[i], &xb, self.b_tb[i]);
            let g = Activation::Sigmoid.apply(-t_a * CFC_DT + t_b);
            *out = (1.0 - g) * ff1 + g * ff2;
        }
        h.copy_from_slice(&h_new);
        h_new
    }
}

tensor_table!(CfcLayer {
    w_bb,
    b_bb,
    w_ff1,
    b_ff1,
    w_ff2,
    b_ff2,
    w_ta,
    b_ta,
    w_tb,
    b_tb
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::neural::LayerWeights;

    fn patterned(input_size: usize, hidden_size: usize, backbone_units: usize) -> CfcLayer {
        let mut l = CfcLayer::zeros(input_size, hidden_size, backbone_units);
        let n = l.n_params();
        let flat: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01 - 0.3).collect();
        assert_eq!(l.from_flat(&flat), n);
        l
    }

    #[test]
    fn flat_round_trip_is_bit_identical() {
        let l = patterned(3, 4, 5);
        let flat = l.to_flat();
        assert_eq!(flat.len(), l.n_params());
        let mut l2 = CfcLayer::zeros(3, 4, 5);
        assert_eq!(l2.from_flat(&flat), flat.len());
        assert_eq!(l2.to_flat(), flat);
    }

    #[test]
    fn n_params_formula() {
        // B(I+H) + B + 4(HB + H) = 5*7 + 5 + 4*(4*5 + 4) = 136
        assert_eq!(CfcLayer::zeros(3, 4, 5).n_params(), 136);
    }

    #[test]
    fn output_is_bounded_and_state_evolves() {
        let l = patterned(3, 4, 5);
        let mut h = vec![0.0; 4];
        let mut prev = h.clone();
        for t in 0..50 {
            let x = vec![0.5 * (t as f64).sin(), -0.2, 0.9];
            let out = l.forward(&x, &mut h);
            assert_eq!(out, h);
            for &v in &out {
                assert!(v.is_finite() && v.abs() < 1.0, "unbounded output {v}");
            }
            if t == 1 {
                assert_ne!(h, prev, "state must evolve");
            }
            prev.clone_from(&h);
        }
    }

    #[test]
    fn zero_weights_give_neutral_gate_output() {
        // All-zero weights: ff1 = ff2 = tanh(0) = 0 -> h' = 0 regardless of g.
        let l = CfcLayer::zeros(2, 3, 2);
        let mut h = vec![0.0; 3];
        let out = l.forward(&[1.0, -1.0], &mut h);
        assert_eq!(out, vec![0.0; 3]);
    }
}
