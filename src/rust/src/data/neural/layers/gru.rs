//! GRU cell layer (PyTorch nn.GRUCell convention).

use super::super::Activation;
use super::helpers::dot_plus_bias;
use super::tensor::tensor_table;

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

impl GruLayer {
    pub fn zeros(input_size: usize, hidden_size: usize) -> Self {
        let three_h = 3 * hidden_size;
        Self {
            input_size,
            hidden_size,
            weight_ih: vec![vec![0.0; input_size]; three_h],
            weight_hh: vec![vec![0.0; hidden_size]; three_h],
            bias_ih: vec![0.0; three_h],
            bias_hh: vec![0.0; three_h],
        }
    }

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

// Flat order: weight_ih (row-major) -> weight_hh (row-major) -> bias_ih -> bias_hh.
tensor_table!(GruLayer {
    weight_ih,
    weight_hh,
    bias_ih,
    bias_hh
});
