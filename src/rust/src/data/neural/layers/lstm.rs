//! LSTM cell layer (PyTorch nn.LSTMCell convention).

use super::super::Activation;
use super::helpers::dot_plus_bias;
use super::tensor::tensor_table;

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
    pub fn zeros(input_size: usize, hidden_size: usize) -> Self {
        let four_h = 4 * hidden_size;
        Self {
            input_size,
            hidden_size,
            weight_ih: vec![vec![0.0; input_size]; four_h],
            weight_hh: vec![vec![0.0; hidden_size]; four_h],
            bias_ih: vec![0.0; four_h],
            bias_hh: vec![0.0; four_h],
        }
    }

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

// Flat order: weight_ih (row-major) -> weight_hh (row-major) -> bias_ih -> bias_hh.
tensor_table!(LstmLayer {
    weight_ih,
    weight_hh,
    bias_ih,
    bias_hh
});
