//! Dense (fully-connected) layer.

use super::super::Activation;
use super::tensor::tensor_table;
use serde::{Deserialize, Serialize};

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

impl DenseLayer {
    pub fn zeros(input_size: usize, output_size: usize, activation: Activation) -> Self {
        Self {
            w: vec![vec![0.0; input_size]; output_size],
            b: vec![0.0; output_size],
            activation,
        }
    }
}

// Flat order: W (row-major) then b.
tensor_table!(DenseLayer { w, b });
