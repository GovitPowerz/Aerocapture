//! Neural network model with modular architecture.
//!
//! Supports arbitrary layer configurations (e.g. [6, 12, 2] or [6, 24, 12, 2])
//! with per-layer activation function choice. Loads from JSON (new format) or
//! legacy Fortran nn_param files (auto-detected).

use super::DataError;
use serde::{Deserialize, Serialize};

/// Activation function for a layer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Tanh,
    Relu,
    Sigmoid,
    Asinh,
    Linear,
}

impl Activation {
    fn apply(self, x: f64) -> f64 {
        match self {
            Activation::Tanh => x.tanh(),
            Activation::Relu => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Asinh => x.asinh(),
            Activation::Linear => x,
        }
    }
}

/// A single layer: weights, biases, and activation function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    /// Weights [n_out × n_in], row-major: w[j][i] = weight from input i to output j.
    pub w: Vec<Vec<f64>>,
    /// Biases [n_out].
    pub b: Vec<f64>,
    /// Activation function applied after affine transform.
    pub activation: Activation,
}

/// JSON file structure for neural network models.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct NnJsonFile {
    format_version: u32,
    architecture: NnArchitecture,
    weights: std::collections::BTreeMap<String, NnLayerWeights>,
    output_interpretation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NnArchitecture {
    layers: Vec<usize>,
    activations: Vec<Activation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NnLayerWeights {
    w: Vec<Vec<f64>>,
    b: Vec<f64>,
}

/// Modular neural network model.
///
/// Replaces the fixed-size `NeuralNetParams`. Supports arbitrary depth and width.
#[derive(Debug, Clone)]
pub struct NeuralNetModel {
    /// Layer sizes: [input_size, hidden1, ..., output_size].
    pub layer_sizes: Vec<usize>,
    /// Network layers (len = layer_sizes.len() - 1).
    pub layers: Vec<Layer>,
    /// Output interpretation (e.g. "atan2").
    pub output_interpretation: String,
}

impl NeuralNetModel {
    /// Load NN from file, auto-detecting format.
    ///
    /// JSON files start with `{`, legacy Fortran files start with whitespace/text.
    pub fn load(path: &str) -> Result<Self, DataError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| DataError(format!("Cannot read {}: {}", path, e)))?;

        let first_char = content.trim_start().chars().next().unwrap_or(' ');
        if first_char == '{' {
            Self::from_json(&content, path)
        } else {
            Self::from_legacy(&content, path)
        }
    }

    /// Load from JSON format.
    fn from_json(content: &str, path: &str) -> Result<Self, DataError> {
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
            let lw = file.weights.get(&key).ok_or_else(|| {
                DataError(format!("Missing {} in weights in {}", key, path))
            })?;

            let n_out = file.architecture.layers[i + 1];
            let n_in = file.architecture.layers[i];

            if lw.w.len() != n_out || lw.b.len() != n_out {
                return Err(DataError(format!(
                    "Layer {} size mismatch: expected {}x{}, got w={}x?, b={} in {}",
                    i, n_out, n_in, lw.w.len(), lw.b.len(), path
                )));
            }

            layers.push(Layer {
                w: lw.w.clone(),
                b: lw.b.clone(),
                activation: file.architecture.activations[i],
            });
        }

        Ok(NeuralNetModel {
            layer_sizes: file.architecture.layers,
            layers,
            output_interpretation: file.output_interpretation,
        })
    }

    /// Load from legacy Fortran nn_param format (6-line header + flat weights).
    ///
    /// Hardcodes the legacy architecture: 6→12(tanh)→2(asinh).
    fn from_legacy(content: &str, path: &str) -> Result<Self, DataError> {
        const N_INPUT: usize = 6;
        const N_HIDDEN: usize = 12;
        const N_OUTPUT: usize = 2;

        let values: Vec<f64> = content
            .lines()
            .skip(6)
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let token = l.split_whitespace().next().unwrap_or("0");
                token
                    .parse::<f64>()
                    .map_err(|_| DataError(format!("Cannot parse '{}' as f64 in {}", token, path)))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let expected = N_INPUT * N_HIDDEN + N_HIDDEN + N_HIDDEN * N_OUTPUT + N_OUTPUT;
        if values.len() < expected {
            return Err(DataError(format!(
                "NN param file too short: {} values, need {} in {}",
                values.len(), expected, path
            )));
        }

        // Read in Fortran column-major order, store as row-major
        let mut idx = 0;

        // lw1: input→hidden [N_HIDDEN × N_INPUT]
        let mut lw1 = vec![vec![0.0; N_INPUT]; N_HIDDEN];
        for i in 0..N_INPUT {
            for j in 0..N_HIDDEN {
                lw1[j][i] = values[idx];
                idx += 1;
            }
        }

        let mut bias1 = vec![0.0; N_HIDDEN];
        for b in bias1.iter_mut() {
            *b = values[idx];
            idx += 1;
        }

        // lw4: hidden→output [N_OUTPUT × N_HIDDEN]
        let mut lw4 = vec![vec![0.0; N_HIDDEN]; N_OUTPUT];
        for i in 0..N_HIDDEN {
            for j in 0..N_OUTPUT {
                lw4[j][i] = values[idx];
                idx += 1;
            }
        }

        let mut bias4 = vec![0.0; N_OUTPUT];
        for b in bias4.iter_mut() {
            *b = values[idx];
            idx += 1;
        }

        Ok(NeuralNetModel {
            layer_sizes: vec![N_INPUT, N_HIDDEN, N_OUTPUT],
            layers: vec![
                Layer { w: lw1, b: bias1, activation: Activation::Tanh },
                Layer { w: lw4, b: bias4, activation: Activation::Asinh },
            ],
            output_interpretation: "atan2".to_string(),
        })
    }

    /// Save to JSON format.
    pub fn save_json(&self, path: &str) -> Result<(), DataError> {
        let mut weights = std::collections::BTreeMap::new();
        let mut activations = Vec::new();

        for (i, layer) in self.layers.iter().enumerate() {
            weights.insert(
                format!("layer_{}", i),
                NnLayerWeights {
                    w: layer.w.clone(),
                    b: layer.b.clone(),
                },
            );
            activations.push(layer.activation);
        }

        let file = NnJsonFile {
            format_version: 1,
            architecture: NnArchitecture {
                layers: self.layer_sizes.clone(),
                activations,
            },
            weights,
            output_interpretation: self.output_interpretation.clone(),
        };

        let json = serde_json::to_string_pretty(&file)
            .map_err(|e| DataError(format!("JSON serialize error: {}", e)))?;
        std::fs::write(path, json)
            .map_err(|e| DataError(format!("Cannot write {}: {}", path, e)))?;

        Ok(())
    }

    /// Generic forward pass through all layers.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut current = input.to_vec();
        for layer in &self.layers {
            let n_out = layer.b.len();
            let mut next = Vec::with_capacity(n_out);
            for j in 0..n_out {
                let sum: f64 = layer.w[j].iter().zip(&current).map(|(w, x)| w * x).sum();
                next.push(layer.activation.apply(sum + layer.b[j]));
            }
            current = next;
        }
        current
    }

    /// Total number of parameters (weights + biases).
    pub fn n_params(&self) -> usize {
        self.layers.iter().map(|l| l.w.len() * l.w[0].len() + l.b.len()).sum()
    }

    /// Flatten all weights and biases into a single vector.
    ///
    /// Order: for each layer, all weights (row-major) then all biases.
    pub fn to_flat_weights(&self) -> Vec<f64> {
        let mut flat = Vec::with_capacity(self.n_params());
        for layer in &self.layers {
            for row in &layer.w {
                flat.extend_from_slice(row);
            }
            flat.extend_from_slice(&layer.b);
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

        let mut idx = 0;
        let mut layers = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let n_in = layer_sizes[i];
            let n_out = layer_sizes[i + 1];

            let mut w = Vec::with_capacity(n_out);
            for _ in 0..n_out {
                if idx + n_in > weights.len() {
                    return Err(DataError("Weight vector too short".to_string()));
                }
                w.push(weights[idx..idx + n_in].to_vec());
                idx += n_in;
            }

            if idx + n_out > weights.len() {
                return Err(DataError("Weight vector too short for biases".to_string()));
            }
            let b = weights[idx..idx + n_out].to_vec();
            idx += n_out;

            layers.push(Layer {
                w,
                b,
                activation: activations[i],
            });
        }

        Ok(NeuralNetModel {
            layer_sizes: layer_sizes.to_vec(),
            layers,
            output_interpretation: "atan2".to_string(),
        })
    }
}
