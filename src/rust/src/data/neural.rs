//! Neural network model with modular architecture.
//!
//! Supports arbitrary layer configurations (e.g. [6, 12, 2] or [6, 24, 12, 2])
//! with per-layer activation function choice. Loads from JSON format.

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
    Swish,
    Mish,
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
    #[serde(default)]
    input_mask: Option<Vec<usize>>,
    #[serde(default)]
    ablated_input: Option<usize>,
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

/// Total number of candidate NN inputs (16 existing + 7 new).
pub const NN_FULL_INPUT_SIZE: usize = 23;

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
    /// Optional input selection mask: indices into the full 23-input vector.
    /// Length must equal layer_sizes[0]. None means use inputs as-is.
    pub input_mask: Option<Vec<usize>>,
    /// Optional index of a single input to zero out (ablation analysis).
    /// Must be in [0, NN_FULL_INPUT_SIZE). None means no ablation.
    pub ablated_input: Option<usize>,
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
        Self::from_json(&content, path)
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
            let lw = file
                .weights
                .get(&key)
                .ok_or_else(|| DataError(format!("Missing {} in weights in {}", key, path)))?;

            let n_out = file.architecture.layers[i + 1];
            let n_in = file.architecture.layers[i];

            if lw.w.len() != n_out || lw.b.len() != n_out {
                return Err(DataError(format!(
                    "Layer {} size mismatch: expected {}x{}, got w={}x?, b={} in {}",
                    i,
                    n_out,
                    n_in,
                    lw.w.len(),
                    lw.b.len(),
                    path
                )));
            }

            layers.push(Layer {
                w: lw.w.clone(),
                b: lw.b.clone(),
                activation: file.architecture.activations[i],
            });
        }

        Self::validate_mask(&file.input_mask, file.architecture.layers[0])?;
        Self::validate_ablated_input(&file.ablated_input)?;

        let output_size = *file.architecture.layers.last().unwrap_or(&0);
        if file.output_interpretation != "direct" && output_size < 2 {
            return Err(DataError(format!(
                "output_interpretation '{}' requires >= 2 outputs, got {} in {}",
                file.output_interpretation, output_size, path
            )));
        }

        Ok(NeuralNetModel {
            layer_sizes: file.architecture.layers,
            layers,
            output_interpretation: file.output_interpretation,
            input_mask: file.input_mask,
            ablated_input: file.ablated_input,
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
            input_mask: self.input_mask.clone(),
            ablated_input: self.ablated_input,
        };

        let json = serde_json::to_string_pretty(&file)
            .map_err(|e| DataError(format!("JSON serialize error: {}", e)))?;
        std::fs::write(path, json)
            .map_err(|e| DataError(format!("Cannot write {}: {}", path, e)))?;

        Ok(())
    }

    /// Generic forward pass through all layers.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        assert_eq!(
            input.len(),
            self.layer_sizes[0],
            "NN input length ({}) does not match expected input size ({})",
            input.len(),
            self.layer_sizes[0],
        );
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
        self.layers
            .iter()
            .map(|l| l.w.len() * l.w[0].len() + l.b.len())
            .sum()
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
            input_mask: None,
            ablated_input: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid NeuralNetModel with a given input size.
    fn make_model(input_size: usize) -> NeuralNetModel {
        NeuralNetModel {
            layer_sizes: vec![input_size, 4, 2],
            layers: vec![
                Layer {
                    w: vec![vec![0.1; input_size]; 4],
                    b: vec![0.0; 4],
                    activation: Activation::Tanh,
                },
                Layer {
                    w: vec![vec![0.1; 4]; 2],
                    b: vec![0.0; 2],
                    activation: Activation::Linear,
                },
            ],
            output_interpretation: "atan2".to_string(),
            input_mask: None,
            ablated_input: None,
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
        // index 23 == NN_FULL_INPUT_SIZE, which is out of range
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
        // index 22 is the last valid index (NN_FULL_INPUT_SIZE - 1)
        let result = NeuralNetModel::validate_ablated_input(&Some(22));
        assert!(result.is_ok());
    }
}
