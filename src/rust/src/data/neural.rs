//! Neural network model with modular architecture.
//!
//! Supports arbitrary layer configurations (e.g. [6, 12, 2] or [6, 24, 12, 2])
//! with per-layer activation function choice. Loads from JSON format.

use super::DataError;
use crate::data::nn_state::NnState;
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

impl LayerWeights for Layer {
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

/// JSON file structure for neural network models (v1 schema).
#[derive(Debug, Clone, Deserialize)]
struct NnJsonFile {
    #[allow(dead_code)]
    format_version: u32,
    architecture: NnArchitecture,
    weights: std::collections::BTreeMap<String, NnLayerWeights>,
    output_interpretation: String,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NnLayerWeights {
    w: Vec<Vec<f64>>,
    b: Vec<f64>,
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
    // Phase 1+: Gru, Lstm, Attention, LayerNorm, Ssm, Window
}

/// JSON file structure for neural network models (v2 schema).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct NnJsonFileV2 {
    format_version: u32,
    architecture: Vec<LayerSpec>,
    weights: std::collections::BTreeMap<String, NnLayerWeights>,
    output_interpretation: String,
    #[serde(default)]
    input_mask: Option<Vec<usize>>,
    #[serde(default)]
    ablated_input: Option<usize>,
}

/// Total number of candidate NN inputs (16 existing + 7 new).
pub const NN_FULL_INPUT_SIZE: usize = 23;

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

        let layer_sizes = file.architecture.layers;
        let architecture: Vec<LayerSpec> = (0..layers.len())
            .map(|i| LayerSpec::Dense {
                input_size: layer_sizes[i],
                output_size: layer_sizes[i + 1],
                activation: layers[i].activation,
            })
            .collect();

        Ok(NeuralNetModel {
            architecture,
            layer_sizes,
            layers,
            output_interpretation: file.output_interpretation,
            input_mask: file.input_mask,
            ablated_input: file.ablated_input,
        })
    }

    /// Load v2 JSON schema (architecture is a tagged-layer list).
    fn from_v2_json(content: &str, path: &str) -> Result<Self, DataError> {
        let file: NnJsonFileV2 = serde_json::from_str(content)
            .map_err(|e| DataError(format!("JSON parse error in {}: {}", path, e)))?;

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

                    if lw.w.len() != *output_size || lw.b.len() != *output_size {
                        return Err(DataError(format!(
                            "Layer {} size mismatch: expected {}x{}, got w={}x?, b={} in {}",
                            i,
                            output_size,
                            input_size,
                            lw.w.len(),
                            lw.b.len(),
                            path
                        )));
                    }
                    for (row_idx, row) in lw.w.iter().enumerate() {
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

                    layers.push(Layer {
                        w: lw.w.clone(),
                        b: lw.b.clone(),
                        activation: *activation,
                    });
                }
            }
        }

        Self::validate_mask(&file.input_mask, layer_sizes[0])?;
        Self::validate_ablated_input(&file.ablated_input)?;

        let output_size = *layer_sizes.last().unwrap_or(&0);
        if file.output_interpretation != "direct" && output_size < 2 {
            return Err(DataError(format!(
                "output_interpretation '{}' requires >= 2 outputs, got {} in {}",
                file.output_interpretation, output_size, path
            )));
        }

        Ok(NeuralNetModel {
            architecture: file.architecture,
            layer_sizes,
            layers,
            output_interpretation: file.output_interpretation,
            input_mask: file.input_mask,
            ablated_input: file.ablated_input,
        })
    }

    /// Save to JSON format (v2 schema: tagged-layer list).
    pub fn save_json(&self, path: &str) -> Result<(), DataError> {
        let mut weights = std::collections::BTreeMap::new();

        for (i, layer) in self.layers.iter().enumerate() {
            weights.insert(
                format!("layer_{}", i),
                NnLayerWeights {
                    w: layer.w.clone(),
                    b: layer.b.clone(),
                },
            );
        }

        let file = NnJsonFileV2 {
            format_version: 2,
            architecture: self.architecture.clone(),
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
        for (layer, _layer_state) in self.layers.iter().zip(state.layer_states.iter_mut()) {
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
            let mut layer = Layer {
                w: vec![vec![0.0; n_in]; n_out],
                b: vec![0.0; n_out],
                activation: activations[i],
            };
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
                Layer {
                    w: vec![
                        vec![0.1, 0.2, 0.3, 0.4],
                        vec![0.5, 0.6, 0.7, 0.8],
                        vec![-0.1, -0.2, -0.3, -0.4],
                    ],
                    b: vec![0.01, 0.02, 0.03],
                    activation: Activation::Tanh,
                },
                Layer {
                    w: vec![vec![0.1, 0.2, 0.3], vec![-0.1, -0.2, -0.3]],
                    b: vec![0.1, -0.1],
                    activation: Activation::Linear,
                },
            ],
            output_interpretation: "atan2".to_string(),
            input_mask: None,
            ablated_input: None,
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
}
