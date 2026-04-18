//! Neural network model with modular architecture.
//!
//! Supports arbitrary layer configurations (e.g. [6, 12, 2] or [6, 24, 12, 2])
//! with per-layer activation function choice. Loads from JSON format.

use super::DataError;
use crate::data::nn_state::{LayerState, NnState};
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

/// Parse an activation name string into the Activation enum.
/// Uses serde's Activation deserialize so the canonical set of names
/// matches Activation's #[serde(rename_all = "snake_case")] derive.
pub fn parse_activation(s: &str) -> Result<Activation, DataError> {
    serde_json::from_str::<Activation>(&format!("\"{}\"", s))
        .map_err(|e| DataError(format!("parse_activation({:?}): {}", s, e)))
}

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

/// Dot product `row . vec + bias`. Helper for per-gate pre-activation sums.
#[inline]
fn dot_plus_bias(row: &[f64], vec: &[f64], bias: f64) -> f64 {
    bias + row.iter().zip(vec).map(|(w, v)| w * v).sum::<f64>()
}

impl GruLayer {
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

/// Layer variant. Phase 1 ships Dense and Gru (added in Task 2).
#[derive(Debug, Clone)]
pub enum Layer {
    Dense(DenseLayer),
    Gru(GruLayer),
    // Phases 2-4 add: Lstm, Attention, LayerNorm, Ssm, Window
}

impl Layer {
    /// Input size of this layer (for forward-pass shape checks).
    pub fn input_size(&self) -> usize {
        match self {
            Layer::Dense(d) => {
                if d.w.is_empty() {
                    0
                } else {
                    d.w[0].len()
                }
            }
            Layer::Gru(g) => g.input_size,
        }
    }
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

impl LayerWeights for DenseLayer {
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

impl LayerWeights for GruLayer {
    fn to_flat(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.n_params());
        for row in &self.weight_ih {
            v.extend_from_slice(row);
        }
        for row in &self.weight_hh {
            v.extend_from_slice(row);
        }
        v.extend_from_slice(&self.bias_ih);
        v.extend_from_slice(&self.bias_hh);
        v
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_flat(&mut self, flat: &[f64]) -> usize {
        let three_h = 3 * self.hidden_size;
        let mut idx = 0;
        for row in self.weight_ih.iter_mut() {
            row.copy_from_slice(&flat[idx..idx + self.input_size]);
            idx += self.input_size;
        }
        for row in self.weight_hh.iter_mut() {
            row.copy_from_slice(&flat[idx..idx + self.hidden_size]);
            idx += self.hidden_size;
        }
        self.bias_ih.copy_from_slice(&flat[idx..idx + three_h]);
        idx += three_h;
        self.bias_hh.copy_from_slice(&flat[idx..idx + three_h]);
        idx += three_h;
        idx
    }

    fn n_params(&self) -> usize {
        3 * self.hidden_size * self.input_size
            + 3 * self.hidden_size * self.hidden_size
            + 2 * 3 * self.hidden_size
    }
}

impl LayerWeights for Layer {
    fn to_flat(&self) -> Vec<f64> {
        match self {
            Layer::Dense(d) => d.to_flat(),
            Layer::Gru(g) => g.to_flat(),
        }
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_flat(&mut self, flat: &[f64]) -> usize {
        match self {
            Layer::Dense(d) => d.from_flat(flat),
            Layer::Gru(g) => g.from_flat(flat),
        }
    }

    fn n_params(&self) -> usize {
        match self {
            Layer::Dense(d) => d.n_params(),
            Layer::Gru(g) => g.n_params(),
        }
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
    #[serde(skip_serializing_if = "Option::is_none", default)]
    w: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    b: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    weight_ih: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    weight_hh: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    bias_ih: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    bias_hh: Option<Vec<f64>>,
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
    Gru {
        input_size: usize,
        hidden_size: usize,
    },
    // Phases 2+: Lstm, Attention, LayerNorm, Ssm, Window
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

            let w =
                lw.w.as_ref()
                    .ok_or_else(|| DataError(format!("Layer {} missing w in {}", i, path)))?;
            let b =
                lw.b.as_ref()
                    .ok_or_else(|| DataError(format!("Layer {} missing b in {}", i, path)))?;

            if w.len() != n_out || b.len() != n_out {
                return Err(DataError(format!(
                    "Layer {} size mismatch: expected {}x{}, got w={}x?, b={} in {}",
                    i,
                    n_out,
                    n_in,
                    w.len(),
                    b.len(),
                    path
                )));
            }

            layers.push(Layer::Dense(DenseLayer {
                w: w.clone(),
                b: b.clone(),
                activation: file.architecture.activations[i],
            }));
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

        let activations = file.architecture.activations;
        let layer_sizes = file.architecture.layers;
        let architecture: Vec<LayerSpec> = (0..layers.len())
            .map(|i| LayerSpec::Dense {
                input_size: layer_sizes[i],
                output_size: layer_sizes[i + 1],
                activation: activations[i],
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

                    let w = lw
                        .w
                        .as_ref()
                        .ok_or_else(|| DataError(format!("Layer {} missing w in {}", i, path)))?;
                    let b = lw
                        .b
                        .as_ref()
                        .ok_or_else(|| DataError(format!("Layer {} missing b in {}", i, path)))?;

                    if w.len() != *output_size || b.len() != *output_size {
                        return Err(DataError(format!(
                            "Layer {} size mismatch: expected {}x{}, got w={}x?, b={} in {}",
                            i,
                            output_size,
                            input_size,
                            w.len(),
                            b.len(),
                            path
                        )));
                    }
                    for (row_idx, row) in w.iter().enumerate() {
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

                    layers.push(Layer::Dense(DenseLayer {
                        w: w.clone(),
                        b: b.clone(),
                        activation: *activation,
                    }));
                }
                LayerSpec::Gru {
                    input_size,
                    hidden_size,
                } => {
                    if i == 0 {
                        layer_sizes.push(*input_size);
                    }
                    layer_sizes.push(*hidden_size);
                    let three_h = 3 * hidden_size;

                    let key = format!("layer_{}", i);
                    let lw = file.weights.get(&key).ok_or_else(|| {
                        DataError(format!("Missing {} in weights in {}", key, path))
                    })?;

                    let w_ih = lw.weight_ih.as_ref().ok_or_else(|| {
                        DataError(format!("Layer {} (gru) missing weight_ih in {}", i, path))
                    })?;
                    let w_hh = lw.weight_hh.as_ref().ok_or_else(|| {
                        DataError(format!("Layer {} (gru) missing weight_hh in {}", i, path))
                    })?;
                    let b_ih = lw.bias_ih.as_ref().ok_or_else(|| {
                        DataError(format!("Layer {} (gru) missing bias_ih in {}", i, path))
                    })?;
                    let b_hh = lw.bias_hh.as_ref().ok_or_else(|| {
                        DataError(format!("Layer {} (gru) missing bias_hh in {}", i, path))
                    })?;

                    if w_ih.len() != three_h {
                        return Err(DataError(format!(
                            "Layer {} (gru) weight_ih must have {} rows, got {} in {}",
                            i,
                            three_h,
                            w_ih.len(),
                            path
                        )));
                    }
                    if w_hh.len() != three_h {
                        return Err(DataError(format!(
                            "Layer {} (gru) weight_hh must have {} rows, got {} in {}",
                            i,
                            three_h,
                            w_hh.len(),
                            path
                        )));
                    }
                    if b_ih.len() != three_h || b_hh.len() != three_h {
                        return Err(DataError(format!(
                            "Layer {} (gru) biases must each have {} elements in {} (got bias_ih={}, bias_hh={})",
                            i,
                            three_h,
                            path,
                            b_ih.len(),
                            b_hh.len()
                        )));
                    }
                    for (r, row) in w_ih.iter().enumerate() {
                        if row.len() != *input_size {
                            return Err(DataError(format!(
                                "Layer {} (gru) weight_ih row {} length: expected {}, got {} in {}",
                                i,
                                r,
                                input_size,
                                row.len(),
                                path
                            )));
                        }
                    }
                    for (r, row) in w_hh.iter().enumerate() {
                        if row.len() != *hidden_size {
                            return Err(DataError(format!(
                                "Layer {} (gru) weight_hh row {} length: expected {}, got {} in {}",
                                i,
                                r,
                                hidden_size,
                                row.len(),
                                path
                            )));
                        }
                    }

                    layers.push(Layer::Gru(GruLayer {
                        input_size: *input_size,
                        hidden_size: *hidden_size,
                        weight_ih: w_ih.clone(),
                        weight_hh: w_hh.clone(),
                        bias_ih: b_ih.clone(),
                        bias_hh: b_hh.clone(),
                    }));
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
            let entry = match layer {
                Layer::Dense(d) => NnLayerWeights {
                    w: Some(d.w.clone()),
                    b: Some(d.b.clone()),
                    weight_ih: None,
                    weight_hh: None,
                    bias_ih: None,
                    bias_hh: None,
                },
                Layer::Gru(g) => NnLayerWeights {
                    w: None,
                    b: None,
                    weight_ih: Some(g.weight_ih.clone()),
                    weight_hh: Some(g.weight_hh.clone()),
                    bias_ih: Some(g.bias_ih.clone()),
                    bias_hh: Some(g.bias_hh.clone()),
                },
            };
            weights.insert(format!("layer_{}", i), entry);
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
        for (layer, layer_state) in self.layers.iter().zip(state.layer_states.iter_mut()) {
            // Matches (Layer, LayerState) pairs. Construction invariant from
            // NnState::for_model: Dense pairs with None, Gru pairs with Gru{h}.
            // The catch-all below catches mismatches caused by future refactors
            // that accidentally break the invariant.
            match (layer, layer_state) {
                (Layer::Dense(d), LayerState::None) => {
                    let n_out = d.b.len();
                    let mut next = Vec::with_capacity(n_out);
                    for j in 0..n_out {
                        let sum: f64 = d.w[j].iter().zip(&current).map(|(w, x)| w * x).sum();
                        next.push(d.activation.apply(sum + d.b[j]));
                    }
                    current = next;
                }
                (Layer::Gru(g), LayerState::Gru { h }) => {
                    let h_new = g.forward(h, &current);
                    *h = h_new.clone();
                    current = h_new;
                }
                _ => unreachable!(
                    "layer/state variant mismatch (construction invariant -- LayerState::for_layer maps Layer::Dense -> None and Layer::Gru -> Gru)"
                ),
            }
        }
        current
    }

    /// Total number of parameters (weights + biases).
    pub fn n_params(&self) -> usize {
        self.layers.iter().map(|l| l.n_params()).sum()
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
            let mut layer = Layer::Dense(DenseLayer {
                w: vec![vec![0.0; n_in]; n_out],
                b: vec![0.0; n_out],
                activation: activations[i],
            });
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

    /// Construct a NeuralNetModel from a flat weight vector and v2 architecture spec.
    /// Used by the PyO3 flat_weights_to_json helper (Task 7) that routes PSO output
    /// through Rust. Unlike `from_flat_weights` (the v1 wrapper), this accepts
    /// heterogeneous architectures via `LayerSpec`.
    pub fn from_flat_weights_v2(
        flat: &[f64],
        architecture: &[LayerSpec],
        output_interpretation: &str,
        input_mask: Option<Vec<usize>>,
    ) -> Result<Self, DataError> {
        if architecture.is_empty() {
            return Err(DataError(
                "from_flat_weights_v2: empty architecture".to_string(),
            ));
        }
        let mut layers: Vec<Layer> = Vec::with_capacity(architecture.len());
        let mut layer_sizes: Vec<usize> = Vec::with_capacity(architecture.len() + 1);
        let mut offset: usize = 0;

        for (i, spec) in architecture.iter().enumerate() {
            let mut layer = match spec {
                LayerSpec::Dense {
                    input_size,
                    output_size,
                    activation,
                } => {
                    if i == 0 {
                        layer_sizes.push(*input_size);
                    }
                    layer_sizes.push(*output_size);
                    Layer::Dense(DenseLayer {
                        w: vec![vec![0.0; *input_size]; *output_size],
                        b: vec![0.0; *output_size],
                        activation: *activation,
                    })
                }
                LayerSpec::Gru {
                    input_size,
                    hidden_size,
                } => {
                    if i == 0 {
                        layer_sizes.push(*input_size);
                    }
                    layer_sizes.push(*hidden_size);
                    let three_h = 3 * hidden_size;
                    Layer::Gru(GruLayer {
                        input_size: *input_size,
                        hidden_size: *hidden_size,
                        weight_ih: vec![vec![0.0; *input_size]; three_h],
                        weight_hh: vec![vec![0.0; *hidden_size]; three_h],
                        bias_ih: vec![0.0; three_h],
                        bias_hh: vec![0.0; three_h],
                    })
                }
            };
            let needed = layer.n_params();
            if offset + needed > flat.len() {
                return Err(DataError(format!(
                    "from_flat_weights_v2: layer {} needs {} params but only {} remaining (total flat len {})",
                    i,
                    needed,
                    flat.len() - offset,
                    flat.len()
                )));
            }
            let consumed = layer.from_flat(&flat[offset..]);
            offset += consumed;
            layers.push(layer);
        }

        if offset != flat.len() {
            return Err(DataError(format!(
                "from_flat_weights_v2: weight vector length mismatch, consumed {} of {}",
                offset,
                flat.len()
            )));
        }

        Self::validate_mask(&input_mask, layer_sizes[0])?;

        let output_size = *layer_sizes.last().unwrap();
        if output_interpretation != "direct" && output_size < 2 {
            return Err(DataError(format!(
                "output_interpretation '{}' requires >= 2 outputs, got {}",
                output_interpretation, output_size
            )));
        }

        Ok(NeuralNetModel {
            architecture: architecture.to_vec(),
            layer_sizes,
            layers,
            output_interpretation: output_interpretation.to_string(),
            input_mask,
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
                Layer::Dense(DenseLayer {
                    w: vec![vec![0.1; input_size]; 4],
                    b: vec![0.0; 4],
                    activation: Activation::Tanh,
                }),
                Layer::Dense(DenseLayer {
                    w: vec![vec![0.1; 4]; 2],
                    b: vec![0.0; 2],
                    activation: Activation::Linear,
                }),
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
                Layer::Dense(DenseLayer {
                    w: vec![
                        vec![0.1, 0.2, 0.3, 0.4],
                        vec![0.5, 0.6, 0.7, 0.8],
                        vec![-0.1, -0.2, -0.3, -0.4],
                    ],
                    b: vec![0.01, 0.02, 0.03],
                    activation: Activation::Tanh,
                }),
                Layer::Dense(DenseLayer {
                    w: vec![vec![0.1, 0.2, 0.3], vec![-0.1, -0.2, -0.3]],
                    b: vec![0.1, -0.1],
                    activation: Activation::Linear,
                }),
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
    fn gru_forward_known_output() {
        // Minimal 2-input, 2-hidden GRU with all-zero weights + biases.
        // r = sigmoid(0) = 0.5, z = sigmoid(0) = 0.5, n = tanh(0 + 0.5 * 0) = 0.
        // h_new[i] = (1 - 0.5) * 0 + 0.5 * h_prev[i] = 0.5 * h_prev[i].
        let gru = GruLayer {
            input_size: 2,
            hidden_size: 2,
            weight_ih: vec![vec![0.0, 0.0]; 6], // 3H=6 rows, 2 cols
            weight_hh: vec![vec![0.0, 0.0]; 6], // 3H=6 rows, 2 cols
            bias_ih: vec![0.0; 6],
            bias_hh: vec![0.0; 6],
        };
        let h_prev = vec![1.0, 2.0];
        let x = vec![0.5, -0.5];
        let h_new = gru.forward(&h_prev, &x);
        assert!((h_new[0] - 0.5).abs() < 1e-12);
        assert!((h_new[1] - 1.0).abs() < 1e-12);
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

    #[test]
    fn gru_flat_weights_roundtrip() {
        // Build a GruLayer with distinct weight values so a buggy to_flat/from_flat
        // would produce visible mismatches.
        let input_size = 2;
        let hidden_size = 3;
        let three_h = 3 * hidden_size;
        let mut w_ih = Vec::with_capacity(three_h);
        let mut w_hh = Vec::with_capacity(three_h);
        for i in 0..three_h {
            w_ih.push(
                (0..input_size)
                    .map(|k| (i * 10 + k) as f64 * 0.01)
                    .collect(),
            );
            w_hh.push(
                (0..hidden_size)
                    .map(|k| (i * 10 + k) as f64 * 0.001)
                    .collect(),
            );
        }
        let b_ih: Vec<f64> = (0..three_h).map(|i| 0.1 + i as f64 * 0.01).collect();
        let b_hh: Vec<f64> = (0..three_h).map(|i| 0.2 + i as f64 * 0.01).collect();

        let original = GruLayer {
            input_size,
            hidden_size,
            weight_ih: w_ih,
            weight_hh: w_hh,
            bias_ih: b_ih,
            bias_hh: b_hh,
        };

        let flat = original.to_flat();
        assert_eq!(flat.len(), original.n_params());

        // Reconstruct an empty-shaped GruLayer and fill via from_flat.
        let mut twin = GruLayer {
            input_size,
            hidden_size,
            weight_ih: vec![vec![0.0; input_size]; three_h],
            weight_hh: vec![vec![0.0; hidden_size]; three_h],
            bias_ih: vec![0.0; three_h],
            bias_hh: vec![0.0; three_h],
        };
        let consumed = twin.from_flat(&flat);
        assert_eq!(consumed, flat.len());

        // Forward outputs must match on a fixed input.
        let h_prev = vec![0.1, -0.2, 0.3];
        let x = vec![0.5, -0.4];
        let out_orig = original.forward(&h_prev, &x);
        let out_twin = twin.forward(&h_prev, &x);
        for (a, b) in out_orig.iter().zip(out_twin.iter()) {
            assert!((a - b).abs() < 1e-15, "{} vs {}", a, b);
        }
    }

    #[test]
    fn v2_gru_json_roundtrip() {
        let input_size = 2;
        let hidden_size = 3;
        let three_h = 9;
        let gru = GruLayer {
            input_size,
            hidden_size,
            weight_ih: (0..three_h)
                .map(|i| (0..input_size).map(|k| (i + k) as f64 * 0.01).collect())
                .collect(),
            weight_hh: (0..three_h)
                .map(|i| (0..hidden_size).map(|k| (i + k) as f64 * 0.02).collect())
                .collect(),
            bias_ih: (0..three_h).map(|i| 0.1 + i as f64 * 0.01).collect(),
            bias_hh: (0..three_h).map(|i| 0.2 + i as f64 * 0.01).collect(),
        };
        let original = NeuralNetModel {
            architecture: vec![LayerSpec::Gru {
                input_size,
                hidden_size,
            }],
            layer_sizes: vec![input_size, hidden_size],
            layers: vec![Layer::Gru(gru)],
            output_interpretation: "atan2".to_string(),
            input_mask: None,
            ablated_input: None,
        };

        let tmpdir = std::env::temp_dir();
        let path = tmpdir.join("gru_roundtrip.json");
        original.save_json(path.to_str().unwrap()).unwrap();

        let loaded = NeuralNetModel::load(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.layers.len(), 1);
        match &loaded.layers[0] {
            Layer::Gru(g) => {
                assert_eq!(g.input_size, input_size);
                assert_eq!(g.hidden_size, hidden_size);
            }
            _ => panic!("expected Gru layer"),
        }
        // Forward parity
        use crate::data::nn_state::NnState;
        let mut s0 = NnState::for_model(&original);
        let mut s1 = NnState::for_model(&loaded);
        let x = vec![0.3, -0.4];
        let o0 = original.forward(&mut s0, &x);
        let o1 = loaded.forward(&mut s1, &x);
        for (a, b) in o0.iter().zip(o1.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn from_flat_weights_v2_mixed_arch() {
        use crate::data::nn_state::NnState;

        // Dense(3->4,tanh) + Gru(4->4) + Dense(4->2,linear)
        let architecture = vec![
            LayerSpec::Dense {
                input_size: 3,
                output_size: 4,
                activation: Activation::Tanh,
            },
            LayerSpec::Gru {
                input_size: 4,
                hidden_size: 4,
            },
            LayerSpec::Dense {
                input_size: 4,
                output_size: 2,
                activation: Activation::Linear,
            },
        ];
        // Per-layer n_params:
        //   Dense 3->4: 3*4 + 4 = 16
        //   Gru H=4, I=4: 3*4*4 + 3*4*4 + 2*3*4 = 48 + 48 + 24 = 120
        //   Dense 4->2: 4*2 + 2 = 10
        // Total: 146
        let flat: Vec<f64> = (0..146).map(|i| 0.001 * i as f64).collect();
        let model =
            NeuralNetModel::from_flat_weights_v2(&flat, &architecture, "atan2", None).unwrap();
        assert_eq!(model.layers.len(), 3);
        assert_eq!(model.layer_sizes, vec![3, 4, 4, 2]);

        // Forward pass produces finite output.
        let mut state = NnState::for_model(&model);
        let out = model.forward(&mut state, &[0.1, 0.2, 0.3]);
        assert_eq!(out.len(), 2);
        for v in out.iter() {
            assert!(v.is_finite());
        }

        // JSON save/load roundtrip for the mixed Dense+Gru+Dense arch.
        // Catches copy-paste swaps between Dense and Gru serialization arms
        // that would survive the forward-is-finite check but diverge here.
        let tmpdir = std::env::temp_dir();
        let path = tmpdir.join("v2_mixed_arch_roundtrip.json");
        model.save_json(path.to_str().unwrap()).unwrap();
        let reloaded = NeuralNetModel::load(path.to_str().unwrap()).unwrap();
        let mut state2 = NnState::for_model(&reloaded);
        let out_reloaded = reloaded.forward(&mut state2, &[0.1, 0.2, 0.3]);
        for (a, b) in out.iter().zip(out_reloaded.iter()) {
            assert!(
                (a - b).abs() < 1e-15,
                "mixed-arch JSON roundtrip: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn from_flat_weights_v2_length_mismatch() {
        let architecture = vec![LayerSpec::Dense {
            input_size: 3,
            output_size: 4,
            activation: Activation::Tanh,
        }];
        // Dense 3->4 needs 16 params. Too short should Err.
        let flat = vec![0.0; 10];
        let err = NeuralNetModel::from_flat_weights_v2(&flat, &architecture, "atan2", None);
        assert!(err.is_err());
        // Too long should also Err.
        let flat = vec![0.0; 20];
        let err = NeuralNetModel::from_flat_weights_v2(&flat, &architecture, "atan2", None);
        assert!(err.is_err());
    }
}
