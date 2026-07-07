//! Neural network model with modular architecture.
//!
//! Supports arbitrary layer configurations (e.g. [6, 12, 2] or [6, 24, 12, 2])
//! with per-layer activation function choice. Loads from JSON format.

use super::DataError;
use crate::data::nn_state::{LayerState, NnState};
use serde::{Deserialize, Serialize};

mod layers;
pub use layers::{
    DenseLayer, GruLayer, LstmLayer, Mamba3Layer, MambaLayer, TransformerLayer, WindowLayer,
};
// Surface the shared numerical helpers at the module root so the `use super::*`
// test module reaches them by their bare names. Test-only: production code in
// this module never calls the helpers directly (the layer impls that do live in
// `layers::*` and import from `layers::helpers`).
#[cfg(test)]
use layers::helpers::{build_pe_table, expm1_over_x, gelu_exact, layer_norm_biased, softplus};

/// Per-input normalization transform applied after the affine `(raw - center)/scale`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NormTransform {
    #[default]
    None,
    Asinh,
    Tanh,
}

/// Uniform per-input normalization: `norm = transform((raw - center) / scale)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NormSpec {
    pub transform: NormTransform,
    pub scale: f64,
    pub center: f64,
}

impl Default for NormSpec {
    fn default() -> Self {
        Self {
            transform: NormTransform::None,
            scale: 1.0,
            center: 0.0,
        }
    }
}

#[inline]
pub fn apply_norm(raw: f64, spec: &NormSpec) -> f64 {
    let v = (raw - spec.center) / spec.scale;
    match spec.transform {
        NormTransform::None => v,
        NormTransform::Asinh => v.asinh(),
        NormTransform::Tanh => v.tanh(),
    }
}

/// Default per-input normalization table (divisor form `(raw - center) / scale`).
/// All 35 entries are calibrated, including DV entries 32-34 (smooth, no sentinel).
pub const DEFAULT_NORMALIZATION: [NormSpec; NN_FULL_INPUT_SIZE] = [
    NormSpec {
        transform: NormTransform::None,
        scale: 0.8754754,
        center: 0.9125593,
    }, // 0  ecc_excess
    NormSpec {
        transform: NormTransform::None,
        scale: 1.443277,
        center: -1.167222,
    }, // 1  inclination_error
    NormSpec {
        transform: NormTransform::Asinh,
        scale: 8.794982e2,
        center: 0.0,
    }, // 2  radial_velocity
    NormSpec {
        transform: NormTransform::Asinh,
        scale: 5.180226e6,
        center: 0.0,
    }, // 3  orbital_energy
    NormSpec {
        transform: NormTransform::None,
        scale: 1178.859,
        center: 4534.045,
    }, // 4  velocity
    NormSpec {
        transform: NormTransform::Asinh,
        scale: 2.494108e1,
        center: 0.0,
    }, // 5  accel_magnitude
    NormSpec {
        transform: NormTransform::None,
        scale: 0.4524197,
        center: 0.4533209,
    }, // 6  heat_flux_fraction
    NormSpec {
        transform: NormTransform::None,
        scale: 0.4363704,
        center: 0.4366122,
    }, // 7  heat_load_fraction
    NormSpec {
        transform: NormTransform::None,
        scale: 43.24290,
        center: 82.93086,
    }, // 8  altitude
    NormSpec {
        transform: NormTransform::None,
        scale: 0.1246266,
        center: -0.05801090,
    }, // 9 fpa
    NormSpec {
        transform: NormTransform::None,
        scale: 0.2803614,
        center: 0.2875094,
    }, // 10 latitude
    NormSpec {
        transform: NormTransform::Asinh,
        scale: 2.367649e1,
        center: 0.0,
    }, // 11 drag_accel
    NormSpec {
        transform: NormTransform::Asinh,
        scale: 7.841004e0,
        center: 0.0,
    }, // 12 lift_accel
    NormSpec {
        transform: NormTransform::Asinh,
        scale: 2.396120e7,
        center: 0.0,
    }, // 13 sma_error
    NormSpec {
        transform: NormTransform::Asinh,
        scale: 4.752185e7,
        center: 0.0,
    }, // 14 apoapsis_alt
    NormSpec {
        transform: NormTransform::None,
        scale: 0.5,
        center: 0.5,
    }, // 15 bounce_flag
    NormSpec {
        transform: NormTransform::None,
        scale: 1.0,
        center: 0.0,
    }, // 16 cos_bank_nominal
    NormSpec {
        transform: NormTransform::None,
        scale: 808.8315,
        center: 812.3864,
    }, // 17 pdyn_nominal
    NormSpec {
        transform: NormTransform::Asinh,
        scale: 7.416992e2,
        center: 0.0,
    }, // 18 hdot_nominal
    NormSpec {
        transform: NormTransform::Asinh,
        scale: 3.373053e2,
        center: 0.0,
    }, // 19 pdyn_error
    NormSpec {
        transform: NormTransform::None,
        scale: std::f64::consts::FRAC_PI_2,
        center: std::f64::consts::FRAC_PI_2,
    }, // 20 exit_bank_teacher
    NormSpec {
        transform: NormTransform::None,
        scale: 0.1,
        center: 0.0,
    }, // 21 inclination_err_rate
    NormSpec {
        transform: NormTransform::None,
        scale: std::f64::consts::PI,
        center: 0.0,
    }, // 22 prev_bank_signed
    NormSpec {
        transform: NormTransform::Tanh,
        scale: 30.0,
        center: 0.0,
    }, // 23 time_since_sign_flip
    NormSpec {
        transform: NormTransform::Tanh,
        scale: 100.0,
        center: 0.0,
    }, // 24 inclination_err_integral
    NormSpec {
        transform: NormTransform::None,
        scale: 1.0,
        center: 0.0,
    }, // 25 exit_bank_teacher_sin
    NormSpec {
        transform: NormTransform::None,
        scale: 1.0,
        center: 0.0,
    }, // 26 exit_bank_teacher_cos
    NormSpec {
        transform: NormTransform::None,
        scale: 1.0,
        center: 0.0,
    }, // 27 prev_bank_signed_sin
    NormSpec {
        transform: NormTransform::None,
        scale: 1.0,
        center: 0.0,
    }, // 28 prev_bank_signed_cos
    NormSpec {
        transform: NormTransform::None,
        scale: 1.0,
        center: 0.0,
    }, // 29 prev_realized_sin
    NormSpec {
        transform: NormTransform::None,
        scale: 1.0,
        center: 0.0,
    }, // 30 prev_realized_cos
    NormSpec {
        transform: NormTransform::Asinh,
        scale: 3.750782e4,
        center: 0.0,
    }, // 31 periapsis_alt
    NormSpec {
        transform: NormTransform::Asinh,
        scale: 1.919853e3,
        center: 0.0,
    }, // 32 predicted_dv1 (energy-close; calibrated on the redefined smooth DV)
    NormSpec {
        transform: NormTransform::Asinh,
        scale: 3.846528e2,
        center: 0.0,
    }, // 33 predicted_dv2 (periapsis; calibrated)
    NormSpec {
        transform: NormTransform::Asinh,
        scale: 3.486664e2,
        center: 0.0,
    }, // 34 predicted_dv3 (inclination; calibrated)
];

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

/// Output parameterization for the NN's bank-angle decoder.
///
/// `Atan2Signed` (default, backward-compatible): emits 2 outputs and
/// `bank = atan2(out[0], out[1]) ∈ (-π, π]`.
///
/// `AcosTanh`: emits 1 output through `tanh` and `bank = acos(out[0]) ∈ [0, π]`.
/// Only legal in `magnitude_only` mode (architecture validates last layer
/// `output_size = 1` with activation `tanh`).
///
/// `ScaledPi`: emits 1 tanh output; `bank = scaled_pi_n * π * out[0] ∈ [-n·π, n·π]`.
///
/// `Delta`: emits 1 tanh output; `bank = prev_realized + delta_max * out[0]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputParam {
    #[default]
    Atan2Signed,
    AcosTanh,
    ScaledPi,
    Delta,
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

/// Layer variant. Phase 1 ships Dense and Gru; Phase 2a adds Lstm; Phase 2b adds Window; Phase 3a adds Transformer; Phase 4a adds Mamba.
#[derive(Debug, Clone)]
pub enum Layer {
    Dense(DenseLayer),
    Gru(GruLayer),
    Lstm(LstmLayer),
    Window(WindowLayer),
    // Boxed: TransformerLayer is 472 bytes vs 112 for GruLayer; boxing keeps enum size uniform.
    Transformer(Box<TransformerLayer>),
    // Boxed: MambaLayer's stack footprint is ~200 bytes (3 DMatrix + 2 DVector
    // headers); weight data lives on the heap behind those pointers regardless
    // of boxing. The box is purely for enum-variant size uniformity against
    // Transformer (472 bytes) -- same `large_enum_variant` clippy motivation.
    Mamba(Box<MambaLayer>),
    // Boxed for enum-variant size uniformity, same as Mamba (adds a_imag + lambda_logit).
    Mamba3(Box<Mamba3Layer>),
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
            Layer::Lstm(l) => l.input_size,
            Layer::Window(w) => w.input_size,
            Layer::Transformer(t) => t.d_model,
            Layer::Mamba(m) => m.input_size,
            Layer::Mamba3(m) => m.input_size,
        }
    }
}

/// Trait for flattening and reconstructing a layer's parameters.
///
/// Each layer type implements its own canonical flat ordering:
/// dense = W (row-major) then b; gru/lstm/window/transformer/mamba defined per variant
/// in the respective impl blocks below. Order MUST match the PyTorch mirror in
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
        match self {
            Layer::Dense(d) => d.to_flat(),
            Layer::Gru(g) => g.to_flat(),
            Layer::Lstm(l) => l.to_flat(),
            Layer::Window(w) => w.to_flat(),
            Layer::Transformer(t) => t.to_flat(),
            Layer::Mamba(m) => m.to_flat(),
            Layer::Mamba3(m) => m.to_flat(),
        }
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_flat(&mut self, flat: &[f64]) -> usize {
        match self {
            Layer::Dense(d) => d.from_flat(flat),
            Layer::Gru(g) => g.from_flat(flat),
            Layer::Lstm(l) => l.from_flat(flat),
            Layer::Window(w) => w.from_flat(flat),
            Layer::Transformer(t) => t.from_flat(flat),
            Layer::Mamba(m) => m.from_flat(flat),
            Layer::Mamba3(m) => m.from_flat(flat),
        }
    }

    fn n_params(&self) -> usize {
        match self {
            Layer::Dense(d) => d.n_params(),
            Layer::Gru(g) => g.n_params(),
            Layer::Lstm(l) => l.n_params(),
            Layer::Window(w) => w.n_params(),
            Layer::Transformer(t) => t.n_params(),
            Layer::Mamba(m) => m.n_params(),
            Layer::Mamba3(m) => m.n_params(),
        }
    }
}

/// JSON file structure for neural network models (v1 schema).
/// v1 always loads with `OutputParam::Atan2Signed` (the bank-decoder
/// parameterization is a v2 feature; v1 files predate it). The legacy
/// `output_interpretation` field is silently ignored. Output_size is
/// validated to match the parameterization at load time.
#[derive(Debug, Clone, Deserialize)]
struct NnJsonFile {
    #[allow(dead_code)]
    format_version: u32,
    architecture: NnArchitecture,
    weights: std::collections::BTreeMap<String, NnLayerWeights>,
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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct NnLayerWeights {
    // Dense fields
    #[serde(skip_serializing_if = "Option::is_none", default)]
    w: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    b: Option<Vec<f64>>,
    // GRU / LSTM fields
    #[serde(skip_serializing_if = "Option::is_none", default)]
    weight_ih: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    weight_hh: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    bias_ih: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    bias_hh: Option<Vec<f64>>,
    // Transformer attention projection fields
    #[serde(skip_serializing_if = "Option::is_none", default)]
    w_q: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    b_q: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    w_k: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    b_k: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    w_v: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    b_v: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    w_o: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    b_o: Option<Vec<f64>>,
    // Transformer FFN fields
    #[serde(skip_serializing_if = "Option::is_none", default)]
    w_ffn1: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    b_ffn1: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    w_ffn2: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    b_ffn2: Option<Vec<f64>>,
    // Transformer LayerNorm fields (ln1 / ln2)
    #[serde(skip_serializing_if = "Option::is_none", default)]
    ln1_gamma: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    ln1_beta: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    ln2_gamma: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    ln2_beta: Option<Vec<f64>>,
    // Mamba SSM fields (Phase 4a)
    #[serde(skip_serializing_if = "Option::is_none", default)]
    x_proj_w: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    dt_proj_w: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    dt_proj_b: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    a_log: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    d_skip: Option<Vec<f64>>,
    // Mamba-3 extra fields (spike): a_imag iff complex, lambda_logit iff trapezoidal.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    a_imag: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    lambda_logit: Option<Vec<f64>>,
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
    Lstm {
        input_size: usize,
        hidden_size: usize,
    },
    Window {
        input_size: usize,
        n_steps: usize,
    },
    Transformer {
        d_model: usize,
        n_heads: usize,
        d_ffn: usize,
        n_seq: usize,
    },
    Mamba {
        input_size: usize,
        d_state: usize,
        dt_rank: usize,
    },
    Mamba3 {
        input_size: usize,
        d_state: usize,
        dt_rank: usize,
        #[serde(default = "default_discretization")]
        discretization: String,
        #[serde(default = "default_state_mode")]
        state_mode: String,
    },
}

fn default_discretization() -> String {
    "euler".to_string()
}

fn default_state_mode() -> String {
    "real".to_string()
}

/// Parse the Mamba-3 `discretization`/`state_mode` strings into the runtime
/// `(trapezoidal, complex)` bools. Shared by the JSON and flat-weights load paths
/// so the serialized string interface stays uniform across TOML and JSON.
pub(crate) fn mamba3_flags(discretization: &str, state_mode: &str) -> Result<(bool, bool), String> {
    let trapezoidal = match discretization {
        "euler" => false,
        "trapezoidal" => true,
        other => {
            return Err(format!(
                "discretization must be euler|trapezoidal, got {other:?}"
            ));
        }
    };
    let complex = match state_mode {
        "real" => false,
        "complex" => true,
        other => return Err(format!("state_mode must be real|complex, got {other:?}")),
    };
    Ok((trapezoidal, complex))
}

impl LayerSpec {
    /// Returns `(input_size, output_size, kind_label)` for chain-consistency validation.
    /// - Dense:       (input_size, output_size, "dense")
    /// - Gru/Lstm:    (input_size, hidden_size, "gru"/"lstm")
    /// - Window:      (input_size, n_steps * input_size, "window")
    /// - Transformer: (d_model,    d_model,              "transformer")
    /// - Mamba:       (input_size, input_size,            "mamba")
    fn io(&self) -> (usize, usize, &'static str) {
        match self {
            LayerSpec::Dense {
                input_size,
                output_size,
                ..
            } => (*input_size, *output_size, "dense"),
            LayerSpec::Gru {
                input_size,
                hidden_size,
            } => (*input_size, *hidden_size, "gru"),
            LayerSpec::Lstm {
                input_size,
                hidden_size,
            } => (*input_size, *hidden_size, "lstm"),
            LayerSpec::Window {
                input_size,
                n_steps,
            } => (*input_size, n_steps * input_size, "window"),
            LayerSpec::Transformer { d_model, .. } => (*d_model, *d_model, "transformer"),
            LayerSpec::Mamba { input_size, .. } => (*input_size, *input_size, "mamba"),
            LayerSpec::Mamba3 { input_size, .. } => (*input_size, *input_size, "mamba3"),
        }
    }
}

fn default_scaled_pi_n() -> f64 {
    1.0
}
fn default_delta_max() -> f64 {
    0.35
}

/// JSON file structure for neural network models (v2 schema).
/// `output_param` selects the bank-angle decoder: `Atan2Signed` (default,
/// 2-output `atan2`) or `AcosTanh` (1-output `acos(tanh(x))`, magnitude_only
/// mode only). When absent in older v2 files, defaults to `Atan2Signed`
/// for backward compat. The legacy `output_interpretation` field is silently
/// ignored.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct NnJsonFileV2 {
    format_version: u32,
    architecture: Vec<LayerSpec>,
    weights: std::collections::BTreeMap<String, NnLayerWeights>,
    #[serde(default)]
    input_mask: Option<Vec<usize>>,
    #[serde(default)]
    ablated_input: Option<usize>,
    #[serde(default)]
    ablated_value: f64,
    #[serde(default)]
    output_param: OutputParam,
    #[serde(default = "default_scaled_pi_n")]
    scaled_pi_n: f64,
    #[serde(default = "default_delta_max")]
    delta_max: f64,
    #[serde(default)]
    normalization: Option<Vec<NormSpec>>,
}

/// Total number of candidate NN inputs (16 baseline + 4 reference trajectory + 1 exit-bank teacher + 4 lateral-state telemetry
/// + 6 (sin,cos) bank-history pairs for exit teacher / prev commanded / prev realized + 1 periapsis_alt
/// + 3 live correction-DV components).
pub const NN_FULL_INPUT_SIZE: usize = 35;

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
    /// Optional input selection mask: indices into the full 35-input vector.
    /// Length must equal layer_sizes[0]. None means use inputs as-is.
    pub input_mask: Option<Vec<usize>>,
    /// Optional index of a single input to freeze (ablation analysis).
    /// Must be in [0, NN_FULL_INPUT_SIZE). None means no ablation.
    /// When set, `build_nn_input` overwrites `full_input[ablated_input]` with
    /// `ablated_value` (default 0.0 => classic zero-ablation).
    pub ablated_input: Option<usize>,
    /// Value to freeze the ablated input to. Default 0.0 (zero-ablation).
    /// Used for flip-ablation: freeze a binary ±1 flag to -1 / +1 instead of
    /// an out-of-distribution 0.
    pub ablated_value: f64,
    /// Output parameterization for the bank-angle decoder.
    /// Default: `Atan2Signed` (2-output atan2, backward-compatible).
    pub output_param: OutputParam,
    /// Half-range multiplier for `ScaledPi`: `bank = scaled_pi_n * π * out[0]`.
    pub scaled_pi_n: f64,
    /// Per-step increment bound for `Delta`: `bank = prev_realized + delta_max * out[0]`.
    pub delta_max: f64,
    /// Per-input normalization table (len == NN_FULL_INPUT_SIZE). Resolved from the
    /// JSON `normalization` block when present and well-sized, else `DEFAULT_NORMALIZATION`.
    pub normalization: Vec<NormSpec>,
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

    /// Validate that the network's final layer produces the right number of outputs
    /// for the given `output_param`:
    /// - `Atan2Signed`: requires output_size == 2 (bank = atan2(out[0], out[1]))
    /// - `AcosTanh`:    requires output_size == 1 (bank = acos(tanh(out[0])))
    /// - `ScaledPi`:    requires output_size == 1 (bank = scaled_pi_n * π * tanh(out[0]))
    /// - `Delta`:       requires output_size == 1 (bank = prev_realized + delta_max * tanh(out[0]))
    pub fn validate_output_size(
        output_size: usize,
        output_param: OutputParam,
        path: &str,
    ) -> Result<(), DataError> {
        let expected = match output_param {
            OutputParam::Atan2Signed => 2,
            OutputParam::AcosTanh | OutputParam::ScaledPi | OutputParam::Delta => 1,
        };
        if output_size != expected {
            return Err(DataError(format!(
                "network output_size must be {} for output_param {:?}, got {} in {}",
                expected, output_param, output_size, path
            )));
        }
        Ok(())
    }

    /// Validate that the last layer's activation matches the output_param
    /// constraint. `AcosTanh`, `ScaledPi`, and `Delta` require `Tanh` so that
    /// `output[0] ∈ [-1, 1]`. `Atan2Signed` has no constraint.
    /// Without this guard a hand-crafted (or trainer-bug-produced) v2 JSON with
    /// `output_param: "acos_tanh"` plus `linear`/`asinh`/`swish` last activation
    /// loads silently and emits NaN at runtime when |out[0]| > 1.
    pub fn validate_output_activation(
        last_activation: Activation,
        output_param: OutputParam,
        path: &str,
    ) -> Result<(), DataError> {
        let needs_tanh = matches!(
            output_param,
            OutputParam::AcosTanh | OutputParam::ScaledPi | OutputParam::Delta
        );
        if needs_tanh && last_activation != Activation::Tanh {
            return Err(DataError(format!(
                "output_param={:?} requires last-layer activation=Tanh, got {:?} in {}. \
                 Without tanh, out[0] is unbounded.",
                output_param, last_activation, path
            )));
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

    /// Resolve the per-input normalization table: use the JSON block when present
    /// and correctly sized, else fall back to `DEFAULT_NORMALIZATION`.
    fn resolve_normalization(block: Option<Vec<NormSpec>>) -> Vec<NormSpec> {
        match block {
            Some(v) if v.len() == NN_FULL_INPUT_SIZE => v,
            _ => DEFAULT_NORMALIZATION.to_vec(),
        }
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
        Self::validate_output_size(output_size, OutputParam::default(), path)?;

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
            input_mask: file.input_mask,
            ablated_input: file.ablated_input,
            // v1 schema has no ablated_value; classic zero-ablation.
            ablated_value: 0.0,
            output_param: OutputParam::default(),
            scaled_pi_n: default_scaled_pi_n(),
            delta_max: default_delta_max(),
            // v1 schema has no normalization block; use the default table.
            normalization: Self::resolve_normalization(None),
        })
    }

    /// Load v2 JSON schema (architecture is a tagged-layer list).
    fn from_v2_json(content: &str, path: &str) -> Result<Self, DataError> {
        let file: NnJsonFileV2 = serde_json::from_str(content)
            .map_err(|e| DataError(format!("JSON parse error in {}: {}", path, e)))?;

        // Chain consistency: layer i's output must feed layer i+1's input.
        // Dense: output_size -> next.input_size; Gru/Lstm: hidden_size -> next.input_size;
        // Window: n_steps * input_size -> next.input_size (zero-param buffer flatten).
        for i in 0..file.architecture.len().saturating_sub(1) {
            let (_, prev_out, prev_label) = file.architecture[i].io();
            let (next_in, _, next_label) = file.architecture[i + 1].io();
            if prev_out != next_in {
                return Err(DataError(format!(
                    "architecture chain mismatch at layer {}->{} in {}: layer {} ({}) produces output={}, but layer {} ({}) expects input={}",
                    i,
                    i + 1,
                    path,
                    i,
                    prev_label,
                    prev_out,
                    i + 1,
                    next_label,
                    next_in
                )));
            }
        }

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
                LayerSpec::Lstm {
                    input_size,
                    hidden_size,
                } => {
                    if i == 0 {
                        layer_sizes.push(*input_size);
                    }
                    layer_sizes.push(*hidden_size);
                    let four_h = 4 * hidden_size;

                    let key = format!("layer_{}", i);
                    let lw = file.weights.get(&key).ok_or_else(|| {
                        DataError(format!("Missing {} in weights in {}", key, path))
                    })?;

                    let w_ih = lw.weight_ih.as_ref().ok_or_else(|| {
                        DataError(format!("Layer {} (lstm) missing weight_ih in {}", i, path))
                    })?;
                    let w_hh = lw.weight_hh.as_ref().ok_or_else(|| {
                        DataError(format!("Layer {} (lstm) missing weight_hh in {}", i, path))
                    })?;
                    let b_ih = lw.bias_ih.as_ref().ok_or_else(|| {
                        DataError(format!("Layer {} (lstm) missing bias_ih in {}", i, path))
                    })?;
                    let b_hh = lw.bias_hh.as_ref().ok_or_else(|| {
                        DataError(format!("Layer {} (lstm) missing bias_hh in {}", i, path))
                    })?;

                    if w_ih.len() != four_h {
                        return Err(DataError(format!(
                            "Layer {} (lstm) weight_ih must have {} rows, got {} in {}",
                            i,
                            four_h,
                            w_ih.len(),
                            path
                        )));
                    }
                    if w_hh.len() != four_h {
                        return Err(DataError(format!(
                            "Layer {} (lstm) weight_hh must have {} rows, got {} in {}",
                            i,
                            four_h,
                            w_hh.len(),
                            path
                        )));
                    }
                    if b_ih.len() != four_h || b_hh.len() != four_h {
                        return Err(DataError(format!(
                            "Layer {} (lstm) biases must each have {} elements in {} (got bias_ih={}, bias_hh={})",
                            i,
                            four_h,
                            path,
                            b_ih.len(),
                            b_hh.len()
                        )));
                    }
                    for (r, row) in w_ih.iter().enumerate() {
                        if row.len() != *input_size {
                            return Err(DataError(format!(
                                "Layer {} (lstm) weight_ih row {} length: expected {}, got {} in {}",
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
                                "Layer {} (lstm) weight_hh row {} length: expected {}, got {} in {}",
                                i,
                                r,
                                hidden_size,
                                row.len(),
                                path
                            )));
                        }
                    }

                    layers.push(Layer::Lstm(LstmLayer {
                        input_size: *input_size,
                        hidden_size: *hidden_size,
                        weight_ih: w_ih.clone(),
                        weight_hh: w_hh.clone(),
                        bias_ih: b_ih.clone(),
                        bias_hh: b_hh.clone(),
                    }));
                }
                LayerSpec::Window {
                    input_size,
                    n_steps,
                } => {
                    if *input_size == 0 || *n_steps == 0 {
                        return Err(DataError(format!(
                            "Layer {} (window) input_size and n_steps must be positive in {}",
                            i, path
                        )));
                    }
                    if i == 0 {
                        layer_sizes.push(*input_size);
                    }
                    // Window's output is n_steps * input_size (flattened buffer).
                    layer_sizes.push(*input_size * *n_steps);
                    // Window has zero trainable parameters, so we don't look up
                    // weights["layer_i"] here -- save_json skips the entry and
                    // any present one (from a hand-crafted JSON) is ignored.
                    layers.push(Layer::Window(WindowLayer {
                        input_size: *input_size,
                        n_steps: *n_steps,
                    }));
                }
                LayerSpec::Transformer {
                    d_model,
                    n_heads,
                    d_ffn,
                    n_seq,
                } => {
                    if *d_model == 0 || *n_heads == 0 || *d_ffn == 0 || *n_seq == 0 {
                        return Err(DataError(format!(
                            "Layer {} (transformer) all shape fields must be positive in {}",
                            i, path
                        )));
                    }
                    if d_model % n_heads != 0 {
                        return Err(DataError(format!(
                            "Layer {} (transformer) d_model={} not divisible by n_heads={} in {}",
                            i, d_model, n_heads, path
                        )));
                    }
                    let d_head = d_model / n_heads;

                    let key = format!("layer_{}", i);
                    let lw = file.weights.get(&key).ok_or_else(|| {
                        DataError(format!("Missing {} in weights in {}", key, path))
                    })?;

                    macro_rules! req_mat {
                        ($field:ident) => {
                            lw.$field.as_ref().ok_or_else(|| {
                                DataError(format!(
                                    "Layer {} (transformer) missing {} in {}",
                                    i,
                                    stringify!($field),
                                    path
                                ))
                            })?
                        };
                    }
                    macro_rules! req_vec {
                        ($field:ident) => {
                            lw.$field.as_ref().ok_or_else(|| {
                                DataError(format!(
                                    "Layer {} (transformer) missing {} in {}",
                                    i,
                                    stringify!($field),
                                    path
                                ))
                            })?
                        };
                    }

                    if i == 0 {
                        layer_sizes.push(*d_model);
                    }
                    layer_sizes.push(*d_model);

                    // Read all weight tensors before shape validation (macros borrow lw).
                    let w_q = req_mat!(w_q);
                    let b_q = req_vec!(b_q);
                    let w_k = req_mat!(w_k);
                    let b_k = req_vec!(b_k);
                    let w_v = req_mat!(w_v);
                    let b_v = req_vec!(b_v);
                    let w_o = req_mat!(w_o);
                    let b_o = req_vec!(b_o);
                    let w_ffn1 = req_mat!(w_ffn1);
                    let b_ffn1 = req_vec!(b_ffn1);
                    let w_ffn2 = req_mat!(w_ffn2);
                    let b_ffn2 = req_vec!(b_ffn2);
                    let ln1_gamma = req_vec!(ln1_gamma);
                    let ln1_beta = req_vec!(ln1_beta);
                    let ln2_gamma = req_vec!(ln2_gamma);
                    let ln2_beta = req_vec!(ln2_beta);

                    // Validate matrix shapes: (name, matrix, expected_rows, expected_cols).
                    for (name, m, exp_rows, exp_cols) in [
                        ("w_q", w_q, *d_model, *d_model),
                        ("w_k", w_k, *d_model, *d_model),
                        ("w_v", w_v, *d_model, *d_model),
                        ("w_o", w_o, *d_model, *d_model),
                        ("w_ffn1", w_ffn1, *d_ffn, *d_model),
                        ("w_ffn2", w_ffn2, *d_model, *d_ffn),
                    ] {
                        if m.len() != exp_rows {
                            return Err(DataError(format!(
                                "Layer {} (transformer) {} must have {} rows, got {} in {}",
                                i,
                                name,
                                exp_rows,
                                m.len(),
                                path
                            )));
                        }
                        for (r, row) in m.iter().enumerate() {
                            if row.len() != exp_cols {
                                return Err(DataError(format!(
                                    "Layer {} (transformer) {} row {} length: expected {}, got {} in {}",
                                    i,
                                    name,
                                    r,
                                    exp_cols,
                                    row.len(),
                                    path
                                )));
                            }
                        }
                    }
                    // Validate vector lengths: (name, vector, expected_length).
                    for (name, v, expected) in [
                        ("b_q", b_q, *d_model),
                        ("b_k", b_k, *d_model),
                        ("b_v", b_v, *d_model),
                        ("b_o", b_o, *d_model),
                        ("b_ffn1", b_ffn1, *d_ffn),
                        ("b_ffn2", b_ffn2, *d_model),
                        ("ln1_gamma", ln1_gamma, *d_model),
                        ("ln1_beta", ln1_beta, *d_model),
                        ("ln2_gamma", ln2_gamma, *d_model),
                        ("ln2_beta", ln2_beta, *d_model),
                    ] {
                        if v.len() != expected {
                            return Err(DataError(format!(
                                "Layer {} (transformer) {} length: expected {}, got {} in {}",
                                i,
                                name,
                                expected,
                                v.len(),
                                path
                            )));
                        }
                    }

                    let mut layer = TransformerLayer {
                        d_model: *d_model,
                        n_heads: *n_heads,
                        d_head,
                        d_ffn: *d_ffn,
                        n_seq: *n_seq,
                        w_q: w_q.clone(),
                        b_q: b_q.clone(),
                        w_k: w_k.clone(),
                        b_k: b_k.clone(),
                        w_v: w_v.clone(),
                        b_v: b_v.clone(),
                        w_o: w_o.clone(),
                        b_o: b_o.clone(),
                        w_ffn1: w_ffn1.clone(),
                        b_ffn1: b_ffn1.clone(),
                        w_ffn2: w_ffn2.clone(),
                        b_ffn2: b_ffn2.clone(),
                        ln1_gamma: ln1_gamma.clone(),
                        ln1_beta: ln1_beta.clone(),
                        ln2_gamma: ln2_gamma.clone(),
                        ln2_beta: ln2_beta.clone(),
                        k_pe_offsets: Vec::new(),
                        v_pe_offsets: Vec::new(),
                    };
                    layer.rebuild_pe_offsets();
                    layers.push(Layer::Transformer(Box::new(layer)));
                }
                LayerSpec::Mamba {
                    input_size,
                    d_state,
                    dt_rank,
                } => {
                    if *input_size == 0 || *d_state == 0 || *dt_rank == 0 {
                        return Err(DataError(format!(
                            "Layer {} (mamba) input_size, d_state, and dt_rank must be positive in {}",
                            i, path
                        )));
                    }
                    if *dt_rank > *input_size {
                        return Err(DataError(format!(
                            "Layer {} (mamba) dt_rank={} must not exceed input_size={} in {}",
                            i, dt_rank, input_size, path
                        )));
                    }

                    if i == 0 {
                        layer_sizes.push(*input_size);
                    }
                    layer_sizes.push(*input_size);

                    let key = format!("layer_{}", i);
                    let lw = file.weights.get(&key).ok_or_else(|| {
                        DataError(format!("Missing {} in weights in {}", key, path))
                    })?;

                    macro_rules! req_mamba_mat {
                        ($field:ident) => {
                            lw.$field.as_ref().ok_or_else(|| {
                                DataError(format!(
                                    "Layer {} (mamba) missing {} in {}",
                                    i,
                                    stringify!($field),
                                    path
                                ))
                            })?
                        };
                    }
                    macro_rules! req_mamba_vec {
                        ($field:ident) => {
                            lw.$field.as_ref().ok_or_else(|| {
                                DataError(format!(
                                    "Layer {} (mamba) missing {} in {}",
                                    i,
                                    stringify!($field),
                                    path
                                ))
                            })?
                        };
                    }

                    let x_proj_w = req_mamba_mat!(x_proj_w);
                    let dt_proj_w = req_mamba_mat!(dt_proj_w);
                    let dt_proj_b = req_mamba_vec!(dt_proj_b);
                    let a_log = req_mamba_mat!(a_log);
                    let d_skip = req_mamba_vec!(d_skip);

                    let rows_x = dt_rank + 2 * d_state;
                    // Shape validation for matrices.
                    for (name, m, exp_rows, exp_cols) in [
                        ("x_proj_w", x_proj_w, rows_x, *input_size),
                        ("dt_proj_w", dt_proj_w, *input_size, *dt_rank),
                        ("a_log", a_log, *input_size, *d_state),
                    ] {
                        if m.len() != exp_rows {
                            return Err(DataError(format!(
                                "Layer {} (mamba) {} must have {} rows, got {} in {}",
                                i,
                                name,
                                exp_rows,
                                m.len(),
                                path
                            )));
                        }
                        for (r, row) in m.iter().enumerate() {
                            if row.len() != exp_cols {
                                return Err(DataError(format!(
                                    "Layer {} (mamba) {} row {} length: expected {}, got {} in {}",
                                    i,
                                    name,
                                    r,
                                    exp_cols,
                                    row.len(),
                                    path
                                )));
                            }
                        }
                    }
                    // Shape validation for vectors.
                    for (name, v, expected) in [
                        ("dt_proj_b", dt_proj_b, *input_size),
                        ("d_skip", d_skip, *input_size),
                    ] {
                        if v.len() != expected {
                            return Err(DataError(format!(
                                "Layer {} (mamba) {} length: expected {}, got {} in {}",
                                i,
                                name,
                                expected,
                                v.len(),
                                path
                            )));
                        }
                    }

                    // Convert Vec<Vec<f64>> -> DMatrix (row-major).
                    let to_dmatrix = |rows_data: &Vec<Vec<f64>>,
                                      nr: usize,
                                      nc: usize|
                     -> nalgebra::DMatrix<f64> {
                        let flat: Vec<f64> =
                            rows_data.iter().flat_map(|r| r.iter().copied()).collect();
                        nalgebra::DMatrix::from_row_slice(nr, nc, &flat)
                    };

                    layers.push(Layer::Mamba(Box::new(MambaLayer {
                        input_size: *input_size,
                        d_state: *d_state,
                        dt_rank: *dt_rank,
                        x_proj_w: to_dmatrix(x_proj_w, rows_x, *input_size),
                        dt_proj_w: to_dmatrix(dt_proj_w, *input_size, *dt_rank),
                        dt_proj_b: nalgebra::DVector::from_vec(dt_proj_b.clone()),
                        a_log: to_dmatrix(a_log, *input_size, *d_state),
                        d_skip: nalgebra::DVector::from_vec(d_skip.clone()),
                    })));
                }
                LayerSpec::Mamba3 {
                    input_size,
                    d_state,
                    dt_rank,
                    discretization,
                    state_mode,
                } => {
                    let (trapezoidal, complex) = mamba3_flags(discretization, state_mode)
                        .map_err(|e| DataError(format!("Layer {i} (mamba3) {e} in {path}")))?;
                    if *input_size == 0 || *d_state == 0 || *dt_rank == 0 {
                        return Err(DataError(format!(
                            "Layer {i} (mamba3) input_size, d_state, dt_rank must be positive in {path}"
                        )));
                    }
                    if *dt_rank > *input_size {
                        return Err(DataError(format!(
                            "Layer {i} (mamba3) dt_rank={dt_rank} must not exceed input_size={input_size} in {path}"
                        )));
                    }
                    if i == 0 {
                        layer_sizes.push(*input_size);
                    }
                    layer_sizes.push(*input_size);

                    let key = format!("layer_{}", i);
                    let lw = file.weights.get(&key).ok_or_else(|| {
                        DataError(format!("Missing {} in weights in {}", key, path))
                    })?;

                    // Assemble the canonical flat slab from named JSON fields, then from_flat.
                    let flat_mat =
                        |name: &str, m: &Option<Vec<Vec<f64>>>| -> Result<Vec<f64>, DataError> {
                            let rows = m.as_ref().ok_or_else(|| {
                                DataError(format!("Layer {i} (mamba3) missing {name} in {path}"))
                            })?;
                            Ok(rows.iter().flat_map(|r| r.iter().copied()).collect())
                        };
                    let flat_vec =
                        |name: &str, v: &Option<Vec<f64>>| -> Result<Vec<f64>, DataError> {
                            v.as_ref().cloned().ok_or_else(|| {
                                DataError(format!("Layer {i} (mamba3) missing {name} in {path}"))
                            })
                        };

                    let mut slab = Vec::new();
                    slab.extend(flat_mat("x_proj_w", &lw.x_proj_w)?);
                    slab.extend(flat_mat("dt_proj_w", &lw.dt_proj_w)?);
                    slab.extend(flat_vec("dt_proj_b", &lw.dt_proj_b)?);
                    slab.extend(flat_mat("a_log", &lw.a_log)?);
                    if complex {
                        slab.extend(flat_mat("a_imag", &lw.a_imag)?);
                    }
                    if trapezoidal {
                        slab.extend(flat_vec("lambda_logit", &lw.lambda_logit)?);
                    }
                    slab.extend(flat_vec("d_skip", &lw.d_skip)?);

                    let mut m =
                        Mamba3Layer::zeros(*input_size, *d_state, *dt_rank, trapezoidal, complex);
                    if slab.len() != m.n_params() {
                        return Err(DataError(format!(
                            "Layer {i} (mamba3) weight count {} != expected {} in {path}",
                            slab.len(),
                            m.n_params()
                        )));
                    }
                    m.from_flat(&slab);
                    layers.push(Layer::Mamba3(Box::new(m)));
                }
            }
        }

        Self::validate_mask(&file.input_mask, layer_sizes[0])?;
        Self::validate_ablated_input(&file.ablated_input)?;

        let output_size = *layer_sizes.last().unwrap_or(&0);
        Self::validate_output_size(output_size, file.output_param, path)?;
        let last_activation = match file.architecture.last() {
            Some(LayerSpec::Dense { activation, .. }) => *activation,
            // Non-dense final layer with AcosTanh would have failed
            // validate_output_size when output_param=AcosTanh expects
            // output_size=1 (only Dense exposes a configurable output_size+activation
            // pair); for Atan2Signed the activation is irrelevant so default is fine.
            _ => Activation::Tanh,
        };
        Self::validate_output_activation(last_activation, file.output_param, path)?;

        let normalization = Self::resolve_normalization(file.normalization);

        Ok(NeuralNetModel {
            architecture: file.architecture,
            layer_sizes,
            layers,
            input_mask: file.input_mask,
            ablated_input: file.ablated_input,
            ablated_value: file.ablated_value,
            output_param: file.output_param,
            scaled_pi_n: file.scaled_pi_n,
            delta_max: file.delta_max,
            normalization,
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
                    ..NnLayerWeights::default()
                },
                Layer::Gru(g) => NnLayerWeights {
                    weight_ih: Some(g.weight_ih.clone()),
                    weight_hh: Some(g.weight_hh.clone()),
                    bias_ih: Some(g.bias_ih.clone()),
                    bias_hh: Some(g.bias_hh.clone()),
                    ..NnLayerWeights::default()
                },
                Layer::Lstm(l) => NnLayerWeights {
                    weight_ih: Some(l.weight_ih.clone()),
                    weight_hh: Some(l.weight_hh.clone()),
                    bias_ih: Some(l.bias_ih.clone()),
                    bias_hh: Some(l.bias_hh.clone()),
                    ..NnLayerWeights::default()
                },
                // Window is zero-param; skip the weights entry entirely.
                Layer::Window(_) => continue,
                Layer::Transformer(t) => NnLayerWeights {
                    w_q: Some(t.w_q.clone()),
                    b_q: Some(t.b_q.clone()),
                    w_k: Some(t.w_k.clone()),
                    b_k: Some(t.b_k.clone()),
                    w_v: Some(t.w_v.clone()),
                    b_v: Some(t.b_v.clone()),
                    w_o: Some(t.w_o.clone()),
                    b_o: Some(t.b_o.clone()),
                    w_ffn1: Some(t.w_ffn1.clone()),
                    b_ffn1: Some(t.b_ffn1.clone()),
                    w_ffn2: Some(t.w_ffn2.clone()),
                    b_ffn2: Some(t.b_ffn2.clone()),
                    ln1_gamma: Some(t.ln1_gamma.clone()),
                    ln1_beta: Some(t.ln1_beta.clone()),
                    ln2_gamma: Some(t.ln2_gamma.clone()),
                    ln2_beta: Some(t.ln2_beta.clone()),
                    ..NnLayerWeights::default()
                },
                Layer::Mamba(m) => {
                    let dmatrix_rows = |mat: &nalgebra::DMatrix<f64>| -> Vec<Vec<f64>> {
                        (0..mat.nrows())
                            .map(|r| (0..mat.ncols()).map(|c| mat[(r, c)]).collect())
                            .collect()
                    };
                    NnLayerWeights {
                        x_proj_w: Some(dmatrix_rows(&m.x_proj_w)),
                        dt_proj_w: Some(dmatrix_rows(&m.dt_proj_w)),
                        dt_proj_b: Some(m.dt_proj_b.iter().copied().collect()),
                        a_log: Some(dmatrix_rows(&m.a_log)),
                        d_skip: Some(m.d_skip.iter().copied().collect()),
                        ..NnLayerWeights::default()
                    }
                }
                Layer::Mamba3(m) => {
                    let dmatrix_rows = |mat: &nalgebra::DMatrix<f64>| -> Vec<Vec<f64>> {
                        (0..mat.nrows())
                            .map(|r| (0..mat.ncols()).map(|c| mat[(r, c)]).collect())
                            .collect()
                    };
                    NnLayerWeights {
                        x_proj_w: Some(dmatrix_rows(&m.x_proj_w)),
                        dt_proj_w: Some(dmatrix_rows(&m.dt_proj_w)),
                        dt_proj_b: Some(m.dt_proj_b.iter().copied().collect()),
                        a_log: Some(dmatrix_rows(&m.a_log)),
                        a_imag: m.a_imag.as_ref().map(dmatrix_rows),
                        lambda_logit: m.lambda_logit.as_ref().map(|v| v.iter().copied().collect()),
                        d_skip: Some(m.d_skip.iter().copied().collect()),
                        ..NnLayerWeights::default()
                    }
                }
            };
            weights.insert(format!("layer_{}", i), entry);
        }

        let file = NnJsonFileV2 {
            format_version: 2,
            architecture: self.architecture.clone(),
            weights,
            input_mask: self.input_mask.clone(),
            ablated_input: self.ablated_input,
            ablated_value: self.ablated_value,
            output_param: self.output_param,
            scaled_pi_n: self.scaled_pi_n,
            delta_max: self.delta_max,
            normalization: Some(self.normalization.clone()),
        };

        let json = serde_json::to_string_pretty(&file)
            .map_err(|e| DataError(format!("JSON serialize error: {}", e)))?;
        std::fs::write(path, json)
            .map_err(|e| DataError(format!("Cannot write {}: {}", path, e)))?;

        Ok(())
    }

    /// Generic forward pass through all layers.
    ///
    /// Takes `&mut NnState` so stateful layers (GRU/LSTM/Window/Transformer/Mamba) can mutate
    /// their per-sim hidden state. Dense layers ignore the state slot.
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
                (Layer::Lstm(l), LayerState::Lstm { h, c }) => {
                    let (h_new, c_new) = l.forward(h, c, &current);
                    *h = h_new.clone();
                    *c = c_new;
                    current = h_new;
                }
                (Layer::Window(w), LayerState::Window { buffer }) => {
                    current = w.forward(&current, buffer);
                }
                (Layer::Transformer(t), LayerState::Transformer { k_cache, v_cache }) => {
                    current = t.forward(&current, k_cache, v_cache);
                }
                (Layer::Mamba(m), LayerState::Mamba { h }) => {
                    current = m.forward(&current, h);
                }
                (
                    Layer::Mamba3(m),
                    LayerState::Mamba3 {
                        h_re,
                        h_im,
                        x_prev,
                        b_prev,
                    },
                ) => {
                    current = m.forward(&current, h_re, h_im, x_prev, b_prev);
                }
                _ => unreachable!(
                    "layer/state variant mismatch (construction invariant -- LayerState::for_layer maps Layer::Dense -> None, Layer::Gru -> Gru, Layer::Lstm -> Lstm, Layer::Window -> Window, Layer::Transformer -> Transformer)"
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
            input_mask: None,
            ablated_input: None,
            ablated_value: 0.0,
            output_param: OutputParam::default(),
            scaled_pi_n: default_scaled_pi_n(),
            delta_max: default_delta_max(),
            normalization: DEFAULT_NORMALIZATION.to_vec(),
        })
    }

    /// Construct a NeuralNetModel from a flat weight vector and v2 architecture spec.
    /// Used by the PyO3 flat_weights_to_json helper (Task 7) that routes PSO output
    /// through Rust. Unlike `from_flat_weights` (the v1 wrapper), this accepts
    /// heterogeneous architectures via `LayerSpec`.
    pub fn from_flat_weights_v2(
        flat: &[f64],
        architecture: &[LayerSpec],
        input_mask: Option<Vec<usize>>,
        output_param: OutputParam,
        scaled_pi_n: f64,
        delta_max: f64,
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
                LayerSpec::Lstm {
                    input_size,
                    hidden_size,
                } => {
                    if i == 0 {
                        layer_sizes.push(*input_size);
                    }
                    layer_sizes.push(*hidden_size);
                    let four_h = 4 * hidden_size;
                    Layer::Lstm(LstmLayer {
                        input_size: *input_size,
                        hidden_size: *hidden_size,
                        weight_ih: vec![vec![0.0; *input_size]; four_h],
                        weight_hh: vec![vec![0.0; *hidden_size]; four_h],
                        bias_ih: vec![0.0; four_h],
                        bias_hh: vec![0.0; four_h],
                    })
                }
                LayerSpec::Window {
                    input_size,
                    n_steps,
                } => {
                    if *input_size == 0 || *n_steps == 0 {
                        return Err(DataError(format!(
                            "from_flat_weights_v2: Window layer {} input_size and n_steps must be positive",
                            i
                        )));
                    }
                    if i == 0 {
                        layer_sizes.push(*input_size);
                    }
                    layer_sizes.push(*input_size * *n_steps);
                    Layer::Window(WindowLayer {
                        input_size: *input_size,
                        n_steps: *n_steps,
                    })
                }
                LayerSpec::Transformer {
                    d_model,
                    n_heads,
                    d_ffn,
                    n_seq,
                } => {
                    if *d_model == 0 || *d_ffn == 0 || *n_seq == 0 {
                        return Err(DataError(format!(
                            "from_flat_weights_v2: Transformer layer {} d_model, d_ffn, and n_seq must be positive (got d_model={}, d_ffn={}, n_seq={})",
                            i, d_model, d_ffn, n_seq
                        )));
                    }
                    if *n_heads == 0 || *d_model % *n_heads != 0 {
                        return Err(DataError(format!(
                            "from_flat_weights_v2: Transformer layer {} d_model={} not divisible by n_heads={}",
                            i, d_model, n_heads
                        )));
                    }
                    let d_head = d_model / n_heads;
                    let f = *d_ffn;
                    let d = *d_model;
                    if i == 0 {
                        layer_sizes.push(d);
                    }
                    layer_sizes.push(d);
                    Layer::Transformer(Box::new(TransformerLayer {
                        d_model: d,
                        n_heads: *n_heads,
                        d_head,
                        d_ffn: f,
                        n_seq: *n_seq,
                        w_q: vec![vec![0.0; d]; d],
                        b_q: vec![0.0; d],
                        w_k: vec![vec![0.0; d]; d],
                        b_k: vec![0.0; d],
                        w_v: vec![vec![0.0; d]; d],
                        b_v: vec![0.0; d],
                        w_o: vec![vec![0.0; d]; d],
                        b_o: vec![0.0; d],
                        w_ffn1: vec![vec![0.0; d]; f],
                        b_ffn1: vec![0.0; f],
                        w_ffn2: vec![vec![0.0; f]; d],
                        b_ffn2: vec![0.0; d],
                        ln1_gamma: vec![1.0; d],
                        ln1_beta: vec![0.0; d],
                        ln2_gamma: vec![1.0; d],
                        ln2_beta: vec![0.0; d],
                        k_pe_offsets: Vec::new(),
                        v_pe_offsets: Vec::new(),
                    }))
                }
                LayerSpec::Mamba {
                    input_size,
                    d_state,
                    dt_rank,
                } => {
                    if *dt_rank == 0 || *dt_rank > *input_size {
                        return Err(DataError(format!(
                            "from_flat_weights_v2: Mamba layer {} dt_rank={} invalid for input_size={}",
                            i, dt_rank, input_size
                        )));
                    }
                    if *d_state == 0 || *input_size == 0 {
                        return Err(DataError(format!(
                            "from_flat_weights_v2: Mamba layer {} input_size and d_state must be positive",
                            i
                        )));
                    }
                    if i == 0 {
                        layer_sizes.push(*input_size);
                    }
                    layer_sizes.push(*input_size);
                    let rows_x = dt_rank + 2 * d_state;
                    Layer::Mamba(Box::new(MambaLayer {
                        input_size: *input_size,
                        d_state: *d_state,
                        dt_rank: *dt_rank,
                        x_proj_w: nalgebra::DMatrix::<f64>::zeros(rows_x, *input_size),
                        dt_proj_w: nalgebra::DMatrix::<f64>::zeros(*input_size, *dt_rank),
                        dt_proj_b: nalgebra::DVector::<f64>::zeros(*input_size),
                        a_log: nalgebra::DMatrix::<f64>::zeros(*input_size, *d_state),
                        d_skip: nalgebra::DVector::<f64>::zeros(*input_size),
                    }))
                }
                LayerSpec::Mamba3 {
                    input_size,
                    d_state,
                    dt_rank,
                    discretization,
                    state_mode,
                } => {
                    let (trapezoidal, complex) =
                        mamba3_flags(discretization, state_mode).map_err(|e| {
                            DataError(format!("from_flat_weights_v2: Mamba3 layer {i} {e}"))
                        })?;
                    if *dt_rank == 0 || *dt_rank > *input_size || *d_state == 0 || *input_size == 0
                    {
                        return Err(DataError(format!(
                            "from_flat_weights_v2: Mamba3 layer {} dims invalid (input_size={}, d_state={}, dt_rank={})",
                            i, input_size, d_state, dt_rank
                        )));
                    }
                    if i == 0 {
                        layer_sizes.push(*input_size);
                    }
                    layer_sizes.push(*input_size);
                    Layer::Mamba3(Box::new(Mamba3Layer::zeros(
                        *input_size,
                        *d_state,
                        *dt_rank,
                        trapezoidal,
                        complex,
                    )))
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
        Self::validate_output_size(output_size, output_param, "<flat_weights_v2>")?;
        let last_activation = match architecture.last() {
            Some(LayerSpec::Dense { activation, .. }) => *activation,
            _ => Activation::Tanh,
        };
        Self::validate_output_activation(last_activation, output_param, "<flat_weights_v2>")?;

        Ok(NeuralNetModel {
            architecture: architecture.to_vec(),
            layer_sizes,
            layers,
            input_mask,
            ablated_input: None,
            ablated_value: 0.0,
            output_param,
            scaled_pi_n,
            delta_max,
            normalization: DEFAULT_NORMALIZATION.to_vec(),
        })
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
