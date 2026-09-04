//! Neural network model with modular architecture.
//!
//! Supports arbitrary layer configurations (e.g. [6, 12, 2] or [6, 24, 12, 2])
//! with per-layer activation function choice. Loads from JSON format.

use super::DataError;
use crate::data::nn_state::{LayerState, NnState};
use serde::{Deserialize, Serialize};

mod layers;
use layers::json_to_flat;
pub use layers::{
    CfcLayer, DenseLayer, GruLayer, LstmLayer, Mamba3Layer, MambaLayer, MlstmLayer, Shape,
    SlstmLayer, Tensor, TensorField, TransformerLayer, WindowLayer,
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

/// Candidate-input names, index-aligned with `DEFAULT_NORMALIZATION` and with the
/// `raw[i]` slots `gnc::guidance::neural::build_nn_input` fills. This is the single
/// source of truth for the 35-wide candidate contract: exported to Python as
/// `aerocapture_rs.NN_INPUT_NAMES` / `candidate_inputs()`, from which the Python side
/// derives its name list, width and index lookups (a pure-Python fallback copy in
/// `training/config.py` is asserted equal element-wise by the drift test).
pub const NN_INPUT_NAMES: [&str; NN_FULL_INPUT_SIZE] = [
    "eccentricity_excess",      // 0
    "inclination_error",        // 1
    "radial_velocity",          // 2
    "orbital_energy",           // 3
    "velocity",                 // 4
    "accel_magnitude",          // 5
    "heat_flux_fraction",       // 6
    "heat_load_fraction",       // 7
    "altitude",                 // 8
    "fpa",                      // 9
    "latitude",                 // 10
    "drag_accel",               // 11
    "lift_accel",               // 12
    "sma_error",                // 13
    "apoapsis_alt",             // 14
    "bounce_flag",              // 15
    "cos_bank_nominal",         // 16
    "pdyn_nominal",             // 17
    "hdot_nominal",             // 18
    "pdyn_error",               // 19
    "exit_bank_teacher",        // 20
    "inclination_err_rate",     // 21
    "prev_bank_signed",         // 22
    "time_since_sign_flip",     // 23
    "inclination_err_integral", // 24
    "exit_bank_teacher_sin",    // 25
    "exit_bank_teacher_cos",    // 26
    "prev_bank_signed_sin",     // 27
    "prev_bank_signed_cos",     // 28
    "prev_realized_sin",        // 29
    "prev_realized_cos",        // 30
    "periapsis_alt",            // 31
    "predicted_dv1",            // 32
    "predicted_dv2",            // 33
    "predicted_dv3",            // 34
];

/// Default per-input normalization table (divisor form `(raw - center) / scale`).
/// All 35 entries are calibrated, including DV entries 32-34 (smooth, no sentinel).
pub const DEFAULT_NORMALIZATION: [NormSpec; NN_FULL_INPUT_SIZE] = [
    NormSpec {
        transform: NormTransform::None,
        scale: 0.8754754,
        center: 0.9125593,
    }, // 0  eccentricity_excess
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
    // Boxed for enum-variant size uniformity (5 matrix + 5 bias vectors, ~264 bytes unboxed).
    Cfc(Box<CfcLayer>),
    Slstm(SlstmLayer),
    // Boxed for enum-variant size uniformity (4 matrix + 4 bias + 2 gate vectors).
    Mlstm(Box<MlstmLayer>),
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
            Layer::Cfc(l) => l.input_size,
            Layer::Slstm(l) => l.input_size,
            Layer::Mlstm(l) => l.input_size,
        }
    }
}

/// A layer's trainable weights as a tensor table.
///
/// `tensors()` lists the named tensors in the canonical flat order, declared
/// once per layer type with `tensor_table!`; `to_flat` / `from_flat` /
/// `n_params` and the JSON codec are generic walks over that table. The order
/// is the PSO chromosome contract shared with the PyTorch mirror in
/// src/python/aerocapture/training/rl/layers/<type>.py -- Python derives it
/// from this same table through `aerocapture_rs.layer_schema`.
///
/// Callers MUST ensure `flat.len() >= self.n_params()` before invoking
/// `from_flat`; it may panic otherwise. Length validation lives at the
/// caller (see `NeuralNetModel::from_flat_weights`) so the trait method
/// stays infallible and layers don't invent per-layer error dialects.
pub trait LayerWeights {
    /// Named tensors in canonical flat order.
    fn tensors(&self) -> Vec<(&'static str, &dyn Tensor)>;
    fn tensors_mut(&mut self) -> Vec<(&'static str, &mut dyn Tensor)>;

    /// Derived-field hook, run at the end of every `from_flat` -- the single
    /// point both load paths (JSON and PSO chromosome) pass through.
    fn post_load(&mut self) {}

    fn n_params(&self) -> usize {
        self.tensors().iter().map(|(_, t)| t.shape().numel()).sum()
    }

    fn to_flat(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n_params());
        for (_, t) in self.tensors() {
            t.write_flat(&mut out);
        }
        out
    }

    // `from_flat` takes `&mut self` by design: it overwrites this layer's
    // weights in place from a flat slice and returns elements consumed.
    #[allow(clippy::wrong_self_convention)]
    fn from_flat(&mut self, flat: &[f64]) -> usize {
        let mut idx = 0;
        for (_, t) in self.tensors_mut() {
            let n = t.shape().numel();
            t.read_flat(&flat[idx..idx + n]);
            idx += n;
        }
        self.post_load();
        idx
    }
}

impl Layer {
    fn as_weights(&self) -> &dyn LayerWeights {
        match self {
            Layer::Dense(l) => l,
            Layer::Gru(l) => l,
            Layer::Lstm(l) => l,
            Layer::Window(l) => l,
            Layer::Transformer(l) => l.as_ref(),
            Layer::Mamba(l) => l.as_ref(),
            Layer::Mamba3(l) => l.as_ref(),
            Layer::Cfc(l) => l.as_ref(),
            Layer::Slstm(l) => l,
            Layer::Mlstm(l) => l.as_ref(),
        }
    }

    fn as_weights_mut(&mut self) -> &mut dyn LayerWeights {
        match self {
            Layer::Dense(l) => l,
            Layer::Gru(l) => l,
            Layer::Lstm(l) => l,
            Layer::Window(l) => l,
            Layer::Transformer(l) => l.as_mut(),
            Layer::Mamba(l) => l.as_mut(),
            Layer::Mamba3(l) => l.as_mut(),
            Layer::Cfc(l) => l.as_mut(),
            Layer::Slstm(l) => l,
            Layer::Mlstm(l) => l.as_mut(),
        }
    }

    /// Zero-weight layer of the shape `spec` describes. The one place a spec's
    /// dimensions are validated, shared by the JSON and flat-weights loaders
    /// (and the PyO3 `layer_schema` accessor).
    pub fn from_spec(spec: &LayerSpec) -> Result<Layer, String> {
        let kind = spec.io().2;
        let positive = |all: bool, fields: &str| -> Result<(), String> {
            if all {
                Ok(())
            } else {
                Err(format!("({kind}) {fields} must be positive"))
            }
        };
        let mamba_dims = |input_size: usize, d_state: usize, dt_rank: usize| {
            positive(
                input_size > 0 && d_state > 0 && dt_rank > 0,
                "input_size, d_state, dt_rank",
            )?;
            if dt_rank > input_size {
                return Err(format!(
                    "({kind}) dt_rank={dt_rank} must not exceed input_size={input_size}"
                ));
            }
            Ok(())
        };
        Ok(match spec {
            LayerSpec::Dense {
                input_size,
                output_size,
                activation,
            } => Layer::Dense(DenseLayer::zeros(*input_size, *output_size, *activation)),
            LayerSpec::Gru {
                input_size,
                hidden_size,
            } => Layer::Gru(GruLayer::zeros(*input_size, *hidden_size)),
            LayerSpec::Lstm {
                input_size,
                hidden_size,
            } => Layer::Lstm(LstmLayer::zeros(*input_size, *hidden_size)),
            LayerSpec::Window {
                input_size,
                n_steps,
            } => {
                positive(*input_size > 0 && *n_steps > 0, "input_size and n_steps")?;
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
                positive(
                    *d_model > 0 && *d_ffn > 0 && *n_seq > 0,
                    "d_model, d_ffn, n_seq",
                )?;
                if *n_heads == 0 || d_model % n_heads != 0 {
                    return Err(format!(
                        "({kind}) d_model={d_model} not divisible by n_heads={n_heads}"
                    ));
                }
                Layer::Transformer(Box::new(TransformerLayer::zeros(
                    *d_model, *n_heads, *d_ffn, *n_seq,
                )))
            }
            LayerSpec::Mamba {
                input_size,
                d_state,
                dt_rank,
            } => {
                mamba_dims(*input_size, *d_state, *dt_rank)?;
                Layer::Mamba(Box::new(MambaLayer::zeros(*input_size, *d_state, *dt_rank)))
            }
            LayerSpec::Mamba3 {
                input_size,
                d_state,
                dt_rank,
                discretization,
                state_mode,
            } => {
                let (trapezoidal, complex) = mamba3_flags(discretization, state_mode)
                    .map_err(|e| format!("({kind}) {e}"))?;
                mamba_dims(*input_size, *d_state, *dt_rank)?;
                Layer::Mamba3(Box::new(Mamba3Layer::zeros(
                    *input_size,
                    *d_state,
                    *dt_rank,
                    trapezoidal,
                    complex,
                )))
            }
            LayerSpec::Cfc {
                input_size,
                hidden_size,
                backbone_units,
            } => {
                positive(
                    *input_size > 0 && *hidden_size > 0 && *backbone_units > 0,
                    "input_size, hidden_size, backbone_units",
                )?;
                Layer::Cfc(Box::new(CfcLayer::zeros(
                    *input_size,
                    *hidden_size,
                    *backbone_units,
                )))
            }
            LayerSpec::Slstm {
                input_size,
                hidden_size,
            } => {
                positive(
                    *input_size > 0 && *hidden_size > 0,
                    "input_size and hidden_size",
                )?;
                Layer::Slstm(SlstmLayer::zeros(*input_size, *hidden_size))
            }
            LayerSpec::Mlstm {
                input_size,
                hidden_size,
            } => {
                positive(
                    *input_size > 0 && *hidden_size > 0,
                    "input_size and hidden_size",
                )?;
                Layer::Mlstm(Box::new(MlstmLayer::zeros(*input_size, *hidden_size)))
            }
        })
    }
}

impl LayerWeights for Layer {
    fn tensors(&self) -> Vec<(&'static str, &dyn Tensor)> {
        self.as_weights().tensors()
    }

    fn tensors_mut(&mut self) -> Vec<(&'static str, &mut dyn Tensor)> {
        self.as_weights_mut().tensors_mut()
    }

    fn post_load(&mut self) {
        self.as_weights_mut().post_load()
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
    weights: WeightsJson,
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
    Cfc {
        input_size: usize,
        hidden_size: usize,
        backbone_units: usize,
    },
    Slstm {
        input_size: usize,
        hidden_size: usize,
    },
    Mlstm {
        input_size: usize,
        hidden_size: usize,
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
            LayerSpec::Cfc {
                input_size,
                hidden_size,
                ..
            } => (*input_size, *hidden_size, "cfc"),
            LayerSpec::Slstm {
                input_size,
                hidden_size,
            } => (*input_size, *hidden_size, "slstm"),
            LayerSpec::Mlstm {
                input_size,
                hidden_size,
            } => (*input_size, *hidden_size, "mlstm"),
        }
    }
}

fn default_scaled_pi_n() -> f64 {
    1.0
}
fn default_delta_max() -> f64 {
    0.35
}

/// The `weights` block: `layer_<i>` -> `{tensor name -> rows | vector | scalar}`.
/// Read as plain JSON objects and looked up by tensor-table name.
type WeightsJson = std::collections::BTreeMap<String, serde_json::Map<String, serde_json::Value>>;

/// Keys the retired fixed-field weights schema serialized after the rest of
/// the layer (the flag-gated Mamba3 tensors, added after `d_skip`). They are
/// stable-sorted last so re-saving any deployed model is byte-identical.
const JSON_KEYS_LAST: &[&str] = &["a_imag", "lambda_logit"];

/// One layer's weights for `save_json`, serialized as an object in the given
/// (table, then `JSON_KEYS_LAST`) order.
#[derive(Debug)]
struct LayerWeightsJson(Vec<(&'static str, serde_json::Value)>);

impl Serialize for LayerWeightsJson {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (name, value) in &self.0 {
            map.serialize_entry(name, value)?;
        }
        map.end()
    }
}

/// JSON file structure for neural network models (v2 schema), generic over
/// the per-layer weights representation (`serde_json::Map` on load,
/// `LayerWeightsJson` on save).
/// `output_param` selects the bank-angle decoder: `Atan2Signed` (default,
/// 2-output `atan2`) or `AcosTanh` (1-output `acos(tanh(x))`, magnitude_only
/// mode only). When absent in older v2 files, defaults to `Atan2Signed`
/// for backward compat. The legacy `output_interpretation` field is silently
/// ignored.
#[derive(Debug, Serialize, Deserialize)]
struct NnJsonFileV2<W> {
    format_version: u32,
    architecture: Vec<LayerSpec>,
    weights: std::collections::BTreeMap<String, W>,
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

/// `[input, out_0, out_1, ...]` from the chain of `LayerSpec::io()` widths.
fn layer_sizes_of(architecture: &[LayerSpec]) -> Vec<usize> {
    let mut sizes = Vec::with_capacity(architecture.len() + 1);
    if let Some(first) = architecture.first() {
        sizes.push(first.io().0);
    }
    sizes.extend(architecture.iter().map(|s| s.io().1));
    sizes
}

/// Activation the output-param check sees: the last Dense layer's, else Tanh
/// (only Dense exposes a configurable output_size + activation pair; a
/// non-Dense tail with a 1-output decoder already failed `validate_output_size`).
fn last_activation(architecture: &[LayerSpec]) -> Activation {
    match architecture.last() {
        Some(LayerSpec::Dense { activation, .. }) => *activation,
        _ => Activation::Tanh,
    }
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

    /// Resolve the per-input normalization table: the JSON block when present
    /// (hard error on wrong length -- `NN_FULL_INPUT_SIZE` has grown across
    /// eras, and silently swapping an older model's embedded calibration for
    /// `DEFAULT_NORMALIZATION` is exactly the train/inference mismatch the
    /// block exists to prevent), else `DEFAULT_NORMALIZATION`. Mirrors the
    /// strictness of the `[network.normalization]` TOML override.
    fn resolve_normalization(
        block: Option<Vec<NormSpec>>,
        path: &str,
    ) -> Result<Vec<NormSpec>, DataError> {
        match block {
            Some(v) if v.len() == NN_FULL_INPUT_SIZE => Ok(v),
            Some(v) => Err(DataError(format!(
                "embedded normalization block has {} entries, expected {} in {}",
                v.len(),
                NN_FULL_INPUT_SIZE,
                path
            ))),
            None => Ok(DEFAULT_NORMALIZATION.to_vec()),
        }
    }

    /// Validate that layer i's output width feeds layer i+1's input width.
    /// Shared by the JSON and flat-weights construction paths -- a mis-chained
    /// architecture would otherwise silently truncate the Dense dot products
    /// (`zip` stops at the shorter operand) instead of erroring.
    fn validate_layer_chain(architecture: &[LayerSpec], context: &str) -> Result<(), DataError> {
        for i in 0..architecture.len().saturating_sub(1) {
            let (_, prev_out, prev_label) = architecture[i].io();
            let (next_in, _, next_label) = architecture[i + 1].io();
            if prev_out != next_in {
                return Err(DataError(format!(
                    "architecture chain mismatch at layer {}->{} in {}: layer {} ({}) produces output={}, but layer {} ({}) expects input={}",
                    i,
                    i + 1,
                    context,
                    i,
                    prev_label,
                    prev_out,
                    i + 1,
                    next_label,
                    next_in
                )));
            }
        }
        Ok(())
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

        let layer_sizes = file.architecture.layers;
        let activations = file.architecture.activations;
        if layer_sizes.len() < 2 {
            return Err(DataError(format!(
                "architecture.layers needs at least [input, output] in {path}"
            )));
        }
        let n_layers = layer_sizes.len() - 1;
        if activations.len() != n_layers {
            return Err(DataError(format!(
                "Activation count ({}) != layer count ({}) in {}",
                activations.len(),
                n_layers,
                path
            )));
        }

        let architecture: Vec<LayerSpec> = (0..n_layers)
            .map(|i| LayerSpec::Dense {
                input_size: layer_sizes[i],
                output_size: layer_sizes[i + 1],
                activation: activations[i],
            })
            .collect();
        let layers = Self::load_layers(&architecture, &file.weights, path)?;

        Self::validate_mask(&file.input_mask, layer_sizes[0])?;
        Self::validate_ablated_input(&file.ablated_input)?;
        Self::validate_output_size(layer_sizes[n_layers], OutputParam::default(), path)?;

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
            normalization: Self::resolve_normalization(None, path)?,
        })
    }

    /// Build every layer of `architecture` from the JSON `weights` block: a
    /// zero layer from the spec, each tensor of its table looked up by name
    /// and shape-checked into a flat slab, then one `from_flat` (which runs
    /// the layer's `post_load` hook).
    fn load_layers(
        architecture: &[LayerSpec],
        weights: &WeightsJson,
        path: &str,
    ) -> Result<Vec<Layer>, DataError> {
        architecture
            .iter()
            .enumerate()
            .map(|(i, spec)| {
                let kind = spec.io().2;
                let mut layer = Layer::from_spec(spec)
                    .map_err(|e| DataError(format!("Layer {i} {e} in {path}")))?;
                let shapes: Vec<(&'static str, Shape)> = layer
                    .tensors()
                    .iter()
                    .map(|(name, t)| (*name, t.shape()))
                    .collect();
                if shapes.is_empty() {
                    // Zero-parameter layer (Window): no weights entry is
                    // written by save_json, and a hand-written one is ignored.
                    return Ok(layer);
                }
                let key = format!("layer_{i}");
                let lw = weights
                    .get(&key)
                    .ok_or_else(|| DataError(format!("Missing {key} in weights in {path}")))?;
                let mut slab = Vec::with_capacity(layer.n_params());
                for (name, shape) in shapes {
                    let value = lw.get(name).ok_or_else(|| {
                        DataError(format!("Layer {i} ({kind}) missing {name} in {path}"))
                    })?;
                    json_to_flat(value, shape, &mut slab).map_err(|e| {
                        DataError(format!("Layer {i} ({kind}) {name}: {e} in {path}"))
                    })?;
                }
                layer.from_flat(&slab);
                Ok(layer)
            })
            .collect()
    }

    /// Load v2 JSON schema (architecture is a tagged-layer list).
    fn from_v2_json(content: &str, path: &str) -> Result<Self, DataError> {
        let file: NnJsonFileV2<serde_json::Map<String, serde_json::Value>> =
            serde_json::from_str(content)
                .map_err(|e| DataError(format!("JSON parse error in {}: {}", path, e)))?;
        if file.architecture.is_empty() {
            return Err(DataError(format!("empty architecture in {path}")));
        }
        Self::validate_layer_chain(&file.architecture, path)?;
        let layers = Self::load_layers(&file.architecture, &file.weights, path)?;
        let layer_sizes = layer_sizes_of(&file.architecture);

        Self::validate_mask(&file.input_mask, layer_sizes[0])?;
        Self::validate_ablated_input(&file.ablated_input)?;
        let output_size = *layer_sizes.last().unwrap_or(&0);
        Self::validate_output_size(output_size, file.output_param, path)?;
        Self::validate_output_activation(
            last_activation(&file.architecture),
            file.output_param,
            path,
        )?;
        let normalization = Self::resolve_normalization(file.normalization, path)?;

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
            let mut entries: Vec<(&'static str, serde_json::Value)> = layer
                .tensors()
                .iter()
                .map(|(name, t)| (*name, t.to_json()))
                .collect();
            if entries.is_empty() {
                // Zero-parameter layer (Window): no weights entry.
                continue;
            }
            entries.sort_by_key(|(name, _)| JSON_KEYS_LAST.contains(name));
            weights.insert(format!("layer_{i}"), LayerWeightsJson(entries));
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
                    // Load paths chain-validate widths; this guards direct
                    // struct construction (zip would silently truncate).
                    debug_assert!(
                        d.w.is_empty() || d.w[0].len() == current.len(),
                        "dense layer expects input width {}, got {}",
                        d.w[0].len(),
                        current.len(),
                    );
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
                (Layer::Cfc(l), LayerState::Cfc { h }) => {
                    current = l.forward(&current, h);
                }
                (Layer::Slstm(l), LayerState::Slstm { h, c, n, m }) => {
                    current = l.forward(&current, h, c, n, m);
                }
                (Layer::Mlstm(l), LayerState::Mlstm { c, n, m }) => {
                    current = l.forward(&current, c, n, m);
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
            let mut layer = Layer::Dense(DenseLayer::zeros(n_in, n_out, activations[i]));
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
    /// Used by the PyO3 `flat_weights_to_json` helper that routes PSO output
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
        Self::validate_layer_chain(architecture, "<flat_weights_v2>")?;
        let mut layers: Vec<Layer> = Vec::with_capacity(architecture.len());
        let mut offset: usize = 0;

        for (i, spec) in architecture.iter().enumerate() {
            let mut layer = Layer::from_spec(spec)
                .map_err(|e| DataError(format!("from_flat_weights_v2: layer {i} {e}")))?;
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
            offset += layer.from_flat(&flat[offset..]);
            layers.push(layer);
        }

        if offset != flat.len() {
            return Err(DataError(format!(
                "from_flat_weights_v2: weight vector length mismatch, consumed {} of {}",
                offset,
                flat.len()
            )));
        }

        let layer_sizes = layer_sizes_of(architecture);
        Self::validate_mask(&input_mask, layer_sizes[0])?;
        let output_size = *layer_sizes.last().unwrap();
        Self::validate_output_size(output_size, output_param, "<flat_weights_v2>")?;
        Self::validate_output_activation(
            last_activation(architecture),
            output_param,
            "<flat_weights_v2>",
        )?;

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
