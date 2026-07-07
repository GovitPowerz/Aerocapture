//! Layer types for the modular neural network.
//!
//! One submodule per layer variant, each owning its struct, `LayerWeights`
//! impl, and forward fn(s). Shared numerical helpers live in `helpers`.

pub(crate) mod helpers;

pub(crate) mod cfc;
pub(crate) mod dense;
pub(crate) mod gru;
pub(crate) mod lstm;
pub(crate) mod mamba;
pub(crate) mod mamba3;
pub(crate) mod slstm;
pub(crate) mod transformer;
pub(crate) mod window;

pub use cfc::CfcLayer;
pub use dense::DenseLayer;
pub use gru::GruLayer;
pub use lstm::LstmLayer;
pub use mamba::MambaLayer;
pub use mamba3::Mamba3Layer;
pub use slstm::SlstmLayer;
pub use transformer::TransformerLayer;
pub use window::WindowLayer;
