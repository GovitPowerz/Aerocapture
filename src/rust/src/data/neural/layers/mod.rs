//! Layer types for the modular neural network.
//!
//! One submodule per layer variant, each owning its struct, `LayerWeights`
//! impl, and forward fn(s). Shared numerical helpers live in `helpers`.

pub(crate) mod helpers;

pub mod dense;
pub mod gru;
pub mod lstm;
pub mod mamba;
pub mod transformer;
pub mod window;

pub use dense::DenseLayer;
pub use gru::GruLayer;
pub use lstm::LstmLayer;
pub use mamba::MambaLayer;
pub use transformer::TransformerLayer;
pub use window::WindowLayer;
