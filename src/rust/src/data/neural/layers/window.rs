//! Window-MLP layer: zero-parameter FIFO ring buffer.

use super::super::LayerWeights;

/// Window-MLP layer: FIFO ring buffer of the last `n_steps` inputs,
/// concatenated into a vector of length `n_steps * input_size`.
///
/// Zero trainable parameters -- all trainable weight lives in the downstream
/// Dense layer. Phase 2b MVP ships PSO-only; PPO use raises
/// NotImplementedError at Python-side `build_layer(WindowSpec)`.
#[derive(Debug, Clone)]
pub struct WindowLayer {
    pub input_size: usize,
    pub n_steps: usize,
}

impl WindowLayer {
    /// Push `input` onto the tail of the ring buffer, drop the oldest slot,
    /// and return the flattened buffer (length = `n_steps * input_size`).
    ///
    /// Buffer is pre-filled with zero vectors at episode start (see
    /// `LayerState::for_layer`) so every tick is branchless: one pop_front,
    /// one push_back, one flatten. Takes the `VecDeque` directly (rather than
    /// a `&mut LayerState`) so the caller can hold the match-destructured
    /// buffer reference across the call without a double-borrow.
    pub fn forward(
        &self,
        input: &[f64],
        buffer: &mut std::collections::VecDeque<Vec<f64>>,
    ) -> Vec<f64> {
        assert_eq!(
            input.len(),
            self.input_size,
            "WindowLayer expected input_size={}, got {}",
            self.input_size,
            input.len()
        );
        buffer.pop_front();
        buffer.push_back(input.to_vec());
        let mut out = Vec::with_capacity(self.n_steps * self.input_size);
        for slot in buffer.iter() {
            out.extend_from_slice(slot);
        }
        out
    }
}

impl LayerWeights for WindowLayer {
    fn to_flat(&self) -> Vec<f64> {
        Vec::new()
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_flat(&mut self, _flat: &[f64]) -> usize {
        // WindowLayer is parameter-free: consume nothing, tolerate any tail slice.
        0
    }

    fn n_params(&self) -> usize {
        0
    }
}
