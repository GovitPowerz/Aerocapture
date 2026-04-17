//! Per-sim mutable state for stateful NN layers.
//!
//! Lives outside NeuralNetModel (which is immutable and shared via Arc).
//! Phase 0 ships only LayerState::None (dense layers are stateless).
//! Phase 1+ adds Gru, Lstm, Window, Ssm variants.

use crate::data::neural::{Layer, NeuralNetModel};

#[derive(Debug, Clone)]
pub enum LayerState {
    None,
    // Phase 1+: Gru { h: Vec<f64> }, Lstm { h: Vec<f64>, c: Vec<f64> },
    // Window { buffer: std::collections::VecDeque<Vec<f64>> }, Ssm { h: Vec<f64> },
}

impl LayerState {
    pub fn for_layer(layer: &Layer) -> Self {
        let _ = layer; // all Phase 0 layers are stateless
        LayerState::None
    }

    pub fn reset(&mut self) {
        match self {
            LayerState::None => {}
        }
    }
}

#[derive(Debug, Clone)]
pub struct NnState {
    pub layer_states: Vec<LayerState>,
}

impl NnState {
    pub fn for_model(model: &NeuralNetModel) -> Self {
        let layer_states = model.layers.iter().map(LayerState::for_layer).collect();
        Self { layer_states }
    }

    pub fn reset(&mut self) {
        for s in self.layer_states.iter_mut() {
            s.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::neural::{Activation, Layer, LayerSpec, NeuralNetModel};

    fn two_layer_model() -> NeuralNetModel {
        NeuralNetModel {
            architecture: vec![
                LayerSpec::Dense {
                    input_size: 3,
                    output_size: 2,
                    activation: Activation::Tanh,
                },
                LayerSpec::Dense {
                    input_size: 2,
                    output_size: 1,
                    activation: Activation::Linear,
                },
            ],
            layer_sizes: vec![3, 2, 1],
            layers: vec![
                Layer {
                    w: vec![vec![0.1; 3]; 2],
                    b: vec![0.0; 2],
                    activation: Activation::Tanh,
                },
                Layer {
                    w: vec![vec![0.1; 2]; 1],
                    b: vec![0.0; 1],
                    activation: Activation::Linear,
                },
            ],
            output_interpretation: "direct".to_string(),
            input_mask: None,
            ablated_input: None,
        }
    }

    #[test]
    fn for_model_produces_one_state_per_layer() {
        let model = two_layer_model();
        let state = NnState::for_model(&model);
        assert_eq!(state.layer_states.len(), 2);
        for s in &state.layer_states {
            assert!(matches!(s, LayerState::None));
        }
    }

    #[test]
    fn clone_is_independent() {
        let model = two_layer_model();
        let state = NnState::for_model(&model);
        let cloned = state.clone();
        // With only LayerState::None, there is nothing mutable to diverge yet;
        // assert structural equivalence to lock the invariant.
        assert_eq!(state.layer_states.len(), cloned.layer_states.len());
    }

    #[test]
    fn reset_is_idempotent_on_none_states() {
        let model = two_layer_model();
        let mut state = NnState::for_model(&model);
        state.reset();
        state.reset();
        assert_eq!(state.layer_states.len(), 2);
    }
}
