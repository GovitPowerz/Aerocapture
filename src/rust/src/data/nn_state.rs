//! Per-sim mutable state for stateful NN layers.
//!
//! Lives outside NeuralNetModel (which is immutable and shared via Arc).
//! Phase 0 ships only LayerState::None (dense layers are stateless).
//! Phase 1+ adds Gru, Lstm, Window; Phase 3+ adds Ssm variants.

use std::collections::VecDeque;

use crate::data::neural::{Layer, NeuralNetModel};

#[derive(Debug, Clone)]
pub enum LayerState {
    None,
    Gru { h: Vec<f64> },
    Lstm { h: Vec<f64>, c: Vec<f64> },
    Window { buffer: VecDeque<Vec<f64>> },
    // Phase 3+: Ssm { h: Vec<f64> }
}

impl LayerState {
    pub fn for_layer(layer: &Layer) -> Self {
        match layer {
            Layer::Dense(_) => LayerState::None,
            Layer::Gru(g) => LayerState::Gru {
                h: vec![0.0; g.hidden_size],
            },
            Layer::Lstm(l) => LayerState::Lstm {
                h: vec![0.0; l.hidden_size],
                c: vec![0.0; l.hidden_size],
            },
            Layer::Window(w) => {
                let mut buffer = VecDeque::with_capacity(w.n_steps);
                for _ in 0..w.n_steps {
                    buffer.push_back(vec![0.0; w.input_size]);
                }
                LayerState::Window { buffer }
            }
        }
    }

    pub fn reset(&mut self) {
        match self {
            LayerState::None => {}
            LayerState::Gru { h } => {
                for v in h.iter_mut() {
                    *v = 0.0;
                }
            }
            LayerState::Lstm { h, c } => {
                for v in h.iter_mut() {
                    *v = 0.0;
                }
                for v in c.iter_mut() {
                    *v = 0.0;
                }
            }
            LayerState::Window { buffer } => {
                for slot in buffer.iter_mut() {
                    for v in slot.iter_mut() {
                        *v = 0.0;
                    }
                }
            }
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
    use crate::data::neural::{Activation, DenseLayer, Layer, LayerSpec, NeuralNetModel};

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
                Layer::Dense(DenseLayer {
                    w: vec![vec![0.1; 3]; 2],
                    b: vec![0.0; 2],
                    activation: Activation::Tanh,
                }),
                Layer::Dense(DenseLayer {
                    w: vec![vec![0.1; 2]; 1],
                    b: vec![0.0; 1],
                    activation: Activation::Linear,
                }),
            ],
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

    #[test]
    fn clone_is_behaviorally_independent_with_gru_state() {
        use crate::data::neural::{GruLayer, Layer};

        let gru = GruLayer {
            input_size: 2,
            hidden_size: 3,
            weight_ih: vec![vec![0.0; 2]; 9],
            weight_hh: vec![vec![0.0; 3]; 9],
            bias_ih: vec![0.0; 9],
            bias_hh: vec![0.0; 9],
        };
        let layer = Layer::Gru(gru);
        let original_state = LayerState::for_layer(&layer);
        let mut cloned_state = original_state.clone();

        if let LayerState::Gru { h } = &mut cloned_state {
            h[0] = 42.0;
        } else {
            panic!("expected LayerState::Gru");
        }

        // Mutating clone must not affect original.
        if let LayerState::Gru { h } = &original_state {
            assert_eq!(h[0], 0.0);
            assert_eq!(h[1], 0.0);
            assert_eq!(h[2], 0.0);
        } else {
            panic!("expected LayerState::Gru");
        }
    }

    #[test]
    fn clone_is_behaviorally_independent_with_lstm_state() {
        use crate::data::neural::{Layer, LstmLayer};

        let lstm = LstmLayer {
            input_size: 2,
            hidden_size: 2,
            weight_ih: vec![vec![0.0, 0.0]; 8],
            weight_hh: vec![vec![0.0, 0.0]; 8],
            bias_ih: vec![0.0; 8],
            bias_hh: vec![0.0; 8],
        };
        let layer = Layer::Lstm(lstm);
        let original_state = LayerState::for_layer(&layer);
        let mut cloned_state = original_state.clone();

        // Mutate the clone
        if let LayerState::Lstm { h, c } = &mut cloned_state {
            h[0] = 1.0;
            h[1] = 2.0;
            c[0] = 3.0;
            c[1] = 4.0;
        } else {
            panic!("expected LayerState::Lstm");
        }

        // Original must remain zeroed
        if let LayerState::Lstm { h, c } = &original_state {
            assert_eq!(h, &vec![0.0, 0.0]);
            assert_eq!(c, &vec![0.0, 0.0]);
        } else {
            panic!("expected LayerState::Lstm");
        }
    }

    #[test]
    fn layer_state_window_for_layer_prefills_buffer_with_zero_vectors() {
        let layer = Layer::Window(crate::data::neural::WindowLayer {
            input_size: 4,
            n_steps: 3,
        });
        let state = LayerState::for_layer(&layer);
        if let LayerState::Window { buffer } = state {
            assert_eq!(buffer.len(), 3);
            for slot in buffer.iter() {
                assert_eq!(slot.len(), 4);
                assert!(slot.iter().all(|&v| v == 0.0));
            }
        } else {
            panic!("expected LayerState::Window");
        }
    }

    #[test]
    fn layer_state_window_reset_clears_buffer_to_zeros() {
        let mut state = LayerState::Window {
            buffer: VecDeque::from(vec![vec![1.0, 2.0], vec![3.0, 4.0]]),
        };
        state.reset();
        if let LayerState::Window { buffer } = state {
            assert_eq!(buffer.len(), 2);
            for slot in buffer.iter() {
                assert!(slot.iter().all(|&v| v == 0.0));
            }
        } else {
            panic!("expected LayerState::Window after reset");
        }
    }

    #[test]
    fn layer_state_window_clone_is_independent() {
        let layer = Layer::Window(crate::data::neural::WindowLayer {
            input_size: 2,
            n_steps: 3,
        });
        let original = LayerState::for_layer(&layer);
        let mut cloned = original.clone();

        if let LayerState::Window { buffer } = &mut cloned {
            buffer.pop_front();
            buffer.push_back(vec![9.9, 8.8]);
        } else {
            panic!("expected LayerState::Window");
        }

        // Original remains zero-filled.
        if let LayerState::Window { buffer } = &original {
            assert_eq!(buffer.len(), 3);
            for slot in buffer.iter() {
                assert!(slot.iter().all(|&v| v == 0.0));
            }
        } else {
            panic!("expected LayerState::Window");
        }
    }
}
