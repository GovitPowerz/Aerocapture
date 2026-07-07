use super::*;

#[test]
fn apply_norm_divisor_forms() {
    assert!(
        (apply_norm(
            50.0,
            &NormSpec {
                transform: NormTransform::None,
                scale: 0.5,
                center: 0.5
            }
        ) - 99.0)
            .abs()
            < 1e-12
    );
    let got = apply_norm(
        880.0,
        &NormSpec {
            transform: NormTransform::Asinh,
            scale: 880.0,
            center: 0.0,
        },
    );
    assert!((got - 1.0_f64.asinh()).abs() < 1e-12); // asinh(1.0)
    let got = apply_norm(
        30.0,
        &NormSpec {
            transform: NormTransform::Tanh,
            scale: 30.0,
            center: 0.0,
        },
    );
    assert!((got - 1.0_f64.tanh()).abs() < 1e-12);
    assert!(
        (apply_norm(
            0.3,
            &NormSpec {
                transform: NormTransform::None,
                scale: 1.0,
                center: 0.0
            }
        ) - 0.3)
            .abs()
            < 1e-12
    );
}

#[test]
fn default_normalization_has_full_width() {
    assert_eq!(DEFAULT_NORMALIZATION.len(), NN_FULL_INPUT_SIZE);
}

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
        input_mask: None,
        ablated_input: None,
        ablated_value: 0.0,
        output_param: OutputParam::default(),
        scaled_pi_n: default_scaled_pi_n(),
        delta_max: default_delta_max(),
        normalization: DEFAULT_NORMALIZATION.to_vec(),
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
    // index == NN_FULL_INPUT_SIZE is out of range
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
    // index 34 is the last valid index (NN_FULL_INPUT_SIZE - 1)
    let result = NeuralNetModel::validate_ablated_input(&Some(34));
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
        input_mask: None,
        ablated_input: None,
        ablated_value: 0.0,
        output_param: OutputParam::default(),
        scaled_pi_n: default_scaled_pi_n(),
        delta_max: default_delta_max(),
        normalization: DEFAULT_NORMALIZATION.to_vec(),
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
fn lstm_flat_weights_roundtrip() {
    let original = LstmLayer {
        input_size: 3,
        hidden_size: 2,
        weight_ih: (0..8)
            .map(|i| (0..3).map(|j| (i * 3 + j) as f64 * 0.01).collect())
            .collect(),
        weight_hh: (0..8)
            .map(|i| (0..2).map(|j| 100.0 + (i * 2 + j) as f64 * 0.01).collect())
            .collect(),
        bias_ih: (0..8).map(|i| 200.0 + i as f64).collect(),
        bias_hh: (0..8).map(|i| 300.0 + i as f64).collect(),
    };

    let flat = original.to_flat();
    assert_eq!(flat.len(), 56); // 4H*I + 4H*H + 2*4H = 24 + 16 + 16
    assert_eq!(flat.len(), original.n_params());

    let mut reconstructed = LstmLayer {
        input_size: 3,
        hidden_size: 2,
        weight_ih: vec![vec![0.0; 3]; 8],
        weight_hh: vec![vec![0.0; 2]; 8],
        bias_ih: vec![0.0; 8],
        bias_hh: vec![0.0; 8],
    };
    let consumed = reconstructed.from_flat(&flat);
    assert_eq!(consumed, 56);

    assert_eq!(reconstructed.weight_ih, original.weight_ih);
    assert_eq!(reconstructed.weight_hh, original.weight_hh);
    assert_eq!(reconstructed.bias_ih, original.bias_ih);
    assert_eq!(reconstructed.bias_hh, original.bias_hh);
}

#[test]
fn v2_gru_json_roundtrip() {
    // Use hidden_size=2 so the GRU's output is a valid network output
    // (atan2 requires the final layer to produce exactly 2 values).
    let input_size = 2;
    let hidden_size = 2;
    let three_h = 6;
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
        input_mask: None,
        ablated_input: None,
        ablated_value: 0.0,
        output_param: OutputParam::default(),
        scaled_pi_n: default_scaled_pi_n(),
        delta_max: default_delta_max(),
        normalization: DEFAULT_NORMALIZATION.to_vec(),
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
    let model = NeuralNetModel::from_flat_weights_v2(
        &flat,
        &architecture,
        None,
        OutputParam::default(),
        default_scaled_pi_n(),
        default_delta_max(),
    )
    .unwrap();
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
fn from_v2_json_chain_mismatch_raises() {
    // Dense(23->32) -> Dense(16->2) -- second layer expects 16, first produces 32.
    let bad = r#"{
            "format_version": 2,
            "architecture": [
                {"type": "dense", "input_size": 23, "output_size": 32, "activation": "tanh"},
                {"type": "dense", "input_size": 16, "output_size": 2, "activation": "linear"}
            ],
            "weights": {
                "layer_0": {"w": [], "b": []},
                "layer_1": {"w": [], "b": []}
            },
            "output_interpretation": "atan2"
        }"#;
    let err = NeuralNetModel::from_v2_json(bad, "<test>");
    assert!(err.is_err(), "expected chain-mismatch error");
    let msg = err.err().unwrap().0;
    assert!(
        msg.contains("chain mismatch"),
        "error message should mention chain mismatch, got: {}",
        msg
    );
    assert!(
        msg.contains("output=32") && msg.contains("input=16"),
        "error message should quote the mismatched sizes, got: {}",
        msg
    );
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
    let err = NeuralNetModel::from_flat_weights_v2(
        &flat,
        &architecture,
        None,
        OutputParam::default(),
        default_scaled_pi_n(),
        default_delta_max(),
    );
    assert!(err.is_err());
    // Too long should also Err.
    let flat = vec![0.0; 20];
    let err = NeuralNetModel::from_flat_weights_v2(
        &flat,
        &architecture,
        None,
        OutputParam::default(),
        default_scaled_pi_n(),
        default_delta_max(),
    );
    assert!(err.is_err());
}

#[test]
fn from_flat_weights_v2_carries_scaled_pi_knobs() {
    // minimal 3->1 tanh dense arch: 3*1 + 1 = 4 params
    let arch = vec![LayerSpec::Dense {
        input_size: 3,
        output_size: 1,
        activation: Activation::Tanh,
    }];
    let flat = vec![0.0_f64; 4];
    let m =
        NeuralNetModel::from_flat_weights_v2(&flat, &arch, None, OutputParam::ScaledPi, 2.0, 0.7)
            .unwrap();
    assert_eq!(m.output_param, OutputParam::ScaledPi);
    assert!((m.scaled_pi_n - 2.0).abs() < 1e-15);
    assert!((m.delta_max - 0.7).abs() < 1e-15);
}

#[test]
fn lstm_json_v2_roundtrip() {
    use crate::data::nn_state::NnState;

    let input_size = 3;
    let hidden_size = 4;
    let four_h = 16;
    let lstm = LstmLayer {
        input_size,
        hidden_size,
        weight_ih: (0..four_h)
            .map(|i| {
                (0..input_size)
                    .map(|j| (i * input_size + j) as f64 * 0.001)
                    .collect()
            })
            .collect(),
        weight_hh: (0..four_h)
            .map(|i| {
                (0..hidden_size)
                    .map(|j| 1.0 + (i * hidden_size + j) as f64 * 0.001)
                    .collect()
            })
            .collect(),
        bias_ih: (0..four_h).map(|i| 2.0 + i as f64 * 0.01).collect(),
        bias_hh: (0..four_h).map(|i| 3.0 + i as f64 * 0.01).collect(),
    };
    let dense_out = DenseLayer {
        w: vec![vec![0.5, -0.5, 0.25, 0.1]; 2],
        b: vec![0.0, 0.1],
        activation: Activation::Linear,
    };
    let original = NeuralNetModel {
        architecture: vec![
            LayerSpec::Lstm {
                input_size,
                hidden_size,
            },
            LayerSpec::Dense {
                input_size: hidden_size,
                output_size: 2,
                activation: Activation::Linear,
            },
        ],
        layer_sizes: vec![input_size, hidden_size, 2],
        layers: vec![Layer::Lstm(lstm), Layer::Dense(dense_out)],
        input_mask: None,
        ablated_input: None,
        ablated_value: 0.0,
        output_param: OutputParam::default(),
        scaled_pi_n: default_scaled_pi_n(),
        delta_max: default_delta_max(),
        normalization: DEFAULT_NORMALIZATION.to_vec(),
    };

    let tmpdir = std::env::temp_dir();
    let path = tmpdir.join("lstm_v2_roundtrip.json");
    original.save_json(path.to_str().unwrap()).unwrap();

    let loaded = NeuralNetModel::load(path.to_str().unwrap()).unwrap();
    assert_eq!(loaded.layers.len(), 2);
    match &loaded.layers[0] {
        Layer::Lstm(l) => {
            assert_eq!(l.input_size, input_size);
            assert_eq!(l.hidden_size, hidden_size);
        }
        _ => panic!("expected Lstm layer at index 0"),
    }

    // Forward parity over multiple steps (stateful)
    let mut s0 = NnState::for_model(&original);
    let mut s1 = NnState::for_model(&loaded);
    let x = vec![0.1, -0.2, 0.3];
    for _ in 0..5 {
        let o0 = original.forward(&mut s0, &x);
        let o1 = loaded.forward(&mut s1, &x);
        for (a, b) in o0.iter().zip(o1.iter()) {
            assert!((a - b).abs() < 1e-14, "{} vs {}", a, b);
        }
    }
}

#[test]
fn lstm_forward_known_output_zero_weights() {
    // Minimal 2-input, 2-hidden LSTM with all weights=0, all biases=0.
    // Then gates are all sigmoid(0)=0.5 (for i, f, o) and tanh(0)=0 (for g).
    // c_new = 0.5 * c_prev + 0.5 * 0 = 0.5 * c_prev
    // h_new = 0.5 * tanh(c_new)
    let lstm = LstmLayer {
        input_size: 2,
        hidden_size: 2,
        weight_ih: vec![vec![0.0, 0.0]; 8], // 4H=8 rows, 2 cols
        weight_hh: vec![vec![0.0, 0.0]; 8],
        bias_ih: vec![0.0; 8],
        bias_hh: vec![0.0; 8],
    };
    let h_prev = vec![0.0, 0.0];
    let c_prev = vec![2.0, -4.0];
    let x = vec![0.5, -0.5];
    let (h_new, c_new) = lstm.forward(&h_prev, &c_prev, &x);
    // c_new = f*c + i*g = 0.5*c_prev + 0.5*0 = 0.5*c_prev
    assert!((c_new[0] - 1.0).abs() < 1e-12);
    assert!((c_new[1] - (-2.0)).abs() < 1e-12);
    // h_new = o*tanh(c_new) = 0.5*tanh(c_new)
    assert!((h_new[0] - 0.5 * 1.0_f64.tanh()).abs() < 1e-12);
    assert!((h_new[1] - 0.5 * (-2.0_f64).tanh()).abs() < 1e-12);
}

// ── WindowLayer tests ─────────────────────────────────────────────

#[test]
fn window_layer_struct_and_spec_variants_construct() {
    let spec = LayerSpec::Window {
        input_size: 4,
        n_steps: 3,
    };
    match spec {
        LayerSpec::Window {
            input_size,
            n_steps,
        } => {
            assert_eq!(input_size, 4);
            assert_eq!(n_steps, 3);
        }
        _ => panic!("expected LayerSpec::Window"),
    }

    let layer = WindowLayer {
        input_size: 4,
        n_steps: 3,
    };
    assert_eq!(layer.input_size, 4);
    assert_eq!(layer.n_steps, 3);

    let enum_layer = Layer::Window(layer);
    match enum_layer {
        Layer::Window(w) => {
            assert_eq!(w.input_size, 4);
            assert_eq!(w.n_steps, 3);
        }
        _ => panic!("expected Layer::Window"),
    }
}

#[test]
fn window_layer_weights_trait_zero_params() {
    let layer = WindowLayer {
        input_size: 4,
        n_steps: 8,
    };
    assert_eq!(layer.n_params(), 0);
    assert_eq!(layer.to_flat(), Vec::<f64>::new());

    let mut layer_mut = layer.clone();
    // from_flat on Window consumes 0 params regardless of remaining slice length;
    // this is load-bearing for from_flat_weights_v2's per-layer offset accounting.
    let consumed = layer_mut.from_flat(&[]);
    assert_eq!(consumed, 0);
    let consumed_with_tail = layer_mut.from_flat(&[0.1, 0.2, 0.3]);
    assert_eq!(consumed_with_tail, 0);
}

#[test]
fn window_layer_forward_push_pop_and_concat_zero_padded() {
    use crate::data::nn_state::LayerState;

    let layer = WindowLayer {
        input_size: 2,
        n_steps: 3,
    };
    let mut state = LayerState::for_layer(&Layer::Window(layer.clone()));

    // Tick 0: first real input [1.0, 2.0]. Buffer becomes [[0,0], [0,0], [1,2]].
    let buffer = match &mut state {
        LayerState::Window { buffer } => buffer,
        _ => panic!("expected Window state"),
    };
    let out0 = layer.forward(&[1.0, 2.0], buffer);
    assert_eq!(out0, vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0]);

    // Tick 1: [3.0, 4.0]. Buffer becomes [[0,0], [1,2], [3,4]].
    let buffer = match &mut state {
        LayerState::Window { buffer } => buffer,
        _ => panic!("expected Window state"),
    };
    let out1 = layer.forward(&[3.0, 4.0], buffer);
    assert_eq!(out1, vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0]);

    // Tick 2: [5.0, 6.0]. Buffer becomes [[1,2], [3,4], [5,6]].
    let buffer = match &mut state {
        LayerState::Window { buffer } => buffer,
        _ => panic!("expected Window state"),
    };
    let out2 = layer.forward(&[5.0, 6.0], buffer);
    assert_eq!(out2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Buffer stays at steady-state capacity (always n_steps=3 entries).
    if let LayerState::Window { buffer } = state {
        assert_eq!(buffer.len(), 3);
    } else {
        panic!("expected Window state");
    }
}

#[test]
fn window_layer_end_to_end_forward_through_neural_net_model() {
    use crate::data::nn_state::NnState;

    // Window(2, 3) -> Dense(6 -> 2, linear). Dense weights are the identity
    // on the first two flat-buffer slots so we can verify the whole chain.
    let arch = vec![
        LayerSpec::Window {
            input_size: 2,
            n_steps: 3,
        },
        LayerSpec::Dense {
            input_size: 6,
            output_size: 2,
            activation: Activation::Linear,
        },
    ];
    // Dense weights: row 0 picks buffer[0][0], row 1 picks buffer[0][1].
    let model = NeuralNetModel {
        architecture: arch,
        layer_sizes: vec![2, 6, 2],
        layers: vec![
            Layer::Window(WindowLayer {
                input_size: 2,
                n_steps: 3,
            }),
            Layer::Dense(DenseLayer {
                w: vec![
                    vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                ],
                b: vec![0.0, 0.0],
                activation: Activation::Linear,
            }),
        ],
        input_mask: None,
        ablated_input: None,
        ablated_value: 0.0,
        output_param: OutputParam::default(),
        scaled_pi_n: default_scaled_pi_n(),
        delta_max: default_delta_max(),
        normalization: DEFAULT_NORMALIZATION.to_vec(),
    };
    let mut state = NnState::for_model(&model);

    // Tick 0: input [1.0, 2.0]. Buffer[0] is the oldest slot = zeros.
    let out = model.forward(&mut state, &[1.0, 2.0]);
    assert_eq!(out, vec![0.0, 0.0]);

    // Tick 1: input [3.0, 4.0]. Buffer[0] is still zeros (popped).
    let out = model.forward(&mut state, &[3.0, 4.0]);
    assert_eq!(out, vec![0.0, 0.0]);

    // Tick 2: input [5.0, 6.0]. Buffer[0] is now [1.0, 2.0].
    let out = model.forward(&mut state, &[5.0, 6.0]);
    assert_eq!(out, vec![1.0, 2.0]);

    // Tick 3: input [7.0, 8.0]. Buffer[0] is now [3.0, 4.0].
    let out = model.forward(&mut state, &[7.0, 8.0]);
    assert_eq!(out, vec![3.0, 4.0]);
}

#[test]
fn window_json_v2_roundtrip_spec_only() {
    let model = NeuralNetModel {
        architecture: vec![
            LayerSpec::Window {
                input_size: 4,
                n_steps: 3,
            },
            LayerSpec::Dense {
                input_size: 12,
                output_size: 2,
                activation: Activation::Linear,
            },
        ],
        layer_sizes: vec![4, 12, 2],
        layers: vec![
            Layer::Window(WindowLayer {
                input_size: 4,
                n_steps: 3,
            }),
            Layer::Dense(DenseLayer {
                w: vec![vec![0.1; 12]; 2],
                b: vec![0.0; 2],
                activation: Activation::Linear,
            }),
        ],
        input_mask: None,
        ablated_input: None,
        ablated_value: 0.0,
        output_param: OutputParam::default(),
        scaled_pi_n: default_scaled_pi_n(),
        delta_max: default_delta_max(),
        normalization: DEFAULT_NORMALIZATION.to_vec(),
    };

    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();
    model.save_json(path).unwrap();
    let content = std::fs::read_to_string(path).unwrap();

    assert!(content.contains("\"type\": \"window\""));
    assert!(content.contains("\"input_size\": 4"));
    assert!(content.contains("\"n_steps\": 3"));
    // Window has no weights entry in the weights dict (only Dense at index 1).
    assert!(content.contains("\"layer_1\""));
    assert!(!content.contains("\"layer_0\""));

    let parsed = NeuralNetModel::from_json_str(&content, path).unwrap();
    match &parsed.architecture[0] {
        LayerSpec::Window {
            input_size,
            n_steps,
        } => {
            assert_eq!(*input_size, 4);
            assert_eq!(*n_steps, 3);
        }
        _ => panic!("expected LayerSpec::Window"),
    }
    match &parsed.layers[0] {
        Layer::Window(w) => {
            assert_eq!(w.input_size, 4);
            assert_eq!(w.n_steps, 3);
        }
        _ => panic!("expected Layer::Window"),
    }
}

#[test]
fn window_from_flat_weights_v2_produces_zero_param_layer() {
    let arch = vec![
        LayerSpec::Window {
            input_size: 4,
            n_steps: 3,
        },
        LayerSpec::Dense {
            input_size: 12,
            output_size: 2,
            activation: Activation::Linear,
        },
    ];
    // Total param count = 0 (window) + 12*2 + 2 = 26.
    let flat: Vec<f64> = (0..26).map(|i| i as f64 * 0.01).collect();
    let model = NeuralNetModel::from_flat_weights_v2(
        &flat,
        &arch,
        None,
        OutputParam::default(),
        default_scaled_pi_n(),
        default_delta_max(),
    )
    .unwrap();

    match &model.layers[0] {
        Layer::Window(w) => {
            assert_eq!(w.input_size, 4);
            assert_eq!(w.n_steps, 3);
        }
        _ => panic!("expected Layer::Window"),
    }
    match &model.layers[1] {
        Layer::Dense(d) => {
            assert_eq!(d.w.len(), 2);
            assert_eq!(d.w[0].len(), 12);
            assert_eq!(d.b.len(), 2);
        }
        _ => panic!("expected Layer::Dense"),
    }
    assert_eq!(model.layer_sizes, vec![4, 12, 2]);
}

#[test]
fn window_from_flat_weights_v2_rejects_zero_fields() {
    let arch = vec![LayerSpec::Window {
        input_size: 0,
        n_steps: 3,
    }];
    let flat: Vec<f64> = Vec::new();
    let err = NeuralNetModel::from_flat_weights_v2(
        &flat,
        &arch,
        None,
        OutputParam::default(),
        default_scaled_pi_n(),
        default_delta_max(),
    );
    assert!(err.is_err());
}

#[test]
fn gelu_exact_matches_spec_values() {
    // Hand-computed f64 values of 0.5 * x * (1 + erf(x / sqrt(2))).
    // Generated with Python: 0.5 * x * (1 + math.erf(x / math.sqrt(2)))
    // Both sides use IEEE-754 correctly-rounded erf, so results are bit-identical.
    assert!((gelu_exact(0.0) - 0.0).abs() < 1e-15);
    assert!((gelu_exact(1.0) - 0.8413447460685429).abs() < 1e-15);
    assert!((gelu_exact(-1.0) - (-0.15865525393145707)).abs() < 1e-15);
    assert!((gelu_exact(2.5) - 2.4844758366855597).abs() < 1e-15);
}

#[test]
fn layer_norm_biased_zero_mean_unit_var() {
    // Input [1,2,3,4]: mean=2.5, biased var=((-1.5)^2+(-0.5)^2+(0.5)^2+(1.5)^2)/4 = 1.25.
    // After normalization the output should be zero-mean with unit variance (up to eps).
    let x = [1.0_f64, 2.0, 3.0, 4.0];
    let gamma = [1.0, 1.0, 1.0, 1.0];
    let beta = [0.0, 0.0, 0.0, 0.0];
    let out = layer_norm_biased(&x, &gamma, &beta, 1e-5);
    let mean: f64 = out.iter().sum::<f64>() / 4.0;
    assert!(mean.abs() < 1e-12); // output should be zero-mean
    let var: f64 = out.iter().map(|v| v * v).sum::<f64>() / 4.0;
    assert!((var - 1.0).abs() < 1e-4); // unit variance (up to eps floor)
}

#[test]
fn layer_norm_applies_gamma_beta() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let gamma = [2.0, 2.0, 2.0, 2.0];
    let beta = [1.0, 1.0, 1.0, 1.0];
    let out = layer_norm_biased(&x, &gamma, &beta, 1e-5);
    // Expected: 2 * normalized + 1
    let plain = layer_norm_biased(&x, &[1.0; 4], &[0.0; 4], 1e-5);
    for (i, v) in out.iter().enumerate() {
        assert!((v - (2.0 * plain[i] + 1.0)).abs() < 1e-12);
    }
}

#[test]
fn pe_table_shape_and_known_entries() {
    let pe = build_pe_table(4, 4);
    assert_eq!(pe.len(), 4);
    assert_eq!(pe[0].len(), 4);
    // PE[0, :] = [sin(0), cos(0), sin(0), cos(0)] = [0, 1, 0, 1]
    assert!((pe[0][0] - 0.0).abs() < 1e-15);
    assert!((pe[0][1] - 1.0).abs() < 1e-15);
    assert!((pe[0][2] - 0.0).abs() < 1e-15);
    assert!((pe[0][3] - 1.0).abs() < 1e-15);
    // PE[1, 0] = sin(1.0), PE[1, 1] = cos(1.0)
    assert!((pe[1][0] - 1.0_f64.sin()).abs() < 1e-15);
    assert!((pe[1][1] - 1.0_f64.cos()).abs() < 1e-15);
    // PE[1, 2] = sin(1.0 / 10000^(2/4)) = sin(1.0 / 100) = sin(0.01)
    assert!((pe[1][2] - 0.01_f64.sin()).abs() < 1e-14);
    assert!((pe[1][3] - 0.01_f64.cos()).abs() < 1e-14);
}

#[test]
fn transformer_layer_rebuild_pe_offsets_matches_matmul() {
    // With W_K = W_V = identity, k_pe_offsets and v_pe_offsets should equal the raw PE table.
    let d_model = 4;
    let n_seq = 3;
    let w_k: Vec<Vec<f64>> = (0..d_model)
        .map(|i| {
            (0..d_model)
                .map(|j| if i == j { 1.0 } else { 0.0 })
                .collect()
        })
        .collect();
    let w_v: Vec<Vec<f64>> = w_k.clone();
    let mut layer = TransformerLayer {
        d_model,
        n_heads: 2,
        d_head: 2,
        d_ffn: 8,
        n_seq,
        w_q: vec![vec![0.0; d_model]; d_model],
        b_q: vec![0.0; d_model],
        w_k: w_k.clone(),
        b_k: vec![0.0; d_model],
        w_v: w_v.clone(),
        b_v: vec![0.0; d_model],
        w_o: vec![vec![0.0; d_model]; d_model],
        b_o: vec![0.0; d_model],
        w_ffn1: vec![vec![0.0; d_model]; 8],
        b_ffn1: vec![0.0; 8],
        w_ffn2: vec![vec![0.0; 8]; d_model],
        b_ffn2: vec![0.0; d_model],
        ln1_gamma: vec![1.0; d_model],
        ln1_beta: vec![0.0; d_model],
        ln2_gamma: vec![1.0; d_model],
        ln2_beta: vec![0.0; d_model],
        k_pe_offsets: Vec::new(),
        v_pe_offsets: Vec::new(),
    };
    layer.rebuild_pe_offsets();
    let pe = build_pe_table(n_seq, d_model);
    for (i, pe_row) in pe.iter().enumerate() {
        for (j, &pe_val) in pe_row.iter().enumerate() {
            assert!((layer.k_pe_offsets[i][j] - pe_val).abs() < 1e-15);
            assert!((layer.v_pe_offsets[i][j] - pe_val).abs() < 1e-15);
        }
    }
}

#[test]
fn transformer_forward_single_token_zero_weights_is_residual() {
    // All projections zero + LN gamma=1, beta=0 + FFN zero means:
    //   x_norm1 = LN(x)
    //   q = k = v = 0
    //   attention output = 0
    //   x1 = x + W_O @ 0 + b_o = x
    //   ffn_out = 0
    //   out = x1 + 0 = x
    let d_model = 4;
    let n_heads = 2;
    let d_ffn = 8;
    let n_seq = 3;
    let layer = make_zero_transformer(d_model, n_heads, d_ffn, n_seq);
    let mut k_cache = std::collections::VecDeque::new();
    let mut v_cache = std::collections::VecDeque::new();
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let out = layer.forward(&x, &mut k_cache, &mut v_cache);
    for i in 0..d_model {
        assert!(
            (out[i] - x[i]).abs() < 1e-12,
            "out[{}]={} x[{}]={}",
            i,
            out[i],
            i,
            x[i]
        );
    }
    assert_eq!(k_cache.len(), 1);
    assert_eq!(v_cache.len(), 1);
}

#[test]
fn transformer_forward_cache_grows_then_saturates() {
    let d_model = 4;
    let n_heads = 2;
    let d_ffn = 8;
    let n_seq = 3;
    let mut layer = make_zero_transformer(d_model, n_heads, d_ffn, n_seq);
    layer.w_k[0][0] = 1.0;
    layer.rebuild_pe_offsets();
    let mut k_cache = std::collections::VecDeque::new();
    let mut v_cache = std::collections::VecDeque::new();
    for step in 0..5 {
        let x = vec![step as f64, 0.0, 0.0, 0.0];
        let _ = layer.forward(&x, &mut k_cache, &mut v_cache);
        let expected_len = (step + 1).min(n_seq);
        assert_eq!(k_cache.len(), expected_len, "step {step}");
        assert_eq!(v_cache.len(), expected_len, "step {step}");
    }
    assert_eq!(k_cache.len(), 3);
}

fn make_zero_transformer(
    d_model: usize,
    n_heads: usize,
    d_ffn: usize,
    n_seq: usize,
) -> TransformerLayer {
    let mut layer = TransformerLayer {
        d_model,
        n_heads,
        d_head: d_model / n_heads,
        d_ffn,
        n_seq,
        w_q: vec![vec![0.0; d_model]; d_model],
        b_q: vec![0.0; d_model],
        w_k: vec![vec![0.0; d_model]; d_model],
        b_k: vec![0.0; d_model],
        w_v: vec![vec![0.0; d_model]; d_model],
        b_v: vec![0.0; d_model],
        w_o: vec![vec![0.0; d_model]; d_model],
        b_o: vec![0.0; d_model],
        w_ffn1: vec![vec![0.0; d_model]; d_ffn],
        b_ffn1: vec![0.0; d_ffn],
        w_ffn2: vec![vec![0.0; d_ffn]; d_model],
        b_ffn2: vec![0.0; d_model],
        ln1_gamma: vec![1.0; d_model],
        ln1_beta: vec![0.0; d_model],
        ln2_gamma: vec![1.0; d_model],
        ln2_beta: vec![0.0; d_model],
        k_pe_offsets: Vec::new(),
        v_pe_offsets: Vec::new(),
    };
    layer.rebuild_pe_offsets();
    layer
}

#[test]
fn layer_spec_transformer_variant_serializes() {
    let spec = LayerSpec::Transformer {
        d_model: 32,
        n_heads: 4,
        d_ffn: 64,
        n_seq: 64,
    };
    let json = serde_json::to_string(&spec).unwrap();
    assert!(json.contains("\"type\":\"transformer\""));
    assert!(json.contains("\"d_model\":32"));
    let round: LayerSpec = serde_json::from_str(&json).unwrap();
    match round {
        LayerSpec::Transformer {
            d_model,
            n_heads,
            d_ffn,
            n_seq,
        } => {
            assert_eq!((d_model, n_heads, d_ffn, n_seq), (32, 4, 64, 64));
        }
        _ => panic!("wrong variant"),
    }
}

#[test]
fn transformer_layer_weights_flat_roundtrip() {
    let d_model = 4usize;
    let n_heads = 2;
    let d_ffn = 6;
    let n_seq = 3;
    // n_params = 4*d_model^2 + 2*d_ffn*d_model + d_ffn + 9*d_model
    //          = 4*16 + 2*24 + 6 + 36 = 64 + 48 + 6 + 36 = 154
    let n_params = 4 * d_model * d_model + 2 * d_ffn * d_model + d_ffn + 9 * d_model;
    assert_eq!(n_params, 154);

    let flat: Vec<f64> = (0..n_params).map(|i| (i as f64) * 0.01 + 0.5).collect();

    let mut layer = TransformerLayer {
        d_model,
        n_heads,
        d_head: d_model / n_heads,
        d_ffn,
        n_seq,
        w_q: vec![vec![0.0; d_model]; d_model],
        b_q: vec![0.0; d_model],
        w_k: vec![vec![0.0; d_model]; d_model],
        b_k: vec![0.0; d_model],
        w_v: vec![vec![0.0; d_model]; d_model],
        b_v: vec![0.0; d_model],
        w_o: vec![vec![0.0; d_model]; d_model],
        b_o: vec![0.0; d_model],
        w_ffn1: vec![vec![0.0; d_model]; d_ffn],
        b_ffn1: vec![0.0; d_ffn],
        w_ffn2: vec![vec![0.0; d_ffn]; d_model],
        b_ffn2: vec![0.0; d_model],
        ln1_gamma: vec![1.0; d_model],
        ln1_beta: vec![0.0; d_model],
        ln2_gamma: vec![1.0; d_model],
        ln2_beta: vec![0.0; d_model],
        k_pe_offsets: Vec::new(),
        v_pe_offsets: Vec::new(),
    };
    let consumed = layer.from_flat(&flat);
    assert_eq!(consumed, n_params);
    assert_eq!(layer.k_pe_offsets.len(), n_seq); // rebuild_pe_offsets ran
    assert_eq!(layer.v_pe_offsets.len(), n_seq);

    let round = layer.to_flat();
    assert_eq!(round.len(), n_params);
    for (i, (a, b)) in flat.iter().zip(round.iter()).enumerate() {
        assert!((a - b).abs() < 1e-15, "mismatch at index {i}: {a} vs {b}");
    }
}

#[test]
fn transformer_layer_weights_n_params_formula() {
    let layer = make_zero_transformer(4, 2, 6, 3);
    // 4*4*4 + 2*6*4 + 6 + 9*4 = 64 + 48 + 6 + 36 = 154
    assert_eq!(layer.n_params(), 154);
}

#[test]
fn transformer_from_flat_weights_v2_roundtrip() {
    let d_model = 4;
    let n_heads = 2;
    let d_ffn = 6;
    let n_seq = 3;
    let arch = vec![
        LayerSpec::Transformer {
            d_model,
            n_heads,
            d_ffn,
            n_seq,
        },
        LayerSpec::Dense {
            input_size: d_model,
            output_size: 2,
            activation: Activation::Linear,
        },
    ];
    // Transformer: 154 params; Dense(4->2): 4*2 + 2 = 10 params
    let total = 154 + 10;
    let flat: Vec<f64> = (0..total).map(|i| (i as f64) * 0.01 + 0.5).collect();
    let model = NeuralNetModel::from_flat_weights_v2(
        &flat,
        &arch,
        None,
        OutputParam::default(),
        default_scaled_pi_n(),
        default_delta_max(),
    )
    .unwrap();
    let round = model.to_flat_weights();
    assert_eq!(round.len(), total);
    for (i, (a, b)) in flat.iter().zip(round.iter()).enumerate() {
        assert!((a - b).abs() < 1e-15, "mismatch at index {i}: {a} vs {b}");
    }
}

#[test]
fn from_flat_weights_v2_transformer_rejects_zero_d_model() {
    // from_flat_weights_v2 must reject Transformer with d_model=0 (parity with from_v2_json).
    let arch = vec![LayerSpec::Transformer {
        d_model: 0,
        n_heads: 1,
        d_ffn: 8,
        n_seq: 4,
    }];
    let err = NeuralNetModel::from_flat_weights_v2(
        &[],
        &arch,
        None,
        OutputParam::default(),
        default_scaled_pi_n(),
        default_delta_max(),
    )
    .unwrap_err();
    assert!(
        err.0.contains("d_model") || err.0.contains("positive"),
        "expected positivity error, got: {err:?}"
    );
}

#[test]
fn from_flat_weights_v2_transformer_rejects_zero_d_ffn() {
    let arch = vec![LayerSpec::Transformer {
        d_model: 4,
        n_heads: 2,
        d_ffn: 0,
        n_seq: 4,
    }];
    let err = NeuralNetModel::from_flat_weights_v2(
        &[],
        &arch,
        None,
        OutputParam::default(),
        default_scaled_pi_n(),
        default_delta_max(),
    )
    .unwrap_err();
    assert!(
        err.0.contains("d_ffn") || err.0.contains("positive"),
        "expected positivity error, got: {err:?}"
    );
}

#[test]
fn from_flat_weights_v2_transformer_rejects_zero_n_seq() {
    let arch = vec![LayerSpec::Transformer {
        d_model: 4,
        n_heads: 2,
        d_ffn: 8,
        n_seq: 0,
    }];
    let err = NeuralNetModel::from_flat_weights_v2(
        &[],
        &arch,
        None,
        OutputParam::default(),
        default_scaled_pi_n(),
        default_delta_max(),
    )
    .unwrap_err();
    assert!(
        err.0.contains("n_seq") || err.0.contains("positive"),
        "expected positivity error, got: {err:?}"
    );
}

#[test]
fn transformer_json_v2_save_load_roundtrip() {
    // Dense(8->4,linear) -> Transformer(d_model=4, n_heads=2, d_ffn=8, n_seq=3) -> Dense(4->2,linear)
    let d_model = 4usize;
    let n_heads = 2usize;
    let d_ffn = 8usize;
    let n_seq = 3usize;

    let architecture = vec![
        LayerSpec::Dense {
            input_size: 8,
            output_size: d_model,
            activation: Activation::Linear,
        },
        LayerSpec::Transformer {
            d_model,
            n_heads,
            d_ffn,
            n_seq,
        },
        LayerSpec::Dense {
            input_size: d_model,
            output_size: 2,
            activation: Activation::Linear,
        },
    ];

    // Dense(8->4): 8*4 + 4 = 36 params
    // Transformer(d=4, f=8): 4*4*4 (QKV each d*d) + 4*4 (w_o, d*d) + 8*4 (w_ffn1, f*d)
    //   + 4*8 (w_ffn2, d*f) + 4 biases each for b_q/b_k/b_v/b_o/b_ffn1/b_ffn2 + 2*4 ln params * 2
    //   = LayerWeights::n_params = 4*4*4 + 2*8*4 + 8 + 9*4 = ...
    // Use n_params() directly from the model.
    let dummy_flat_len = {
        // Build a zero model to get n_params without needing the exact formula.
        let mut sizes = vec![0usize];
        for spec in &architecture {
            let out = match spec {
                LayerSpec::Dense { output_size, .. } => *output_size,
                LayerSpec::Transformer { d_model, .. } => *d_model,
                _ => 0,
            };
            sizes.push(out);
        }
        // Calculate: Dense 8->4 = 36, Transformer = n_params() by formula, Dense 4->2 = 10
        // TransformerLayer::n_params: 4*d*d + 2*f*d + f + 9*d
        //   = 4*16 + 2*8*4 + 8 + 9*4 = 64 + 64 + 8 + 36 = 172
        36 + 172 + 10
    };
    let flat: Vec<f64> = (0..dummy_flat_len)
        .map(|i| (i as f64) * 0.003 - 0.7)
        .collect();
    let model = NeuralNetModel::from_flat_weights_v2(
        &flat,
        &architecture,
        None,
        OutputParam::default(),
        default_scaled_pi_n(),
        default_delta_max(),
    )
    .unwrap();
    assert_eq!(model.n_params(), dummy_flat_len);

    let tmpdir = std::env::temp_dir();
    let path = tmpdir.join("transformer_v2_roundtrip.json");
    model.save_json(path.to_str().unwrap()).unwrap();

    let loaded = NeuralNetModel::load(path.to_str().unwrap()).unwrap();
    assert_eq!(loaded.architecture.len(), 3);
    assert_eq!(loaded.n_params(), model.n_params());

    // Flat-weight round-trip must be bit-identical.
    let orig_flat = model.to_flat_weights();
    let loaded_flat = loaded.to_flat_weights();
    assert_eq!(orig_flat.len(), loaded_flat.len());
    for (i, (a, b)) in orig_flat.iter().zip(loaded_flat.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-15,
            "roundtrip mismatch at {i}: {a} vs {b}"
        );
    }

    // Architecture spec must be identical.
    assert_eq!(
        format!("{:?}", model.architecture),
        format!("{:?}", loaded.architecture),
    );

    // Spot-check: middle layer is Transformer with correct shape.
    match &loaded.layers[1] {
        Layer::Transformer(t) => {
            assert_eq!(t.d_model, d_model);
            assert_eq!(t.n_heads, n_heads);
            assert_eq!(t.d_ffn, d_ffn);
            assert_eq!(t.n_seq, n_seq);
            // PE offsets must be rebuilt (non-empty after load).
            assert_eq!(t.k_pe_offsets.len(), n_seq);
            assert_eq!(t.v_pe_offsets.len(), n_seq);
        }
        _ => panic!("expected Transformer at layer 1"),
    }
}

#[test]
fn neural_net_model_forward_transformer_threads_state() {
    // Dense(4->4) -> Transformer(d_model=4, n_heads=2, d_ffn=8, n_seq=3) -> Dense(4->2)
    // n_params: Dense=20, Transformer=4*4*4 + 2*8*4 + 8 + 9*4 = 172, Dense=10, total=202
    let architecture = vec![
        LayerSpec::Dense {
            input_size: 4,
            output_size: 4,
            activation: Activation::Linear,
        },
        LayerSpec::Transformer {
            d_model: 4,
            n_heads: 2,
            d_ffn: 8,
            n_seq: 3,
        },
        LayerSpec::Dense {
            input_size: 4,
            output_size: 2,
            activation: Activation::Linear,
        },
    ];
    let flat: Vec<f64> = (0..202).map(|i| ((i % 7) as f64) * 0.01).collect();
    let model = NeuralNetModel::from_flat_weights_v2(
        &flat,
        &architecture,
        None,
        OutputParam::default(),
        default_scaled_pi_n(),
        default_delta_max(),
    )
    .unwrap();
    assert_eq!(model.n_params(), 202);

    let mut state = NnState::for_model(&model);
    let x = vec![0.5, -0.3, 0.7, 0.1];

    // Drive for 5 steps; cache should saturate at n_seq=3.
    let mut outputs = Vec::new();
    for _ in 0..5 {
        outputs.push(model.forward(&mut state, &x));
    }

    // All finite, correct output shape.
    for o in &outputs {
        assert_eq!(o.len(), 2);
        for v in o {
            assert!(v.is_finite(), "output contains non-finite value: {v}");
        }
    }

    // Cache saturated at n_seq=3 after 5 steps.
    match &state.layer_states[1] {
        LayerState::Transformer { k_cache, .. } => assert_eq!(k_cache.len(), 3),
        _ => panic!("expected Transformer state at layer 1"),
    }
}

#[test]
fn transformer_from_v2_json_rejects_wrong_w_q_shape() {
    let d_model = 4usize;
    let d_ffn = 8usize;
    let n_seq = 3usize;
    let n_heads = 2usize;

    let row_dm = vec![0.0_f64; d_model];
    let row_ffn_in = vec![0.0_f64; d_model];
    let row_ffn_out = vec![0.0_f64; d_ffn];
    let zero_bias_dm = vec![0.0_f64; d_model];
    let zero_bias_ffn = vec![0.0_f64; d_ffn];
    let gamma = vec![1.0_f64; d_model];
    let beta = vec![0.0_f64; d_model];

    // CORRUPT: w_q has d_model+1 rows instead of d_model.
    let bad_w_q: Vec<Vec<f64>> = (0..d_model + 1).map(|_| row_dm.clone()).collect();

    let json = serde_json::json!({
        "format_version": 2,
        "architecture": [{"type": "transformer", "d_model": d_model, "n_heads": n_heads, "d_ffn": d_ffn, "n_seq": n_seq}],
        "weights": {
            "layer_0": {
                "w_q": bad_w_q,
                "b_q": zero_bias_dm.clone(),
                "w_k": (0..d_model).map(|_| row_dm.clone()).collect::<Vec<_>>(),
                "b_k": zero_bias_dm.clone(),
                "w_v": (0..d_model).map(|_| row_dm.clone()).collect::<Vec<_>>(),
                "b_v": zero_bias_dm.clone(),
                "w_o": (0..d_model).map(|_| row_dm.clone()).collect::<Vec<_>>(),
                "b_o": zero_bias_dm.clone(),
                "w_ffn1": (0..d_ffn).map(|_| row_ffn_in.clone()).collect::<Vec<_>>(),
                "b_ffn1": zero_bias_ffn.clone(),
                "w_ffn2": (0..d_model).map(|_| row_ffn_out.clone()).collect::<Vec<_>>(),
                "b_ffn2": zero_bias_dm.clone(),
                "ln1_gamma": gamma.clone(), "ln1_beta": beta.clone(),
                "ln2_gamma": gamma.clone(), "ln2_beta": beta.clone(),
            }
        }
    });
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), serde_json::to_string(&json).unwrap()).unwrap();

    let result = NeuralNetModel::load(tmp.path().to_str().unwrap());
    assert!(
        result.is_err(),
        "expected error for wrong w_q row count, got Ok"
    );
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("w_q") || err.contains("transformer"),
        "expected error to mention w_q or transformer, got: {err}"
    );
}

#[test]
fn transformer_from_v2_json_rejects_wrong_b_q_length() {
    let d_model = 4usize;
    let d_ffn = 8usize;
    let n_seq = 3usize;
    let n_heads = 2usize;

    let row_dm = vec![0.0_f64; d_model];
    let row_ffn_in = vec![0.0_f64; d_model];
    let row_ffn_out = vec![0.0_f64; d_ffn];
    let zero_bias_dm = vec![0.0_f64; d_model];
    let zero_bias_ffn = vec![0.0_f64; d_ffn];
    let gamma = vec![1.0_f64; d_model];
    let beta = vec![0.0_f64; d_model];

    // CORRUPT: b_q is one element too long.
    let bad_b_q = vec![0.0_f64; d_model + 1];

    let json = serde_json::json!({
        "format_version": 2,
        "architecture": [{"type": "transformer", "d_model": d_model, "n_heads": n_heads, "d_ffn": d_ffn, "n_seq": n_seq}],
        "weights": {
            "layer_0": {
                "w_q": (0..d_model).map(|_| row_dm.clone()).collect::<Vec<_>>(),
                "b_q": bad_b_q,
                "w_k": (0..d_model).map(|_| row_dm.clone()).collect::<Vec<_>>(),
                "b_k": zero_bias_dm.clone(),
                "w_v": (0..d_model).map(|_| row_dm.clone()).collect::<Vec<_>>(),
                "b_v": zero_bias_dm.clone(),
                "w_o": (0..d_model).map(|_| row_dm.clone()).collect::<Vec<_>>(),
                "b_o": zero_bias_dm.clone(),
                "w_ffn1": (0..d_ffn).map(|_| row_ffn_in.clone()).collect::<Vec<_>>(),
                "b_ffn1": zero_bias_ffn.clone(),
                "w_ffn2": (0..d_model).map(|_| row_ffn_out.clone()).collect::<Vec<_>>(),
                "b_ffn2": zero_bias_dm.clone(),
                "ln1_gamma": gamma.clone(), "ln1_beta": beta.clone(),
                "ln2_gamma": gamma.clone(), "ln2_beta": beta.clone(),
            }
        }
    });
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), serde_json::to_string(&json).unwrap()).unwrap();

    let result = NeuralNetModel::load(tmp.path().to_str().unwrap());
    assert!(
        result.is_err(),
        "expected error for wrong b_q length, got Ok"
    );
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("b_q") || err.contains("transformer"),
        "expected error to mention b_q or transformer, got: {err}"
    );
}

#[test]
fn softplus_matches_stable_form_small_x() {
    // softplus(0) = log(2) ≈ 0.6931471805599453
    assert!((softplus(0.0) - std::f64::consts::LN_2).abs() < 1e-15);
    // softplus(1) = log(1 + e) ≈ 1.3132616875182228
    assert!((softplus(1.0) - 1.3132616875182228).abs() < 1e-14);
    // softplus(-1) = log(1 + 1/e) ≈ 0.3132616875182228
    assert!((softplus(-1.0) - 0.3132616875182228).abs() < 1e-14);
}

#[test]
fn softplus_no_overflow_at_large_magnitude() {
    // For x = 100, softplus(x) must stay finite and ≈ x (not Inf from naive exp).
    let y = softplus(100.0);
    assert!(y.is_finite());
    assert!((y - 100.0).abs() < 1e-10);
    // For x = -100, softplus(x) ≈ exp(-100) ≈ 3.72e-44, still finite.
    let y_neg = softplus(-100.0);
    assert!(y_neg.is_finite());
    assert!(y_neg > 0.0);
    assert!(y_neg < 1e-40);
}

#[test]
fn expm1_over_x_matches_exact_for_moderate_z() {
    // For |z| >= 1e-8, use expm1(z) / z directly.
    for &z in &[0.5_f64, -0.5, 1.0, -1.0, 5.0, -5.0, 0.01, -0.01] {
        let expected = z.exp_m1() / z;
        let got = expm1_over_x(z);
        assert!(
            (got - expected).abs() < 1e-15,
            "z={z}: got {got}, expected {expected}"
        );
    }
}

#[test]
fn expm1_over_x_taylor_branch_at_tiny_z() {
    // Taylor: 1 + z/2 + z^2/6 (error ~ z^3/24)
    // At z = 1e-10, Taylor and exact should agree to machine epsilon.
    let z = 1e-10;
    let taylor = 1.0 + z * 0.5 + z * z / 6.0;
    let got = expm1_over_x(z);
    assert!(
        (got - taylor).abs() < 1e-16,
        "z=1e-10: got {got}, taylor {taylor}"
    );
    // At z = 0, result should be 1.0 (the limit).
    assert_eq!(expm1_over_x(0.0), 1.0);
}

#[test]
fn expm1_over_x_crossover_is_smooth() {
    // Adjacent values across the crossover should not jump.
    let z1 = 0.99e-8;
    let z2 = 1.01e-8;
    let y1 = expm1_over_x(z1);
    let y2 = expm1_over_x(z2);
    // The two branches evaluate different formulas at z values ~1e-8 apart, so the
    // maximum expected delta is O(z) ≈ O(1e-8). 1e-9 is well within that bound.
    assert!((y1 - y2).abs() < 1e-9, "crossover jump: y1={y1}, y2={y2}");
}

#[test]
fn mamba_to_flat_from_flat_roundtrip() {
    use rand::{RngExt, SeedableRng};

    let (input_size, d_state, dt_rank) = (8usize, 4usize, 2usize);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut rand_vec =
        |n: usize| -> Vec<f64> { (0..n).map(|_| rng.random_range(-1.0..1.0)).collect() };

    let x_proj_rows = dt_rank + 2 * d_state;
    let original = MambaLayer {
        input_size,
        d_state,
        dt_rank,
        x_proj_w: nalgebra::DMatrix::from_row_slice(
            x_proj_rows,
            input_size,
            &rand_vec(x_proj_rows * input_size),
        ),
        dt_proj_w: nalgebra::DMatrix::from_row_slice(
            input_size,
            dt_rank,
            &rand_vec(input_size * dt_rank),
        ),
        dt_proj_b: nalgebra::DVector::from_row_slice(&rand_vec(input_size)),
        a_log: nalgebra::DMatrix::from_row_slice(
            input_size,
            d_state,
            &rand_vec(input_size * d_state),
        ),
        d_skip: nalgebra::DVector::from_row_slice(&rand_vec(input_size)),
    };

    let expected_n = input_size * (3 * d_state + 2 * dt_rank + 2);
    assert_eq!(original.n_params(), expected_n);

    let flat = original.to_flat();
    assert_eq!(flat.len(), expected_n);

    // Build a zero-initialized MambaLayer with same shape, then from_flat in place.
    let mut reconstructed = MambaLayer {
        input_size,
        d_state,
        dt_rank,
        x_proj_w: nalgebra::DMatrix::zeros(x_proj_rows, input_size),
        dt_proj_w: nalgebra::DMatrix::zeros(input_size, dt_rank),
        dt_proj_b: nalgebra::DVector::zeros(input_size),
        a_log: nalgebra::DMatrix::zeros(input_size, d_state),
        d_skip: nalgebra::DVector::zeros(input_size),
    };
    let cursor = reconstructed.from_flat(&flat);
    assert_eq!(cursor, expected_n);

    assert_eq!(reconstructed.input_size, original.input_size);
    assert_eq!(reconstructed.d_state, original.d_state);
    assert_eq!(reconstructed.dt_rank, original.dt_rank);
    for i in 0..x_proj_rows {
        for j in 0..input_size {
            assert_eq!(reconstructed.x_proj_w[(i, j)], original.x_proj_w[(i, j)]);
        }
    }
    for i in 0..input_size {
        for j in 0..dt_rank {
            assert_eq!(reconstructed.dt_proj_w[(i, j)], original.dt_proj_w[(i, j)]);
        }
    }
    for i in 0..input_size {
        assert_eq!(reconstructed.dt_proj_b[i], original.dt_proj_b[i]);
        assert_eq!(reconstructed.d_skip[i], original.d_skip[i]);
    }
    for i in 0..input_size {
        for j in 0..d_state {
            assert_eq!(reconstructed.a_log[(i, j)], original.a_log[(i, j)]);
        }
    }
}

#[test]
#[should_panic]
fn mamba_from_flat_panics_on_short_slice() {
    // 4 * (3*2 + 2*1 + 2) == 4 * 10 == 40; one less = 39 should panic
    let (input_size, d_state, dt_rank) = (4usize, 2usize, 1usize);
    let x_proj_rows = dt_rank + 2 * d_state;
    let mut layer = MambaLayer {
        input_size,
        d_state,
        dt_rank,
        x_proj_w: nalgebra::DMatrix::zeros(x_proj_rows, input_size),
        dt_proj_w: nalgebra::DMatrix::zeros(input_size, dt_rank),
        dt_proj_b: nalgebra::DVector::zeros(input_size),
        a_log: nalgebra::DMatrix::zeros(input_size, d_state),
        d_skip: nalgebra::DVector::zeros(input_size),
    };
    let too_short = vec![0.0_f64; 39]; // one less than 40
    layer.from_flat(&too_short); // must panic
}

#[test]
fn mamba_json_v2_save_load_roundtrip() {
    // Dense(8 -> 4, linear) -> Mamba(4, 2, 1) -> Dense(4 -> 2, linear)
    let architecture = vec![
        LayerSpec::Dense {
            input_size: 8,
            output_size: 4,
            activation: Activation::Linear,
        },
        LayerSpec::Mamba {
            input_size: 4,
            d_state: 2,
            dt_rank: 1,
        },
        LayerSpec::Dense {
            input_size: 4,
            output_size: 2,
            activation: Activation::Linear,
        },
    ];
    // Dense(8->4) = 36, Mamba(4, 2, 1) = 4*(6+2+2) = 40, Dense(4->2) = 10; total 86.
    let flat: Vec<f64> = (0..86).map(|i| (i as f64) * 0.017 - 0.9).collect();
    let model = NeuralNetModel::from_flat_weights_v2(
        &flat,
        &architecture,
        None,
        OutputParam::default(),
        default_scaled_pi_n(),
        default_delta_max(),
    )
    .unwrap();
    assert_eq!(model.n_params(), 86);

    let tmpdir = std::env::temp_dir();
    let path = tmpdir.join("mamba_v2_roundtrip.json");
    model.save_json(path.to_str().unwrap()).unwrap();

    // Sanity-check: JSON always includes dt_rank for Mamba layers
    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(
        raw.contains("\"dt_rank\""),
        "save_json output must contain dt_rank field; got: {raw}"
    );

    let loaded = NeuralNetModel::load(path.to_str().unwrap()).unwrap();
    assert_eq!(loaded.architecture.len(), 3);
    assert_eq!(loaded.n_params(), model.n_params());

    // Flat-weight round-trip must be bit-identical.
    let orig_flat = model.to_flat_weights();
    let loaded_flat = loaded.to_flat_weights();
    assert_eq!(orig_flat.len(), loaded_flat.len());
    for (i, (a, b)) in orig_flat.iter().zip(loaded_flat.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-15,
            "roundtrip mismatch at {i}: {a} vs {b}"
        );
    }

    // Architecture spec must be identical.
    assert_eq!(
        format!("{:?}", model.architecture),
        format!("{:?}", loaded.architecture),
    );

    // Spot-check: middle layer is Mamba with correct shape.
    match &loaded.layers[1] {
        Layer::Mamba(m) => {
            assert_eq!(m.input_size, 4);
            assert_eq!(m.d_state, 2);
            assert_eq!(m.dt_rank, 1);
            assert_eq!(m.x_proj_w.nrows(), 1 + 2 * 2); // dt_rank + 2*d_state = 5
            assert_eq!(m.x_proj_w.ncols(), 4); // input_size
            assert_eq!(m.dt_proj_w.shape(), (4, 1));
            assert_eq!(m.a_log.shape(), (4, 2));
            assert_eq!(m.dt_proj_b.len(), 4);
            assert_eq!(m.d_skip.len(), 4);
        }
        _ => panic!("expected Mamba at layer 1"),
    }
}

#[test]
fn mamba3_json_v2_save_load_roundtrip_all_flags() {
    // Dense(8 -> 4) -> Mamba3(4, 2, 1, flags) -> Dense(4 -> 2), all 4 flag combos.
    for &(disc, sm) in &[
        ("euler", "real"),
        ("trapezoidal", "real"),
        ("euler", "complex"),
        ("trapezoidal", "complex"),
    ] {
        let trapezoidal = disc == "trapezoidal";
        let complex = sm == "complex";
        let architecture = vec![
            LayerSpec::Dense {
                input_size: 8,
                output_size: 4,
                activation: Activation::Linear,
            },
            LayerSpec::Mamba3 {
                input_size: 4,
                d_state: 2,
                dt_rank: 1,
                discretization: disc.to_string(),
                state_mode: sm.to_string(),
            },
            LayerSpec::Dense {
                input_size: 4,
                output_size: 2,
                activation: Activation::Linear,
            },
        ];
        // Dense=36, Mamba3 base=40 (+8 complex, +4 trapz), Dense=10.
        let n = 36 + 40 + if complex { 8 } else { 0 } + if trapezoidal { 4 } else { 0 } + 10;
        let flat: Vec<f64> = (0..n).map(|i| (i as f64) * 0.013 - 0.7).collect();
        let model = NeuralNetModel::from_flat_weights_v2(
            &flat,
            &architecture,
            None,
            OutputParam::default(),
            default_scaled_pi_n(),
            default_delta_max(),
        )
        .unwrap();
        assert_eq!(model.n_params(), n, "trap={trapezoidal} cplx={complex}");

        let tmpdir = std::env::temp_dir();
        let path = tmpdir.join(format!("mamba3_v2_roundtrip_{trapezoidal}_{complex}.json"));
        model.save_json(path.to_str().unwrap()).unwrap();

        let loaded = NeuralNetModel::load(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.n_params(), model.n_params());
        let orig_flat = model.to_flat_weights();
        let loaded_flat = loaded.to_flat_weights();
        for (i, (a, b)) in orig_flat.iter().zip(loaded_flat.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-15,
                "trap={trapezoidal} cplx={complex} roundtrip mismatch at {i}: {a} vs {b}"
            );
        }
        assert_eq!(
            format!("{:?}", model.architecture),
            format!("{:?}", loaded.architecture),
        );
        match &loaded.layers[1] {
            Layer::Mamba3(m) => {
                assert_eq!(m.trapezoidal, trapezoidal);
                assert_eq!(m.complex, complex);
                assert_eq!(m.a_imag.is_some(), complex);
                assert_eq!(m.lambda_logit.is_some(), trapezoidal);
            }
            _ => panic!("expected Mamba3 at layer 1"),
        }
    }
}

#[test]
fn mamba_from_v2_json_rejects_zero_dt_rank() {
    // Build a minimal v2 JSON by hand with dt_rank = 0 in the Mamba spec.
    // The constructor validators in from_v2_json must reject this.
    let dir = std::env::temp_dir();
    let path = dir.join("mamba_zero_dt_rank.json");
    let bad_json = r#"{
            "format_version": 2,
            "architecture": [
                {"type": "dense", "input_size": 4, "output_size": 4, "activation": "linear"},
                {"type": "mamba", "input_size": 4, "d_state": 2, "dt_rank": 0},
                {"type": "dense", "input_size": 4, "output_size": 2, "activation": "linear"}
            ],
            "weights": {
                "layer_0": {"w": [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], "b": [0.0, 0.0, 0.0, 0.0]},
                "layer_1": {
                    "x_proj_w": [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                    "dt_proj_w": [[], [], [], []],
                    "dt_proj_b": [0.0, 0.0, 0.0, 0.0],
                    "a_log": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    "d_skip": [0.0, 0.0, 0.0, 0.0]
                },
                "layer_2": {"w": [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], "b": [0.0, 0.0]}
            }
        }"#;
    std::fs::write(&path, bad_json).unwrap();
    let result = NeuralNetModel::load(path.to_str().unwrap());
    assert!(
        result.is_err(),
        "from_v2_json must reject Mamba with dt_rank = 0"
    );
}

#[test]
fn mamba_forward_two_step_hand_verified() {
    use nalgebra::{DMatrix, DVector};

    // Minimal layer: d_inner=2, d_state=2, dt_rank=1
    //
    // x_proj: (5, 2) -- rows [dt_pre; B_0; B_1; C_0; C_1]
    // For x = [1, 0]: proj = [0, 1, 0, 1, 0] -> dt_pre=0, B=[1, 0], C=[1, 0]
    // For x = [0, 1]: proj = [0, 0, 1, 0, 1] -> dt_pre=0, B=[0, 1], C=[0, 1]
    //
    // dt_proj_w zero, bias such that softplus(b) = 0.5 -> b = log(exp(0.5) - 1)
    // Δ = 0.5 (per channel, constant)
    //
    // a_log = 0 -> A = -exp(0) = -1 (per (d, n))
    // Ā[d, n] = exp(Δ * A) = exp(-0.5) ≈ 0.6065306597126334
    // expm1_over_x(Δ * A) = (exp(-0.5) - 1) / (-0.5) ≈ 0.7869386805747332
    // B̄[d, n] = Δ * B[n] * expm1_over_x(Δ * A) = 0.5 * B[n] * 0.7869
    //
    // d_skip = 0 (no skip, isolate SSM)

    let x_proj_w = DMatrix::from_row_slice(
        5,
        2,
        &[
            0.0, 0.0, // dt_pre row
            1.0, 0.0, // B_0
            0.0, 1.0, // B_1
            1.0, 0.0, // C_0
            0.0, 1.0, // C_1
        ],
    );
    let dt_proj_w = DMatrix::from_row_slice(2, 1, &[0.0, 0.0]);
    let b_val = (0.5_f64.exp() - 1.0).ln(); // inv_softplus(0.5)
    let dt_proj_b = DVector::from_row_slice(&[b_val, b_val]);
    let a_log = DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 0.0]);
    let d_skip = DVector::from_row_slice(&[0.0, 0.0]);

    let layer = MambaLayer {
        input_size: 2,
        d_state: 2,
        dt_rank: 1,
        x_proj_w,
        dt_proj_w,
        dt_proj_b,
        a_log,
        d_skip,
    };

    let mut h = DMatrix::<f64>::zeros(2, 2);

    // Step 1: x = [1, 0]
    let x1 = [1.0, 0.0];
    let y1 = layer.forward(&x1, &mut h);
    assert!(
        (y1[0] - 0.3934693402873666).abs() < 1e-12,
        "step 1 y[0] = {}",
        y1[0]
    );
    assert!((y1[1] - 0.0).abs() < 1e-15, "step 1 y[1] = {}", y1[1]);

    // Step 2: x = [0, 1], h = [[0.39347, 0], [0, 0]]
    let x2 = [0.0, 1.0];
    let y2 = layer.forward(&x2, &mut h);
    assert!((y2[0] - 0.0).abs() < 1e-15, "step 2 y[0] = {}", y2[0]);
    assert!(
        (y2[1] - 0.3934693402873666).abs() < 1e-12,
        "step 2 y[1] = {}",
        y2[1]
    );
    // State h[0, 0] should now be ~0.23865 (exp(-0.5) * prev value)
    // Exact: exp(-0.5) * B_bar_step1 = 0.6065306597126334 * 0.3934693402873666
    assert!(
        (h[(0, 0)] - 0.2386512185411911).abs() < 1e-12,
        "h[0, 0] = {}",
        h[(0, 0)]
    );
}

mod mamba_proptest {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn mamba_flat_roundtrip_proptest(
            input_size in 1usize..=8,
            d_state in 1usize..=8,
            dt_rank in 1usize..=4,
            seed in 0u64..200,
        ) {
            use rand::{RngExt, SeedableRng};
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let n = input_size * (3 * d_state + 2 * dt_rank + 2);
            let flat: Vec<f64> = (0..n).map(|_| rng.random_range(-5.0..5.0)).collect();

            let x_proj_rows = dt_rank + 2 * d_state;
            let mut layer = MambaLayer {
                input_size,
                d_state,
                dt_rank,
                x_proj_w: nalgebra::DMatrix::zeros(x_proj_rows, input_size),
                dt_proj_w: nalgebra::DMatrix::zeros(input_size, dt_rank),
                dt_proj_b: nalgebra::DVector::zeros(input_size),
                a_log: nalgebra::DMatrix::zeros(input_size, d_state),
                d_skip: nalgebra::DVector::zeros(input_size),
            };
            let cursor = layer.from_flat(&flat);
            prop_assert_eq!(cursor, n);
            prop_assert_eq!(layer.n_params(), n);

            let back = layer.to_flat();
            prop_assert_eq!(back.len(), n);
            for i in 0..n {
                prop_assert_eq!(back[i], flat[i]);
            }
        }
    }

    proptest! {
        #[test]
        fn mamba_forward_finite_on_finite_inputs(
            d_inner in 1usize..=4,
            d_state in 1usize..=4,
            dt_rank in 1usize..=3,
            seed in 0u64..1000,
        ) {
            use rand::{RngExt, SeedableRng};
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let rand_vec = |n: usize, rng: &mut rand::rngs::StdRng| -> Vec<f64> {
                (0..n).map(|_| rng.random_range(-1.0..1.0)).collect()
            };
            let x_proj_w = nalgebra::DMatrix::from_row_slice(dt_rank + 2 * d_state, d_inner,
                &rand_vec((dt_rank + 2 * d_state) * d_inner, &mut rng));
            let dt_proj_w = nalgebra::DMatrix::from_row_slice(d_inner, dt_rank,
                &rand_vec(d_inner * dt_rank, &mut rng));
            let dt_proj_b = nalgebra::DVector::from_row_slice(&rand_vec(d_inner, &mut rng));
            let a_log = nalgebra::DMatrix::from_row_slice(d_inner, d_state, &rand_vec(d_inner * d_state, &mut rng));
            let d_skip = nalgebra::DVector::from_row_slice(&rand_vec(d_inner, &mut rng));

            let layer = MambaLayer {
                input_size: d_inner, d_state, dt_rank,
                x_proj_w, dt_proj_w, dt_proj_b, a_log, d_skip,
            };
            let x: Vec<f64> = rand_vec(d_inner, &mut rng);
            let mut h = nalgebra::DMatrix::<f64>::zeros(d_inner, d_state);

            for _ in 0..50 {
                let y = layer.forward(&x, &mut h);
                for v in &y {
                    prop_assert!(v.is_finite(), "y not finite: {v}");
                }
                for i in 0..d_inner {
                    for j in 0..d_state {
                        prop_assert!(h[(i, j)].is_finite(), "h[{i}, {j}] not finite");
                    }
                }
            }
        }
    }
}

#[test]
fn scaled_pi_requires_output_size_1() {
    assert!(NeuralNetModel::validate_output_size(1, OutputParam::ScaledPi, "<t>").is_ok());
    assert!(NeuralNetModel::validate_output_size(2, OutputParam::ScaledPi, "<t>").is_err());
}

#[test]
fn delta_requires_output_size_1() {
    assert!(NeuralNetModel::validate_output_size(1, OutputParam::Delta, "<t>").is_ok());
    assert!(NeuralNetModel::validate_output_size(2, OutputParam::Delta, "<t>").is_err());
}

#[test]
fn scaled_pi_and_delta_require_tanh_last_activation() {
    for p in [OutputParam::ScaledPi, OutputParam::Delta] {
        assert!(NeuralNetModel::validate_output_activation(Activation::Tanh, p, "<t>").is_ok());
        assert!(NeuralNetModel::validate_output_activation(Activation::Linear, p, "<t>").is_err());
    }
}

#[test]
fn output_param_default_is_atan2_signed() {
    let p: OutputParam = OutputParam::default();
    assert_eq!(p, OutputParam::Atan2Signed);
}

#[test]
fn output_param_serde_round_trip() {
    let p = OutputParam::AcosTanh;
    let s = serde_json::to_string(&p).unwrap();
    assert_eq!(s, "\"acos_tanh\"");
    let back: OutputParam = serde_json::from_str(&s).unwrap();
    assert_eq!(back, p);

    let p2 = OutputParam::Atan2Signed;
    let s2 = serde_json::to_string(&p2).unwrap();
    assert_eq!(s2, "\"atan2_signed\"");

    let p3 = OutputParam::ScaledPi;
    let s3 = serde_json::to_string(&p3).unwrap();
    assert_eq!(s3, "\"scaled_pi\"");
    let back3: OutputParam = serde_json::from_str(&s3).unwrap();
    assert_eq!(back3, p3);

    let p4 = OutputParam::Delta;
    let s4 = serde_json::to_string(&p4).unwrap();
    assert_eq!(s4, "\"delta\"");
    let back4: OutputParam = serde_json::from_str(&s4).unwrap();
    assert_eq!(back4, p4);
}

#[test]
fn output_param_persists_through_v2_json_round_trip() {
    let arch = vec![LayerSpec::Dense {
        input_size: 3,
        output_size: 1,
        activation: Activation::Tanh,
    }];
    let layers = vec![Layer::Dense(DenseLayer {
        w: vec![vec![0.1, 0.2, 0.3]],
        b: vec![0.4],
        activation: Activation::Tanh,
    })];
    let original = NeuralNetModel {
        architecture: arch,
        layer_sizes: vec![3, 1],
        layers,
        input_mask: None,
        ablated_input: None,
        ablated_value: 0.0,
        output_param: OutputParam::AcosTanh,
        scaled_pi_n: default_scaled_pi_n(),
        delta_max: default_delta_max(),
        normalization: DEFAULT_NORMALIZATION.to_vec(),
    };

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.json");
    original.save_json(path.to_str().unwrap()).unwrap();
    let loaded = NeuralNetModel::load(path.to_str().unwrap()).unwrap();

    assert_eq!(loaded.output_param, OutputParam::AcosTanh);
}

#[test]
fn scaled_pi_knobs_persist_through_v2_json_round_trip() {
    let arch = vec![LayerSpec::Dense {
        input_size: 3,
        output_size: 1,
        activation: Activation::Tanh,
    }];
    let layers = vec![Layer::Dense(DenseLayer {
        w: vec![vec![0.1, 0.2, 0.3]],
        b: vec![0.4],
        activation: Activation::Tanh,
    })];
    let original = NeuralNetModel {
        architecture: arch,
        layer_sizes: vec![3, 1],
        layers,
        input_mask: None,
        ablated_input: None,
        ablated_value: 0.0,
        output_param: OutputParam::ScaledPi,
        scaled_pi_n: 2.0,
        delta_max: 0.7,
        normalization: DEFAULT_NORMALIZATION.to_vec(),
    };

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.json");
    original.save_json(path.to_str().unwrap()).unwrap();
    let loaded = NeuralNetModel::load(path.to_str().unwrap()).unwrap();

    assert_eq!(loaded.output_param, OutputParam::ScaledPi);
    assert!(
        (loaded.scaled_pi_n - 2.0).abs() < 1e-15,
        "scaled_pi_n: {}",
        loaded.scaled_pi_n
    );
    assert!(
        (loaded.delta_max - 0.7).abs() < 1e-15,
        "delta_max: {}",
        loaded.delta_max
    );
}

#[test]
fn cfc_json_round_trip_bit_identical() {
    let arch = vec![
        LayerSpec::Dense {
            input_size: 3,
            output_size: 4,
            activation: Activation::Tanh,
        },
        LayerSpec::Cfc {
            input_size: 4,
            hidden_size: 4,
            backbone_units: 5,
        },
        LayerSpec::Dense {
            input_size: 4,
            output_size: 2,
            activation: Activation::Linear,
        },
    ];
    let n: usize = 3 * 4 + 4 + (5 * 8 + 5 + 4 * (4 * 5 + 4)) + (4 * 2 + 2);
    let flat: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001 - 0.2).collect();
    let model = NeuralNetModel::from_flat_weights_v2(
        &flat,
        &arch,
        None,
        OutputParam::default(),
        default_scaled_pi_n(),
        default_delta_max(),
    )
    .unwrap();
    assert_eq!(model.n_params(), n);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cfc_rt.json");
    model.save_json(path.to_str().unwrap()).unwrap();
    let loaded = NeuralNetModel::load(path.to_str().unwrap()).unwrap();
    assert_eq!(loaded.to_flat_weights(), model.to_flat_weights());
}

#[test]
fn output_param_absent_in_json_loads_as_atan2_signed() {
    let json = r#"{
            "format_version": 2,
            "architecture": [{"type": "dense", "input_size": 2, "output_size": 2, "activation": "linear"}],
            "weights": {
                "layer_0": {"w": [[0.1, 0.2], [0.3, 0.4]], "b": [0.0, 0.0]}
            }
        }"#;
    let m = NeuralNetModel::from_json_str(json, "<test>").unwrap();
    assert_eq!(m.output_param, OutputParam::Atan2Signed);
}

#[test]
fn acos_tanh_with_non_tanh_activation_rejected_at_v2_json_load() {
    let json = r#"{
            "format_version": 2,
            "output_param": "acos_tanh",
            "architecture": [{"type": "dense", "input_size": 2, "output_size": 1, "activation": "linear"}],
            "weights": {
                "layer_0": {"w": [[0.1, 0.2]], "b": [0.0]}
            }
        }"#;
    let result = NeuralNetModel::from_json_str(json, "<test>");
    assert!(result.is_err());
    let msg = format!("{:?}", result.unwrap_err());
    assert!(
        msg.contains("AcosTanh"),
        "expected AcosTanh in error, got: {}",
        msg
    );
    assert!(msg.contains("Tanh"), "expected Tanh in error, got: {}", msg);
}

#[test]
fn acos_tanh_with_asinh_activation_rejected_at_v2_json_load() {
    let json = r#"{
            "format_version": 2,
            "output_param": "acos_tanh",
            "architecture": [{"type": "dense", "input_size": 2, "output_size": 1, "activation": "asinh"}],
            "weights": {
                "layer_0": {"w": [[0.1, 0.2]], "b": [0.0]}
            }
        }"#;
    let result = NeuralNetModel::from_json_str(json, "<test>");
    assert!(result.is_err());
}

#[test]
fn acos_tanh_with_tanh_activation_accepted_at_v2_json_load() {
    let json = r#"{
            "format_version": 2,
            "output_param": "acos_tanh",
            "architecture": [{"type": "dense", "input_size": 2, "output_size": 1, "activation": "tanh"}],
            "weights": {
                "layer_0": {"w": [[0.1, 0.2]], "b": [0.0]}
            }
        }"#;
    let m = NeuralNetModel::from_json_str(json, "<test>").unwrap();
    assert_eq!(m.output_param, OutputParam::AcosTanh);
}
