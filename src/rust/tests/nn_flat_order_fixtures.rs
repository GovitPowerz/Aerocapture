//! Frozen flat-order fixtures for every NN layer type.
//!
//! Every deployed `best_model.json` and every PSO checkpoint chromosome encodes
//! the canonical per-layer flat order, so that order can never move. Each case
//! here freezes, for one small layer of each type, (a) the flat vector and
//! (b) the exact `save_json` bytes. The test pins both directions of the
//! name <-> flat-position mapping:
//!   - load `<case>.model.json` -> `to_flat_weights()` == `<case>.flat.json`
//!   - `from_flat_weights_v2(flat)` -> `save_json` bytes == `<case>.model.json`
//!   - load `<case>.model.json` -> `save_json` bytes == `<case>.model.json`
//!
//! Regenerate (only when the format is DELIBERATELY changed) with
//! `NN_FIXTURE_WRITE=1 cargo test --release --test nn_flat_order_fixtures`.

use std::path::PathBuf;

use aerocapture::data::neural::{Activation, LayerSpec, NeuralNetModel, OutputParam};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/flat_order")
}

fn dense(input_size: usize, output_size: usize, activation: Activation) -> LayerSpec {
    LayerSpec::Dense {
        input_size,
        output_size,
        activation,
    }
}

/// (case name, architecture, expected n_params). Output width is 2 everywhere
/// (Atan2Signed head). The hand-computed counts freeze `n_params` too.
fn cases() -> Vec<(&'static str, Vec<LayerSpec>, usize)> {
    let mut v = vec![
        ("dense", vec![dense(3, 2, Activation::Tanh)], 8),
        (
            "gru",
            vec![LayerSpec::Gru {
                input_size: 3,
                hidden_size: 2,
            }],
            42,
        ),
        (
            "lstm",
            vec![LayerSpec::Lstm {
                input_size: 3,
                hidden_size: 2,
            }],
            56,
        ),
        (
            "window",
            vec![
                LayerSpec::Window {
                    input_size: 2,
                    n_steps: 3,
                },
                dense(6, 2, Activation::Linear),
            ],
            14,
        ),
        (
            "transformer",
            vec![
                LayerSpec::Transformer {
                    d_model: 4,
                    n_heads: 2,
                    d_ffn: 6,
                    n_seq: 3,
                },
                dense(4, 2, Activation::Linear),
            ],
            164,
        ),
        (
            "mamba",
            vec![
                LayerSpec::Mamba {
                    input_size: 3,
                    d_state: 2,
                    dt_rank: 1,
                },
                dense(3, 2, Activation::Linear),
            ],
            38,
        ),
        (
            "cfc",
            vec![LayerSpec::Cfc {
                input_size: 3,
                hidden_size: 2,
                backbone_units: 4,
            }],
            64,
        ),
        (
            "slstm",
            vec![LayerSpec::Slstm {
                input_size: 3,
                hidden_size: 2,
            }],
            48,
        ),
        (
            "mlstm",
            vec![LayerSpec::Mlstm {
                input_size: 3,
                hidden_size: 2,
            }],
            40,
        ),
    ];
    for (name, disc, sm, n) in [
        ("mamba3_euler_real", "euler", "real", 38),
        ("mamba3_trapezoidal_real", "trapezoidal", "real", 41),
        ("mamba3_euler_complex", "euler", "complex", 44),
        ("mamba3_trapezoidal_complex", "trapezoidal", "complex", 47),
    ] {
        v.push((
            name,
            vec![
                LayerSpec::Mamba3 {
                    input_size: 3,
                    d_state: 2,
                    dt_rank: 1,
                    discretization: disc.to_string(),
                    state_mode: sm.to_string(),
                },
                dense(3, 2, Activation::Linear),
            ],
            n,
        ));
    }
    v
}

fn build(arch: &[LayerSpec], flat: &[f64]) -> NeuralNetModel {
    NeuralNetModel::from_flat_weights_v2(flat, arch, None, OutputParam::default(), 1.0, 0.35)
        .expect("from_flat_weights_v2")
}

fn save_bytes(model: &NeuralNetModel) -> String {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    model.save_json(tmp.path().to_str().unwrap()).unwrap();
    std::fs::read_to_string(tmp.path()).unwrap()
}

/// Distinct, non-zero values so any permutation of the flat order is visible.
fn pattern(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.5 + 0.001 * i as f64).collect()
}

#[test]
fn flat_order_fixtures_pin_every_layer_type() {
    let dir = fixture_dir();
    let write = std::env::var("NN_FIXTURE_WRITE").is_ok();
    if write {
        std::fs::create_dir_all(&dir).unwrap();
    }
    for (name, arch, n) in cases() {
        let model_path = dir.join(format!("{name}.model.json"));
        let flat_path = dir.join(format!("{name}.flat.json"));

        let flat = pattern(n);
        let model = build(&arch, &flat);
        assert_eq!(model.n_params(), n, "{name}: n_params");
        assert_eq!(model.to_flat_weights(), flat, "{name}: pattern round-trip");

        if write {
            std::fs::write(&model_path, save_bytes(&model)).unwrap();
            std::fs::write(&flat_path, serde_json::to_string(&flat).unwrap()).unwrap();
        }

        let frozen_json = std::fs::read_to_string(&model_path)
            .unwrap_or_else(|e| panic!("{name}: missing fixture {}: {e}", model_path.display()));
        let frozen_flat: Vec<f64> =
            serde_json::from_str(&std::fs::read_to_string(&flat_path).unwrap()).unwrap();
        assert_eq!(frozen_flat.len(), n, "{name}: fixture width");

        // JSON -> flat pins name -> position.
        let loaded = NeuralNetModel::from_json_str(&frozen_json, name).unwrap();
        assert_eq!(
            loaded.to_flat_weights(),
            frozen_flat,
            "{name}: json -> flat"
        );
        // flat -> JSON pins position -> name (and the exact serialized bytes).
        assert_eq!(
            save_bytes(&build(&arch, &frozen_flat)),
            frozen_json,
            "{name}: flat -> json bytes"
        );
        // JSON -> JSON must be byte-stable.
        assert_eq!(
            save_bytes(&loaded),
            frozen_json,
            "{name}: json -> json bytes"
        );
    }
}
