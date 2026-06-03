use super::*;
use std::io::Write;
use toml::Value;

fn val(s: &str) -> Value {
    toml::from_str::<Value>(s).unwrap()
}

fn write_temp_toml(dir: &std::path::Path, name: &str, content: &str) -> std::path::PathBuf {
    let path = dir.join(name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::File::create(&path)
        .unwrap()
        .write_all(content.as_bytes())
        .unwrap();
    path
}

// ─── piecewise_constant resolver tests ───

#[test]
fn piecewise_flat_keys_override_bank_angles_array() {
    // Regression: a `bank_angles = [...]` array used to early-return,
    // silently ignoring `bank_angle_N` overrides. The GA writes its
    // chromosome as flat `bank_angle_N` keys, so flat keys must overlay
    // the array (precedence: flat keys > array > default).
    let pc: TomlPiecewiseConstantParams =
        toml::from_str("bank_angles = [10.0, 20.0, 30.0]\nbank_angle_1 = 99.0").unwrap();
    assert_eq!(
        pc.resolve_bank_angles_deg().unwrap(),
        vec![10.0, 99.0, 30.0]
    );
}

#[test]
fn piecewise_bank_angles_array_without_overrides_unchanged() {
    let pc: TomlPiecewiseConstantParams =
        toml::from_str("bank_angles = [10.0, 20.0, 30.0]").unwrap();
    assert_eq!(
        pc.resolve_bank_angles_deg().unwrap(),
        vec![10.0, 20.0, 30.0]
    );
}

// ─── deep_merge tests ───

#[test]
fn test_deep_merge_scalar_replacement() {
    let mut base = val("x = 1");
    let overlay = val("x = 99");
    deep_merge(&mut base, overlay);
    assert_eq!(base["x"].as_integer().unwrap(), 99);
}

#[test]
fn test_deep_merge_array_replacement() {
    let mut base = val("x = [1, 2, 3]");
    let overlay = val("x = [42]");
    deep_merge(&mut base, overlay);
    let arr = base["x"].as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0].as_integer().unwrap(), 42);
}

#[test]
fn test_deep_merge_table_recursion() {
    let mut base = val("[a]\nx = 1\ny = 2");
    let overlay = val("[a]\ny = 99\nz = 3");
    deep_merge(&mut base, overlay);
    let a = base["a"].as_table().unwrap();
    assert_eq!(a["x"].as_integer().unwrap(), 1); // kept from base
    assert_eq!(a["y"].as_integer().unwrap(), 99); // overlay wins
    assert_eq!(a["z"].as_integer().unwrap(), 3); // added from overlay
}

#[test]
fn test_deep_merge_nested_tables() {
    let mut base = val("[a.b]\nx = 1");
    let overlay = val("[a.b]\ny = 2\n[a.c]\nz = 3");
    deep_merge(&mut base, overlay);
    assert_eq!(base["a"]["b"]["x"].as_integer().unwrap(), 1);
    assert_eq!(base["a"]["b"]["y"].as_integer().unwrap(), 2);
    assert_eq!(base["a"]["c"]["z"].as_integer().unwrap(), 3);
}

#[test]
fn test_deep_merge_overlay_adds_new_top_level() {
    let mut base = val("x = 1");
    let overlay = val("y = 2");
    deep_merge(&mut base, overlay);
    assert_eq!(base["x"].as_integer().unwrap(), 1);
    assert_eq!(base["y"].as_integer().unwrap(), 2);
}

// ─── resolve_toml_bases tests ───

#[test]
fn test_resolve_single_base() {
    let dir = tempfile::tempdir().unwrap();
    write_temp_toml(dir.path(), "parent.toml", "x = 1\ny = 2");
    let child_path = write_temp_toml(
        dir.path(),
        "child.toml",
        "base = \"parent.toml\"\ny = 99\nz = 3",
    );

    let content = std::fs::read_to_string(&child_path).unwrap();
    let root: Value = toml::from_str(&content).unwrap();
    let mut visited = HashSet::new();
    let result = resolve_toml_bases(root, &child_path, &mut visited).unwrap();

    assert_eq!(result["x"].as_integer().unwrap(), 1); // from parent
    assert_eq!(result["y"].as_integer().unwrap(), 99); // child wins
    assert_eq!(result["z"].as_integer().unwrap(), 3); // child only
    assert!(result.get("base").is_none()); // base key stripped
}

#[test]
fn test_resolve_multiple_bases_merge_order() {
    let dir = tempfile::tempdir().unwrap();
    write_temp_toml(dir.path(), "a.toml", "x = 1\ny = 10");
    write_temp_toml(dir.path(), "b.toml", "y = 20\nz = 30");
    let child_path = write_temp_toml(
        dir.path(),
        "child.toml",
        "base = [\"a.toml\", \"b.toml\"]\nz = 99",
    );

    let content = std::fs::read_to_string(&child_path).unwrap();
    let root: Value = toml::from_str(&content).unwrap();
    let mut visited = HashSet::new();
    let result = resolve_toml_bases(root, &child_path, &mut visited).unwrap();

    assert_eq!(result["x"].as_integer().unwrap(), 1); // from a
    assert_eq!(result["y"].as_integer().unwrap(), 20); // b wins over a
    assert_eq!(result["z"].as_integer().unwrap(), 99); // child wins over b
}

#[test]
fn test_resolve_recursive_base() {
    let dir = tempfile::tempdir().unwrap();
    write_temp_toml(dir.path(), "grandparent.toml", "x = 1");
    write_temp_toml(
        dir.path(),
        "parent.toml",
        "base = \"grandparent.toml\"\ny = 2",
    );
    let child_path = write_temp_toml(dir.path(), "child.toml", "base = \"parent.toml\"\nz = 3");

    let content = std::fs::read_to_string(&child_path).unwrap();
    let root: Value = toml::from_str(&content).unwrap();
    let mut visited = HashSet::new();
    let result = resolve_toml_bases(root, &child_path, &mut visited).unwrap();

    assert_eq!(result["x"].as_integer().unwrap(), 1);
    assert_eq!(result["y"].as_integer().unwrap(), 2);
    assert_eq!(result["z"].as_integer().unwrap(), 3);
}

#[test]
fn test_resolve_cycle_detection() {
    let dir = tempfile::tempdir().unwrap();
    write_temp_toml(dir.path(), "a.toml", "base = \"b.toml\"\nx = 1");
    write_temp_toml(dir.path(), "b.toml", "base = \"a.toml\"\ny = 2");

    let a_path = dir.path().join("a.toml");
    let content = std::fs::read_to_string(&a_path).unwrap();
    let root: Value = toml::from_str(&content).unwrap();
    let mut visited = HashSet::new();
    let result = resolve_toml_bases(root, &a_path, &mut visited);

    assert!(result.is_err());
    let err_msg = result.unwrap_err().0;
    assert!(err_msg.contains("Cycle detected") || err_msg.contains("already visited"));
}

#[test]
fn test_resolve_missing_base_error() {
    let dir = tempfile::tempdir().unwrap();
    let child_path = write_temp_toml(
        dir.path(),
        "child.toml",
        "base = \"nonexistent.toml\"\nx = 1",
    );

    let content = std::fs::read_to_string(&child_path).unwrap();
    let root: Value = toml::from_str(&content).unwrap();
    let mut visited = HashSet::new();
    let result = resolve_toml_bases(root, &child_path, &mut visited);

    assert!(result.is_err());
    let err_msg = result.unwrap_err().0;
    assert!(err_msg.contains("Cannot read base"));
    assert!(err_msg.contains("nonexistent.toml"));
}

#[test]
fn test_resolve_no_base_passthrough() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_temp_toml(dir.path(), "standalone.toml", "x = 1\ny = 2");

    let content = std::fs::read_to_string(&path).unwrap();
    let root: Value = toml::from_str(&content).unwrap();
    let mut visited = HashSet::new();
    let result = resolve_toml_bases(root.clone(), &path, &mut visited).unwrap();

    assert_eq!(result, root);
}

#[test]
fn test_resolve_base_single_string() {
    let dir = tempfile::tempdir().unwrap();
    write_temp_toml(dir.path(), "parent.toml", "x = 1");
    let child_path = write_temp_toml(dir.path(), "child.toml", "base = \"parent.toml\"\ny = 2");

    let content = std::fs::read_to_string(&child_path).unwrap();
    let root: Value = toml::from_str(&content).unwrap();
    let mut visited = HashSet::new();
    let result = resolve_toml_bases(root, &child_path, &mut visited).unwrap();

    assert_eq!(result["x"].as_integer().unwrap(), 1);
    assert_eq!(result["y"].as_integer().unwrap(), 2);
}

// ─── integration section tests ───

#[test]
fn parse_integration_section_adaptive() {
    let toml_str = r#"
            [mission]
            type = "aerocapture"
            phase = "full"

            [planet]
            name = "mars"
            mu = 4.282829e13
            equatorial_radius = 3393940.0
            polar_radius = 3376780.0
            omega = 7.088218e-5
            j2 = 1.958616e-3

            [guidance]
            type = "ftc"

            [data]
            base_dir = "."
            output_dir = "."

            [integration]
            mode = "adaptive"
            rtol = 1e-8
            initial_dt = 0.05
            min_dt = 1e-8
            max_dt = 1.5
        "#;
    let (_, toml) = SimInput::from_toml(toml_str).expect("parse");
    let integ = toml.integration.unwrap();
    assert_eq!(integ.mode, "adaptive");
    assert!((integ.rtol.unwrap() - 1e-8).abs() < 1e-15);
    assert!((integ.initial_dt.unwrap() - 0.05).abs() < 1e-15);
    assert!((integ.min_dt.unwrap() - 1e-8).abs() < 1e-15);
    assert!((integ.max_dt.unwrap() - 1.5).abs() < 1e-15);
}

#[test]
fn parse_integration_section_absent_defaults_to_none() {
    let toml_str = r#"
            [mission]
            type = "aerocapture"
            phase = "full"

            [planet]
            name = "mars"
            mu = 4.282829e13
            equatorial_radius = 3393940.0
            polar_radius = 3376780.0
            omega = 7.088218e-5
            j2 = 1.958616e-3

            [guidance]
            type = "ftc"

            [data]
            base_dir = "."
            output_dir = "."
        "#;
    let (_, toml) = SimInput::from_toml(toml_str).expect("parse");
    assert!(toml.integration.is_none());
}

#[test]
fn integration_mode_from_toml_none_gives_fixed() {
    let mode = IntegrationMode::from_toml(&None, 1.0).unwrap();
    assert!(matches!(mode, IntegrationMode::FixedGill));
}

#[test]
fn integration_mode_from_toml_fixed_gives_fixed() {
    let cfg = Some(TomlIntegration {
        mode: "fixed".to_string(),
        rtol: None,
        initial_dt: None,
        min_dt: None,
        max_dt: None,
    });
    let mode = IntegrationMode::from_toml(&cfg, 1.0).unwrap();
    assert!(matches!(mode, IntegrationMode::FixedGill));
}

#[test]
fn integration_mode_from_toml_adaptive_defaults() {
    let cfg = Some(TomlIntegration {
        mode: "adaptive".to_string(),
        rtol: None,
        initial_dt: None,
        min_dt: None,
        max_dt: None,
    });
    let mode = IntegrationMode::from_toml(&cfg, 2.0).unwrap();
    match mode {
        IntegrationMode::AdaptiveDopri45(ac) => {
            assert!((ac.rtol - 1e-6).abs() < 1e-15);
            assert!((ac.initial_dt - 0.1).abs() < 1e-15);
            assert!((ac.min_dt - 1e-6).abs() < 1e-15);
            assert!((ac.max_dt - 2.0).abs() < 1e-15); // falls back to integration_period
        }
        _ => panic!("expected AdaptiveDopri45"),
    }
}

#[test]
fn unknown_integration_mode_errors() {
    let cfg = Some(TomlIntegration {
        mode: "adaptiv".to_string(),
        rtol: None,
        initial_dt: None,
        min_dt: None,
        max_dt: None,
    });
    assert!(IntegrationMode::from_toml(&cfg, 1.0).is_err());
}

#[test]
fn integration_mode_from_toml_adaptive_explicit() {
    let cfg = Some(TomlIntegration {
        mode: "adaptive".to_string(),
        rtol: Some(1e-8),
        initial_dt: Some(0.05),
        min_dt: Some(1e-8),
        max_dt: Some(1.5),
    });
    let mode = IntegrationMode::from_toml(&cfg, 2.0).unwrap();
    match mode {
        IntegrationMode::AdaptiveDopri45(ac) => {
            assert!((ac.rtol - 1e-8).abs() < 1e-15);
            assert!((ac.initial_dt - 0.05).abs() < 1e-15);
            assert!((ac.min_dt - 1e-8).abs() < 1e-15);
            assert!((ac.max_dt - 1.5).abs() < 1e-15);
        }
        _ => panic!("expected AdaptiveDopri45"),
    }
}

// ─── density_perturbation TOML parsing tests ───

#[test]
fn test_density_perturbation_toml_parsing() {
    let toml_str = r#"
            seed = 42
            [density_perturbation]
            level = "high"
        "#;
    let mc: TomlMonteCarlo = toml::from_str(toml_str).unwrap();
    assert!(mc.density_perturbation.is_some());
    let dp = mc.density_perturbation.unwrap();
    assert_eq!(dp.level, "high");
}

#[test]
fn test_density_perturbation_toml_custom() {
    let toml_str = r#"
            seed = 42
            [density_perturbation]
            level = "custom"
            tau = 45.0
            sigma = 0.15
        "#;
    let mc: TomlMonteCarlo = toml::from_str(toml_str).unwrap();
    let dp = mc.density_perturbation.unwrap();
    assert_eq!(dp.level, "custom");
    assert_eq!(*dp.custom.get("tau").unwrap(), 45.0);
    assert_eq!(*dp.custom.get("sigma").unwrap(), 0.15);
}

#[test]
fn test_density_perturbation_toml_absent() {
    let toml_str = r#"
            seed = 42
        "#;
    let mc: TomlMonteCarlo = toml::from_str(toml_str).unwrap();
    assert!(mc.density_perturbation.is_none());
}

// --- wind domain tests ---

#[test]
fn test_wind_toml_level() {
    let toml_str = r#"
            seed = 42
            [wind]
            level = "high"
        "#;
    let mc: TomlMonteCarlo = toml::from_str(toml_str).unwrap();
    assert!(mc.wind.is_some());
    assert_eq!(mc.wind.unwrap().level, "high");
}

#[test]
fn test_wind_toml_backward_compat() {
    // Old-style config without level field should still parse
    let toml_str = r#"
            seed = 42
            [wind]
            scale_min = 0.3
            scale_max = 1.7
            direction_bias_deg = 15.0
        "#;
    let mc: TomlMonteCarlo = toml::from_str(toml_str).unwrap();
    let w = mc.wind.unwrap();
    assert_eq!(w.level, "medium"); // default
    assert_eq!(*w.custom.get("scale_min").unwrap(), 0.3);
    assert_eq!(*w.custom.get("scale_max").unwrap(), 1.7);
    assert_eq!(*w.custom.get("direction_bias_deg").unwrap(), 15.0);
}

#[test]
fn test_wind_toml_absent() {
    let toml_str = r#"
            seed = 42
        "#;
    let mc: TomlMonteCarlo = toml::from_str(toml_str).unwrap();
    assert!(mc.wind.is_none());
}

// ─── v2 [[network.architecture]] parser tests ───

#[test]
fn network_architecture_v2_parses() {
    let toml = r#"
[network]
input_mask = [0, 1, 2]

[[network.architecture]]
type = "dense"
input_size = 3
output_size = 4
activation = "tanh"

[[network.architecture]]
type = "gru"
input_size = 4
hidden_size = 4

[[network.architecture]]
type = "dense"
input_size = 4
output_size = 2
activation = "linear"
"#;
    #[derive(Deserialize)]
    struct Wrapper {
        network: TomlNetwork,
    }
    let wrapper: Wrapper = toml::from_str(toml).expect("TOML parse");
    let arch = wrapper
        .network
        .architecture
        .expect("architecture v2 path present");
    assert_eq!(arch.len(), 3);
    match &arch[1] {
        TomlLayerSpec::Gru {
            input_size,
            hidden_size,
        } => {
            assert_eq!(*input_size, 4);
            assert_eq!(*hidden_size, 4);
        }
        _ => panic!("expected Gru at index 1"),
    }
    match &arch[0] {
        TomlLayerSpec::Dense {
            input_size,
            output_size,
            activation,
        } => {
            assert_eq!(*input_size, 3);
            assert_eq!(*output_size, 4);
            assert_eq!(activation, "tanh");
        }
        _ => panic!("expected Dense at index 0"),
    }
}

#[test]
fn toml_normalization_override_parses() {
    let toml_str = r#"
[network]
normalization = [ { transform = "asinh", scale = 10.0, center = 0.0 }, { transform = "none", scale = 2.0, center = 1.0 } ]
"#;
    #[derive(Deserialize)]
    struct Wrapper {
        network: TomlNetwork,
    }
    let parsed: Wrapper = toml::from_str(toml_str).unwrap();
    let n = parsed.network.normalization.expect("normalization parsed");
    assert_eq!(n.len(), 2);
    assert_eq!(n[0].scale, 10.0);
    assert_eq!(n[0].transform, crate::data::neural::NormTransform::Asinh);
    assert_eq!(n[1].center, 1.0);
}

#[test]
fn network_architecture_v2_absent_stays_none() {
    // v1 path: no [[network.architecture]] block, existing configs must still parse
    let toml = r#"
[network]
input_mask = [0, 1, 2]
"#;
    #[derive(Deserialize)]
    struct Wrapper {
        network: TomlNetwork,
    }
    let wrapper: Wrapper = toml::from_str(toml).expect("TOML parse");
    assert!(wrapper.network.architecture.is_none());
    assert_eq!(wrapper.network.input_mask, Some(vec![0, 1, 2]));
}

#[test]
fn toml_layer_spec_to_layer_spec_dense() {
    use crate::data::neural::{Activation, LayerSpec};
    let toml_spec = TomlLayerSpec::Dense {
        input_size: 3,
        output_size: 4,
        activation: "tanh".to_string(),
    };
    match toml_spec.to_layer_spec().unwrap() {
        LayerSpec::Dense {
            input_size,
            output_size,
            activation,
        } => {
            assert_eq!(input_size, 3);
            assert_eq!(output_size, 4);
            assert_eq!(activation, Activation::Tanh);
        }
        _ => panic!("expected Dense"),
    }
}

#[test]
fn toml_layer_spec_to_layer_spec_gru() {
    use crate::data::neural::LayerSpec;
    let toml_spec = TomlLayerSpec::Gru {
        input_size: 4,
        hidden_size: 8,
    };
    match toml_spec.to_layer_spec().unwrap() {
        LayerSpec::Gru {
            input_size,
            hidden_size,
        } => {
            assert_eq!(input_size, 4);
            assert_eq!(hidden_size, 8);
        }
        _ => panic!("expected Gru"),
    }
}

#[test]
fn network_architecture_v2_parses_lstm() {
    let toml = r#"
[[network.architecture]]
type = "dense"
input_size = 3
output_size = 4
activation = "tanh"

[[network.architecture]]
type = "lstm"
input_size = 4
hidden_size = 8

[[network.architecture]]
type = "dense"
input_size = 8
output_size = 2
activation = "linear"
"#;
    #[derive(Deserialize)]
    struct Wrapper {
        network: TomlNetwork,
    }
    let wrapper: Wrapper = toml::from_str(toml).expect("TOML parse");
    let arch = wrapper
        .network
        .architecture
        .expect("architecture v2 path present");
    assert_eq!(arch.len(), 3);
    match &arch[1] {
        TomlLayerSpec::Lstm {
            input_size,
            hidden_size,
        } => {
            assert_eq!(*input_size, 4);
            assert_eq!(*hidden_size, 8);
        }
        _ => panic!("expected Lstm at index 1"),
    }

    // Also verify to_layer_spec() converts correctly
    let converted = arch[1].to_layer_spec().unwrap();
    match converted {
        crate::data::neural::LayerSpec::Lstm {
            input_size,
            hidden_size,
        } => {
            assert_eq!(input_size, 4);
            assert_eq!(hidden_size, 8);
        }
        _ => panic!("expected LayerSpec::Lstm"),
    }
}

#[test]
fn toml_layer_spec_dense_unknown_activation_errors() {
    let toml_spec = TomlLayerSpec::Dense {
        input_size: 3,
        output_size: 4,
        activation: "not_an_activation".to_string(),
    };
    assert!(toml_spec.to_layer_spec().is_err());
}

#[test]
fn toml_layer_spec_to_layer_spec_window() {
    let toml_spec = TomlLayerSpec::Window {
        input_size: 4,
        n_steps: 8,
    };
    match toml_spec.to_layer_spec().unwrap() {
        crate::data::neural::LayerSpec::Window {
            input_size,
            n_steps,
        } => {
            assert_eq!(input_size, 4);
            assert_eq!(n_steps, 8);
        }
        _ => panic!("expected LayerSpec::Window"),
    }
}

#[test]
fn toml_layer_spec_window_parses_from_toml_string() {
    let toml_str = r#"
[[network.architecture]]
type = "window"
input_size = 4
n_steps = 8
"#;
    #[derive(serde::Deserialize)]
    struct Wrapper {
        network: NetworkArch,
    }
    #[derive(serde::Deserialize)]
    struct NetworkArch {
        architecture: Vec<TomlLayerSpec>,
    }
    let parsed: Wrapper = toml::from_str(toml_str).unwrap();
    match &parsed.network.architecture[0] {
        TomlLayerSpec::Window {
            input_size,
            n_steps,
        } => {
            assert_eq!(*input_size, 4);
            assert_eq!(*n_steps, 8);
        }
        _ => panic!("expected TomlLayerSpec::Window"),
    }
}

#[test]
fn toml_layer_spec_window_rejects_zero_fields() {
    let zero_input = TomlLayerSpec::Window {
        input_size: 0,
        n_steps: 8,
    };
    assert!(zero_input.to_layer_spec().is_err());
    let zero_n_steps = TomlLayerSpec::Window {
        input_size: 4,
        n_steps: 0,
    };
    assert!(zero_n_steps.to_layer_spec().is_err());
}

#[test]
fn toml_layer_spec_transformer_parses() {
    let toml_str = r#"
[[network.architecture]]
type = "transformer"
d_model = 32
n_heads = 4
d_ffn = 64
n_seq = 64
"#;
    #[derive(serde::Deserialize)]
    struct NetworkWrapper {
        network: Network,
    }
    #[derive(serde::Deserialize)]
    struct Network {
        architecture: Vec<TomlLayerSpec>,
    }
    let w: NetworkWrapper = toml::from_str(toml_str).unwrap();
    assert_eq!(w.network.architecture.len(), 1);
    let spec = w.network.architecture[0].to_layer_spec().unwrap();
    match spec {
        crate::data::neural::LayerSpec::Transformer {
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
fn toml_layer_spec_transformer_rejects_bad_heads() {
    let toml_str = r#"
[[network.architecture]]
type = "transformer"
d_model = 33
n_heads = 4
d_ffn = 64
n_seq = 64
"#;
    #[derive(serde::Deserialize)]
    struct NetworkWrapper {
        network: Network,
    }
    #[derive(serde::Deserialize)]
    struct Network {
        architecture: Vec<TomlLayerSpec>,
    }
    let w: NetworkWrapper = toml::from_str(toml_str).unwrap();
    let err = w.network.architecture[0].to_layer_spec().unwrap_err();
    assert!(format!("{err}").contains("not divisible"));
}

#[test]
fn mamba_toml_resolves_dt_rank_from_input_size() {
    use crate::data::neural::LayerSpec;
    // input_size=32, omitted dt_rank -> max(1, 32/16) = 2
    let parsed: TomlLayerSpec = toml::from_str(
        r#"type = "mamba"
input_size = 32
d_state = 16
"#,
    )
    .unwrap();
    let spec = parsed.to_layer_spec().unwrap();
    match spec {
        LayerSpec::Mamba {
            input_size,
            d_state,
            dt_rank,
        } => {
            assert_eq!(input_size, 32);
            assert_eq!(d_state, 16);
            assert_eq!(dt_rank, 2);
        }
        _ => panic!("expected Mamba spec"),
    }
}

#[test]
fn mamba_toml_explicit_dt_rank_overrides_default() {
    use crate::data::neural::LayerSpec;
    let parsed: TomlLayerSpec = toml::from_str(
        r#"type = "mamba"
input_size = 32
d_state = 16
dt_rank = 8
"#,
    )
    .unwrap();
    let spec = parsed.to_layer_spec().unwrap();
    match spec {
        LayerSpec::Mamba { dt_rank, .. } => assert_eq!(dt_rank, 8),
        _ => panic!("expected Mamba spec"),
    }
}

#[test]
fn mamba_toml_rejects_dt_rank_larger_than_input_size() {
    let parsed: TomlLayerSpec = toml::from_str(
        r#"type = "mamba"
input_size = 8
d_state = 4
dt_rank = 16
"#,
    )
    .unwrap();
    let result = parsed.to_layer_spec();
    assert!(result.is_err());
    let msg = result.unwrap_err();
    assert!(
        msg.0.contains("dt_rank"),
        "error message should mention dt_rank: {msg}"
    );
}

#[test]
fn mamba_toml_rejects_zero_dims() {
    let parsed: TomlLayerSpec = toml::from_str(
        r#"type = "mamba"
input_size = 8
d_state = 0
"#,
    )
    .unwrap();
    assert!(parsed.to_layer_spec().is_err());
}

#[test]
fn mamba_toml_defaults_dt_rank_to_one_for_small_input() {
    use crate::data::neural::LayerSpec;
    // input_size=8, omitted dt_rank -> max(1, 8/16) = max(1, 0) = 1
    let parsed: TomlLayerSpec = toml::from_str(
        r#"type = "mamba"
input_size = 8
d_state = 4
"#,
    )
    .unwrap();
    let spec = parsed.to_layer_spec().unwrap();
    match spec {
        LayerSpec::Mamba { dt_rank, .. } => assert_eq!(dt_rank, 1),
        _ => panic!("expected Mamba"),
    }
}

// ─── per-field default value regression tests (Fix 7.3a) ───
// Pin each de-shared default so a future value change must update BOTH
// the default fn AND this test.

#[test]
fn defaults_simulation_fields() {
    let sim: TomlSimulation = toml::from_str("").unwrap();
    assert_eq!(sim.n_sims, 1);
    assert!(sim.save_results);
}

#[test]
fn defaults_ftc_security_capture() {
    let ftc: TomlFtcParams = toml::from_str("").unwrap();
    assert_eq!(ftc.security_capture, 1);
}

#[test]
fn defaults_pilot_time_constant() {
    // TomlPilot only deserializes as part of [vehicle.pilot], but we can use
    // the stand-alone Default impl as a proxy for the serde default.
    let p = TomlPilot::default();
    assert!((p.time_constant - 1.0).abs() < 1e-15);
}

#[test]
fn defaults_eq_glide_params() {
    let eg: TomlEqGlideParams = toml::from_str("").unwrap();
    assert!((eg.k_hdot_scale - 0.3).abs() < 1e-15);
    assert!((eg.velocity_bias_low - 0.3).abs() < 1e-15);
}

#[test]
fn defaults_energy_ctrl_kp() {
    let ec: TomlEnergyCtrlParams = toml::from_str("").unwrap();
    assert!((ec.kp - 1.0).abs() < 1e-15);
}

#[test]
fn defaults_pred_guid_params() {
    let pg: TomlPredGuidParams = toml::from_str("").unwrap();
    assert!((pg.k_drag_low - 0.3).abs() < 1e-15);
    assert!((pg.pdyn_threshold - 100.0).abs() < 1e-15);
}

#[test]
fn defaults_fnpag_bank_max_low_deg() {
    let fn_: TomlFnpagParams = toml::from_str("").unwrap();
    assert!((fn_.bank_max_low_deg - 100.0).abs() < 1e-15);
}

#[test]
fn defaults_thermal_limiter_all_one() {
    let tl: TomlThermalLimiterParams = toml::from_str("").unwrap();
    assert!((tl.heat_flux_activation - 1.0).abs() < 1e-15);
    assert!((tl.heat_load_activation - 1.0).abs() < 1e-15);
    assert!((tl.heat_flux_ramp_exponent - 1.0).abs() < 1e-15);
    assert!((tl.heat_load_ramp_exponent - 1.0).abs() < 1e-15);
}

#[test]
fn defaults_command_shaping_enabled() {
    let cs: TomlCommandShapingParams = toml::from_str("max_bank_acceleration = 5.0").unwrap();
    assert!(cs.enabled);
}
