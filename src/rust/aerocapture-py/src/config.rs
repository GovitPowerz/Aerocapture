//! TOML config loading with runtime overrides.
//!
//! Provides `apply_override` to patch a parsed TOML value tree, and
//! `load_and_override` to go from raw TOML text + overrides to fully
//! constructed `SimInput` + `SimData`.

use aerocapture::config::SimInput;
use aerocapture::data::SimData;
use toml::{Table, Value};

/// Typed override value extracted from a Python dict.
#[derive(Debug, Clone)]
pub enum OverrideValue {
    Float(f64),
    Int(i64),
    Str(String),
    Bool(bool),
}

/// Walk a dot-separated `key_path` into a TOML value tree, creating
/// intermediate tables as needed, then set the leaf to `value`.
///
/// Type coercion rules:
/// - Setting an existing Float field with an `Int` value → coerce to Float.
/// - Setting an existing field with a mismatched type → `Err`.
/// - Setting a new (non-existent) key → accept any type.
pub fn apply_override(root: &mut Value, key_path: &str, value: &OverrideValue) -> Result<(), String> {
    let parts: Vec<&str> = key_path.split('.').collect();
    if parts.is_empty() {
        return Err("Empty key path".to_string());
    }

    // Walk to the parent table, creating intermediates.
    let mut current = root;
    for &part in &parts[..parts.len() - 1] {
        if !current.is_table() {
            return Err(format!("Path component '{}' is not a table", part));
        }
        let table = current.as_table_mut().unwrap();
        if !table.contains_key(part) {
            table.insert(part.to_string(), Value::Table(toml::map::Map::new()));
        }
        current = table.get_mut(part).unwrap();
    }

    let leaf_key = parts.last().unwrap();

    if !current.is_table() {
        return Err(format!("Parent of '{}' is not a table", key_path));
    }
    let table = current.as_table_mut().unwrap();

    let new_value = match value {
        OverrideValue::Float(v) => Value::Float(*v),
        OverrideValue::Int(v) => Value::Integer(*v),
        OverrideValue::Str(v) => Value::String(v.clone()),
        OverrideValue::Bool(v) => Value::Boolean(*v),
    };

    // Check type compatibility with existing value.
    if let Some(existing) = table.get(*leaf_key) {
        match (existing, value) {
            // Same-type cases — always OK.
            (Value::Float(_), OverrideValue::Float(_))
            | (Value::Integer(_), OverrideValue::Int(_))
            | (Value::String(_), OverrideValue::Str(_))
            | (Value::Boolean(_), OverrideValue::Bool(_)) => {}
            // Int → Float coercion.
            (Value::Float(_), OverrideValue::Int(v)) => {
                table.insert((*leaf_key).to_string(), Value::Float(*v as f64));
                return Ok(());
            }
            // Type mismatch.
            _ => {
                return Err(format!(
                    "Type mismatch for '{}': existing type {:?}, override type {:?}",
                    key_path,
                    existing.type_str(),
                    override_type_name(value),
                ));
            }
        }
    }

    table.insert((*leaf_key).to_string(), new_value);
    Ok(())
}

fn override_type_name(v: &OverrideValue) -> &'static str {
    match v {
        OverrideValue::Float(_) => "float",
        OverrideValue::Int(_) => "integer",
        OverrideValue::Str(_) => "string",
        OverrideValue::Bool(_) => "boolean",
    }
}

/// Parse TOML content, apply a list of overrides, and construct `SimInput` + `SimData`.
pub fn load_and_override(
    toml_content: &str,
    overrides: &[(String, OverrideValue)],
) -> Result<(SimInput, SimData), String> {
    // Parse into a generic TOML value tree so we can patch it.
    let table: Table =
        toml::from_str(toml_content).map_err(|e| format!("TOML parse error: {}", e))?;
    let mut root = Value::Table(table);

    for (key, value) in overrides {
        apply_override(&mut root, key, value)?;
    }

    // Serialize back to a TOML string so we can feed it through the normal pipeline.
    let patched = toml::to_string(&root).map_err(|e| format!("TOML serialize error: {}", e))?;

    let (sim_input, toml_config) =
        SimInput::from_toml(&patched).map_err(|e| format!("Config parse error: {}", e))?;
    let sim_data =
        SimData::from_toml(&toml_config, &sim_input).map_err(|e| format!("Data load error: {}", e))?;

    Ok((sim_input, sim_data))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_toml_tree() -> Value {
        let toml_str = r#"
[mission]
type = "aerocapture"
planet = "mars"
n_sims = 5

[guidance]
type = "ftc"
reference_trajectory = true
reference_bank_angle = 0.1

[guidance.equilibrium_glide]
k_hdot = 1.5
k_alt = 2.0
"#;
        let table: Table = toml::from_str(toml_str).unwrap();
        Value::Table(table)
    }

    #[test]
    fn apply_override_simple_float() {
        let mut root = sample_toml_tree();
        apply_override(&mut root, "guidance.reference_bank_angle", &OverrideValue::Float(45.0)).unwrap();
        assert_eq!(
            root["guidance"]["reference_bank_angle"].as_float().unwrap(),
            45.0
        );
    }

    #[test]
    fn apply_override_int_to_float_coercion() {
        let mut root = sample_toml_tree();
        // k_hdot is a float (1.5); setting it with Int should coerce to float.
        apply_override(
            &mut root,
            "guidance.equilibrium_glide.k_hdot",
            &OverrideValue::Int(3),
        )
        .unwrap();
        assert_eq!(
            root["guidance"]["equilibrium_glide"]["k_hdot"]
                .as_float()
                .unwrap(),
            3.0
        );
    }

    #[test]
    fn apply_override_type_mismatch_errors() {
        let mut root = sample_toml_tree();
        // reference_bank_angle is a float; setting with Str should fail.
        let result = apply_override(
            &mut root,
            "guidance.reference_bank_angle",
            &OverrideValue::Str("oops".to_string()),
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Type mismatch"));
    }

    #[test]
    fn apply_override_nested_deep_path() {
        let mut root = sample_toml_tree();
        apply_override(
            &mut root,
            "guidance.equilibrium_glide.k_hdot",
            &OverrideValue::Float(9.99),
        )
        .unwrap();
        assert_eq!(
            root["guidance"]["equilibrium_glide"]["k_hdot"]
                .as_float()
                .unwrap(),
            9.99
        );
    }

    #[test]
    fn apply_override_integer_field() {
        let mut root = sample_toml_tree();
        apply_override(&mut root, "mission.n_sims", &OverrideValue::Int(100)).unwrap();
        assert_eq!(root["mission"]["n_sims"].as_integer().unwrap(), 100);
    }

    #[test]
    fn apply_override_creates_new_key() {
        let mut root = sample_toml_tree();
        apply_override(
            &mut root,
            "guidance.equilibrium_glide.brand_new_param",
            &OverrideValue::Float(42.0),
        )
        .unwrap();
        assert_eq!(
            root["guidance"]["equilibrium_glide"]["brand_new_param"]
                .as_float()
                .unwrap(),
            42.0
        );
    }
}
