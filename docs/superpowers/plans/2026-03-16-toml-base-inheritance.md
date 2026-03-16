# TOML Base Inheritance Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate TOML config duplication by adding a `base` key for inheritance — mission-level and training-level shared configs get factored out, each leaf config shrinks ~70%.

**Architecture:** A `base` key in any TOML file references parent configs. The loader resolves bases recursively (relative to the declaring file), deep-merges them left-to-right, overlays the current file, strips `base`, then hands the flat tree to the existing serde pipeline. Both Rust (core crate) and Python (`load_toml_with_bases()`) implement the same logic.

**Tech Stack:** Rust (toml 0.9, serde), Python (tomllib), PyO3/maturin

**Spec:** `docs/superpowers/specs/2026-03-16-toml-base-inheritance-design.md`

---

## Chunk 1: Rust Core — deep_merge + resolve_toml_bases + from_toml_file

### Task 1: Implement `deep_merge()` with tests

**Files:**
- Modify: `src/rust/src/config.rs` — append new function + tests

- [ ] **Step 1: Write failing tests for `deep_merge()`**

Add to the bottom of `src/rust/src/config.rs`, inside a new or existing `#[cfg(test)] mod tests` block:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use toml::Value;

    fn val(s: &str) -> Value {
        s.parse::<Value>().unwrap()
    }

    #[test]
    fn test_deep_merge_scalar_replacement() {
        let mut base = val(r#"x = 1"#);
        let overlay = val(r#"x = 99"#);
        deep_merge(&mut base, overlay);
        assert_eq!(base["x"].as_integer().unwrap(), 99);
    }

    #[test]
    fn test_deep_merge_array_replacement() {
        let mut base = val(r#"a = [1, 2]"#);
        let overlay = val(r#"a = [3]"#);
        deep_merge(&mut base, overlay);
        assert_eq!(base["a"].as_array().unwrap().len(), 1);
        assert_eq!(base["a"][0].as_integer().unwrap(), 3);
    }

    #[test]
    fn test_deep_merge_table_recursion() {
        let mut base = val(r#"
            [a]
            x = 1
            y = 2
        "#);
        let overlay = val(r#"
            [a]
            y = 99
            z = 3
        "#);
        deep_merge(&mut base, overlay);
        assert_eq!(base["a"]["x"].as_integer().unwrap(), 1);  // kept from base
        assert_eq!(base["a"]["y"].as_integer().unwrap(), 99); // overridden
        assert_eq!(base["a"]["z"].as_integer().unwrap(), 3);  // added
    }

    #[test]
    fn test_deep_merge_nested_tables() {
        let mut base = val(r#"
            [a.b]
            x = 1
        "#);
        let overlay = val(r#"
            [a.b]
            y = 2
            [a.c]
            z = 3
        "#);
        deep_merge(&mut base, overlay);
        assert_eq!(base["a"]["b"]["x"].as_integer().unwrap(), 1);
        assert_eq!(base["a"]["b"]["y"].as_integer().unwrap(), 2);
        assert_eq!(base["a"]["c"]["z"].as_integer().unwrap(), 3);
    }

    #[test]
    fn test_deep_merge_overlay_adds_new_top_level() {
        let mut base = val(r#"x = 1"#);
        let overlay = val(r#"y = 2"#);
        deep_merge(&mut base, overlay);
        assert_eq!(base["x"].as_integer().unwrap(), 1);
        assert_eq!(base["y"].as_integer().unwrap(), 2);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/rust && cargo test deep_merge -- --nocapture 2>&1 | head -30`
Expected: FAIL — `deep_merge` function not found

- [ ] **Step 3: Implement `deep_merge()`**

Add to `src/rust/src/config.rs` (before the `impl SimInput` block, around line 583):

```rust
/// Deep-merge `overlay` into `base`. Tables merge recursively;
/// all other types (scalars, arrays) replace the base value.
pub fn deep_merge(base: &mut toml::Value, overlay: toml::Value) {
    match (base.is_table(), overlay.is_table()) {
        (true, true) => {
            let base_table = base.as_table_mut().unwrap();
            if let toml::Value::Table(overlay_table) = overlay {
                for (key, overlay_val) in overlay_table {
                    if let Some(base_val) = base_table.get_mut(&key) {
                        deep_merge(base_val, overlay_val);
                    } else {
                        base_table.insert(key, overlay_val);
                    }
                }
            }
        }
        _ => {
            *base = overlay;
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src/rust && cargo test deep_merge -- --nocapture`
Expected: All 5 `deep_merge` tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/config.rs
git commit -m "feat: add deep_merge() for TOML value tree merging"
```

---

### Task 2: Implement `resolve_toml_bases()` with tests

**Files:**
- Modify: `src/rust/src/config.rs` (add function + tests)

- [ ] **Step 1: Write failing tests for `resolve_toml_bases()`**

These tests need temp files on disk. Add to the `#[cfg(test)] mod tests` block:

```rust
    use std::io::Write;

    /// Helper: write a temp TOML file, return its path.
    fn write_temp_toml(dir: &std::path::Path, name: &str, content: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    #[test]
    fn test_resolve_single_base() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_toml(dir.path(), "base.toml", r#"
            [mission]
            type = "aerocapture"
            planet = "mars"
        "#);
        let child_path = write_temp_toml(dir.path(), "child.toml", r#"
            base = ["base.toml"]
            [guidance]
            type = "ftc"
        "#);
        let content = std::fs::read_to_string(&child_path).unwrap();
        let root: toml::Value = content.parse().unwrap();
        let mut visited = HashSet::new();
        let resolved = resolve_toml_bases(root, &child_path, &mut visited).unwrap();
        assert_eq!(resolved["mission"]["planet"].as_str().unwrap(), "mars");
        assert_eq!(resolved["guidance"]["type"].as_str().unwrap(), "ftc");
        assert!(resolved.get("base").is_none()); // stripped
    }

    #[test]
    fn test_resolve_multiple_bases_merge_order() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_toml(dir.path(), "a.toml", r#"
            [section]
            x = 1
            y = 10
        "#);
        write_temp_toml(dir.path(), "b.toml", r#"
            [section]
            y = 20
            z = 30
        "#);
        let child_path = write_temp_toml(dir.path(), "child.toml", r#"
            base = ["a.toml", "b.toml"]
            [section]
            z = 99
        "#);
        let content = std::fs::read_to_string(&child_path).unwrap();
        let root: toml::Value = content.parse().unwrap();
        let mut visited = HashSet::new();
        let resolved = resolve_toml_bases(root, &child_path, &mut visited).unwrap();
        assert_eq!(resolved["section"]["x"].as_integer().unwrap(), 1);  // from a
        assert_eq!(resolved["section"]["y"].as_integer().unwrap(), 20); // b overrides a
        assert_eq!(resolved["section"]["z"].as_integer().unwrap(), 99); // child overrides b
    }

    #[test]
    fn test_resolve_recursive_base() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_toml(dir.path(), "grandparent.toml", r#"
            [mission]
            type = "aerocapture"
        "#);
        write_temp_toml(dir.path(), "parent.toml", r#"
            base = ["grandparent.toml"]
            [mission]
            planet = "mars"
        "#);
        let child_path = write_temp_toml(dir.path(), "child.toml", r#"
            base = ["parent.toml"]
            [guidance]
            type = "ftc"
        "#);
        let content = std::fs::read_to_string(&child_path).unwrap();
        let root: toml::Value = content.parse().unwrap();
        let mut visited = HashSet::new();
        let resolved = resolve_toml_bases(root, &child_path, &mut visited).unwrap();
        assert_eq!(resolved["mission"]["type"].as_str().unwrap(), "aerocapture");
        assert_eq!(resolved["mission"]["planet"].as_str().unwrap(), "mars");
        assert_eq!(resolved["guidance"]["type"].as_str().unwrap(), "ftc");
    }

    #[test]
    fn test_resolve_cycle_detection() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_toml(dir.path(), "a.toml", r#"
            base = ["b.toml"]
            x = 1
        "#);
        write_temp_toml(dir.path(), "b.toml", r#"
            base = ["a.toml"]
            y = 2
        "#);
        let a_path = dir.path().join("a.toml");
        let content = std::fs::read_to_string(&a_path).unwrap();
        let root: toml::Value = content.parse().unwrap();
        let mut visited = HashSet::new();
        let result = resolve_toml_bases(root, &a_path, &mut visited);
        assert!(result.is_err());
        let err = result.unwrap_err().0;
        assert!(err.contains("Cycle"), "Error should mention cycle: {}", err);
    }

    #[test]
    fn test_resolve_missing_base_error() {
        let dir = tempfile::tempdir().unwrap();
        let child_path = write_temp_toml(dir.path(), "child.toml", r#"
            base = ["nonexistent.toml"]
            x = 1
        "#);
        let content = std::fs::read_to_string(&child_path).unwrap();
        let root: toml::Value = content.parse().unwrap();
        let mut visited = HashSet::new();
        let result = resolve_toml_bases(root, &child_path, &mut visited);
        assert!(result.is_err());
        let err = result.unwrap_err().0;
        assert!(err.contains("nonexistent.toml"), "Error should mention file: {}", err);
    }

    #[test]
    fn test_resolve_no_base_passthrough() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp_toml(dir.path(), "plain.toml", r#"
            x = 1
            [section]
            y = 2
        "#);
        let content = std::fs::read_to_string(&path).unwrap();
        let root: toml::Value = content.parse().unwrap();
        let mut visited = HashSet::new();
        let resolved = resolve_toml_bases(root, &path, &mut visited).unwrap();
        assert_eq!(resolved["x"].as_integer().unwrap(), 1);
        assert_eq!(resolved["section"]["y"].as_integer().unwrap(), 2);
    }

    #[test]
    fn test_resolve_base_single_string() {
        // base as a single string instead of array
        let dir = tempfile::tempdir().unwrap();
        write_temp_toml(dir.path(), "base.toml", r#"x = 1"#);
        let child_path = write_temp_toml(dir.path(), "child.toml", r#"
            base = "base.toml"
            y = 2
        "#);
        let content = std::fs::read_to_string(&child_path).unwrap();
        let root: toml::Value = content.parse().unwrap();
        let mut visited = HashSet::new();
        let resolved = resolve_toml_bases(root, &child_path, &mut visited).unwrap();
        assert_eq!(resolved["x"].as_integer().unwrap(), 1);
        assert_eq!(resolved["y"].as_integer().unwrap(), 2);
    }
```

Add `tempfile` as a dev-dependency in `src/rust/Cargo.toml`:

```toml
[dev-dependencies]
tempfile = "3"
# ... existing dev-deps
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/rust && cargo test resolve_toml -- --nocapture 2>&1 | head -30`
Expected: FAIL — `resolve_toml_bases` function not found

- [ ] **Step 3: Implement `resolve_toml_bases()`**

Add imports at the top of `src/rust/src/config.rs` (alongside existing `use` statements):

```rust
use std::collections::HashSet;
use std::path::Path;
```

Then add to `src/rust/src/config.rs` (right after `deep_merge()`):

```rust
/// Resolve `base` references in a TOML value tree.
///
/// If `root` contains a top-level `base` key (string or array of strings),
/// each referenced file is loaded, recursively resolved, and deep-merged
/// left-to-right. The current file's own keys overlay the merged base.
/// The `base` key is stripped from the result.
///
/// Paths in `base` are resolved relative to `file_path`'s parent directory.
pub fn resolve_toml_bases(
    mut root: toml::Value,
    file_path: &Path,
    visited: &mut HashSet<std::path::PathBuf>,
) -> Result<toml::Value, ParseError> {
    let base_dir = file_path
        .parent()
        .unwrap_or_else(|| Path::new("."));

    // Extract and remove the `base` key.
    let base_paths: Vec<String> = match root.as_table_mut().and_then(|t| t.remove("base")) {
        None => return Ok(root), // No base — passthrough.
        Some(toml::Value::String(s)) => vec![s],
        Some(toml::Value::Array(arr)) => {
            arr.into_iter()
                .map(|v| {
                    v.as_str()
                        .map(|s| s.to_string())
                        .ok_or_else(|| ParseError("base array elements must be strings".into()))
                })
                .collect::<Result<Vec<_>, _>>()?
        }
        Some(_) => return Err(ParseError("base must be a string or array of strings".into())),
    };

    // Register this file to detect cycles.
    let canonical = file_path
        .canonicalize()
        .map_err(|e| ParseError(format!("Cannot canonicalize '{}': {}", file_path.display(), e)))?;
    if !visited.insert(canonical.clone()) {
        return Err(ParseError(format!(
            "Cycle detected: '{}' was already visited",
            file_path.display()
        )));
    }

    // Load and merge bases left-to-right.
    let mut merged = toml::Value::Table(toml::map::Map::new());
    for base_rel in &base_paths {
        let base_abs = base_dir.join(base_rel);
        let base_content = std::fs::read_to_string(&base_abs).map_err(|e| {
            ParseError(format!(
                "Cannot read base '{}' (referenced from '{}'): {}",
                base_abs.display(),
                file_path.display(),
                e
            ))
        })?;
        let base_value: toml::Value = base_content
            .parse()
            .map_err(|e| ParseError(format!("TOML parse error in '{}': {}", base_abs.display(), e)))?;
        let resolved_base = resolve_toml_bases(base_value, &base_abs, visited)?;
        deep_merge(&mut merged, resolved_base);
    }

    // Overlay the current file's own keys on top of merged bases.
    deep_merge(&mut merged, root);

    // Remove canonical from visited so sibling references work
    // (only direct ancestor cycles should be blocked).
    visited.remove(&canonical);

    Ok(merged)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src/rust && cargo test resolve_toml -- --nocapture`
Expected: All 7 `resolve_toml` tests PASS

Run: `cd src/rust && cargo test deep_merge -- --nocapture`
Expected: All 5 `deep_merge` tests still PASS

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/config.rs src/rust/Cargo.toml src/rust/Cargo.lock
git commit -m "feat: add resolve_toml_bases() with cycle detection and recursive resolution"
```

---

### Task 3: Add `from_toml_file()` and update `main.rs`

**Files:**
- Modify: `src/rust/src/config.rs:587` (add `from_toml_file`)
- Modify: `src/rust/src/main.rs:7-30` (use `from_toml_file`)

- [ ] **Step 1: Implement `from_toml_file()`**

Add to `impl SimInput` in `src/rust/src/config.rs`, right before `from_toml()`:

```rust
    /// Load a TOML config file with base inheritance resolution.
    ///
    /// Reads the file, resolves any `base` references, then parses
    /// via the normal `from_toml()` pipeline.
    pub fn from_toml_file(path: &Path) -> Result<(Self, TomlConfig), ParseError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| ParseError(format!("Cannot read '{}': {}", path.display(), e)))?;
        let root: toml::Value = content
            .parse()
            .map_err(|e| ParseError(format!("TOML parse error in '{}': {}", path.display(), e)))?;
        let mut visited = HashSet::new();
        let resolved = resolve_toml_bases(root, path, &mut visited)?;
        let resolved_str = toml::to_string(&resolved)
            .map_err(|e| ParseError(format!("TOML serialize error: {}", e)))?;
        Self::from_toml(&resolved_str)
    }
```

- [ ] **Step 2: Update `main.rs` to use `from_toml_file()`**

Replace the file reading + parsing in `src/rust/src/main.rs`. Change lines 14-30 from:

```rust
    let toml_path = &args[1];
    let content = match std::fs::read_to_string(toml_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Cannot read {}: {}", toml_path, e);
            process::exit(1);
        }
    };
    let (sim_config, toml_config) = match config::SimInput::from_toml(&content) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error parsing TOML config: {}", e);
            process::exit(1);
        }
    };
```

To:

```rust
    let toml_path = std::path::Path::new(&args[1]);
    let (sim_config, toml_config) = match config::SimInput::from_toml_file(toml_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error loading config: {}", e);
            process::exit(1);
        }
    };
```

- [ ] **Step 3: Run full Rust test suite**

Run: `cd src/rust && cargo test`
Expected: All existing tests PASS (including golden regressions — no configs changed yet)

- [ ] **Step 4: Run the CLI binary with an existing flat config to verify it still works**

Run: `cd /Users/govit/Git/Govit/Aerocapture && ./src/rust/target/release/aerocapture configs/test/test_ref_orig.toml`
Expected: Runs successfully, same output as before

- [ ] **Step 5: Commit**

```bash
git add src/rust/src/config.rs src/rust/src/main.rs
git commit -m "feat: add from_toml_file() and use it in main.rs"
```

---

## Chunk 2: PyO3 Bindings — base resolution in load_and_override, batch, load_config

### Task 4: Update PyO3 `load_and_override()` to accept file path

**Files:**
- Modify: `src/rust/aerocapture-py/src/config.rs:103-125`
- Modify: `src/rust/aerocapture-py/src/lib.rs:60-80,98-119`

- [ ] **Step 1: Change `load_and_override()` signature to take `&Path`**

In `src/rust/aerocapture-py/src/config.rs`, change `load_and_override` from:

```rust
pub fn load_and_override(
    toml_content: &str,
    overrides: &[(String, OverrideValue)],
) -> Result<(SimInput, SimData), String> {
    let table: Table =
        toml::from_str(toml_content).map_err(|e| format!("TOML parse error: {}", e))?;
    let mut root = Value::Table(table);
```

To:

```rust
pub fn load_and_override(
    toml_path: &std::path::Path,
    overrides: &[(String, OverrideValue)],
) -> Result<(SimInput, SimData), String> {
    let toml_content = std::fs::read_to_string(toml_path)
        .map_err(|e| format!("Cannot read '{}': {}", toml_path.display(), e))?;
    let root: Value = toml_content
        .parse()
        .map_err(|e| format!("TOML parse error in '{}': {}", toml_path.display(), e))?;
    let mut root = {
        let mut visited = std::collections::HashSet::new();
        aerocapture::config::resolve_toml_bases(root, toml_path, &mut visited)
            .map_err(|e| format!("Base resolution error: {}", e))?
    };
```

The rest of the function (apply overrides, serialize, parse) stays the same.

- [ ] **Step 2: Update `run()` and `run_mc()` in `lib.rs`**

In `src/rust/aerocapture-py/src/lib.rs`, update `run()` — remove the `read_to_string` call and pass path to `load_and_override`:

```rust
fn run(toml_path: &str, overrides: Option<&Bound<'_, PyDict>>) -> PyResult<SimResult> {
    let path = std::path::Path::new(toml_path);
    let overrides = extract_overrides(overrides)?;

    let (sim_input, sim_data) = config::load_and_override(path, &overrides)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    // ... rest unchanged
```

Same pattern for `run_mc()`.

- [ ] **Step 3: Update `run_batch()` in `lib.rs`**

Remove the `read_to_string` call and pass path string to `batch::run_batch`:

```rust
fn run_batch(
    toml_path: &str,
    overrides_list: &Bound<'_, PyList>,
    n_threads: Option<usize>,
    include_trajectories: bool,
) -> PyResult<BatchResults> {
    let path = std::path::Path::new(toml_path);
    // ... extract n_threads, overrides_vec as before ...
    let outputs = batch::run_batch(path, overrides_vec, n_threads)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    // ... rest unchanged
```

- [ ] **Step 4: Update `load_config()` in `lib.rs`**

Add base resolution before converting to Python dict:

```rust
fn load_config(py: Python<'_>, toml_path: &str) -> PyResult<Py<PyAny>> {
    let path = std::path::Path::new(toml_path);
    let content = std::fs::read_to_string(path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Cannot read '{}': {}", toml_path, e))
    })?;

    let root: toml::Value = content.parse::<toml::Table>()
        .map(toml::Value::Table)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("TOML parse error: {}", e)))?;

    let resolved = {
        let mut visited = std::collections::HashSet::new();
        aerocapture::config::resolve_toml_bases(root, path, &mut visited)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Base resolution error: {}", e)))?
    };

    toml_to_py(py, &resolved)
}
```

- [ ] **Step 5: Commit**

```bash
git add src/rust/aerocapture-py/src/config.rs src/rust/aerocapture-py/src/lib.rs
git commit -m "refactor: PyO3 load_and_override takes file path, resolves bases"
```

---

### Task 5: Update `batch.rs` to resolve bases

**Files:**
- Modify: `src/rust/aerocapture-py/src/batch.rs:24-32`

- [ ] **Step 1: Change `run_batch()` to take `&Path` and resolve bases**

Change the signature and initial parse in `src/rust/aerocapture-py/src/batch.rs`:

From:
```rust
pub fn run_batch(
    toml_content: &str,
    overrides_list: Vec<Vec<(String, OverrideValue)>>,
    n_threads: usize,
) -> Result<Vec<RunOutput>, String> {
    let base_table: Table =
        toml::from_str(toml_content).map_err(|e| format!("TOML parse error: {}", e))?;
    let base_value = Value::Table(base_table);
```

To:
```rust
pub fn run_batch(
    toml_path: &std::path::Path,
    overrides_list: Vec<Vec<(String, OverrideValue)>>,
    n_threads: usize,
) -> Result<Vec<RunOutput>, String> {
    // Parse and resolve bases once for the batch.
    let content = std::fs::read_to_string(toml_path)
        .map_err(|e| format!("Cannot read '{}': {}", toml_path.display(), e))?;
    let root: Value = content
        .parse()
        .map_err(|e| format!("TOML parse error in '{}': {}", toml_path.display(), e))?;
    let base_value = {
        let mut visited = std::collections::HashSet::new();
        aerocapture::config::resolve_toml_bases(root, toml_path, &mut visited)
            .map_err(|e| format!("Base resolution error: {}", e))?
    };
```

The rest of the function (clone, apply overrides, run) stays the same.

- [ ] **Step 2: Build PyO3 to verify compilation**

Run: `cd src/rust/aerocapture-py && maturin develop --release 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 3: Run existing PyO3 tests**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/test_pyo3.py -v`
Expected: All PyO3 tests PASS (flat configs still work)

- [ ] **Step 4: Commit**

```bash
git add src/rust/aerocapture-py/src/batch.rs
git commit -m "refactor: batch.rs resolves TOML bases on initial parse"
```

---

## Chunk 3: Python — load_toml_with_bases + update call sites

### Task 6: Implement `load_toml_with_bases()` with tests

**Files:**
- Create: `src/python/aerocapture/training/toml_utils.py`
- Create: `tests/test_toml_utils.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_toml_utils.py`:

```python
"""Tests for TOML base inheritance resolution (Python side)."""

from pathlib import Path

import pytest


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


class TestLoadTomlWithBases:
    def test_single_base(self, tmp_path: Path) -> None:
        from aerocapture.training.toml_utils import load_toml_with_bases

        _write(tmp_path / "base.toml", '[mission]\nplanet = "mars"\n')
        child = _write(tmp_path / "child.toml", 'base = ["base.toml"]\n[guidance]\ntype = "ftc"\n')
        result = load_toml_with_bases(child)
        assert result["mission"]["planet"] == "mars"
        assert result["guidance"]["type"] == "ftc"
        assert "base" not in result

    def test_multiple_bases_merge_order(self, tmp_path: Path) -> None:
        from aerocapture.training.toml_utils import load_toml_with_bases

        _write(tmp_path / "a.toml", "[s]\nx = 1\ny = 10\n")
        _write(tmp_path / "b.toml", "[s]\ny = 20\nz = 30\n")
        child = _write(tmp_path / "child.toml", 'base = ["a.toml", "b.toml"]\n[s]\nz = 99\n')
        result = load_toml_with_bases(child)
        assert result["s"]["x"] == 1   # from a
        assert result["s"]["y"] == 20  # b overrides a
        assert result["s"]["z"] == 99  # child overrides b

    def test_recursive_base(self, tmp_path: Path) -> None:
        from aerocapture.training.toml_utils import load_toml_with_bases

        _write(tmp_path / "gp.toml", '[mission]\ntype = "aerocapture"\n')
        _write(tmp_path / "parent.toml", 'base = ["gp.toml"]\n[mission]\nplanet = "mars"\n')
        child = _write(tmp_path / "child.toml", 'base = ["parent.toml"]\n[guidance]\ntype = "ftc"\n')
        result = load_toml_with_bases(child)
        assert result["mission"]["type"] == "aerocapture"
        assert result["mission"]["planet"] == "mars"
        assert result["guidance"]["type"] == "ftc"

    def test_cycle_detection(self, tmp_path: Path) -> None:
        from aerocapture.training.toml_utils import load_toml_with_bases

        _write(tmp_path / "a.toml", 'base = ["b.toml"]\nx = 1\n')
        _write(tmp_path / "b.toml", 'base = ["a.toml"]\ny = 2\n')
        with pytest.raises(ValueError, match="[Cc]ycle"):
            load_toml_with_bases(tmp_path / "a.toml")

    def test_missing_base(self, tmp_path: Path) -> None:
        from aerocapture.training.toml_utils import load_toml_with_bases

        child = _write(tmp_path / "child.toml", 'base = ["nope.toml"]\nx = 1\n')
        with pytest.raises(FileNotFoundError):
            load_toml_with_bases(child)

    def test_no_base_passthrough(self, tmp_path: Path) -> None:
        from aerocapture.training.toml_utils import load_toml_with_bases

        f = _write(tmp_path / "plain.toml", "x = 1\n[s]\ny = 2\n")
        result = load_toml_with_bases(f)
        assert result["x"] == 1
        assert result["s"]["y"] == 2

    def test_base_single_string(self, tmp_path: Path) -> None:
        from aerocapture.training.toml_utils import load_toml_with_bases

        _write(tmp_path / "base.toml", "x = 1\n")
        child = _write(tmp_path / "child.toml", 'base = "base.toml"\ny = 2\n')
        result = load_toml_with_bases(child)
        assert result["x"] == 1
        assert result["y"] == 2

    def test_deep_merge_nested(self, tmp_path: Path) -> None:
        from aerocapture.training.toml_utils import load_toml_with_bases

        _write(tmp_path / "base.toml", "[a.b]\nx = 1\n")
        child = _write(tmp_path / "child.toml", 'base = ["base.toml"]\n[a.b]\ny = 2\n[a.c]\nz = 3\n')
        result = load_toml_with_bases(child)
        assert result["a"]["b"]["x"] == 1
        assert result["a"]["b"]["y"] == 2
        assert result["a"]["c"]["z"] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_toml_utils.py -v 2>&1 | head -20`
Expected: FAIL — `aerocapture.training.toml_utils` not found

- [ ] **Step 3: Implement `load_toml_with_bases()`**

Create `src/python/aerocapture/training/toml_utils.py`:

```python
"""TOML config loading with base inheritance resolution."""

import tomllib
from pathlib import Path


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Deep-merge overlay into base. Tables merge recursively; scalars/arrays replace."""
    result = dict(base)
    for key, val in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_toml_with_bases(path: Path, *, _visited: frozenset[Path] | None = None) -> dict:
    """Load a TOML file, recursively resolving ``base`` references.

    ``base`` can be a single string or array of strings, resolved relative
    to the declaring file's directory. Cycle detection via canonical paths.
    """
    path = Path(path).resolve()
    if _visited is None:
        _visited = frozenset()

    if path in _visited:
        msg = f"Cycle detected: '{path}' was already visited"
        raise ValueError(msg)

    _visited = _visited | {path}

    with open(path, "rb") as f:
        data = tomllib.load(f)

    base_refs = data.pop("base", None)
    if base_refs is None:
        return data

    if isinstance(base_refs, str):
        base_refs = [base_refs]

    base_dir = path.parent
    merged: dict = {}
    for ref in base_refs:
        base_path = (base_dir / ref).resolve()
        base_data = load_toml_with_bases(base_path, _visited=_visited)
        merged = _deep_merge(merged, base_data)

    return _deep_merge(merged, data)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_toml_utils.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/python/aerocapture/training/toml_utils.py tests/test_toml_utils.py
git commit -m "feat: add Python load_toml_with_bases() with tests"
```

---

### Task 7: Update Python training pipeline call sites

**Files:**
- Modify: `src/python/aerocapture/training/train.py` (2 call sites)
- Modify: `src/python/aerocapture/training/compare_guidance.py` (2 call sites)
- Modify: `src/python/aerocapture/training/final_report.py` (3 call sites)
- Modify: `src/python/aerocapture/training/evaluate.py` (2 call sites)

- [ ] **Step 1: Update `train.py`**

At line 242, change:
```python
    import tomllib
    ...
    if config.sim.toml_config:
        toml_path = Path(cwd or config.sim.exec_dir) / config.sim.toml_config
        with open(toml_path, "rb") as f:
            _toml = tomllib.load(f)
```
To:
```python
    from aerocapture.training.toml_utils import load_toml_with_bases
    ...
    if config.sim.toml_config:
        toml_path = Path(cwd or config.sim.exec_dir) / config.sim.toml_config
        _toml = load_toml_with_bases(toml_path)
```

At line 674, change:
```python
        import tomllib

        with open(args.toml, "rb") as _f:
            _toml_data = tomllib.load(_f)
```
To:
```python
        from aerocapture.training.toml_utils import load_toml_with_bases

        _toml_data = load_toml_with_bases(Path(args.toml))
```

- [ ] **Step 2: Update `compare_guidance.py`**

At line 42, change:
```python
    import tomllib

    with open(base_toml, "rb") as f:
        toml_data = tomllib.load(f)
```
To:
```python
    from aerocapture.training.toml_utils import load_toml_with_bases

    toml_data = load_toml_with_bases(base_toml)
```

At line 208, change:
```python
    import tomllib
    ...
    with open(base_toml, "rb") as f:
        _toml = tomllib.load(f)
```
To:
```python
    from aerocapture.training.toml_utils import load_toml_with_bases
    ...
    _toml = load_toml_with_bases(base_toml)
```

- [ ] **Step 3: Update `final_report.py`**

At line 48 (`_read_target_inclination`), change:
```python
    import tomllib

    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
```
To:
```python
    from aerocapture.training.toml_utils import load_toml_with_bases

    data = load_toml_with_bases(toml_path)
```

At line 63 (`_patch_toml_for_final_eval`), change:
```python
    import tomllib

    with open(base_toml_path, "rb") as f:
        toml_data = tomllib.load(f)
```
To:
```python
    from aerocapture.training.toml_utils import load_toml_with_bases

    toml_data = load_toml_with_bases(base_toml_path)
```

At line 387, change:
```python
        import tomllib
        ...
        with open(args.toml, "rb") as f:
            toml_data = tomllib.load(f)
```
To:
```python
        from aerocapture.training.toml_utils import load_toml_with_bases
        ...
        toml_data = load_toml_with_bases(Path(args.toml))
```

- [ ] **Step 4: Update `evaluate.py`**

At line 349 (`write_guidance_toml`), change:
```python
    import tomllib
    ...
    with open(base_toml_path, "rb") as f:
        toml_data = tomllib.load(f)
```
To:
```python
    from aerocapture.training.toml_utils import load_toml_with_bases
    ...
    toml_data = load_toml_with_bases(base_toml_path)
```

At line 486 (`patch_toml_mc_seed`), change:
```python
    import tomllib
    ...
    with open(base_toml_path, "rb") as f:
        toml_data = tomllib.load(f)
```
To:
```python
    from aerocapture.training.toml_utils import load_toml_with_bases
    ...
    toml_data = load_toml_with_bases(base_toml_path)
```

- [ ] **Step 5: Run existing Python test suite**

Run: `uv run pytest tests/ -v --tb=short 2>&1 | tail -30`
Expected: All tests PASS (flat configs still work — `load_toml_with_bases` is a no-op when there's no `base` key)

- [ ] **Step 6: Run linter**

Run: `./lint_code.sh`
Expected: Clean (no unused `tomllib` imports remain)

Note: if any files still have `import tomllib` that's now unused, remove those import lines.

- [ ] **Step 7: Commit**

```bash
git add src/python/aerocapture/training/train.py src/python/aerocapture/training/compare_guidance.py src/python/aerocapture/training/final_report.py src/python/aerocapture/training/evaluate.py
git commit -m "refactor: replace tomllib.load() with load_toml_with_bases() at all 9 call sites"
```

---

## Chunk 4: Config Migration — extract shared bases + rewrite all 20 configs

### Task 8: Create shared mission base configs

**Files:**
- Create: `configs/missions/mars.toml`
- Create: `configs/missions/earth.toml`

- [ ] **Step 1: Create `configs/missions/` directory**

Run: `mkdir -p configs/missions`

- [ ] **Step 2: Create `configs/missions/mars.toml`**

Extract from any existing Mars config (e.g., `configs/training/msr_aller_ftc_train.toml`). This file contains ALL sections shared across 18 Mars configs:

```toml
# Mars Sample Return — shared mission parameters
# Used as base by all MSR training, test, and nominal configs.

[mission]
type = "aerocapture"
planet = "mars"

[entry]
altitude = 130.0           # km
longitude = 0.0            # deg
latitude = 0.0             # deg
velocity = 5687.15586      # m/s
flight_path_angle = -10.81251  # deg
azimuth = 38.04069         # deg
initial_time = 0.0         # s
initial_bank_angle = 64.77026  # deg
initial_aoa = -27.5        # deg

[vehicle]
mass = 1089.0              # kg
reference_area = 14.7      # m²
cq = 0.00008242            # heat flux coefficient
max_bank_rate = 15.0       # deg/s

[vehicle.periods]
navigation = 1.0
guidance = 1.0
pilot = 1.0
prediction = 1.0
integration = 1.0
photo = 1.0

[vehicle.pilot]
model = "perfect"
time_constant = 1.0
damping = 0.7
frequency = 0.072

[aerodynamics]
equilibrium_aoa = -27.5    # deg
points = [
    { aoa = -27.5, ca = 1.269, cn = -0.205 },
    { aoa = -27.5, ca = 1.269, cn = -0.205 },
]

[flight]
wind = false

[flight.constraints]
max_heat_flux = 151.571        # kW/m²
max_load_factor = 1.912        # g
max_dynamic_pressure = 1.081   # kPa

[flight.final_conditions]
altitude = 130.988             # km
longitude = 36.334             # deg
latitude = 30.634              # deg
velocity = 3360.258            # m/s
flight_path_angle = 3.446      # deg
azimuth = 299.401              # deg
energy = -5.867                # MJ/kg
radial_velocity = 201.978      # m/s

[flight.target_orbit]
apoapsis = 500.130             # km
periapsis = 11.233             # km
semi_major_axis = 3649.622     # km
eccentricity = 0.067
inclination = 50.0             # deg
raan = -7.612                  # deg

[flight.parking_orbit]
apoapsis = 500.0               # km
periapsis = 500.0              # km

[success]
inclination_tolerance = 0.5    # deg
velocity_tolerance = 170.0     # m/s
apoapsis_tolerance = 100.0     # km
periapsis_tolerance = 25.0     # km

[incidence]
altitudes = [-10.0, 50.0, 80.0, 150.0]   # km
angles = [-27.5, -27.5, -27.5, -27.5]    # deg

[data]
atmosphere = "data/atmosphere/mars.dat"
reference_trajectory = "data/reference_trajectory/msr_aller.dat"
```

- [ ] **Step 3: Create `configs/missions/earth.toml`**

Extract from `configs/nominal/esr_aller_ftc_nominal.toml`:

```toml
# Earth Sample Return — shared mission parameters

[mission]
type = "aerocapture"
planet = "earth"

[entry]
altitude = 600.0
longitude = 0.0
latitude = 0.0
velocity = 47295.97517
flight_path_angle = -4.88388
azimuth = 89.40733
initial_time = 0.0
initial_bank_angle = 90.0
initial_aoa = -27.5

[vehicle]
mass = 5600.0
reference_area = 29.0
cq = 0.00008242
max_bank_rate = 15.0

[vehicle.periods]
navigation = 1.0
guidance = 1.0
pilot = 1.0
prediction = 1.0
integration = 1.0
photo = 20.0

[vehicle.pilot]
model = "perfect"
time_constant = 1.0
damping = 0.7
frequency = 0.072

[aerodynamics]
equilibrium_aoa = -27.5
points = [
    { aoa = -27.5, ca = 1.269, cn = -0.205 },
    { aoa = -27.5, ca = 1.269, cn = -0.205 },
]

[flight]
wind = false

[flight.constraints]
max_heat_flux = 33721.096
max_load_factor = 4.421
max_dynamic_pressure = 6.492

[flight.final_conditions]
altitude = 608.535
longitude = 12.569
latitude = -0.045
velocity = 43682.546
flight_path_angle = 4.701
azimuth = 269.086
energy = -170.501
radial_velocity = 3580.320

[flight.target_orbit]
apoapsis = 599757.465
periapsis = 282.373
semi_major_axis = 371511.919
eccentricity = 0.807
inclination = 0.464
raan = -0.321

[flight.parking_orbit]
apoapsis = 599742.0
periapsis = 599474.0

[success]
inclination_tolerance = 0.5
velocity_tolerance = 9500.0
apoapsis_tolerance = 100000.0
periapsis_tolerance = 1000.0

[incidence]
altitudes = [-10.0, 50.0, 80.0, 650.0]
angles = [-27.5, -27.5, -27.5, -27.5]

[data]
atmosphere = "data/atmosphere/earth.dat"
reference_trajectory = "data/reference_trajectory/esr_aller.dat"
```

- [ ] **Step 4: Commit**

```bash
git add configs/missions/mars.toml configs/missions/earth.toml
git commit -m "feat: extract shared mission base configs for Mars and Earth"
```

---

### Task 9: Create shared training base config

**Files:**
- Create: `configs/training/common.toml`

- [ ] **Step 1: Create `configs/training/common.toml`**

Extract the Monte Carlo and cost function sections shared by all 6 training configs:

```toml
# Shared training parameters — MC dispersions + cost function
# Used as base by all training configs alongside a mission base.

[simulation]
random_seed = 0.6866

[monte_carlo]
seed = 42

[monte_carlo.initial_state]
level = "medium"

[monte_carlo.atmosphere]
level = "high"

[monte_carlo.aerodynamics]
level = "medium"

[monte_carlo.navigation]
level = "low"

[monte_carlo.mass]
level = "off"

[cost_function]
g_load_limit = 15.0          # g (Earth g's)
heat_flux_limit = 200.0      # kW/m²
g_load_weight = 1000.0       # penalty weight on normalized squared exceedance
heat_flux_weight = 1000.0    # penalty weight on normalized squared exceedance
```

- [ ] **Step 2: Commit**

```bash
git add configs/training/common.toml
git commit -m "feat: extract shared training base config (MC dispersions + cost function)"
```

---

### Task 10: Migrate all 6 training configs

**Files:**
- Modify: all 6 files in `configs/training/msr_aller_*.toml`

- [ ] **Step 1: Rewrite each training config**

Each training config becomes `base` + guidance type + n_sims + results_suffix + any scheme-specific extras.

**`configs/training/msr_aller_ftc_train.toml`:**
```toml
# MSR outbound — FTC GA training
base = ["../missions/mars.toml", "common.toml"]

[guidance]
type = "ftc"
reference_bank_angle = 64.77026

[simulation]
n_sims = 10

[data]
results_suffix = ".train_nn_temp"
```

**`configs/training/msr_aller_eqglide_train.toml`:**
```toml
# MSR outbound — Equilibrium Glide GA training
base = ["../missions/mars.toml", "common.toml"]

[guidance]
type = "equilibrium_glide"
reference_bank_angle = 64.77026

[simulation]
n_sims = 10

[data]
results_suffix = ".train_nn_temp"
```

**`configs/training/msr_aller_energy_controller_train.toml`:**
```toml
# MSR outbound — Energy Controller GA training
base = ["../missions/mars.toml", "common.toml"]

[guidance]
type = "energy_controller"
reference_bank_angle = 64.77026

[simulation]
n_sims = 10

[data]
results_suffix = ".train_nn_temp"
```

**`configs/training/msr_aller_pred_guid_train.toml`:**
```toml
# MSR outbound — PredGuid GA training
base = ["../missions/mars.toml", "common.toml"]

[guidance]
type = "pred_guid"
reference_bank_angle = 64.77026

[simulation]
n_sims = 10

[data]
results_suffix = ".train_nn_temp"
```

**`configs/training/msr_aller_fnpag_train.toml`:**
```toml
# MSR outbound — FNPAG GA training
base = ["../missions/mars.toml", "common.toml"]

[guidance]
type = "fnpag"
reference_bank_angle = 64.77026

[simulation]
n_sims = 10

[data]
results_suffix = ".train_nn_temp"
```

**`configs/training/msr_aller_nn_train_consolidated.toml`:**
```toml
# MSR outbound — NN training, 50 MC sims per evaluation
base = ["../missions/mars.toml", "common.toml"]

[guidance]
type = "neural_network"
reference_bank_angle = 64.77026

[network]
layer_sizes = [6, 8, 8, 2]
activations = ["tanh", "tanh", "asinh"]

[simulation]
n_sims = 50
max_time = 3000.0

[data]
neural_network = "data/neural_network/nn_model.json"
results_suffix = ".train_nn_temp"
```

- [ ] **Step 2: Run Rust tests to verify training configs still parse correctly**

Run: `cd src/rust && cargo test`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add configs/training/
git commit -m "refactor: migrate 6 training configs to use base inheritance"
```

---

### Task 11: Migrate all 9 test configs

**Files:**
- Modify: all 9 files in `configs/test/`

- [ ] **Step 1: Rewrite the 6 golden test configs**

All golden tests follow the same pattern: `base` + guidance type + n_sims=3 + MC dispersions + results_suffix. The MC dispersions are identical to training common, but test configs don't use the training base (no cost_function).

**`test_ftc_golden.toml`:**
```toml
# Golden regression test — FTC, 3 MC sims, deterministic seed
base = ["../missions/mars.toml"]

[guidance]
type = "ftc"
reference_bank_angle = 64.77026

[simulation]
n_sims = 3
random_seed = 0.6866

[data]
results_suffix = ".golden_ftc"

[monte_carlo]
seed = 42

[monte_carlo.initial_state]
level = "medium"

[monte_carlo.atmosphere]
level = "high"

[monte_carlo.aerodynamics]
level = "medium"

[monte_carlo.navigation]
level = "low"

[monte_carlo.mass]
level = "off"
```

**`test_eqglide_golden.toml`:** Same as above but `type = "equilibrium_glide"`, `results_suffix = ".golden_eqglide"`

**`test_energy_ctrl_golden.toml`:** Same but `type = "energy_controller"`, `results_suffix = ".golden_energy_ctrl"`

**`test_pred_guid_golden.toml`:** Same but `type = "pred_guid"`, `results_suffix = ".golden_pred_guid"`

**`test_fnpag_golden.toml`:** Same but `type = "fnpag"`, `results_suffix = ".golden_fnpag"`

**`test_neural_golden.toml`:** Same but `type = "neural_network"`, `results_suffix = ".golden_neural"`, plus:
```toml
[data]
results_suffix = ".golden_neural"
neural_network = "tests/reference_data/rust_golden/neural/nn_model_golden.json"
```

- [ ] **Step 2: Rewrite the 3 original test configs**

**`test_ref_orig.toml`:**
```toml
# Test: reference trajectory, constant bank angle 0.1°
base = ["../missions/mars.toml"]

[guidance]
type = "ftc"
reference_trajectory = true
reference_bank_angle = 0.1

[simulation]
n_sims = 1
random_seed = 0.6866

[data]
results_suffix = ".test_ref_orig"
```

**`test_high_bank_orig.toml`:**
```toml
# Test: reference trajectory, constant bank angle 64.77°
base = ["../missions/mars.toml"]

[guidance]
type = "ftc"
reference_trajectory = true
reference_bank_angle = 64.77

[simulation]
n_sims = 1
random_seed = 0.6866

[data]
results_suffix = ".test_high_bank_orig"
```

**`test_guided_orig.toml`:** This one keeps the full `[guidance.ftc]` section including the 26-entry pdyn_table:
```toml
# Test: FTC guided trajectory
base = ["../missions/mars.toml"]

[guidance]
type = "ftc"
reference_trajectory = false
reference_bank_angle = 64.77026

[guidance.ftc]
capture_damping = 0.7
capture_frequency = 0.072
capture_pdyn_margin = 1.75
altitude_damping = 0.7
altitude_frequency = 0.08
exit_velocity_threshold = 4400.0
exit_pdyn_margin = 1.75
exit_altitude_threshold = 60.0
exit_radial_vel_gain = 10.0
exit_apoapsis_threshold = 100.0
corridor_slope = 13080.458
corridor_intercept = 0.0
max_reversals = 5
security_capture = 1
security_exit = 3
density_filter_gain = 0.8
longi_activation = 1000.0
longi_inhibition = -1000.0
lateral_activation = 1.311
lateral_inhibition = 1000.0
pdyn_min = 0.0
pdyn_table = [
    { altitude =  0.0000000000, a = -0.1645497562, b = 1.4897963360 },
    { altitude = 45.6282285400, a = -0.1965988262, b = 1.3408173570 },
    { altitude = 46.3171209300, a = -0.1412271905, b = 1.2053819220 },
    { altitude = 47.1889298400, a = -0.1527424374, b = 1.0822587990 },
    { altitude = 47.9217328100, a = -0.1078032389, b = 0.9703286871 },
    { altitude = 48.8656251100, a = -0.1073334457, b = 0.8685740400 },
    { altitude = 49.7274648100, a = -0.1141791608, b = 0.7760698154 },
    { altitude = 50.4639805600, a = -0.0775047379, b = 0.6919750657 },
    { altitude = 51.4503689300, a = -0.0835505551, b = 0.6155252933 },
    { altitude = 52.2821981500, a = -0.0628220168, b = 0.5460255002 },
    { altitude = 53.2879224700, a = -0.0631744053, b = 0.4828438701 },
    { altitude = 54.1971173500, a = -0.0440526032, b = 0.4254060246 },
    { altitude = 55.3824326100, a = -0.0484318959, b = 0.3731898013 },
    { altitude = 56.3625572400, a = -0.0314772782, b = 0.3257205075 },
    { altitude = 57.7335113400, a = -0.0375532950, b = 0.2825666040 },
    { altitude = 58.7781818900, a = -0.0245224347, b = 0.2433357827 },
    { altitude = 60.2325392300, a = -0.0268146977, b = 0.2076713996 },
    { altitude = 61.4416584500, a = -0.0179607869, b = 0.1752492332 },
    { altitude = 63.0827166600, a = -0.0149100858, b = 0.1457745365 },
    { altitude = 64.8798343400, a = -0.0132187541, b = 0.1189793576 },
    { altitude = 66.7226141800, a = -0.0090497830, b = 0.0946201041 },
    { altitude = 69.1696094000, a = -0.0073162743, b = 0.0724753282 },
    { altitude = 71.9212304600, a = -0.0046864259, b = 0.0523437137 },
    { altitude = 75.8264384400, a = -0.0030096889, b = 0.0340422460 },
    { altitude = 81.3544842100, a = -0.0010156255, b = 0.0174045481 },
    { altitude = 96.2469613900, a = -0.0000010000, b = 0.0022793682 },
]

[simulation]
n_sims = 1
random_seed = 0.6866

[data]
results_suffix = ".test_guided_orig"
```

- [ ] **Step 3: Run Rust golden regression tests**

Run: `cd src/rust && cargo test`
Expected: All golden regression tests PASS — this is the critical correctness gate

- [ ] **Step 4: Commit**

```bash
git add configs/test/
git commit -m "refactor: migrate 9 test configs to use base inheritance"
```

---

### Task 12: Migrate all 5 nominal configs

**Files:**
- Modify: all 5 files in `configs/nominal/`

- [ ] **Step 1: Rewrite each nominal config**

**`msr_aller_reference.toml`:**
```toml
# MSR outbound — Reference trajectory (constant bank angle 0.1°)
base = ["../missions/mars.toml"]

[guidance]
type = "ftc"
reference_trajectory = true
reference_bank_angle = 0.1

[simulation]
n_sims = 1
screen_output = true
random_seed = 0.6866

[data]
results_suffix = ".test_ref"
```

**`msr_aller_ftc_nominal.toml`:** Mars FTC with full `[guidance.ftc]` params + pdyn_table (same FTC block as `test_guided_orig.toml` above):
```toml
# MSR outbound — FTC guided, single nominal run
base = ["../missions/mars.toml"]

[guidance]
type = "ftc"
reference_bank_angle = 64.77026

[guidance.ftc]
# ... (identical 20 params + 26-entry pdyn_table as test_guided_orig.toml)
# Copy the full [guidance.ftc] block from the original file.

[simulation]
n_sims = 1
screen_output = true
random_seed = 0.6866

[data]
results_suffix = ".test_guided"
```

**`msr_aller_ftc_consolidated.toml`:** Identical structure to `msr_aller_ftc_nominal.toml` (same FTC params, different results_suffix):
```toml
# MSR outbound — FTC guided, single nominal run (consolidated)
base = ["../missions/mars.toml"]

[guidance]
type = "ftc"
reference_bank_angle = 64.77026

[guidance.ftc]
# ... (identical FTC block)

[simulation]
n_sims = 1
screen_output = true
random_seed = 0.6866

[data]
results_suffix = ".test_consolidated"
```

**`msr_aller_ftc_mc_domain.toml`:** 100-sim MC with 8 dispersion domains (3 more than training common). Note: this config originally had NO `[vehicle.periods]` or `[vehicle.pilot]` — those come from the mars.toml base now via deep merge, which is correct since they have serde defaults:
```toml
# MSR outbound — FTC guided, 100-sim Monte Carlo
base = ["../missions/mars.toml"]

[guidance]
type = "ftc"
reference_bank_angle = 64.77026

[guidance.ftc]
# ... (identical FTC block)

[simulation]
n_sims = 100
random_seed = 0.6866

[data]
results_suffix = ".mc100_domain"

[monte_carlo]
seed = 42

[monte_carlo.initial_state]
level = "medium"

[monte_carlo.atmosphere]
level = "high"

[monte_carlo.aerodynamics]
level = "medium"

[monte_carlo.navigation]
level = "low"

[monte_carlo.mass]
level = "off"

[monte_carlo.vehicle]
level = "medium"

[monte_carlo.pilot]
level = "low"

[monte_carlo.nav_filter]
level = "medium"
```

**`esr_aller_ftc_nominal.toml`:** Earth mission — uses `earth.toml` base. The FTC params are Earth-specific (different corridor_slope, pdyn_table, activation thresholds):
```toml
# ESR outbound — FTC guided, single nominal run (Earth aerocapture)
base = ["../missions/earth.toml"]

[guidance]
type = "ftc"
reference_bank_angle = 64.77026

[guidance.ftc]
capture_damping = 0.7
capture_frequency = 0.072
capture_pdyn_margin = 1.75
altitude_damping = 0.7
altitude_frequency = 0.08
exit_velocity_threshold = 4400.0
exit_pdyn_margin = 1.75
exit_altitude_threshold = 60.0
exit_radial_vel_gain = 10.0
exit_apoapsis_threshold = 100.0
corridor_slope = 108000.0
corridor_intercept = 0.0
max_reversals = 5
security_capture = 1
security_exit = 3
density_filter_gain = 0.8
longi_activation = 50.0
longi_inhibition = -200.0
lateral_activation = -200.0
lateral_inhibition = -200.0
pdyn_min = 0.0
pdyn_table = [
    { altitude =   0.0000000000, a = -3978.560682,    b = 4.254960540 },
    { altitude = 294.2486630,    a = -2021.990388,    b = 1.985120540 },
    { altitude = 314.2486630,    a = -948.9086717,    b = 0.9399715279 },
    { altitude = 334.2486630,    a = -471.0430133,    b = 0.4488293474 },
    { altitude = 354.2486630,    a = -227.0980177,    b = 0.2189307700 },
    { altitude = 374.2486630,    a = -108.5468725,    b = 0.09801385362 },
    { altitude = 394.2486630,    a = -52.62852619,    b = 0.04763612425 },
    { altitude = 414.2486630,    a = -24.08581596,    b = 0.02390645503 },
    { altitude = 434.2486630,    a = -11.59351106,    b = 0.01146473399 },
    { altitude = 454.2486630,    a = -5.331006423,    b = 0.005251233998 },
    { altitude = 474.2486630,    a = -2.639476039,    b = 0.002557150918 },
    { altitude = 494.2486630,    a = -1.264027292,    b = 0.001201952452 },
    { altitude = 514.2486630,    a = -0.5883005931,   b = 0.0005462162713 },
    { altitude = 534.2486630,    a = -0.2634240038,   b = 0.0002700433351 },
    { altitude = 554.2486630,    a = -0.1317516793,   b = 0.0001302086771 },
    { altitude = 574.2486630,    a = -0.06436022892,  b = 0.00006122233561 },
    { altitude = 594.2486630,    a = -0.04445302154,  b = 0.00003411902363 },
    { altitude = 614.2486630,    a = -0.0000001,      b = 0.00003411902363 },
]

[simulation]
n_sims = 1
screen_output = true
random_seed = 0.6866

[data]
results_suffix = ".esr_aller_ftc"
```

- [ ] **Step 3: Run full test suite**

Run: `cd src/rust && cargo test`
Run: `uv run pytest tests/ -v --tb=short 2>&1 | tail -30`
Expected: ALL tests PASS

- [ ] **Step 4: Commit**

```bash
git add configs/nominal/
git commit -m "refactor: migrate 5 nominal configs to use base inheritance"
```

---

## Chunk 5: Rebuild PyO3 + Full Verification

### Task 13: Rebuild PyO3 bindings and run full test suite

**Files:** None (verification only)

- [ ] **Step 1: Rebuild Rust binary**

Run: `cd src/rust && cargo build --release`
Expected: Build succeeds

- [ ] **Step 2: Rebuild PyO3 bindings**

Run: `cd src/rust/aerocapture-py && maturin develop --release`
Expected: Build succeeds

- [ ] **Step 3: Run full Rust test suite**

Run: `cd src/rust && cargo test`
Expected: All tests PASS including golden regressions

- [ ] **Step 4: Run full Python test suite**

Run: `cd /Users/govit/Git/Govit/Aerocapture && uv run pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 5: Run linter**

Run: `./lint_code.sh`
Expected: Clean

- [ ] **Step 6: Run Rust checks**

Run: `./check_all.sh`
Expected: All checks pass (fmt, clippy, test, release build)

- [ ] **Step 7: Run CLI smoke test with a base-inherited config**

Run: `./src/rust/target/release/aerocapture configs/training/msr_aller_ftc_train.toml`
Expected: Runs successfully (resolves bases, runs 10 MC sims)

- [ ] **Step 8: Commit (if any fixes were needed)**

Only if prior steps required fixes.

---

## Chunk 6: Smart Commit

### Task 14: Final commit with documentation sync

- [ ] **Step 1: Invoke smart-commit skill**

Invoke the `smart-commit` skill, telling it to take the whole `feature/toml-base-inheritance` git branch into account (not just staged changes). This will update CLAUDE.md and README.md to reflect the new base inheritance system, then commit everything.
