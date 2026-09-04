//! Every committed NN model JSON must load and re-save stably.
//!
//! `committed_models_load_and_resave_stably` walks `models/` and
//! `articles/paper/data/runs/` for `best_model.json` (plus the two v1 files)
//! and, for each: load -> save -> reload, asserting the flat weights are
//! bit-identical, the architecture is unchanged, and a second save reproduces
//! the first save's bytes.
//!
//! `resave_models_from_list` (ignored) is the before/after byte-diff tool for
//! any serialization change: `NN_RESAVE_LIST=<file of repo-relative paths>`
//! `NN_RESAVE_DIR=<out dir>` `cargo test --release --test nn_model_roundtrip --
//! --ignored`. Run it before and after the change and `diff -r` the two dirs.

mod common;

use std::path::{Path, PathBuf};

use aerocapture::data::neural::NeuralNetModel;

fn walk(dir: &Path, file_name: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, file_name, out);
        } else if path.file_name().and_then(|n| n.to_str()) == Some(file_name) {
            out.push(path);
        }
    }
}

fn save_bytes(model: &NeuralNetModel) -> String {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    model.save_json(tmp.path().to_str().unwrap()).unwrap();
    std::fs::read_to_string(tmp.path()).unwrap()
}

#[test]
fn committed_models_load_and_resave_stably() {
    let root = common::repo_root();
    let mut paths = vec![
        root.join("data/neural_network/nn_model.json"),
        root.join("tests/reference_data/rust_golden/neural/nn_model_golden.json"),
    ];
    walk(&root.join("models"), "best_model.json", &mut paths);
    walk(
        &root.join("articles/paper/data/runs"),
        "best_model.json",
        &mut paths,
    );
    paths.sort();
    assert!(
        paths.len() > 50,
        "expected the committed paper models, found {}",
        paths.len()
    );

    for path in &paths {
        let p = path.to_str().unwrap();
        let model = NeuralNetModel::load(p).unwrap_or_else(|e| panic!("{p}: {e}"));
        let first = save_bytes(&model);
        let reloaded = NeuralNetModel::from_json_str(&first, p).unwrap();
        assert_eq!(
            reloaded.to_flat_weights(),
            model.to_flat_weights(),
            "{p}: flat weights changed across save/reload"
        );
        assert_eq!(
            format!("{:?}", reloaded.architecture),
            format!("{:?}", model.architecture),
            "{p}: architecture changed across save/reload"
        );
        assert_eq!(save_bytes(&reloaded), first, "{p}: re-save not byte-stable");
    }
}

#[test]
#[ignore = "byte-diff tool: set NN_RESAVE_LIST + NN_RESAVE_DIR"]
fn resave_models_from_list() {
    let list = std::env::var("NN_RESAVE_LIST").expect("NN_RESAVE_LIST");
    let out_dir = PathBuf::from(std::env::var("NN_RESAVE_DIR").expect("NN_RESAVE_DIR"));
    std::fs::create_dir_all(&out_dir).unwrap();
    let root = common::repo_root();
    let mut n = 0;
    for line in std::fs::read_to_string(&list).unwrap().lines() {
        let rel = line.trim();
        if rel.is_empty() {
            continue;
        }
        let src = root.join(rel);
        let model =
            NeuralNetModel::load(src.to_str().unwrap()).unwrap_or_else(|e| panic!("{rel}: {e}"));
        let dst = out_dir.join(rel.replace('/', "__"));
        model.save_json(dst.to_str().unwrap()).unwrap();
        n += 1;
    }
    println!("resaved {n} models into {}", out_dir.display());
}
