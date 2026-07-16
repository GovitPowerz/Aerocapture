#!/usr/bin/env bash
# Quantization study campaign (paper Appendix D). Phase-gated: run `ptq` first,
# inspect the verdict, copy it into the two QAT configs, then run the rest.
#   ./experiments/paper/17_quantization.sh ptq        # PTQ sweep + LOO on the champion (~minutes)
#   ./experiments/paper/17_quantization.sh bench      # criterion microbench (run BEFORE trainings for clean numbers)
#   ./experiments/paper/17_quantization.sh qat_finetune   # +3000 gens from the champion checkpoint (~0.5 day)
#   ./experiments/paper/17_quantization.sh qat_scratch    # GA 512 x 20000 from scratch (~2.5-3 days)
#   ./experiments/paper/17_quantization.sh finalists  # n=10000 re-score of the four finalist rows
#   ./experiments/paper/17_quantization.sh collect    # bundle JSONs into articles/paper/data/quant/
set -euo pipefail
cd "$(dirname "$0")/../.."

CHAMPION_DIR=training_output/mamba_p962_long
SWEEP_TOML=configs/training/sweep/mamba_p962.toml
QUANT_DIR=training_output/quant

case "${1:-}" in
ptq)
    uv run python -m aerocapture.training.quantize "$QUANT_DIR/ptq_sweep" \
        --toml "$SWEEP_TOML" \
        --model "$CHAMPION_DIR/best_model.json" \
        --params-dir "$CHAMPION_DIR" \
        --n-sims 1000 --loo-bits 4 --sim-timeout 120
    echo
    echo "GATE: read the verdict above; copy granularity/tensor_policy into"
    echo "configs/training/quant/mamba962_qat4_{finetune,scratch}.toml before launching QAT."
    ;;
bench)
    cargo bench --bench quant_forward --manifest-path src/rust/Cargo.toml
    ;;
qat_finetune)
    uv run python - "$(basename "$0" .sh)" <<'PY'
import json, sys
from pathlib import Path
from aerocapture.training.toml_utils import load_toml_with_bases

res = Path("training_output/quant/ptq_sweep/quantization_results.json")
if not res.exists():
    sys.exit("PTQ sweep has not run: execute the ptq phase first (its verdict pins the QAT cell)")
verdict = json.loads(res.read_text())["verdict"]
for cfg in ("configs/training/quant/mamba962_qat4_finetune.toml", "configs/training/quant/mamba962_qat4_scratch.toml"):
    net = load_toml_with_bases(Path(cfg))["network"]
    got = (net["qat_granularity"], net["qat_tensor_policy"])
    want = (verdict["granularity"], verdict["tensor_policy"])
    if got != want:
        sys.exit(f"{cfg}: qat cell {got} != PTQ verdict {want} -- edit the config before launching (Task 10 Step 2)")
print(f"verdict pre-flight OK: {verdict['granularity']}/{verdict['tensor_policy']}")
PY
    mkdir -p "$QUANT_DIR/mamba962_qat4_finetune"
    [ -f "$QUANT_DIR/mamba962_qat4_finetune/checkpoint_g20000.json" ] || cp "$CHAMPION_DIR/checkpoint_g20000.json" "$QUANT_DIR/mamba962_qat4_finetune/"
    [ -f "$QUANT_DIR/mamba962_qat4_finetune/checkpoint_g20000.npz" ] || cp "$CHAMPION_DIR/checkpoint_g20000.npz" "$QUANT_DIR/mamba962_qat4_finetune/"
    latest=$(ls "$QUANT_DIR/mamba962_qat4_finetune"/checkpoint_g*.json 2>/dev/null | sed -E 's/.*_g0*([0-9]+)\.json/\1/' | sort -n | tail -1)
    latest=${latest:-20000}
    if [ "$latest" -ge 23000 ]; then
        echo "qat_finetune already at gen $latest >= 23000 (champion 20000 + 3000): nothing to do"
        exit 0
    fi
    # --sim-timeout: 4-bit-rounded individuals can produce sims that never
    # terminate (the recorded NaN-hang lesson); 120 s only kills pathological
    # ones (nominal sims are ~4 ms) and they cost out as virtual-DV timeouts.
    uv run python -m aerocapture.training.train configs/training/quant/mamba962_qat4_finetune.toml \
        --n-gen $((23000 - latest)) --output-dir "$QUANT_DIR/mamba962_qat4_finetune" --no-tui --sim-timeout 120
    ;;
qat_scratch)
    uv run python - "$(basename "$0" .sh)" <<'PY'
import json, sys
from pathlib import Path
from aerocapture.training.toml_utils import load_toml_with_bases

res = Path("training_output/quant/ptq_sweep/quantization_results.json")
if not res.exists():
    sys.exit("PTQ sweep has not run: execute the ptq phase first (its verdict pins the QAT cell)")
verdict = json.loads(res.read_text())["verdict"]
for cfg in ("configs/training/quant/mamba962_qat4_finetune.toml", "configs/training/quant/mamba962_qat4_scratch.toml"):
    net = load_toml_with_bases(Path(cfg))["network"]
    got = (net["qat_granularity"], net["qat_tensor_policy"])
    want = (verdict["granularity"], verdict["tensor_policy"])
    if got != want:
        sys.exit(f"{cfg}: qat cell {got} != PTQ verdict {want} -- edit the config before launching (Task 10 Step 2)")
print(f"verdict pre-flight OK: {verdict['granularity']}/{verdict['tensor_policy']}")
PY
    latest=$(ls "$QUANT_DIR/mamba962_qat4_scratch"/checkpoint_g*.json 2>/dev/null | sed -E 's/.*_g0*([0-9]+)\.json/\1/' | sort -n | tail -1 || echo "")
    # --sim-timeout: see qat_finetune note (NaN-hang lesson; nominal sims ~4 ms).
    if [ -z "$latest" ]; then
        uv run python -m aerocapture.training.train configs/training/quant/mamba962_qat4_scratch.toml \
            --output-dir "$QUANT_DIR/mamba962_qat4_scratch" --from-scratch --no-tui --sim-timeout 120
    else
        if [ "$latest" -ge 20000 ]; then
            echo "qat_scratch already at gen $latest >= 20000 (matched budget): nothing to do"
            exit 0
        fi
        echo "existing checkpoints found: resuming qat_scratch to gen 20000 (+$((20000 - latest)))"
        uv run python -m aerocapture.training.train configs/training/quant/mamba962_qat4_scratch.toml \
            --n-gen $((20000 - latest)) --output-dir "$QUANT_DIR/mamba962_qat4_scratch" --no-tui --sim-timeout 120
    fi
    ;;
finalists)
    # QAT arms pass quantize=null (their deployed best_model.json is already on-grid);
    # the PTQ finalist quantizes the champion at the verdict cell on the fly.
    uv run python - <<'PY'
import json
from pathlib import Path

verdict = json.loads(Path("training_output/quant/ptq_sweep/quantization_results.json").read_text())["verdict"]
entries = [
    {"label": "champion_fp", "model": "training_output/mamba_p962_long/best_model.json", "params_dir": "training_output/mamba_p962_long", "quantize": None},
    {"label": "ptq4_verdict", "model": "training_output/mamba_p962_long/best_model.json", "params_dir": "training_output/mamba_p962_long",
     "quantize": {"bits": 4, "granularity": verdict["granularity"], "tensor_policy": verdict["tensor_policy"]}},
    {"label": "qat4_finetune", "model": "training_output/quant/mamba962_qat4_finetune/best_model.json", "params_dir": "training_output/quant/mamba962_qat4_finetune", "quantize": None},
    {"label": "qat4_scratch", "model": "training_output/quant/mamba962_qat4_scratch/best_model.json", "params_dir": "training_output/quant/mamba962_qat4_scratch", "quantize": None},
]
Path("training_output/quant/finalists_entries.json").write_text(json.dumps(entries, indent=2))
PY
    uv run python -m aerocapture.training.quantize "$QUANT_DIR/finalists" \
        --toml "$SWEEP_TOML" \
        --model "$CHAMPION_DIR/best_model.json" \
        --n-sims 10000 --sim-timeout 120 \
        --finalists "$QUANT_DIR/finalists_entries.json"
    ;;
collect)
    mkdir -p articles/paper/data/quant
    cp "$QUANT_DIR/ptq_sweep/quantization_results.json" articles/paper/data/quant/
    cp "$QUANT_DIR/finalists/finalists_results.json" articles/paper/data/quant/
    cp "$QUANT_DIR/ptq_sweep/quantization_sweep.svg" articles/paper/figures/ 2>/dev/null || true
    # criterion medians -> one compact JSON
    uv run python - <<'PY'
import json
from pathlib import Path

rows = {}
for d in Path("src/rust/target/criterion/forward").iterdir():
    est = d / "new" / "estimates.json"
    if est.exists():
        e = json.loads(est.read_text())
        rows[d.name] = {"median_ns": e["median"]["point_estimate"], "ci95": [e["median"]["confidence_interval"]["lower_bound"], e["median"]["confidence_interval"]["upper_bound"]]}
Path("articles/paper/data/quant/bench_forward.json").write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
PY
    ;;
*)
    echo "usage: $0 {ptq|bench|qat_finetune|qat_scratch|finalists|collect}" >&2
    exit 1
    ;;
esac
