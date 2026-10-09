#!/usr/bin/env bash
# Quantization study campaign (paper Appendix C). Phase-gated: run `ptq` first,
# inspect the verdict, copy it into the two QAT configs, then run the rest.
#   ./experiments/paper/17_quantization.sh [v4] ptq           # PTQ sweep + LOO on the champion (~minutes)
#   ./experiments/paper/17_quantization.sh bench              # criterion microbench (architecture-only; run BEFORE trainings for clean numbers)
#   ./experiments/paper/17_quantization.sh [v4] qat_finetune  # +3000 gens from the champion's final checkpoint (v4: ~1.1 h)
#   ./experiments/paper/17_quantization.sh [v4] qat_scratch   # GA from scratch at the champion's budget (v4: 512 x 2 x 20000, ~7.5 h)
#   ./experiments/paper/17_quantization.sh [v4] finalists     # n=10000 re-score of the four finalist rows; v4 also the 10^6 per-scenario rows
#   ./experiments/paper/17_quantization.sh [v4] collect       # bundle JSONs into articles/paper/data/quant{,_v4}/
#
# Two campaigns share this runner. Without `v4` every default is the arxiv-v3 campaign
# (shared-path champion mamba_p962_long, `legacy` noise, training_output/quant/,
# articles/paper/data/quant/), so the committed data still reproduces. `v4` (issue #178)
# flips every default to the deployed per-scenario champion (ou_marginal/hl_mamba_p962,
# `per_draw`, training_output/quant_v4/, articles/paper/data/quant_v4/) and scores the
# finalists on the 10^6 confirmatory through confirmatory_marginal.py. Each default is one
# environment variable (CHAMPION_DIR, SWEEP_TOML, QUANT_DIR, QAT_CONFIG_PREFIX,
# QAT_CELL_PREFIX, NOISE_SEEDING, PAPER_DATA, CONFIRMATORY_MANIFEST; SCRATCH_TARGET_GEN
# defaults to the champion's final generation), overridable per run.
# Both QAT arms resume from the latest checkpoint_g* on rerun (never --from-scratch twice).
set -euo pipefail
cd "$(dirname "$0")/../.."

CAMPAIGN=v3
if [ "${1:-}" = "v4" ]; then
    CAMPAIGN=v4
    shift
    : "${CHAMPION_DIR:=training_output/ou_marginal/hl_mamba_p962}"
    : "${SWEEP_TOML:=configs/training/ou_marginal/hl_mamba_p962.toml}"
    : "${QUANT_DIR:=training_output/quant_v4}"
    : "${QAT_CONFIG_PREFIX:=configs/training/quant/v4_}"
    : "${QAT_CELL_PREFIX:=}"
    : "${NOISE_SEEDING:=per_draw}"
    : "${PAPER_DATA:=articles/paper/data/quant_v4}"
    : "${CONFIRMATORY_MANIFEST:=experiments/ou_marginal/quant_cells_v4.txt}"
else
    : "${CHAMPION_DIR:=training_output/mamba_p962_long}"
    : "${SWEEP_TOML:=configs/training/sweep/mamba_p962.toml}"
    : "${QUANT_DIR:=training_output/quant}"
    : "${QAT_CONFIG_PREFIX:=configs/training/quant/mamba962_}"
    : "${QAT_CELL_PREFIX:=mamba962_}"
    : "${NOISE_SEEDING:=legacy}"
    : "${PAPER_DATA:=articles/paper/data/quant}"
    : "${CONFIRMATORY_MANIFEST:=}"
fi
FINETUNE_TOML="${QAT_CONFIG_PREFIX}qat4_finetune.toml"
SCRATCH_TOML="${QAT_CONFIG_PREFIX}qat4_scratch.toml"
FINETUNE_DIR="$QUANT_DIR/${QAT_CELL_PREFIX}qat4_finetune"
SCRATCH_DIR="$QUANT_DIR/${QAT_CELL_PREFIX}qat4_scratch"
PTQ_RESULTS="$QUANT_DIR/ptq_sweep/quantization_results.json"
export CHAMPION_DIR SWEEP_TOML QUANT_DIR FINETUNE_TOML SCRATCH_TOML FINETUNE_DIR SCRATCH_DIR PTQ_RESULTS NOISE_SEEDING PAPER_DATA CONFIRMATORY_MANIFEST

# latest_gen <dir>: the highest checkpoint_g* generation (0 when there is none), shared with the campaigns.
source experiments/ou_marginal/campaign_lib.sh
ckpt_name() { printf 'checkpoint_g%05d' "$1"; }  # the trainer's zero-padded label (checkpoint.py)

# The PTQ verdict pins the QAT cell: refuse to launch either arm on a config that disagrees.
verdict_gate() {
    uv run python - <<'PY'
import json, os, sys
from pathlib import Path
from aerocapture.training.toml_utils import load_toml_with_bases

res = Path(os.environ["PTQ_RESULTS"])
if not res.exists():
    sys.exit(f"PTQ sweep has not run ({res} missing): execute the ptq phase first (its verdict pins the QAT cell)")
sweep = json.loads(res.read_text())
if sweep.get("noise_seeding", "legacy") != os.environ["NOISE_SEEDING"]:
    sys.exit(f"{res} was scored under {sweep.get('noise_seeding', 'legacy')}, this campaign runs {os.environ['NOISE_SEEDING']}: rerun the ptq phase")
verdict = sweep["verdict"]
for cfg in (os.environ["FINETUNE_TOML"], os.environ["SCRATCH_TOML"]):
    net = load_toml_with_bases(Path(cfg))["network"]
    got = (net["qat_bits"], net["qat_granularity"], net["qat_tensor_policy"])
    want = (verdict["bits"], verdict["granularity"], verdict["tensor_policy"])
    if got != want:
        sys.exit(f"{cfg}: qat cell {got} != PTQ verdict {want} -- edit the config before launching")
print(f"verdict pre-flight OK: {verdict['bits']}b {verdict['granularity']}/{verdict['tensor_policy']}")
PY
}

case "${1:-}" in
ptq)
    uv run python -m aerocapture.training.quantize "$QUANT_DIR/ptq_sweep" \
        --toml "$SWEEP_TOML" \
        --model "$CHAMPION_DIR/best_model.json" \
        --params-dir "$CHAMPION_DIR" \
        --noise-seeding "$NOISE_SEEDING" \
        --n-sims 1000 --loo-bits 4 --sim-timeout 120
    echo
    echo "GATE: read the verdict above; copy granularity/tensor_policy into"
    echo "$FINETUNE_TOML and $SCRATCH_TOML before launching QAT."
    ;;
bench)
    cargo bench --bench quant_forward --manifest-path src/rust/Cargo.toml
    ;;
qat_finetune)
    verdict_gate
    champ_gen=$(latest_gen "$CHAMPION_DIR")
    [ "$champ_gen" -gt 0 ] || { echo "no checkpoint_g*.json in $CHAMPION_DIR: the fine-tune resumes the champion's final checkpoint" >&2; exit 1; }
    target=$((champ_gen + 3000))
    mkdir -p "$FINETUNE_DIR"
    # Seed an empty arm with the champion's final checkpoint pair (npz first: the json is the resume key).
    if [ "$(latest_gen "$FINETUNE_DIR")" -eq 0 ]; then
        cp "$CHAMPION_DIR/$(ckpt_name "$champ_gen").npz" "$FINETUNE_DIR/"
        cp "$CHAMPION_DIR/$(ckpt_name "$champ_gen").json" "$FINETUNE_DIR/"
    fi
    latest=$(latest_gen "$FINETUNE_DIR")
    if [ "$latest" -ge "$target" ]; then
        echo "qat_finetune already at gen $latest >= $target (champion $champ_gen + 3000): nothing to do"
        exit 0
    fi
    # --sim-timeout: 4-bit-rounded individuals can produce sims that never
    # terminate (the recorded NaN-hang lesson); 120 s only kills pathological
    # ones (nominal sims are ~4 ms) and they cost out as virtual-DV timeouts.
    uv run python -u -m aerocapture.training.train "$FINETUNE_TOML" \
        --n-gen $((target - latest)) --output-dir "$FINETUNE_DIR" --no-tui --sim-timeout 120
    ;;
qat_scratch)
    verdict_gate
    target=${SCRATCH_TARGET_GEN:-$(latest_gen "$CHAMPION_DIR")}
    [ "$target" -gt 0 ] || { echo "no checkpoint_g*.json in $CHAMPION_DIR to match the budget of; set SCRATCH_TARGET_GEN" >&2; exit 1; }
    latest=$(latest_gen "$SCRATCH_DIR")
    # --sim-timeout: see qat_finetune note (NaN-hang lesson; nominal sims ~4 ms).
    if [ "$latest" -eq 0 ]; then
        uv run python -u -m aerocapture.training.train "$SCRATCH_TOML" \
            --n-gen "$target" --output-dir "$SCRATCH_DIR" --from-scratch --no-tui --sim-timeout 120
    else
        if [ "$latest" -ge "$target" ]; then
            echo "qat_scratch already at gen $latest >= $target (matched budget): nothing to do"
            exit 0
        fi
        echo "existing checkpoints found: resuming qat_scratch to gen $target (+$((target - latest)))"
        uv run python -u -m aerocapture.training.train "$SCRATCH_TOML" \
            --n-gen $((target - latest)) --output-dir "$SCRATCH_DIR" --no-tui --sim-timeout 120
    fi
    ;;
finalists)
    # QAT arms pass quantize=null (their deployed best_model.json is already on-grid);
    # the PTQ finalist quantizes the champion at the verdict cell on the fly.
    verdict_gate
    uv run python - <<'PY'
import json, os, sys
from pathlib import Path

verdict = json.loads(Path(os.environ["PTQ_RESULTS"]).read_text())["verdict"]
champ, ft, sc = (os.environ[k] for k in ("CHAMPION_DIR", "FINETUNE_DIR", "SCRATCH_DIR"))
if manifest := os.environ["CONFIRMATORY_MANIFEST"]:
    # The 10^6 scorer reads the manifest's dirs, not this run's: refuse a dir override the manifest does not follow.
    sys.path.insert(0, "experiments/ou_marginal")
    from confirmatory_marginal import read_manifest

    want = [champ, os.path.join(os.environ["QUANT_DIR"], "ptq4_verdict"), ft, sc]
    got = [d or f"training_output/{label}" for label, _, d in read_manifest(Path(manifest))]
    if [Path(p).resolve() for p in got] != [Path(p).resolve() for p in want]:
        sys.exit(f"{manifest} scores {got}, this run's dirs are {want}: edit the manifest or drop the dir overrides")
entries = [
    {"label": "champion_fp", "model": f"{champ}/best_model.json", "params_dir": champ, "quantize": None},
    {"label": "ptq4_verdict", "model": f"{champ}/best_model.json", "params_dir": champ,
     "quantize": {"bits": 4, "granularity": verdict["granularity"], "tensor_policy": verdict["tensor_policy"]}},
    {"label": "qat4_finetune", "model": f"{ft}/best_model.json", "params_dir": ft, "quantize": None},
    {"label": "qat4_scratch", "model": f"{sc}/best_model.json", "params_dir": sc, "quantize": None},
]
Path(os.environ["QUANT_DIR"], "finalists_entries.json").write_text(json.dumps(entries, indent=2))
PY
    uv run python -m aerocapture.training.quantize "$QUANT_DIR/finalists" \
        --toml "$SWEEP_TOML" \
        --model "$CHAMPION_DIR/best_model.json" \
        --noise-seeding "$NOISE_SEEDING" \
        --n-sims 10000 --sim-timeout 120 \
        --finalists "$QUANT_DIR/finalists_entries.json"
    if [ -n "$CONFIRMATORY_MANIFEST" ]; then
        # The 10^6 per-scenario rows (#171's scorer: manifest rows, per-replicate persistence, the
        # champion's row already scored by #174 is skipped). The scorer reads a cell's best_model.json
        # from its dir, so the PTQ verdict is materialized once as a cell of its own; a materialized
        # model that no longer matches the verdict is refused (the scored row pins its model_sha256).
        uv run python - <<'PY'
import hashlib, json, os, shutil, sys
from pathlib import Path
from aerocapture.training.quantize import quantize_model_weights

sweep = json.loads(Path(os.environ["PTQ_RESULTS"]).read_text())
verdict = sweep["verdict"]
champ = Path(os.environ["CHAMPION_DIR"])
cell = Path(os.environ["QUANT_DIR"]) / "ptq4_verdict"
model = json.dumps(quantize_model_weights(json.loads((champ / "best_model.json").read_text()), 4, verdict["granularity"], verdict["tensor_policy"]))
out = cell / "best_model.json"
if out.exists() and out.read_text() != model:
    sys.exit(f"{out} differs from the champion quantized at the current verdict: delete {cell} (and its scored row) to re-materialize")
cell.mkdir(parents=True, exist_ok=True)
out.write_text(model)
shutil.copyfile(champ / "best_params.json", cell / "best_params.json")
# confirmatory_marginal.py quotes a cell only once its run is over (a final_selection.json); this
# cell has no run, its provenance is the verdict rule applied to the champion.
(cell / "final_selection.json").write_text(json.dumps({
    "winner": {"provenance": "ptq4_verdict", "source_cell": str(champ), "source_model_sha256": hashlib.sha256((champ / "best_model.json").read_bytes()).hexdigest()},
    "verdict": {"bits": 4, "granularity": verdict["granularity"], "tensor_policy": verdict["tensor_policy"]},
    "ptq_results": os.environ["PTQ_RESULTS"],
    "noise_seeding": sweep.get("noise_seeding", "legacy"),
}, indent=2))
print(f"materialized {out} at {verdict['granularity']}/{verdict['tensor_policy']}")
PY
        uv run python -u experiments/ou_marginal/confirmatory_marginal.py --manifest "$CONFIRMATORY_MANIFEST"
        uv run python -u experiments/ou_marginal/confirmatory_marginal.py --manifest "$CONFIRMATORY_MANIFEST" --table
    fi
    ;;
collect)
    mkdir -p "$PAPER_DATA"
    cp "$QUANT_DIR/ptq_sweep/quantization_results.json" "$PAPER_DATA/"
    cp "$QUANT_DIR/finalists/finalists_results.json" "$PAPER_DATA/"
    if [ "$CAMPAIGN" = v4 ]; then
        # The forward-pass micro-benchmark is architecture-only: data/quant/bench_forward.json stays.
        cp "$QUANT_DIR/ptq_sweep/quantization_sweep.svg" articles/paper/figures/quantization_sweep_v4.svg 2>/dev/null || true
        uv run python articles/paper/scripts/extract_quant_v4.py
        echo "then: make -C articles/paper sums provenance, commit"
        exit 0
    fi
    cp "$QUANT_DIR/ptq_sweep/quantization_sweep.svg" articles/paper/figures/ 2>/dev/null || true
    # criterion medians -> one compact JSON
    uv run python - <<'PY'
import json, os
from pathlib import Path

rows = {}
for d in Path("src/rust/target/criterion/forward").iterdir():
    est = d / "new" / "estimates.json"
    if est.exists():
        e = json.loads(est.read_text())
        rows[d.name] = {"median_ns": e["median"]["point_estimate"], "ci95": [e["median"]["confidence_interval"]["lower_bound"], e["median"]["confidence_interval"]["upper_bound"]]}
Path(os.environ["PAPER_DATA"], "bench_forward.json").write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
PY
    ;;
*)
    echo "usage: $0 [v4] {ptq|bench|qat_finetune|qat_scratch|finalists|collect}" >&2
    exit 1
    ;;
esac
