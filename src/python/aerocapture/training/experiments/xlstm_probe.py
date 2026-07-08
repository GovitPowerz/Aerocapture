"""xLSTM probe: {lstm, slstm, mlstm} matched-budget controlled arms.

Mechanism decomposition: lstm -> slstm isolates exponential gating (can the
cell sharply REVISE a stored estimate when surprise arrives -- the bounce, a
density shock); slstm -> mlstm isolates matrix memory (vs Mamba's diagonal
state). The lstm baseline arm is the paper's sweep cell lstm_p1082 VERBATIM
(Dense(17->10) -> LSTM(10,10) -> Dense(10->2)), anchoring the probe at the sweep's
~1k operating point; the sweep's single trained run cross-checks the baseline
repeats as a reference row. Budgets: lstm 1082 / slstm 1042 / mlstm 1078. slstm
runs the SAME H=10 as lstm -- its -40 params are the single-bias delta inherent
to the cell definition (an axis cost, like mamba3's +192 complex params), not a
budget mismatch; mlstm runs H=19 because it has no recurrent matrices. Arms
inherit the paper's atan2 training environment (msr_aller_nn_atan2_train.toml:
17-input calibrated mask + normalization, scaffolding = "live") AND its sweep
training regime (GA n_pop 300, seed_strategy = "adaptive" with bucket = "max"
curation, training_n_sims 2, n_gen 5000); sigma_run comes from seed-repeats +
GA/curation stochasticity.

CLI (from repo root):
    python -m aerocapture.training.experiments.xlstm_probe --generate --repeats 3
    python -m aerocapture.training.experiments.xlstm_probe --train  --repeats 3 --n-gen 5000 --training-n-sims 2
    python -m aerocapture.training.experiments.xlstm_probe --eval --report --repeats 3 --n-sims 1000
    python -m aerocapture.training.experiments.xlstm_probe --all --repeats 3 --n-gen 5000 --n-sims 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aerocapture.training.experiments import probe_common as pc

BASE_SEED = 20260707
CONFIG_DIR = Path("configs/training/xlstm_probe")
OUT_DIR = Path("training_output/xlstm_probe")

# Sandwich = the sweep cell lstm_p1082's, verbatim (input_size 17 = the atan2
# base's calibrated input_mask length, inherited, not respecified).
_DENSE_IN = {"type": "dense", "input_size": 17, "output_size": 10, "activation": "swish"}
_HEAD_10 = {"type": "dense", "input_size": 10, "output_size": 2, "activation": "asinh"}
_HEAD_19 = {"type": "dense", "input_size": 19, "output_size": 2, "activation": "asinh"}

ARMS: dict[str, list[dict[str, Any]]] = {
    "lstm": [_DENSE_IN, {"type": "lstm", "input_size": 10, "hidden_size": 10}, _HEAD_10],
    "slstm": [_DENSE_IN, {"type": "slstm", "input_size": 10, "hidden_size": 10}, _HEAD_10],
    "mlstm": [_DENSE_IN, {"type": "mlstm", "input_size": 10, "hidden_size": 19}, _HEAD_19],
}
BASELINE = "lstm"
TREATMENTS = ["slstm", "mlstm"]

REFERENCES: dict[str, tuple[Path, Path]] = {
    "lstm_p1082_sweep": (Path("configs/training/sweep/lstm_p1082.toml"), Path("training_output/sweep_lstm_p1082")),
    "lstm_champion": (Path("configs/training/msr_aller_lstm_pso_train.toml"), Path("training_output/neural_network_lstm_pso")),
    "mamba_champion": (Path("configs/training/msr_aller_mamba_pso_train.toml"), Path("training_output/neural_network_mamba_pso")),
}


def generate_configs(repeats: int, n_gen: int, training_n_sims: int) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for arm, arch in ARMS.items():
        for r in range(repeats):
            out_dir = OUT_DIR / f"{arm}_s{r}"
            path = CONFIG_DIR / f"{arm}_s{r}.toml"
            path.write_text(pc.leaf_toml("xlstm_probe", arm, arch, BASE_SEED + r, BASE_SEED, out_dir, n_gen, training_n_sims))
    manifest = pc.write_manifest(ARMS, CONFIG_DIR, {"repeats": repeats, "n_gen": n_gen, "training_n_sims": training_n_sims})
    print(f"Wrote {len(ARMS) * repeats} arm configs to {CONFIG_DIR}/")
    for arm, m in manifest["arms"].items():
        print(f"  {arm}: cell {m['cell_params']}, total {m['total_params']} trainable params")


def eval_all(repeats: int, n_sims: int, sim_timeout: float | None) -> dict[str, Any]:
    from aerocapture.training.evaluate import PROBE_EVAL_SEED_OFFSET, make_reserved_seeds

    seeds = make_reserved_seeds(0, PROBE_EVAL_SEED_OFFSET, n_sims)
    results: dict[str, Any] = {
        "n_sims": n_sims,
        "repeats": repeats,
        "arms": pc.eval_arms(list(ARMS), repeats, CONFIG_DIR, OUT_DIR, seeds, sim_timeout),
        "references": pc.score_references(REFERENCES, seeds, sim_timeout),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "probe_results.json").write_text(json.dumps(results, indent=2))
    print(f"Wrote eval results to {OUT_DIR / 'probe_results.json'}")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="xLSTM probe (lstm vs slstm vs mlstm, matched budget)")
    p.add_argument("--generate", action="store_true", help="write arm configs + manifest")
    p.add_argument("--train", action="store_true", help="GA-train each arm x repeat (subprocess, sweep regime)")
    p.add_argument("--eval", action="store_true", help="score deployed models + references on the reserved pool")
    p.add_argument("--report", action="store_true", help="print the arm comparison table + significance")
    p.add_argument("--all", action="store_true", help="generate -> train -> eval -> report")
    p.add_argument("--repeats", type=int, default=3, help="seed-repeats per arm (sigma_run sample)")
    p.add_argument("--n-gen", type=int, default=5000, help="GA generations per training run (sweep regime)")
    p.add_argument("--training-n-sims", type=int, default=2, help="sims per individual per generation (sweep regime)")
    p.add_argument("--n-sims", type=int, default=1000, help="reserved eval pool size")
    p.add_argument("--sim-timeout", type=float, default=None, help="per-sim wall-clock timeout (s)")
    p.add_argument("--force", action="store_true", help="retrain even if best_model.json exists")
    p.add_argument("--from-scratch", action="store_true", help="wipe checkpoints + retrain")
    args = p.parse_args()

    if not any((args.generate, args.train, args.eval, args.report, args.all)):
        p.error("pass at least one of --generate/--train/--eval/--report/--all")

    if args.generate or args.all:
        generate_configs(args.repeats, args.n_gen, args.training_n_sims)
    if args.train or args.all:
        pc.train_jobs(list(ARMS), args.repeats, CONFIG_DIR, OUT_DIR, args.n_gen, args.training_n_sims, args.sim_timeout, args.force, args.from_scratch)
    results: dict[str, Any] | None = None
    if args.eval or args.all:
        results = eval_all(args.repeats, args.n_sims, args.sim_timeout)
    if args.report or args.all:
        if results is None:
            results = json.loads((OUT_DIR / "probe_results.json").read_text())
        pc.print_report(results, list(ARMS), BASELINE, TREATMENTS, "xLSTM probe (lstm vs slstm vs mlstm, matched budget)")


if __name__ == "__main__":
    main()
