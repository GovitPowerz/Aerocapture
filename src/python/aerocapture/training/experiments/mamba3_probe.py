"""Mamba-3 probe: the 2x2 (discretization x state_mode) under the shared probe regime.

Supersedes the original reduced-budget spike (PSO n_pop 64, fixed seeds, 21-input
default normalization): arms now inherit the paper's atan2 training environment
(msr_aller_nn_atan2_train.toml: 17-input calibrated mask + normalization,
scaffolding = "live") AND its sweep training regime (GA n_pop 300, seed_strategy =
"adaptive" with bucket = "max" curation, training_n_sims 2, n_gen 5000) -- the same
regime as cfc_probe / xlstm_probe, so all probe rows share one paper methodology.

Arms run at the Mamba_962 anchor dims: the euler+real baseline arm is BIT-identical
to the deployed Mamba_962 cell, so the 2x2 isolates the two Mamba-3 axes at the
paper's headline operating point. Arms are NOT budget-matched to each other
(baseline 962 / trapz 978 / complex 1154 / both 1170 NN params) -- the axis costs
are inherent to the ablation and recorded in the manifest. Seed-repeats give the
sigma_run error bars the full-budget mamba3_962 campaign (GA 512 x 10k, single run
per arm) could not; its four arms appear here as reference rows alongside the
deployed Mamba champion.

CLI (from repo root):
    python -m aerocapture.training.experiments.mamba3_probe --generate --repeats 3
    python -m aerocapture.training.experiments.mamba3_probe --train  --repeats 3 --n-gen 5000 --training-n-sims 2
    python -m aerocapture.training.experiments.mamba3_probe --eval --report --repeats 3 --n-sims 1000
    python -m aerocapture.training.experiments.mamba3_probe --all --repeats 3 --n-gen 5000 --n-sims 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aerocapture.training.experiments import probe_common as pc

BASE_SEED = 20260707  # same as the other probes; per-repeat monte_carlo.seed = BASE_SEED + r
CONFIG_DIR = Path("configs/training/mamba3_probe")
OUT_DIR = Path("training_output/mamba3_probe")

# Mamba_962 anchor dims (dt_rank omitted -> resolves to max(1, 16 // 16) = 1).
_DENSE_IN = {"type": "dense", "input_size": 17, "output_size": 16, "activation": "swish"}
_DENSE_OUT = {"type": "dense", "input_size": 16, "output_size": 2, "activation": "asinh"}


def _m3(discretization: str, state_mode: str) -> dict[str, Any]:
    return {"type": "mamba3", "input_size": 16, "d_state": 12, "discretization": discretization, "state_mode": state_mode}


# arm -> full architecture. euler+real == the deployed Mamba_962 cell (962 NN params);
# the axis costs (+16 trapz, +192 complex) are inherent, not a budget mismatch.
ARMS: dict[str, list[dict[str, Any]]] = {
    "baseline": [_DENSE_IN, _m3("euler", "real"), _DENSE_OUT],
    "trapz": [_DENSE_IN, _m3("trapezoidal", "real"), _DENSE_OUT],
    "complex": [_DENSE_IN, _m3("euler", "complex"), _DENSE_OUT],
    "both": [_DENSE_IN, _m3("trapezoidal", "complex"), _DENSE_OUT],
}
BASELINE = "baseline"
TREATMENTS = ["trapz", "complex", "both"]

# Deployed champion + the four full-budget (GA 512 x 10k, single-run) 962 arms.
REFERENCES: dict[str, tuple[Path, Path]] = {
    "mamba_champion": (Path("configs/training/msr_aller_mamba_pso_train.toml"), Path("training_output/neural_network_mamba_pso")),
    "962_baseline": (Path("configs/training/mamba3_962/baseline.toml"), Path("training_output/mamba3_962/baseline")),
    "962_trapz": (Path("configs/training/mamba3_962/trapz.toml"), Path("training_output/mamba3_962/trapz")),
    "962_complex": (Path("configs/training/mamba3_962/complex.toml"), Path("training_output/mamba3_962/complex")),
    "962_both": (Path("configs/training/mamba3_962/both.toml"), Path("training_output/mamba3_962/both")),
}


def generate_configs(repeats: int, n_gen: int, training_n_sims: int) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for arm, arch in ARMS.items():
        for r in range(repeats):
            out_dir = OUT_DIR / f"{arm}_s{r}"
            path = CONFIG_DIR / f"{arm}_s{r}.toml"
            path.write_text(pc.leaf_toml("mamba3_probe", arm, arch, BASE_SEED + r, BASE_SEED, out_dir, n_gen, training_n_sims))
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
    p = argparse.ArgumentParser(description="Mamba-3 2x2 probe at the 962-dim anchor (sweep regime)")
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
        pc.print_report(results, list(ARMS), BASELINE, TREATMENTS, "Mamba-3 probe (2x2 at the 962-dim anchor, sweep regime)")


if __name__ == "__main__":
    main()
