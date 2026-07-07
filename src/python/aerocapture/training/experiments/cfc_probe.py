"""CfC probe: {gru, cfc} matched-budget controlled arms -> tail DV with sigma_run.

Hypothesis under test: input-dependent time constants (CfC) match or beat the
closest scalar-state baseline (GRU) on the sizing tail at the same param budget
(gru 7106 vs cfc 7074 total trainable, -0.5%). Both arms train on identical
fixed seeds; sigma_run comes from seed-repeats + PSO stochasticity. Deployed
GRU/Mamba champions are scored on the same reserved pool as reference rows
(NOT budget-matched -- own masks/settings).

CLI (from repo root):
    python -m aerocapture.training.experiments.cfc_probe --generate --repeats 3
    python -m aerocapture.training.experiments.cfc_probe --train  --repeats 3 --n-gen 500 --training-n-sims 10
    python -m aerocapture.training.experiments.cfc_probe --eval --report --repeats 3 --n-sims 1000
    python -m aerocapture.training.experiments.cfc_probe --all --repeats 3 --n-gen 500 --n-sims 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aerocapture.training.experiments import probe_common as pc

BASE_SEED = 20260707  # same as mamba3_ablation -- identical training seed lists across probes
CONFIG_DIR = Path("configs/training/cfc_probe")
OUT_DIR = Path("training_output/cfc_probe")
INPUT_MASK = list(range(21))

_DENSE_IN = {"type": "dense", "input_size": 21, "output_size": 32, "activation": "swish"}
_DENSE_OUT = {"type": "dense", "input_size": 32, "output_size": 2, "activation": "asinh"}

# arm -> full architecture (budget-matched: gru 6336 vs cfc 6304 cell params)
ARMS: dict[str, list[dict[str, Any]]] = {
    "gru": [_DENSE_IN, {"type": "gru", "input_size": 32, "hidden_size": 32}, _DENSE_OUT],
    "cfc": [_DENSE_IN, {"type": "cfc", "input_size": 32, "hidden_size": 32, "backbone_units": 32}, _DENSE_OUT],
}
BASELINE = "gru"
TREATMENTS = ["cfc"]

# Deployed champions scored on the same pool (reference rows, not budget-matched).
REFERENCES: dict[str, tuple[Path, Path]] = {
    "gru_champion": (Path("configs/training/msr_aller_gru_pso_train.toml"), Path("training_output/neural_network_gru_pso")),
    "mamba_champion": (Path("configs/training/msr_aller_mamba_pso_train.toml"), Path("training_output/neural_network_mamba_pso")),
}


def generate_configs(repeats: int, n_gen: int, training_n_sims: int) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for arm, arch in ARMS.items():
        for r in range(repeats):
            out_dir = OUT_DIR / f"{arm}_s{r}"
            path = CONFIG_DIR / f"{arm}_s{r}.toml"
            path.write_text(pc.leaf_toml("cfc_probe", arm, arch, BASE_SEED + r, BASE_SEED, out_dir, n_gen, training_n_sims, INPUT_MASK))
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
    p = argparse.ArgumentParser(description="CfC vs GRU matched-budget probe")
    p.add_argument("--generate", action="store_true", help="write arm configs + manifest")
    p.add_argument("--train", action="store_true", help="PSO-train each arm x repeat (subprocess)")
    p.add_argument("--eval", action="store_true", help="score deployed models + references on the reserved pool")
    p.add_argument("--report", action="store_true", help="print the arm comparison table + significance")
    p.add_argument("--all", action="store_true", help="generate -> train -> eval -> report")
    p.add_argument("--repeats", type=int, default=3, help="seed-repeats per arm (sigma_run sample)")
    p.add_argument("--n-gen", type=int, default=500, help="PSO generations per training run")
    p.add_argument("--training-n-sims", type=int, default=10, help="sims per individual per generation")
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
        pc.print_report(results, list(ARMS), BASELINE, TREATMENTS, "CfC probe (cfc vs gru, matched budget)")


if __name__ == "__main__":
    main()
