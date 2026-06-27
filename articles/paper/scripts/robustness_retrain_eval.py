"""exp-13 robustness retrain eval: does training ON the high-dispersion regime
close the off-nominal gap the paper reports?

Evaluates four policies on the SAME reserved 9M stress pool and high regime that
`robustness_stress.py` uses (atmosphere / density_perturbation / navigation /
nav_filter = high), so the numbers are directly comparable to
`robustness_stress.json`:

    NN-medium       deployed Mamba_962 (medium-trained)   -- the paper's headline
    NN-high         Mamba_962 retrained on the high regime
    jointFTC-medium deployed joint-FTC (medium-trained)   -- the paper's robust classical
    jointFTC-high   joint-FTC retrained on the high regime

Each is scored with its run-local best_model.json + co-trained scaffolding, on
identical paired scenarios (offset 9M), with the high overrides applied uniformly
(idempotent for the high-trained configs, a shift for the medium-trained ones).

Reports per-scheme stress stats plus a summary: how much retraining helps the NN
(NN-high vs NN-medium) and whether the retrained NN beats the retrained classical
off-nominal (NN-high vs jointFTC-high). Skips cells not deployed yet.

Usage:
    uv run python articles/paper/scripts/robustness_retrain_eval.py [--n-sims 1000]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src/python"))

from aerocapture.training.paper_stats import run_stats  # noqa: E402

# (label, run_dir under training_output/, training TOML). Same machinery as
# robustness_stress.py; the high-trained cells point at their exp-13 configs.
SCHEMES = [
    ("NN-medium", "mamba_p962_long", "configs/training/sweep/mamba_p962.toml"),
    ("NN-high", "paper/robustness_retrain/mamba_p962", "configs/training/paper/robustness_retrain/mamba_p962_high.toml"),
    ("jointFTC-medium", "paper/joint_reference/ftc", "configs/training/msr_aller_ftc_joint_ref_train.toml"),
    ("jointFTC-high", "paper/robustness_retrain/ftc_joint", "configs/training/paper/robustness_retrain/ftc_joint_high.toml"),
]
# The stress regime -- identical to robustness_stress.py so the pools line up.
STRESS_OVERRIDES = {
    "monte_carlo.atmosphere.level": "high",
    "monte_carlo.density_perturbation.level": "high",
    "monte_carlo.navigation.level": "high",
    "monte_carlo.nav_filter.level": "high",
}
OUT = REPO / "articles/paper/data/robustness_retrain.json"


def _eval_one(label: str, run_dir: str, toml: str, n_sims: int) -> dict:
    import aerocapture_rs
    from aerocapture.training.evaluate import STRESS_EVAL_SEED_OFFSET, make_reserved_seeds
    from aerocapture.training.parquet_output import FINAL_COLUMNS, FINAL_RECORD_INDICES
    from aerocapture.training.report import _resolve_eval_toml
    from aerocapture.training.toml_utils import load_toml_with_bases

    scheme_dir = REPO / "training_output" / run_dir
    eval_toml, scaffolding = _resolve_eval_toml(Path(toml), scheme_dir)
    base_mc_seed = load_toml_with_bases(eval_toml).get("monte_carlo", {}).get("seed", 42)
    seeds = make_reserved_seeds(base_mc_seed, STRESS_EVAL_SEED_OFFSET, n_sims)

    base: dict = {"simulation.n_sims": 1, **STRESS_OVERRIDES, **scaffolding}
    local_model = scheme_dir / "best_model.json"
    if local_model.exists():
        base["data.neural_network"] = str(local_model.resolve())
    overrides = [{**base, "monte_carlo.seed": s} for s in seeds]

    results = aerocapture_rs.run_batch(toml_path=str(eval_toml.resolve()), overrides_list=overrides, sim_timeout_secs=5.0)
    recs = np.asarray(results.final_records)
    col = {name: recs[:, idx] for name, idx in zip(FINAL_COLUMNS, FINAL_RECORD_INDICES, strict=True)}
    return {"label": label, **run_stats(col["ifinal"], col["eccentricity"], col["dv_total_m_s"], n_boot=2000)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sims", type=int, default=1000)
    args = parser.parse_args(argv)

    out: dict[str, dict] = {}
    for label, run_dir, toml in SCHEMES:
        if not (REPO / "training_output" / run_dir / "final_eval.parquet").exists():
            print(f"  skip {label} ({run_dir} not deployed yet)")
            continue
        s = _eval_one(label, run_dir, toml, args.n_sims)
        out[label] = s
        print(f"  {label:16s} stress: capture {s['capture_pct']:5.1f}% | mean {s['dv_mean']:7.1f} | CVaR95 {s.get('dv_cvar95'):7.1f} | CVaR99 {s.get('dv_cvar99'):7.1f}")

    # Summary: how much retraining helps the NN, and the off-nominal head-to-head.
    def delta(a: str, b: str, field: str) -> float | None:
        if a in out and b in out and out[a].get(field) is not None and out[b].get(field) is not None:
            return round(out[a][field] - out[b][field], 2)
        return None

    summary = {
        "nn_retrain_capture_gain_pts": delta("NN-high", "NN-medium", "capture_pct"),
        "nn_retrain_cvar95_change": delta("NN-high", "NN-medium", "dv_cvar95"),
        "nn_high_vs_jointftc_high_capture_pts": delta("NN-high", "jointFTC-high", "capture_pct"),
        "nn_high_vs_jointftc_high_cvar95": delta("NN-high", "jointFTC-high", "dv_cvar95"),
        "nn_high_vs_jointftc_medium_capture_pts": delta("NN-high", "jointFTC-medium", "capture_pct"),
        "nn_high_vs_jointftc_medium_cvar95": delta("NN-high", "jointFTC-medium", "dv_cvar95"),
    }

    if out:
        OUT.write_text(json.dumps({"stress_overrides": STRESS_OVERRIDES, "n_sims": args.n_sims, "pool": "STRESS_EVAL 9M", "schemes": list(out.values()), "summary": summary}, indent=2))
        print("\n  summary:")
        for k, v in summary.items():
            print(f"    {k:42s} {v}")
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
