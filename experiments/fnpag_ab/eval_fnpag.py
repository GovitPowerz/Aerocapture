"""FNPAG A/B eval: deployed scaffolding, fixed validation seeds, paired output.

Usage: uv run python experiments/fnpag_ab/eval_fnpag.py <label> [n_sims]
Writes experiments/fnpag_ab/result_<label>.json (per-seed dv + summary).
"""

import json
import sys
from pathlib import Path

import numpy as np

import aerocapture_rs
from aerocapture.training.evaluate import VALIDATION_SEED_OFFSET, make_reserved_seeds
from aerocapture.training.param_spaces import route_scaffolding_param

import os

TOML = os.environ.get("AERO_TOML", "configs/training/msr_aller_fnpag_train.toml")
BEST = os.environ.get("AERO_BEST", "training_output/fnpag/best_params.json")
SCHEME = os.environ.get("AERO_SCHEME", "fnpag")
BASE_SEED = 42

IDX = aerocapture_rs.final_record_indices()
C_DV, C_DV1, C_DV2, C_DV3 = IDX["dv_total_ms"], IDX["dv1_ms"], IDX["dv2_ms"], IDX["dv3_ms"]
C_APOERR, C_PERIERR = IDX["apoapsis_err_km"], IDX["periapsis_err_km"]
C_IFINAL, C_ECC = IDX["ifinal"], IDX["ecc"]


def pct(a, label):
    a = np.asarray(a, float)
    return {
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "mean": float(a.mean()),
        "max": float(a.max()),
    }


def main():
    label = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 256

    best = json.loads(Path(BEST).read_text())
    routed = dict(route_scaffolding_param(k, v, SCHEME) for k, v in best.items())

    seeds = make_reserved_seeds(BASE_SEED, VALIDATION_SEED_OFFSET, n)
    ovr = [{**routed, "monte_carlo.seed": int(s), "simulation.n_sims": 1} for s in seeds]

    res = aerocapture_rs.run_batch(TOML, ovr, sim_timeout_secs=10.0)
    fr = np.asarray(res.final_records)  # (n, 52)

    captured = (fr[:, C_IFINAL] == 3) & (fr[:, C_ECC] < 1.0)
    cap = fr[captured]

    out = {
        "label": label,
        "n_sims": int(n),
        "capture_rate": float(captured.mean()),
        "seeds": [int(s) for s in seeds],
        "captured_mask": captured.astype(int).tolist(),
        "dv_total": fr[:, C_DV].tolist(),  # all sims, per-seed (paired key)
        "summary_captured": {
            "dv": pct(cap[:, C_DV], "dv"),
            "dv1": pct(cap[:, C_DV1], "dv1"),
            "dv2": pct(cap[:, C_DV2], "dv2"),
            "dv3": pct(cap[:, C_DV3], "dv3"),
            "apoapsis_err_km": pct(np.abs(cap[:, C_APOERR]), "apo"),
            "periapsis_err_km": pct(np.abs(cap[:, C_PERIERR]), "peri"),
        },
    }
    Path("experiments/fnpag_ab").mkdir(parents=True, exist_ok=True)
    Path(f"experiments/fnpag_ab/result_{label}.json").write_text(json.dumps(out, indent=2))

    s = out["summary_captured"]
    print(f"[{label}] n={n} capture={out['capture_rate']*100:.1f}%")
    for k in ("dv", "dv1", "dv2", "dv3", "apoapsis_err_km", "periapsis_err_km"):
        v = s[k]
        print(f"  {k:18s} p50={v['p50']:9.2f}  p95={v['p95']:9.2f}  mean={v['mean']:9.2f}  max={v['max']:9.2f}")


if __name__ == "__main__":
    main()
