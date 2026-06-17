"""Off-nominal robustness stress: deployed schemes on a HARDER MC pool.

Evaluates each DEPLOYED policy (no retraining) on a disjoint, harder regime
(atmosphere / density-perturbation / navigation / nav_filter at level=high) on
the reserved STRESS_EVAL_SEED_OFFSET (9M) pool. Tests DEPLOYMENT robustness:
FNPAG's forward predictor depends on the onboard density estimate, so degraded
nav + a wider density regime should degrade it MORE than FTC's analytic feedback
law (no predictor-divergence mode) -- the evidence the paper needs before
calling FTC "more robust than FNPAG". The trained policies all saw the medium
regime, so this is a generalization stress, identical scenarios across schemes
(paired on the 9M pool). FNPAG's predictor adapts online to the measured
density, so if it still degrades that is a genuine finding, not under-tuning.

Reports capture-rate drop + tail-DV inflation vs each scheme's nominal (2M)
numbers from results.json. Gated on 01 (classical) + 02 (NN).

Usage:
    uv run python articles/paper/scripts/robustness_stress.py [--n-sims 1000]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src/python"))

from aerocapture.training.paper_stats import run_stats  # noqa: E402

# (label, run_dir, training TOML). NN + the two relevant classicals + pred_guid
# (a second predictor-corrector data point for the FNPAG comparison).
SCHEMES = [
    ("NN", "paper/optimizer_budget/ga_300", "configs/training/paper/dense_p3998_ga.toml"),
    ("joint-FTC", "paper/joint_reference/ftc", "configs/training/msr_aller_ftc_joint_ref_train.toml"),
    ("FTC-fixed", "ftc", "configs/training/msr_aller_ftc_train.toml"),
    ("FNPAG", "fnpag", "configs/training/msr_aller_fnpag_train.toml"),
    ("PredGuid", "pred_guid", "configs/training/msr_aller_pred_guid_train.toml"),
]
# The stress regime: bump the density/nav-coupled domains to high; leave the
# rest at the campaign default (medium) so the stress isolates FNPAG's weak point.
STRESS_OVERRIDES = {
    "monte_carlo.atmosphere.level": "high",
    "monte_carlo.density_perturbation.level": "high",
    "monte_carlo.navigation.level": "high",
    "monte_carlo.nav_filter.level": "high",
}
OUT = REPO / "articles/paper/data/robustness_stress.json"


def _stress_one(label: str, run_dir: str, toml: str, n_sims: int) -> dict:
    import aerocapture_rs
    from aerocapture.training.evaluate import STRESS_EVAL_SEED_OFFSET, make_reserved_seeds
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

    from aerocapture.training.parquet_output import FINAL_COLUMNS, FINAL_RECORD_INDICES

    col = {name: recs[:, idx] for name, idx in zip(FINAL_COLUMNS, FINAL_RECORD_INDICES, strict=True)}
    return {"label": label, **run_stats(col["ifinal"], col["eccentricity"], col["dv_total_m_s"], n_boot=2000)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sims", type=int, default=1000)
    args = parser.parse_args(argv)

    nominal = {}
    res_path = REPO / "articles/paper/data/results.json"
    if res_path.exists():
        runs = json.loads(res_path.read_text())["runs"]
        for label, run_dir, _ in SCHEMES:
            # results.json keys drop the "paper/" prefix; classical dirs nest under classical_baselines/.
            key = run_dir[len("paper/") :] if run_dir.startswith("paper/") else f"classical_baselines/{run_dir}"
            nominal[label] = runs.get(key, {})

    out = []
    for label, run_dir, toml in SCHEMES:
        if not (REPO / "training_output" / run_dir / "final_eval.parquet").exists():
            print(f"  skip {label} ({run_dir} not deployed yet)")
            continue
        s = _stress_one(label, run_dir, toml, args.n_sims)
        nom = nominal.get(label, {})
        s["nominal_capture_pct"] = nom.get("capture_pct")
        s["nominal_dv_cvar95"] = nom.get("dv_cvar95")
        nom_cap, nom_cv = nom.get("capture_pct"), nom.get("dv_cvar95")
        s["capture_drop_pts"] = round(nom_cap - s["capture_pct"], 2) if nom_cap is not None else None
        s["cvar95_inflation"] = round(s.get("dv_cvar95", float("nan")) - nom_cv, 1) if nom_cv is not None and s.get("dv_cvar95") is not None else None
        out.append(s)
        print(f"  {label:8s} stress: capture {s['capture_pct']:5.1f}% (drop {s['capture_drop_pts']}) | CVaR95 {s.get('dv_cvar95')} (+{s['cvar95_inflation']})")

    if out:
        OUT.write_text(json.dumps({"stress_overrides": STRESS_OVERRIDES, "n_sims": args.n_sims, "schemes": out}, indent=2))
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
