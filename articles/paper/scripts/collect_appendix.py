"""Collect the per-scheme appendix mission-report data (Appendix A).

For each benchmarked guidance scheme, re-run 1000-sim MC on the reserved
FINAL_EVAL (2M) pool with trajectories (pinned to the run-local deployed model
+ co-trained scaffolding, so the numbers reproduce Table 3 / results.json), then
render the report-style corridor + constraint SVGs and a stats.json into
articles/paper/figures/appendix/<slug>/. Collector-vs-figure split: this reads
training_output/; the committed SVGs + stats.json are the durable artifacts.

Usage:
    uv run python articles/paper/scripts/collect_appendix.py [--schemes SLUG ...] [--n-sims 1000]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src/python"))

FIGROOT = REPO / "articles/paper/figures/appendix"
RESULTS = REPO / "articles/paper/data/results.json"
CORRIDOR_CACHE = REPO / "training_output/mars/corridor_boundaries.npz"
N_TRAJ_SPAGHETTI = 300
POINT_STRIDE = 3

# (slug, title, run_dir under training_output/, training TOML, results.json key)
SCHEMES = [
    ("nn_mamba", "NN -- Mamba (962 params)", "mamba_p962_long",
     "configs/training/sweep/mamba_p962.toml", "headline/mamba_p962"),
    ("nn_lstm", "NN -- LSTM (1082 params)", "lstm_p1082_long",
     "configs/training/sweep/lstm_p1082.toml", "headline/lstm_p1082"),
    ("nn_gru", "NN -- GRU (1014 params)", "gru_p1014_long",
     "configs/training/sweep/gru_p1014.toml", "headline/gru_p1014"),
    ("nn_dense", "NN -- Dense (515 params)", "dense_p515_ga_paper_best",
     "configs/training/msr_aller_nn_atan2_best_paper.toml", "headline/dense_p515"),
    ("ftc", "FTC (joint reference)", "paper/joint_reference/ftc",
     "configs/training/msr_aller_ftc_joint_ref_train.toml", "joint_reference/ftc"),
    ("fnpag", "FNPAG", "fnpag",
     "configs/training/msr_aller_fnpag_train.toml", "classical_baselines/fnpag"),
    ("predguid", "PredGuid (joint reference)", "paper/joint_reference/pred_guid",
     "configs/training/msr_aller_pred_guid_joint_ref_train.toml", "joint_reference/pred_guid"),
    ("energyctl", "Energy controller (joint reference)", "paper/joint_reference/energy_controller",
     "configs/training/msr_aller_energy_controller_joint_ref_train.toml", "joint_reference/energy_controller"),
    ("eqglide", "Equilibrium glide", "equilibrium_glide",
     "configs/training/msr_aller_eqglide_train.toml", "classical_baselines/equilibrium_glide"),
    ("piecewise", "Piecewise constant", "piecewise_constant",
     "configs/training/msr_aller_piecewise_constant_train.toml", "classical_baselines/piecewise_constant"),
]


def chart_dv_cdf_overlay(final_records, output):
    """4-curve ECDF: total correction DV (bold) + the 3 burns, captured only."""
    from aerocapture.training import charts

    cap = charts.is_captured(final_records)
    rec = final_records[cap]
    series = [
        ("total Δv", np.abs(rec[:, charts._FR_DV_TOTAL]), "#111111", 2.0),
        ("dv1 periapsis raise", np.abs(rec[:, charts._FR_DV1]), "#4878cf", 1.2),
        ("dv2 circularization", np.abs(rec[:, charts._FR_DV2]), "#d1701f", 1.2),
        ("dv3 plane change", np.abs(rec[:, charts._FR_DV3]), "#6a51a3", 1.2),
    ]
    sns.set_theme(style="whitegrid", palette="muted", rc={"axes.facecolor": "#f5f5f5"})
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for label, vals, color, lw in series:
        v = np.sort(vals)
        y = np.arange(1, len(v) + 1) / len(v)
        ax.plot(v, y, color=color, lw=lw, label=label)
    ax.set_xlabel("correction Δv (m/s)")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("Correction Δv -- empirical CDF (total + burns)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize="small", loc="lower right")
    sns.despine(fig=fig)
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def collect_one(slug, title, run_dir, toml, results_key, n_sims):
    import aerocapture_rs
    from aerocapture.training import charts
    from aerocapture.training.evaluate import FINAL_EVAL_SEED_OFFSET, make_reserved_seeds
    from aerocapture.training.reference import _MC_DISPERSION_DOMAINS
    from aerocapture.training.report import _read_constraint_limits, _resolve_eval_toml, compute_eval_summary, read_cost_kwargs
    from aerocapture.training.toml_utils import load_toml_with_bases

    scheme_dir = REPO / "training_output" / run_dir
    eval_toml, scaffolding = _resolve_eval_toml(REPO / toml, scheme_dir)
    base_mc_seed = load_toml_with_bases(eval_toml).get("monte_carlo", {}).get("seed", 42)
    seeds = make_reserved_seeds(base_mc_seed, FINAL_EVAL_SEED_OFFSET, n_sims)

    pin = dict(scaffolding)
    local_model = scheme_dir / "best_model.json"
    if local_model.exists():
        pin["data.neural_network"] = str(local_model.resolve())
    overrides = [{"simulation.n_sims": 1, "monte_carlo.seed": s, **pin} for s in seeds]
    batch = aerocapture_rs.run_batch(
        toml_path=str(eval_toml.resolve()), overrides_list=overrides,
        include_trajectories=True, sim_timeout_secs=5.0,
    )
    recs = np.asarray(batch.final_records)
    trajs = [np.asarray(t) for t in batch.trajectories]

    # drift self-check vs results.json (the far_tail mislabel trap)
    ref = json.loads(RESULTS.read_text())["runs"][results_key]
    cap = charts.is_captured(recs)
    got_cap = 100.0 * float(cap.mean())
    got_mean = float(np.abs(recs[cap, charts._FR_DV_TOTAL]).mean())
    tag = "OK" if (abs(got_cap - ref["capture_pct"]) <= 0.6 and abs(got_mean - ref["dv_mean"]) <= 2.0) else "DRIFT"
    print(f"[{slug}] capture {got_cap:.1f} vs {ref['capture_pct']} | mean {got_mean:.1f} vs {ref['dv_mean']}  [{tag}]")

    hfl, gll, hll = _read_constraint_limits(eval_toml)
    traj_class = charts.classify_trajectories(recs, heat_flux_limit=hfl, g_load_limit=gll, heat_load_limit=hll)

    # spaghetti subsample (stats use all sims; lines use a subset + point stride)
    k = max(1, len(trajs) // N_TRAJ_SPAGHETTI)
    idx = list(range(0, len(trajs), k))[:N_TRAJ_SPAGHETTI]
    sub_trajs = [trajs[i][::POINT_STRIDE] for i in idx]
    sub_class = traj_class[idx]

    zones = dict(np.load(CORRIDOR_CACHE))
    corridor_data = {key: zones[key] for key in (
        "energy_bins", "envelope_crash_pdyn", "envelope_restricted_max_pdyn",
        "envelope_restricted_min_pdyn", "envelope_capture_pdyn")}

    nom_ov = {"simulation.n_sims": 1,
              **{f"monte_carlo.{d}.level": "off" for d in _MC_DISPERSION_DOMAINS}, **pin}
    nom = aerocapture_rs.run_mc(toml_path=str(eval_toml.resolve()), overrides=nom_ov,
                                include_trajectories=True, sim_timeout_secs=5.0)
    undispersed = np.asarray(nom.trajectories[0]) if nom.trajectories else None
    nk = {"undispersed_nominal": undispersed}

    out = FIGROOT / slug
    out.mkdir(parents=True, exist_ok=True)
    charts.chart_corridor_pdyn(sub_trajs, sub_class, out / "corridor_pdyn.svg", corridor_data=corridor_data, **nk)
    charts.chart_corridor_inclination(sub_trajs, sub_class, out / "corridor_inclination.svg", **nk)
    charts.chart_corridor_bank(sub_trajs, sub_class, out / "corridor_bank.svg", **nk)
    chart_dv_cdf_overlay(recs, out / "dv_cdf.svg")
    charts.chart_heat_flux_time(sub_trajs, sub_class, out / "heat_flux.svg", limit_kw_m2=hfl, **nk)
    charts.chart_gload_time(sub_trajs, sub_class, out / "g_load.svg", limit_g=gll, **nk)
    charts.chart_heat_load_time(sub_trajs, sub_class, out / "heat_load.svg", limit_kj_m2=hll, **nk)

    summary = compute_eval_summary(recs, n_sims=len(recs), cost_kwargs=read_cost_kwargs(eval_toml))
    dvc = np.abs(recs[cap, charts._FR_DV_TOTAL])
    summary["dv_p99"] = float(np.percentile(dvc, 99))
    summary["dv_cvar95"] = float(np.sort(dvc)[-max(1, round(len(dvc) * 0.05)):].mean())
    summary["title"] = title
    (out / "stats.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"  wrote {out.relative_to(REPO)} (7 svg + stats.json)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schemes", nargs="*", default=None, help="slugs to collect (default all)")
    parser.add_argument("--n-sims", type=int, default=1000)
    args = parser.parse_args()
    wanted = set(args.schemes) if args.schemes else None
    for row in SCHEMES:
        if wanted is None or row[0] in wanted:
            collect_one(*row, n_sims=args.n_sims)


if __name__ == "__main__":
    main()
