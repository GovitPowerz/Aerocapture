"""Collect the reachable-corridor data (articles/paper/data/corridor.npz).

Traces the aerocapture capture corridor in the (orbital energy, dynamic pressure)
plane from a large DISPERSED Monte-Carlo of RANDOMIZED piecewise-constant bank
profiles (positive / in-plane, no roll reversals). Per energy bin: the upper
boundary is the p99.5 of pdyn among CAPTURING trajectories (crash-side limit),
the lower boundary the p0.5 among captures below --apoapsis-max-km (escape-side
limit); both Gaussian-smoothed. Overlays the deployed Mamba ensemble + nominal.

Collector-vs-figure split: reads training_output/ + configs; the committed
corridor.npz is the durable artifact.

Usage:
    uv run python articles/paper/scripts/collect_corridor.py \
        [--n-sims 300000] [--n-segments 10] [--apoapsis-max-km 5000] \
        [--n-energy-bins 200] [--n-pdyn-buckets 400] [--chunk-size 20000] \
        [--upper-pct 99.5] [--lower-pct 0.5] [--smooth-sigma 2.5] [--ensemble-sims 200]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src/python"))

BASE_TOML = REPO / "configs/training/msr_aller_piecewise_constant_train.toml"
MAMBA_RUN = REPO / "training_output/mamba_p962_long"
MAMBA_TOML = REPO / "configs/training/sweep/mamba_p962.toml"
MAMBA_BUNDLE = REPO / "articles/paper/data/runs/headline/mamba_p962"
OUT = REPO / "articles/paper/data/corridor.npz"

CORRIDOR_SEED_OFFSET = 10_000_000   # env-dispersion seeds, disjoint from reserved pools (<= 9M)
CORRIDOR_BANK_SEED = 20260706       # fixed -> reproducible bank draws

FR_ECC, FR_APO, FR_IFINAL = 9, 15, 31   # final record (52,)
TC_ENERGY, TC_PDYN = 8, 9               # trajectory (N, 17)

E_LO, E_HI = -6.0, 5.0
PDYN_MAX = 2.8   # kPa top histogram edge (captures do not dive past this; deep divers crash)
DOWNSAMPLE = 3   # ensemble trajectory point stride


def _percentile_from_hist(hist, p_centers, pct):
    """Per energy bin (row): pdyn at cumulative fraction pct/100; NaN if the bin is empty."""
    out = np.full(hist.shape[0], np.nan)
    frac = pct / 100.0
    for e in range(hist.shape[0]):
        tot = hist[e].sum()
        if tot == 0:
            continue
        cum = np.cumsum(hist[e]) / tot
        out[e] = p_centers[min(int(np.searchsorted(cum, frac)), len(p_centers) - 1)]
    return out


def _smooth(e_centers, y, sigma):
    from scipy.ndimage import gaussian_filter1d

    valid = ~np.isnan(y)
    if valid.sum() < 2:
        return y
    y = y.copy()
    y[~valid] = np.interp(e_centers[~valid], e_centers[valid], y[valid])
    return gaussian_filter1d(y, sigma=sigma)


def build_corridor(args):
    import aerocapture_rs

    e_edges = np.linspace(E_LO, E_HI, args.n_energy_bins + 1)
    p_edges = np.linspace(0.0, PDYN_MAX, args.n_pdyn_buckets + 1)
    e_centers = (e_edges[:-1] + e_edges[1:]) / 2
    p_centers = (p_edges[:-1] + p_edges[1:]) / 2
    hist_up = np.zeros((args.n_energy_bins, args.n_pdyn_buckets), dtype=np.int64)
    hist_lo = np.zeros((args.n_energy_bins, args.n_pdyn_buckets), dtype=np.int64)
    rng = np.random.default_rng(CORRIDOR_BANK_SEED)

    done = n_cap = 0
    while done < args.n_sims:
        m = min(args.chunk_size, args.n_sims - done)
        banks = rng.uniform(0.0, 180.0, size=(m, args.n_segments))
        ov = []
        for j in range(m):
            d = {"simulation.n_sims": 1,
                 "monte_carlo.seed": CORRIDOR_SEED_OFFSET + done + j,
                 "guidance.piecewise_constant.n_segments": args.n_segments}
            for i in range(args.n_segments):
                d[f"guidance.piecewise_constant.bank_angle_{i}"] = float(banks[j, i])
            ov.append(d)
        batch = aerocapture_rs.run_batch(toml_path=str(BASE_TOML.resolve()), overrides_list=ov,
                                         include_trajectories=True, sim_timeout_secs=5.0)
        recs = np.asarray(batch.final_records)
        cap = (recs[:, FR_IFINAL] == 3) & (recs[:, FR_ECC] < 1.0)
        apo = recs[:, FR_APO]
        n_cap += int(cap.sum())
        for j in np.nonzero(cap)[0]:
            t = np.asarray(batch.trajectories[j])
            ei = np.clip(np.digitize(t[:, TC_ENERGY], e_edges) - 1, 0, args.n_energy_bins - 1)
            pi = np.clip(np.digitize(t[:, TC_PDYN], p_edges) - 1, 0, args.n_pdyn_buckets - 1)
            np.add.at(hist_up, (ei, pi), 1)
            if apo[j] < args.apoapsis_max_km:
                np.add.at(hist_lo, (ei, pi), 1)
        done += m
        del batch
        print(f"  {done}/{args.n_sims} sims, capture {100 * n_cap / max(done, 1):.1f}%", end="\r")
    print()
    upper = _smooth(e_centers, _percentile_from_hist(hist_up, p_centers, args.upper_pct), args.smooth_sigma)
    lower = _smooth(e_centers, _percentile_from_hist(hist_lo, p_centers, args.lower_pct), args.smooth_sigma)
    pop = int((hist_up.sum(axis=1) > 0).sum())
    print(f"corridor: {n_cap} captures, {pop}/{args.n_energy_bins} energy bins populated")
    return e_centers, lower, upper, n_cap


def build_overlay(n_ens):
    """Deployed Mamba dispersed ensemble (energy, pdyn per trajectory) + undispersed nominal."""
    import aerocapture_rs
    from aerocapture.training.evaluate import FINAL_EVAL_SEED_OFFSET, make_reserved_seeds
    from aerocapture.training.reference import _MC_DISPERSION_DOMAINS
    from aerocapture.training.report import _resolve_eval_toml
    from aerocapture.training.toml_utils import load_toml_with_bases

    eval_toml, scaffolding = _resolve_eval_toml(MAMBA_TOML, MAMBA_RUN)
    pin = dict(scaffolding)
    bundle_model = MAMBA_BUNDLE / "best_model.json"
    if bundle_model.exists():
        pin["data.neural_network"] = str(bundle_model.resolve())
    base_seed = load_toml_with_bases(eval_toml).get("monte_carlo", {}).get("seed", 42)
    seeds = make_reserved_seeds(base_seed, FINAL_EVAL_SEED_OFFSET, n_ens)
    ov = [{"simulation.n_sims": 1, "monte_carlo.seed": s, **pin} for s in seeds]
    batch = aerocapture_rs.run_batch(toml_path=str(eval_toml.resolve()), overrides_list=ov,
                                     include_trajectories=True, sim_timeout_secs=5.0)
    ens_e, ens_p = [], []
    for t in batch.trajectories:
        a = np.asarray(t)[::DOWNSAMPLE]
        ens_e.append(a[:, TC_ENERGY])
        ens_p.append(a[:, TC_PDYN])
    nom_ov = {"simulation.n_sims": 1,
              **{f"monte_carlo.{dom}.level": "off" for dom in _MC_DISPERSION_DOMAINS}, **pin}
    nom = aerocapture_rs.run_mc(toml_path=str(eval_toml.resolve()), overrides=nom_ov,
                                include_trajectories=True, sim_timeout_secs=5.0)
    nt = np.asarray(nom.trajectories[0])
    return (np.array(ens_e, dtype=object), np.array(ens_p, dtype=object),
            nt[:, TC_ENERGY], nt[:, TC_PDYN])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-sims", type=int, default=300_000)
    p.add_argument("--n-segments", type=int, default=10)
    p.add_argument("--apoapsis-max-km", type=float, default=5000.0)
    p.add_argument("--n-energy-bins", type=int, default=200)
    p.add_argument("--n-pdyn-buckets", type=int, default=400)
    p.add_argument("--chunk-size", type=int, default=20_000)
    p.add_argument("--upper-pct", type=float, default=99.5)
    p.add_argument("--lower-pct", type=float, default=0.5)
    p.add_argument("--smooth-sigma", type=float, default=2.5)
    p.add_argument("--ensemble-sims", type=int, default=200)
    args = p.parse_args()

    e_centers, lower, upper, n_cap = build_corridor(args)
    ens_e, ens_p, nom_e, nom_p = build_overlay(args.ensemble_sims)
    np.savez_compressed(
        OUT, energy_bins=e_centers, lower_pdyn=lower, upper_pdyn=upper,
        nominal_energy=nom_e, nominal_pdyn=nom_p, ens_energy=ens_e, ens_pdyn=ens_p,
        n_sims=args.n_sims, n_segments=args.n_segments, apoapsis_max_km=args.apoapsis_max_km,
        upper_pct=args.upper_pct, lower_pct=args.lower_pct)
    print(f"wrote {OUT.relative_to(REPO)}: {n_cap} captures, ensemble {len(ens_e)}, nominal {len(nom_e)} pts")


if __name__ == "__main__":
    main()
