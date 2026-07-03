"""Collect the corridor-figure data bundle (articles/paper/data/corridor.npz).

Report-grade corridor data for fig_corridor: a Monte-Carlo ensemble of the
DEPLOYED headline policy (Mamba_962, pinned run-local best_model.json +
co-trained scaffolding) on the reserved FINAL_EVAL (2M) pool with full
trajectories, classified three-way (captured-OK / constraint-violation /
failed) exactly like report.py's corridor panels; the policy's own
undispersed nominal; and the 4-layer corridor zone envelopes from the
mission cache (training_output/mars/corridor_boundaries.npz).

Usage:
    uv run python articles/paper/scripts/collect_corridor.py [--n-sims 200]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src/python"))

RUN_DIR = REPO / "training_output/mamba_p962_long"
TOML = REPO / "configs/training/sweep/mamba_p962.toml"
CORRIDOR_CACHE = REPO / "training_output/mars/corridor_boundaries.npz"
OUT = REPO / "articles/paper/data/corridor.npz"
DOWNSAMPLE = 3  # keep every 3rd trajectory point (plus the last)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sims", type=int, default=200)
    args = parser.parse_args()

    import aerocapture_rs
    from aerocapture.training import charts
    from aerocapture.training.evaluate import FINAL_EVAL_SEED_OFFSET, make_reserved_seeds
    from aerocapture.training.reference import nominal_flight_overrides
    from aerocapture.training.report import _resolve_eval_toml
    from aerocapture.training.toml_utils import load_toml_with_bases

    eval_toml, scaffolding = _resolve_eval_toml(TOML, RUN_DIR)
    cfg = load_toml_with_bases(eval_toml)
    base_mc_seed = cfg.get("monte_carlo", {}).get("seed", 42)
    limits = cfg["flight"]["constraints"]

    pin = dict(scaffolding)
    local_model = RUN_DIR / "best_model.json"
    if local_model.exists():
        pin["data.neural_network"] = str(local_model.resolve())

    # MC ensemble on the final-eval pool, one sim per seed, with trajectories.
    seeds = make_reserved_seeds(base_mc_seed, FINAL_EVAL_SEED_OFFSET, args.n_sims)
    overrides = [{"simulation.n_sims": 1, "monte_carlo.seed": s, **pin} for s in seeds]
    batch = aerocapture_rs.run_batch(
        toml_path=str(eval_toml.resolve()), overrides_list=overrides,
        include_trajectories=True, sim_timeout_secs=5.0,
    )
    recs = np.asarray(batch.final_records)
    traj_class = charts.classify_trajectories(
        recs,
        heat_flux_limit=limits["max_heat_flux"],
        g_load_limit=limits["max_load_factor"],
        heat_load_limit=limits["max_heat_load"],
    )

    def downsample(traj: np.ndarray) -> np.ndarray:
        idx = np.r_[np.arange(0, len(traj) - 1, DOWNSAMPLE), len(traj) - 1]
        return traj[idx]

    energy, pdyn = [], []
    for i in range(len(batch.trajectories)):
        t = downsample(np.asarray(batch.trajectories[i]))
        energy.append(t[:, charts._TC_ENERGY])
        pdyn.append(t[:, charts._TC_PDYN])

    # The deployed policy's undispersed nominal (all dispersion domains off).
    nom_ov = nominal_flight_overrides({}, "neural_network", cfg.get("monte_carlo", {}))
    nom = aerocapture_rs.run_batch(
        toml_path=str(eval_toml.resolve()), overrides_list=[{**nom_ov, **pin}],
        include_trajectories=True, sim_timeout_secs=5.0,
    )
    nom_t = np.asarray(nom.trajectories[0])

    zones = np.load(CORRIDOR_CACHE)
    np.savez_compressed(
        OUT,
        energy=np.array(energy, dtype=object),
        pdyn=np.array(pdyn, dtype=object),
        traj_class=traj_class,
        nominal_energy=nom_t[:, charts._TC_ENERGY],
        nominal_pdyn=nom_t[:, charts._TC_PDYN],
        energy_bins=zones["energy_bins"],
        envelope_crash_pdyn=zones["envelope_crash_pdyn"],
        envelope_restricted_max_pdyn=zones["envelope_restricted_max_pdyn"],
        envelope_restricted_min_pdyn=zones["envelope_restricted_min_pdyn"],
        envelope_capture_pdyn=zones["envelope_capture_pdyn"],
    )
    n_ok = int((traj_class == charts.TRAJ_OK).sum())
    n_con = int((traj_class == charts.TRAJ_CONSTRAINED).sum())
    n_fail = int((traj_class == charts.TRAJ_FAILED).sum())
    print(f"wrote {OUT.relative_to(REPO)}: n={len(energy)} (ok {n_ok} / constrained {n_con} / failed {n_fail}), nominal {len(nom_t)} pts")


if __name__ == "__main__":
    main()
