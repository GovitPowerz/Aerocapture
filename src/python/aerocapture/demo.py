"""Clone-to-figure demo: fly the paper's deployed NN guidance cell over a Monte
Carlo batch and render one figure (DV CDF + flown corridor).

    uv run python -m aerocapture.demo [--n-sims N] [--output PATH] [--legacy]

By default this flies the per-scenario density-noise regime (``per_draw``, the
simulator default since ADR-0006) with the committed per-scenario fine-tune
``models/demo/ft_mamba_962/`` (the paper's Appendix E champion; see its README
for provenance). ``--legacy`` reproduces the historical shared-noise-path regime
with the champion that was trained and quoted under it
(``models/demo/mamba_962_legacy/``), the conditioning defect Appendix E
discloses. Runs on an arbitrary fixed seed, deliberately outside the reserved
training/validation/final-eval pools: illustrative output, not paper numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import aerocapture_rs
import matplotlib.pyplot as plt
import numpy as np

from aerocapture.training.deploy_overrides import load_scaffolding_overrides

REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_TOML = REPO_ROOT / "configs/training/sweep/mamba_p962.toml"
DEMO_MODEL_DIR = REPO_ROOT / "models/demo/ft_mamba_962"  # per-scenario fine-tune (Appendix E champion)
LEGACY_MODEL_DIR = REPO_ROOT / "models/demo/mamba_962_legacy"  # shared-path champion (historical headline)
# Arbitrary constant, NOT drawn from the reserved seed pools in seeds.py.
DEMO_SEED = 424242

# Trajectory matrix columns (see BatchResults docs): energy [8] MJ/kg, pdyn [9] kPa.
TRAJ_ENERGY, TRAJ_PDYN = 8, 9


def run_demo(n_sims: int, output: Path, legacy: bool = False) -> None:
    # One sim per seed, matching the paper's evaluation methodology (run_batch
    # per-seed, as in report.py / fresh_pool_requote.py) so the demo's numbers
    # are comparable to the quoted ones.
    model_dir, mode, regime = (
        (LEGACY_MODEL_DIR, "legacy", "shared density-noise path (legacy, historical headline)")
        if legacy
        else (DEMO_MODEL_DIR, "per_draw", "per-scenario density noise (per_draw, ADR-0006)")
    )
    base: dict[str, object] = {
        "data.neural_network": str(model_dir / "best_model.json"),
        "simulation.n_sims": 1,
        "monte_carlo.noise_seeding": mode,
    }
    base.update(load_scaffolding_overrides(model_dir))
    seeds = np.random.default_rng(DEMO_SEED).integers(0, 2**31, size=n_sims)

    print(f"Flying {n_sims} dispersed MSR aerocapture scenarios with the Mamba-962 guidance NN ({model_dir.name})...")
    print(f"Noise regime: {regime}")
    results = aerocapture_rs.run_batch(
        str(DEMO_TOML),
        overrides_list=[{**base, "monte_carlo.seed": int(s)} for s in seeds],
        include_trajectories=True,
    )

    idx = aerocapture_rs.final_record_indices()
    fr = np.asarray(results.final_records)
    captured = (fr[:, idx["ifinal"]] == 3) & (fr[:, idx["ecc"]] < 1.0)
    dv = fr[captured, idx["dv_total_ms"]]

    n_cap = int(captured.sum())
    print(f"Captured {n_cap}/{n_sims} ({100.0 * n_cap / n_sims:.1f}%)")
    if n_cap:
        p50, p95 = np.percentile(dv, [50, 95])
        cvar95 = dv[dv >= p95].mean()
        print(f"Correction DV: p50 {p50:.1f} m/s, p95 {p95:.1f} m/s, CVaR95 {cvar95:.1f} m/s")

    fig, (ax_cdf, ax_corr) = plt.subplots(1, 2, figsize=(12, 4.5))

    if n_cap:
        dv_sorted = np.sort(dv)
        ax_cdf.step(dv_sorted, np.arange(1, n_cap + 1) / n_cap, color="tab:blue", lw=1.5)
    ax_cdf.set_xlabel("correction DV [m/s]")
    ax_cdf.set_ylabel("CDF over captured scenarios")
    ax_cdf.set_title(f"DV cost, n={n_sims}, capture {100.0 * n_cap / n_sims:.1f}%")
    ax_cdf.grid(alpha=0.3)

    for traj, ok in zip(results.trajectories, captured, strict=True):
        t = np.asarray(traj)
        if t.size == 0:
            continue
        color, alpha = ("tab:blue", 0.08) if ok else ("tab:red", 0.5)
        ax_corr.plot(t[:, TRAJ_ENERGY], t[:, TRAJ_PDYN], color=color, alpha=alpha, lw=0.6)
    ax_corr.set_xlabel("orbital energy [MJ/kg]")
    ax_corr.set_ylabel("dynamic pressure [kPa]")
    ax_corr.set_title("Flown corridor (blue = captured, red = failed)")
    ax_corr.grid(alpha=0.3)

    fig.suptitle(f"Aerocapture demo: Mamba-962 NN guidance ({model_dir.name}), dispersed Mars Sample Return entry ({regime})")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(f"Figure written to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sims", type=int, default=500)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "demo_output/demo.svg")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="fly the historical shared-noise-path regime with the champion trained under it (models/demo/mamba_962_legacy), not the per-scenario default",
    )
    args = parser.parse_args()
    run_demo(args.n_sims, args.output, legacy=args.legacy)


if __name__ == "__main__":
    main()
