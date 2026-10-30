"""Clone-to-figure demo: fly the paper's deployed NN guidance cell over a Monte
Carlo batch and render one figure (DV CDF + flown corridor).

    uv run python -m aerocapture.demo [--n-sims N] [--output PATH] [--finetune | --legacy]

By default this flies the per-scenario density-noise regime (``per_draw``, the
simulator default since ADR-0006) with the paper's deployed seed
``models/demo/hl_mamba_962/`` (trained from scratch under that regime; see its
README for provenance). ``--finetune`` flies the arxiv-v3 per-scenario fine-tune
``models/demo/ft_mamba_962/`` under the same regime; ``--legacy`` reproduces the
historical shared-noise-path regime with the champion that was trained and quoted
under it (``models/demo/mamba_962_legacy/``), the conditioning defect the paper's
Section 6.4 discloses. Runs on an arbitrary fixed seed, deliberately outside the
reserved training/validation/final-eval pools: illustrative output, not paper numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aerocapture.training.cell_eval import evaluate_cell
from aerocapture.training.report import read_cost_kwargs
from aerocapture.training.sizing_tail import score_cell

REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_TOML = REPO_ROOT / "configs/training/sweep/mamba_p962.toml"
DEMO_MODEL_DIR = REPO_ROOT / "models/demo/hl_mamba_962"  # deployed seed (paper v4, scratch under per_draw)
FINETUNE_MODEL_DIR = REPO_ROOT / "models/demo/ft_mamba_962"  # arxiv-v3 per-scenario fine-tune of the shared-path champion
LEGACY_MODEL_DIR = REPO_ROOT / "models/demo/mamba_962_legacy"  # shared-path champion (arxiv-v3 main body)
# Arbitrary constant, NOT drawn from the reserved seed pools in seeds.py.
DEMO_SEED = 424242

# Trajectory matrix columns (see BatchResults docs): energy [8] MJ/kg, pdyn [9] kPa.
TRAJ_ENERGY, TRAJ_PDYN = 8, 9


def select_model(finetune: bool = False, legacy: bool = False) -> tuple[Path, str, str]:
    """The demo cell, its `[monte_carlo] noise_seeding` and the regime label the output carries."""
    if finetune and legacy:
        raise ValueError("--finetune and --legacy are exclusive")
    if legacy:
        return LEGACY_MODEL_DIR, "legacy", "shared density-noise path (legacy, arxiv-v3 main body)"
    if finetune:
        return FINETUNE_MODEL_DIR, "per_draw", "per-scenario density noise (per_draw, ADR-0006; arxiv-v3 fine-tune)"
    return DEMO_MODEL_DIR, "per_draw", "per-scenario density noise (per_draw, ADR-0006)"


def run_demo(n_sims: int, output: Path, finetune: bool = False, legacy: bool = False) -> None:
    # One sim per seed through `evaluate_cell` and the paper's own scorer (`sizing_tail.score_cell`,
    # the estimator every quoted CVaR uses), so the demo's numbers are comparable to the quoted
    # ones; only the pool is illustrative.
    model_dir, mode, regime = select_model(finetune=finetune, legacy=legacy)
    seeds = np.random.default_rng(DEMO_SEED).integers(0, 2**31, size=n_sims)

    print(f"Flying {n_sims} dispersed MSR aerocapture scenarios with the Mamba-962 guidance NN ({model_dir.name})...")
    print(f"Noise regime: {regime}")
    results = evaluate_cell(
        model_dir, DEMO_TOML, seeds, model=model_dir / "best_model.json", extra_overrides={"monte_carlo.noise_seeding": mode}, include_trajectories=True
    )
    captured = results.captured
    dv = results.dv

    rec = score_cell(results, read_cost_kwargs(DEMO_TOML), pool=f"demo rng {DEMO_SEED}")
    n_cap = rec["n_captured"]
    print(f"Captured {n_cap}/{n_sims} ({rec['capture_pct']:.1f}%)")
    if n_cap:
        print(f"Correction DV: p50 {rec['dv_p50']:.1f} m/s, p95 {rec['dv_p95']:.1f} m/s, CVaR95 {rec['dv_cvar95']:.1f} m/s")

    fig, (ax_cdf, ax_corr) = plt.subplots(1, 2, figsize=(12, 4.5))

    if n_cap:
        dv_sorted = np.sort(dv)
        ax_cdf.step(dv_sorted, np.arange(1, n_cap + 1) / n_cap, color="tab:blue", lw=1.5)
    ax_cdf.set_xlabel("correction DV [m/s]")
    ax_cdf.set_ylabel("CDF over captured scenarios")
    ax_cdf.set_title(f"DV cost, n={n_sims}, capture {100.0 * n_cap / n_sims:.1f}%")
    ax_cdf.grid(alpha=0.3)

    assert results.trajectories is not None
    for t, ok in zip(results.trajectories, captured, strict=True):
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
    which = parser.add_mutually_exclusive_group()
    which.add_argument(
        "--finetune",
        action="store_true",
        help="fly the arxiv-v3 per-scenario fine-tune of the shared-path champion (models/demo/ft_mamba_962) instead of the deployed seed",
    )
    which.add_argument(
        "--legacy",
        action="store_true",
        help="fly the historical shared-noise-path regime with the champion trained under it (models/demo/mamba_962_legacy), not the per-scenario default",
    )
    args = parser.parse_args()
    run_demo(args.n_sims, args.output, finetune=args.finetune, legacy=args.legacy)


if __name__ == "__main__":
    main()
