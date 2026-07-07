"""fig_training_n_sims -- Study F (sims/individual allocation).

One half-column panel, sizing tail (CVaR_95, n=1000) -- the mean orders both
series identically, so a single metric carries the figure at half-column width:
- rotating, fixed n_gen: training_n_sims in {2,5,10,20,40} -- the noise floor;
  the sweet spot is ~10 (too few sims = noisy fitness, too many = wasted
  gradient on redundant scenarios).
- adaptive, fixed total compute (n_sims * n_gen = 20000): {2,5,20,100} --
  adaptive_2 DOMINATES (CVaR_95 117.5, mean 109.9): the curated-CDF seed list
  lets a 2-sim-per-individual budget buy far more generations than a
  coverage-heavy 100-sim allocation.
"""

import figlib as fl
import matplotlib.pyplot as plt

# (n_sims, results.json cell key) -- ordered by n_sims.
ROTATING = [
    (2, "training_n_sims/rotating_2"),
    (5, "training_n_sims/rotating_5"),
    (10, "training_n_sims/rotating_10"),
    (20, "training_n_sims/rotating_20"),
    (40, "training_n_sims/rotating_40"),
]
ADAPTIVE = [
    (2, "training_n_sims/adaptive_2"),
    (5, "training_n_sims/adaptive_5"),
    (20, "training_n_sims/adaptive_20"),
    (100, "training_n_sims/adaptive_100"),
]


def main():
    fl.style()
    runs = fl.results()["runs"]

    fig, ax = plt.subplots(figsize=fl.SIZE_HALF)
    series = (
        (ROTATING, "dense", "--s", "rotating, fixed $n_{gen}$"),
        (ADAPTIVE, "mamba", "-o", "adaptive, $n_{sims}{\\times}n_{gen}$ fixed"),
    )
    for pts, ckey, style, label in series:
        n = [p[0] for p in pts]
        cv = [runs[p[1]]["dv_cvar95"] for p in pts]
        ax.plot(n, cv, style, color=fl.C[ckey], lw=1.9, ms=5.5, zorder=3, label=label)

    # annotate the two optima (lowest CVaR_95 per series)
    rot_sweet = runs["training_n_sims/rotating_10"]["dv_cvar95"]
    ada_best = runs["training_n_sims/adaptive_2"]["dv_cvar95"]
    ax.scatter([10], [rot_sweet], s=130, facecolors="none", edgecolors=fl.C["accent"],
               lw=1.8, zorder=4)
    ax.annotate(f"sweet spot\n{rot_sweet:.1f}", (10, rot_sweet), textcoords="offset points",
                xytext=(12, -24), color=fl.C["accent"], fontsize=8, fontweight="bold")
    ax.scatter([2], [ada_best], s=130, facecolors="none", edgecolors=fl.C["accent"],
               lw=1.8, zorder=4)
    ax.annotate(f"dominates\n{ada_best:.1f}", (2, ada_best), textcoords="offset points",
                xytext=(8, -22), color=fl.C["accent"], fontsize=8, fontweight="bold")

    ax.set_xscale("log")
    ticks = sorted({p[0] for p in ROTATING + ADAPTIVE})
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(v) for v in ticks])
    ax.set_xlabel("training sims / individual")
    ax.set_ylabel("CVaR$_{95}$ (m/s)")
    ax.set_title("Allocation: scenarios vs generations")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.margins(x=0.1, y=0.16)
    fig.tight_layout()
    fl.save(fig, "fig_training_n_sims")


if __name__ == "__main__":
    main()
