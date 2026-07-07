"""fig_training_n_sims -- Study F (sims/individual allocation).

Two panels, both leading the mission-SIZING TAIL (CVaR_95, mean secondary).
(A) Rotating-seed noise floor: training_n_sims in {2,5,10,20,40} at a fixed
    per-gen budget -- the sweet spot is ~10 (too few sims = noisy fitness,
    too many = wasted gradient on redundant scenarios).
(B) Adaptive allocation under a fixed total compute (n_sims * n_gen = 20000):
    {2,5,20,100} -- adaptive_2 DOMINATES (CVaR_95 117.5, mean 109.9): the
    curated-CDF seed list lets a 2-sim-per-individual budget buy far more
    generations than a coverage-heavy 100-sim allocation.
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


def _panel(ax, runs, points, title, sweet_idx, sweet_note):
    n = [p[0] for p in points]
    cvar95 = [runs[p[1]]["dv_cvar95"] for p in points]
    mean = [runs[p[1]]["dv_mean"] for p in points]

    ax.plot(n, cvar95, "-o", color=fl.C["mamba"], lw=2.0, ms=6, zorder=3, label="CVaR$_{95}$")
    ax.plot(n, mean, "--s", color=fl.C["dense"], lw=1.4, ms=4.5, zorder=2, alpha=0.85, label="mean")

    # mark + annotate the best (lowest CVaR_95) point
    sx, sy = n[sweet_idx], cvar95[sweet_idx]
    ax.scatter([sx], [sy], s=130, facecolors="none", edgecolors=fl.C["jointftc"], lw=2.0, zorder=4)
    ax.annotate(sweet_note, (sx, sy), textcoords="offset points", xytext=(8, 10),
                color=fl.C["jointftc"], fontsize=8, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([str(v) for v in n])
    ax.set_xlabel("training sims / individual")
    ax.set_title(title, fontsize=10, loc="left")
    ax.margins(x=0.12)


def main():
    fl.style()
    runs = fl.results()["runs"]

    fig, axes = plt.subplots(1, 2, figsize=fl.SIZE_HALF)

    _panel(axes[0], runs, ROTATING, "(A) Rotating noise floor", sweet_idx=2, sweet_note="sweet spot\n133.5 m/s")
    _panel(axes[1], runs, ADAPTIVE, "(B) Adaptive, $n_{sims}\\!\\times\\!n_{gen}=20000$", sweet_idx=0,
           sweet_note="dominates\n117.5 m/s")

    axes[0].set_ylabel("correction $\\Delta v$ (m/s)")
    axes[0].legend(loc="upper center")
    fig.tight_layout()
    fl.save(fig, "fig_training_n_sims")


if __name__ == "__main__":
    main()
