"""fig_seed_strategy -- Study C, the load-bearing methodology result.

Grouped bars per optimizer (GA, islands, PSO, CMA-ES), FIXED vs ROTATING training
seeds, on the mission-SIZING tail (dv_cvar95 primary, dv_mean annotated). THE story:
GA is the WORST optimizer under fixed seeds (cvar95 215.8) and is RESCUED by rotating
seeds (144.5, -71 m/s on the tail); CMA-ES is essentially FLAT (it already resamples
internally). So GA NEEDS a non-stationary objective -- the adaptive/rotating seed
methodology is the contribution, not the optimizer choice.
"""

import figlib as fl
import matplotlib.pyplot as plt
import numpy as np

# (display label, results.json cell stem) ordered by fixed->rotating tail gain (largest first).
OPTS = [
    ("GA", "ga"),
    ("islands", "islands"),
    ("PSO", "pso"),
    ("CMA-ES", "cmaes"),
]
FIXED = "#c44e52"     # fixed seeds -- red (the trap)
ROTATING = "#1f6f3f"  # rotating seeds -- green (the rescue)


def main():
    fl.style()
    runs = fl.results()["runs"]

    fixed_cvar = np.array([runs[f"seed_strategy/{s}_fixed"]["dv_cvar95"] for _, s in OPTS])
    rot_cvar = np.array([runs[f"seed_strategy/{s}_rotating"]["dv_cvar95"] for _, s in OPTS])
    fixed_mean = np.array([runs[f"seed_strategy/{s}_fixed"]["dv_mean"] for _, s in OPTS])
    rot_mean = np.array([runs[f"seed_strategy/{s}_rotating"]["dv_mean"] for _, s in OPTS])

    x = np.arange(len(OPTS))
    w = 0.38

    fig, ax = plt.subplots(figsize=fl.SIZE1)
    bf = ax.bar(x - w / 2, fixed_cvar, w, color=FIXED, label="fixed seeds", zorder=2)
    br = ax.bar(x + w / 2, rot_cvar, w, color=ROTATING, label="rotating seeds", zorder=2,
                hatch="//", edgecolor="white", linewidth=0.4)  # hatched so the pair is not encoded by red/green alone

    # dv_mean tick markers on each bar (secondary metric)
    for xi, m in zip(x - w / 2, fixed_mean, strict=True):
        ax.plot([xi - w / 2, xi + w / 2], [m, m], color="#333333", lw=1.4, zorder=4)
    for xi, m in zip(x + w / 2, rot_mean, strict=True):
        ax.plot([xi - w / 2, xi + w / 2], [m, m], color="#333333", lw=1.4, zorder=4)

    # bar value labels (cvar95)
    for rects in (bf, br):
        for r in rects:
            ax.annotate(f"{r.get_height():.0f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                        ha="center", va="bottom", fontsize=7.5, xytext=(0, 1), textcoords="offset points")

    # visual focus: the GA fixed->rotating tail collapse
    ga_xf, ga_xr = x[0] - w / 2, x[0] + w / 2
    ax.annotate("", xy=(ga_xr, rot_cvar[0] + 6), xytext=(ga_xf, fixed_cvar[0] + 6),
                arrowprops={"arrowstyle": "->", "color": "#b8860b", "lw": 2.2,
                            "connectionstyle": "arc3,rad=-0.35"}, zorder=6)
    drop = fixed_cvar[0] - rot_cvar[0]
    ax.text(x[0], fixed_cvar[0] + 14, f"rotating rescues GA\n-{drop:.0f} m/s on the tail",
            color="#b8860b", fontsize=8.5, ha="center", va="bottom", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([o[0] for o in OPTS])
    ax.set_ylabel("CVaR$_{95}$ correction $\\Delta v$ (m/s)")
    ax.set_ylim(0, fixed_cvar.max() + 36)
    ax.set_title("Study C: GA needs non-stationary seeds; CMA-ES is flat "
                 "(bars = CVaR$_{95}$, black tick = mean)", fontsize=9.5, loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fl.save(fig, "fig_seed_strategy")


if __name__ == "__main__":
    main()
