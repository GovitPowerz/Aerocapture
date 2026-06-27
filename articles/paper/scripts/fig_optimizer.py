"""fig_optimizer -- Study A: optimizer x budget at dense_p3998.

Grouped bars: x = eval budget {60, 150, 300} generations, one bar per optimizer
{GA, islands, PSO, DE, CMA-ES, QPSO}, y = sizing-tail CVaR_95 (m/s). THE story:
GA@60 COLLAPSES (CVaR95 205.6 m/s, mean 166.3) while GA@150/300 are the best
overall (130.9 / 137.6), and islands is budget-robust (145.6 / 136.4 / 144.8).
A broken y-axis makes GA@60's collapse the visual focus.
"""

import figlib as fl
import matplotlib.pyplot as plt
import numpy as np

# Optimizer order (display label, color, cell prefix).
OPTS = [
    ("GA", fl.C["mamba"], "ga"),
    ("islands", fl.C["dense"], "islands"),
    ("PSO", fl.C["lstm"], "pso"),
    ("DE", fl.C["jointftc"], "de"),
    ("CMA-ES", fl.C["fnpag"], "cmaes"),
    ("QPSO", fl.C["gru"], "qpso"),
]
BUDGETS = [60, 150, 300]
METRIC = "dv_cvar95"

# Broken-axis split: lower panel holds the bulk, upper panel just the GA@60 spike.
LOW_LO, LOW_HI = 125.0, 175.0
UP_LO, UP_HI = 200.0, 212.0


def main():
    fl.style()
    runs = fl.results()["runs"]

    n_opt = len(OPTS)
    width = 0.8 / n_opt
    x = np.arange(len(BUDGETS))

    fig, (ax_t, ax_b) = plt.subplots(
        2, 1, figsize=(8.0, 4.0), sharex=True,
        gridspec_kw={"height_ratios": [1, 4], "hspace": 0.08},
    )

    for i, (label, color, prefix) in enumerate(OPTS):
        vals = np.array([runs[f"optimizer_budget/{prefix}_{b}"][METRIC] for b in BUDGETS])
        offs = x + (i - (n_opt - 1) / 2) * width
        for ax in (ax_t, ax_b):
            ax.bar(offs, vals, width, color=color, label=label if ax is ax_b else None,
                   edgecolor="white", linewidth=0.4, zorder=3)
        # value labels on the lower panel (clipped bars get labeled in the upper panel)
        for xo, v in zip(offs, vals, strict=True):
            tgt = ax_t if v > LOW_HI else ax_b
            tgt.annotate(f"{v:.0f}", (xo, min(v, UP_HI if tgt is ax_t else LOW_HI) + 0.4),
                         ha="center", va="bottom", fontsize=6.0, color=color, rotation=90)

    # broken-axis limits
    ax_t.set_ylim(UP_LO, UP_HI)
    ax_b.set_ylim(LOW_LO, LOW_HI)
    ax_t.spines["bottom"].set_visible(False)
    ax_b.spines["top"].set_visible(False)
    ax_t.tick_params(bottom=False)
    ax_t.set_yticks([205])

    # diagonal break marks
    d = 0.008
    kw = dict(transform=ax_t.transAxes, color="0.4", clip_on=False, lw=0.9)
    ax_t.plot((-d, +d), (-d * 4, +d * 4), **kw)
    ax_t.plot((1 - d, 1 + d), (-d * 4, +d * 4), **kw)
    kw["transform"] = ax_b.transAxes
    ax_b.plot((-d, +d), (1 - d, 1 + d), **kw)
    ax_b.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)

    # GA@60 collapse callout
    ga60_x = x[0] + (0 - (n_opt - 1) / 2) * width
    ax_t.annotate(
        "GA@60 collapses\n(206 vs 131 @150)",
        xy=(ga60_x, 205.6), xytext=(0.7, 209),
        fontsize=7.5, color=fl.C["mamba"], ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=fl.C["mamba"], lw=1.0),
    )

    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f"{b} gens" for b in BUDGETS])
    ax_b.set_xlabel("evaluation budget")
    fig.supylabel("sizing tail  CVaR$_{95}$  (m/s)", fontsize=10, x=0.02)
    ax_t.set_title("Optimizer $\\times$ budget at dense-3998 (sizing-tail CVaR$_{95}$)",
                   fontsize=10, loc="left")
    ax_b.legend(ncol=6, fontsize=7.5, loc="upper center", frameon=True,
                columnspacing=1.0, handlelength=1.2)

    fig.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.12)
    fl.save(fig, "fig_optimizer")


if __name__ == "__main__":
    main()
