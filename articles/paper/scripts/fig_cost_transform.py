"""fig_cost_transform -- Study D: cost-transform sweep on the sizing tail.

Grouped bars over the five monotonic cost transforms (linear, sqrt, log, squared,
cubed) at the depth the mission is sized on: the far-tail n=10000 pool, CVaR99.9
(primary) and the sample maximum (secondary). The transform only rescales the
per-sim cost the optimizer minimizes (argmin is preserved under monotonicity),
but the tail-weight changes which captures separate under rank-free PSO/GA.
Ordered by tail-weight linear<sqrt<log<squared<cubed; cubed
(= optimizer_budget/ga_300) is the DEPLOYED choice and wins BOTH far-tail
statistics -- a shallow CVaR95 read would have mildly favored sqrt, which is
exactly why the sizing depth must decide (paper section 4.2).
"""

import figlib as fl
import matplotlib.pyplot as plt
import numpy as np

# (display label, far_tail_eval cell label) ordered by tail-weight.
TRANSFORMS = [
    ("linear", "cost_transform/linear"),
    ("sqrt", "cost_transform/sqrt"),
    ("log", "cost_transform/log"),
    ("squared", "cost_transform/squared"),
    ("cubed", "optimizer_budget/ga_300"),  # deployed choice -- best far tail
]


def main():
    fl.style()
    ft = fl.far_tail()

    labels = [t[0] for t in TRANSFORMS]
    cvar999 = np.array([ft[t[1]]["cvar999"] for t in TRANSFORMS])
    vmax = np.array([ft[t[1]]["max"] for t in TRANSFORMS])

    x = np.arange(len(TRANSFORMS))
    w = 0.38

    fig, ax = plt.subplots(figsize=fl.SIZE_HALF)
    # cubed (deployed) gets the headline color; the others muted.
    cvar_colors = [fl.C["dense"]] * (len(TRANSFORMS) - 1) + [fl.C["mamba"]]
    max_colors = [fl.C["gru"]] * (len(TRANSFORMS) - 1) + ["#2e8b57"]

    ax.bar(x - w / 2, cvar999, w, color=cvar_colors, label="CVaR$_{99.9}$", zorder=3)
    ax.bar(x + w / 2, vmax, w, color=max_colors, alpha=0.65, label="max", zorder=3)

    for xi, v in zip(x - w / 2, cvar999, strict=True):
        ax.annotate(f"{v:.0f}", (xi, v), ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    for xi, v in zip(x + w / 2, vmax, strict=True):
        ax.annotate(f"{v:.0f}", (xi, v), ha="center", va="bottom", fontsize=7.5, color="#555555")

    # mark the deployed choice
    ax.annotate("deployed", (x[-1], max(cvar999[-1], vmax[-1]) + 9), ha="center", va="bottom",
                fontsize=8, color=fl.C["mamba"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("cost transform (increasing tail-weight $\\rightarrow$)")
    ax.set_ylabel("correction $\\Delta v$ (m/s)")
    ax.set_ylim(0, vmax.max() + 34)
    ax.set_title("Far tail vs cost transform (n=10000)")
    ax.legend(loc="upper left", fontsize=7.5, ncols=2)
    fig.tight_layout()
    fl.save(fig, "fig_cost_transform")


if __name__ == "__main__":
    main()
