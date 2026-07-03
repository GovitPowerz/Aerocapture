"""fig_cost_transform -- Study D: cost-transform sweep on the sizing tail.

Grouped bars over the five monotonic cost transforms (linear, sqrt, log, squared,
cubed) on the mission-SIZING TAIL: dv_cvar95 (primary) and dv_p99 (secondary).
The transform only rescales the per-sim cost the optimizer minimizes (argmin is
preserved under monotonicity), but the tail-weight changes which captures separate
under rank-free PSO/GA. Ordered by tail-weight linear<sqrt<log<squared<cubed;
cubed (= optimizer_budget/ga_300) is the DEPLOYED choice -- it wins the FAR tail
(the n=10000 CVaR99.9 decision), not visible at this CVaR95 resolution.
"""

import figlib as fl
import matplotlib.pyplot as plt
import numpy as np

# (display label, runs-key) ordered by tail-weight: linear < sqrt < log < squared < cubed.
TRANSFORMS = [
    ("linear", "cost_transform/linear"),
    ("sqrt", "cost_transform/sqrt"),
    ("log", "cost_transform/log"),
    ("squared", "cost_transform/squared"),
    ("cubed", "optimizer_budget/ga_300"),  # deployed choice -- best far tail
]


def main():
    fl.style()
    runs = fl.results()["runs"]

    labels = [t[0] for t in TRANSFORMS]
    cvar95 = np.array([runs[t[1]]["dv_cvar95"] for t in TRANSFORMS])
    p99 = np.array([runs[t[1]]["dv_p99"] for t in TRANSFORMS])

    x = np.arange(len(TRANSFORMS))
    w = 0.38

    fig, ax = plt.subplots(figsize=fl.SIZE_HALF)
    # cubed (deployed) gets the headline color; the others muted.
    cvar_colors = [fl.C["dense"]] * (len(TRANSFORMS) - 1) + [fl.C["mamba"]]
    p99_colors = [fl.C["gru"]] * (len(TRANSFORMS) - 1) + ["#2e8b57"]

    ax.bar(x - w / 2, cvar95, w, color=cvar_colors, label="CVaR$_{95}$", zorder=3)
    ax.bar(x + w / 2, p99, w, color=p99_colors, alpha=0.65, label="p99", zorder=3)

    for xi, v in zip(x - w / 2, cvar95, strict=False):
        ax.annotate(f"{v:.1f}", (xi, v), ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    for xi, v in zip(x + w / 2, p99, strict=False):
        ax.annotate(f"{v:.1f}", (xi, v), ha="center", va="bottom", fontsize=7.5, color="#555555")

    # mark the deployed choice
    ax.annotate("deployed", (x[-1], max(cvar95[-1], p99[-1]) + 3.2), ha="center", va="bottom",
                fontsize=8, color=fl.C["mamba"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("cost transform (increasing tail-weight $\\rightarrow$)")
    ax.set_ylabel("correction $\\Delta v$ (m/s)")
    ax.set_ylim(0, max(p99.max(), cvar95.max()) + 12)
    ax.set_title("Sizing tail vs cost transform (NN weights, n=1000)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fl.save(fig, "fig_cost_transform")


if __name__ == "__main__":
    main()
