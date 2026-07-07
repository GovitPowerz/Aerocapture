"""fig_robustness -- off-nominal stress test (study 5c).

The honest caveat figure. Under a 1000-sim off-nominal MC (degraded nav + heavier
dispersions), every scheme loses capture and inflates its sizing tail relative to
the nominal pool. The point of the figure: the NN-mamba headline is NOT the most
robust -- joint-FTC degrades LESS on both axes (capture drop 5.5 vs 9.9 pts,
CVaR95 inflation +197 vs +402 m/s). FTC-fixed (the un-retuned classical) collapses
(33 pts capture lost), confirming the off-nominal pool genuinely bites.

Panel A: capture drop (pts), lower = more robust. Panel B: CVaR95 inflation (m/s),
lower = more robust.
"""

import figlib as fl
import matplotlib.pyplot as plt

# Display label, robustness-data label, color key. Ordered most -> least robust
# by capture drop so joint-FTC (the robustness winner) leads and NN sits behind it.
SCHEMES = [
    ("joint-FTC", "joint-FTC", "jointftc"),
    ("FNPAG", "FNPAG", "fnpag"),
    ("PredGuid", "PredGuid", "classical"),
    ("NN-mamba", "NN", "mamba"),
    ("FTC-fixed", "FTC-fixed", "ftc"),
]


def main():
    fl.style()
    rows = {r["label"]: r for r in fl.robustness()}

    fig, axes = plt.subplots(1, 2, figsize=fl.SIZE_HALF)
    panels = (
        (axes[0], "capture_drop_pts", "capture drop (pts)", "Capture robustness"),
        (axes[1], "cvar95_inflation", "CVaR$_{95}$ inflation (m/s)", "Sizing-tail robustness"),
    )
    for ax, metric, xlabel, title in panels:
        labels = [d for d, _, _ in SCHEMES]
        vals = [rows[key][metric] for _, key, _ in SCHEMES]
        colors = [fl.C[ck] for _, _, ck in SCHEMES]
        ypos = range(len(SCHEMES))
        ax.barh(ypos, vals, color=colors, alpha=0.9, zorder=3)
        for y, v in zip(ypos, vals, strict=True):
            ax.annotate(f"{v:.1f}", (v, y), xytext=(4, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=8, fontweight="bold")
        ax.set_yticks(list(ypos))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()  # most robust on top
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10, loc="left")
        ax.margins(x=0.18)

    # honest-caveat callout: joint-FTC beats the NN headline on both axes
    axes[0].annotate("joint-FTC most robust\n(NN headline is not)", xy=(5.5, 0),
                     xytext=(14, 1.3), fontsize=7.5, color=fl.C["jointftc"], va="center",
                     arrowprops=dict(arrowstyle="->", color=fl.C["jointftc"], lw=1.0))

    fig.suptitle("Off-nominal stress (degraded nav + heavy dispersions, n=1000)",
                 fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fl.save(fig, "fig_robustness")


if __name__ == "__main__":
    main()
