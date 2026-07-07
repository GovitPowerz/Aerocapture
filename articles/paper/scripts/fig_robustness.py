"""fig_robustness -- off-nominal stress test (study 5c).

The honest caveat figure. Under a 1000-sim off-nominal MC (degraded nav + heavier
dispersions), every scheme loses capture and inflates its sizing tail relative to
the nominal pool. The point: the NN-mamba headline is NOT the most robust --
joint-FTC degrades LESS on both axes (capture drop 5.5 vs 9.9 pts, CVaR95
inflation +197 vs +402 m/s). FTC-fixed (the un-retuned classical) collapses
(33 pts capture lost), confirming the off-nominal pool genuinely bites.

Half-column single panel: capture drop (x) vs CVaR95 inflation (y), one labeled
point per scheme -- lower-left is robust. Replaces the old two-bar-panel layout
that was illegible at half-column width.
"""

import figlib as fl
import matplotlib.pyplot as plt

# (display label, robustness-data label, color key, label offset (dx, dy) in points)
SCHEMES = [
    ("joint-FTC", "joint-FTC", "jointftc", (8, -2)),
    ("FNPAG", "FNPAG", "fnpag", (8, -2)),
    ("PredGuid", "PredGuid", "classical", (8, -2)),
    ("NN-mamba", "NN", "mamba", (8, -2)),
    ("FTC-fixed", "FTC-fixed", "ftc", (-8, -8)),
]


def main():
    fl.style()
    rows = {r["label"]: r for r in fl.robustness()}

    fig, ax = plt.subplots(figsize=fl.SIZE_HALF)
    for label, key, ckey, (dx, dy) in SCHEMES:
        xv = rows[key]["capture_drop_pts"]
        yv = rows[key]["cvar95_inflation"]
        ax.scatter([xv], [yv], color=fl.C[ckey], s=90, zorder=4,
                   edgecolor="white", linewidth=1.0)
        ax.annotate(f"{label}\n$-{xv:.1f}$ pts, $+{yv:.0f}$ m/s",
                    (xv, yv), textcoords="offset points", xytext=(dx, dy),
                    fontsize=8, color=fl.C[ckey], fontweight="bold",
                    ha="left" if dx >= 0 else "right",
                    va="bottom" if dy >= 0 else "top")

    # lower-left = robust guide
    ax.annotate("robust\n(small drop + small inflation)", xy=(0.03, 0.05),
                xycoords="axes fraction", fontsize=8, color="#555555",
                ha="left", va="bottom", style="italic")

    ax.set_xlabel("capture-rate drop (pts)")
    ax.set_ylabel("CVaR$_{95}$ inflation (m/s)")
    ax.set_title("Off-nominal stress (n=1000)")
    ax.set_xlim(0, 38)
    ax.set_ylim(100, 560)
    fig.tight_layout()
    fl.save(fig, "fig_robustness")


if __name__ == "__main__":
    main()
