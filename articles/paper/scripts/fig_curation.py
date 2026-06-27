"""fig_curation -- Study C-sub: adaptive-seed curation bucket + trim ablation.

Holds everything fixed except the per-bin seed pick (bucket_selection) and the
cost-CDF trim fraction, on the mission-SIZING TAIL (CVaR95, primary) plus the
median-ish dv_mean (secondary). The deployed choice is the max-bucket
(optimizer_budget/ga_300, "bucket_max"), and it is the lowest-tail variant --
picking the WORST seed per quantile bin (max-bucket, no trim) hardens the
optimizer against the sizing tail better than middle/min/random or trimming.
"""

import figlib as fl
import matplotlib.pyplot as plt
import numpy as np

# (display label, results.json runs key) ordered so the deployed choice is first.
CELLS = [
    ("bucket_max\n(deployed)", "optimizer_budget/ga_300"),
    ("bucket_middle", "curation_shaping/bucket_middle"),
    ("bucket_random", "curation_shaping/bucket_random"),
    ("trim_10", "curation_shaping/trim_10"),
    ("trim_20", "curation_shaping/trim_20"),
    ("bucket_min", "curation_shaping/bucket_min"),
]


def main():
    fl.style()
    runs = fl.results()["runs"]
    labels = [lab for lab, _ in CELLS]
    cvar95 = np.array([runs[k]["dv_cvar95"] for _, k in CELLS])
    mean = np.array([runs[k]["dv_mean"] for _, k in CELLS])

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    x = np.arange(len(CELLS))
    for ax, vals, title in ((axes[0], cvar95, "CVaR$_{95}$ (m/s)"), (axes[1], mean, "mean (m/s)")):
        colors = [fl.C["jointftc"] if i == 0 else fl.C["dense"] for i in range(len(CELLS))]
        bars = ax.bar(x, vals, color=colors, width=0.66, zorder=3)
        # deployed (max-bucket) reference line so the tail gap to each variant is legible.
        ax.axhline(vals[0], color=fl.C["jointftc"], lw=0.9, ls="--", zorder=2)
        for b, v in zip(bars, vals, strict=True):
            ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom",
                        fontsize=7.5, fontweight="bold", color="#333333")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(title)
        ax.set_ylim(0, vals.max() * 1.12)
        ax.margins(x=0.04)
    axes[0].set_title("Seed-curation bucket + trim ablation (sizing tail)", fontsize=10, loc="left")
    fig.tight_layout()
    fl.save(fig, "fig_curation")


if __name__ == "__main__":
    main()
