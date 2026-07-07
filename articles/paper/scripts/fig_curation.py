"""fig_curation -- Study C-sub: adaptive-seed curation bucket + trim ablation.

Panel A carries the DECIDING metric: far-tail CVaR99.9 (n=10000) per curation
bucket (the per-bin seed pick). Picking the HARDEST seed per cost-CDF bin (the
max bucket, deployed = optimizer_budget/ga_300) compresses the sizing tail; the
easiest-seed (min) bucket blows it up (sample max 245 m/s) while winning the
mean -- the canonical optimize-the-average failure. Panel B shows the n=1000
mean for all cells including the two trim variants: trimming matched the max
bucket on the shallow pool and was not carried to the far-tail pool.
"""

import figlib as fl
import matplotlib.pyplot as plt
import numpy as np

# (display label, far_tail_eval cell label) -- deployed max-bucket first.
BUCKETS = [
    ("bucket_max\n(deployed)", "optimizer_budget/ga_300"),
    ("bucket_middle", "curation_shaping/bucket_middle"),
    ("bucket_random", "curation_shaping/bucket_random"),
    ("bucket_min", "curation_shaping/bucket_min"),
]
# (display label, results.json runs key) -- the n=1000 mean panel, incl. trims.
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
    ft = fl.far_tail()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=fl.SIZE2)

    # ---- Panel A: far-tail CVaR99.9 per bucket (the sizing decision) ----
    labA = [lab for lab, _ in BUCKETS]
    valA = np.array([ft[k]["cvar999"] for _, k in BUCKETS])
    colA = [fl.C["jointftc"]] + [fl.C["dense"]] * (len(BUCKETS) - 1)
    xA = np.arange(len(BUCKETS))
    barsA = axA.bar(xA, valA, color=colA, width=0.62, zorder=3)
    axA.axhline(valA[0], color=fl.C["jointftc"], lw=0.9, ls="--", zorder=2)
    for b, v in zip(barsA, valA, strict=True):
        axA.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom",
                     fontsize=7.5, fontweight="bold", color="#333333")
    # the min bucket's catastrophic worst case, quoted in the body
    axA.annotate(f"worst case {ft['curation_shaping/bucket_min']['max']:.0f} m/s",
                 (xA[-1], valA[-1]), xytext=(10, 16), textcoords="offset points",
                 ha="right", fontsize=7.5, color=fl.C["fnpag"], fontweight="bold")
    axA.set_xticks(xA)
    axA.set_xticklabels(labA, rotation=20, ha="right", fontsize=8)
    axA.set_ylabel("CVaR$_{99.9}$ (m/s)")
    axA.set_ylim(0, valA.max() * 1.18)
    axA.set_title("(A) Sizing tail (n=10000) vs curation bucket", fontsize=10, loc="left")
    axA.margins(x=0.04)

    # ---- Panel B: n=1000 mean, all cells incl. the trim variants ----
    labB = [lab for lab, _ in CELLS]
    valB = np.array([runs[k]["dv_mean"] for _, k in CELLS])
    colB = [fl.C["jointftc"] if i == 0 else fl.C["dense"] for i in range(len(CELLS))]
    xB = np.arange(len(CELLS))
    barsB = axB.bar(xB, valB, color=colB, width=0.66, zorder=3)
    axB.axhline(valB[0], color=fl.C["jointftc"], lw=0.9, ls="--", zorder=2)
    for b, v in zip(barsB, valB, strict=True):
        axB.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom",
                     fontsize=7.5, fontweight="bold", color="#333333")
    axB.set_xticks(xB)
    axB.set_xticklabels(labB, rotation=20, ha="right", fontsize=8)
    axB.set_ylabel("mean (m/s)")
    axB.set_ylim(0, valB.max() * 1.12)
    axB.set_title("(B) Mean (n=1000), incl. trim variants", fontsize=10, loc="left")
    axB.margins(x=0.04)

    fig.tight_layout()
    fl.save(fig, "fig_curation")


if __name__ == "__main__":
    main()
