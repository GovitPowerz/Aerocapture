"""fig_classical_vs_nn -- the deployability scatter (compute vs sizing tail).

x = per-sim compute cost (ms/sim, LOG), y = far-tail CVaR99.9 (m/s). Lower-left is
better: cheap AND tight-tailed. The recurrent NN (Mamba-962) sits lower-left of
BOTH classical references (joint-FTC ~164, FNPAG ~165) -- it sizes a smaller
ergol margin (lower tail DV) at a fraction of FNPAG's per-sim cost, while staying
within ~3x of FTC's compute. Dense-515 is the cheapest NN but carries a fatter tail.
Half-column figure: point labels carry the name + tail value only (compute is the
x-axis), so the annotations stay legible at ~2.7in display width.
"""

import figlib as fl
import matplotlib.pyplot as plt

# (display label, color key, label offset (dx, dy) in points)
POINTS = [
    ("Mamba-962", "mamba", (7, 4)),
    ("Dense-515", "dense", (7, -4)),
    ("joint-FTC", "jointftc", (7, 6)),
    ("FNPAG", "fnpag", (-7, -14)),
]


def main():
    fl.style()
    import json
    conf = {c["label"]: c["pooled"] for c in json.loads((fl.DATA / "confirmatory_eval.json").read_text())["cells"]}
    ms = {s["label"]: s["ms_per_sim"] for s in fl.compute()}

    # far-tail CVaR99.9 per point, on the frozen confirmatory pool. Mamba and
    # dense on the SAME 3-seed-mean basis as the tail-reversal section (not a
    # lucky single seed); classical references from their confirmatory cells.
    mamba_mean = sum(
        conf[k]["cvar999"]
        for k in ("mamba_p962_long", "paper/tail_repeats/mamba962_s2", "paper/tail_repeats/mamba962_s3")
    ) / 3.0
    dense515_mean = sum(
        conf[k]["cvar999"]
        for k in ("dense_p515_ga_paper_best", "paper/tail_repeats/dense515_s2", "paper/tail_repeats/dense515_s3")
    ) / 3.0
    y = {
        "Mamba-962": mamba_mean,                                  # 125.5 (3-seed mean)
        "Dense-515": dense515_mean,                                # 140.5 (3-seed mean)
        "joint-FTC": conf["joint_reference/ftc"]["cvar999"],       # 165.1
        "FNPAG": conf["fnpag"]["cvar999"],                         # 198.7
    }
    # compute label -> the actual per-sim ms; joint-FTC rides FTC's compute cost.
    x = {
        "Mamba-962": ms["NN-mamba"],   # 3.68
        "Dense-515": ms["NN-dense"],   # 2.40
        "joint-FTC": ms["FTC"],        # 1.25
        "FNPAG": ms["FNPAG"],          # 86.1
    }

    fig, ax = plt.subplots(figsize=fl.SIZE_HALF)

    for label, ckey, (dx, dy) in POINTS:
        xv, yv = x[label], y[label]
        ax.scatter([xv], [yv], color=fl.C[ckey], s=90, zorder=4,
                   edgecolor="white", linewidth=1.0)
        ax.annotate(f"{label}\n{yv:.0f} m/s",
                    (xv, yv), textcoords="offset points", xytext=(dx, dy),
                    fontsize=8, color=fl.C[ckey], fontweight="bold",
                    ha="left" if dx >= 0 else "right",
                    va="bottom" if dy >= 0 else "top")

    # lower-left = better guide
    ax.annotate("better\n(cheap + tight tail)", xy=(0.04, 0.05), xycoords="axes fraction",
                fontsize=8, color="#555555", ha="left", va="bottom", style="italic")

    ax.set_xscale("log")
    ax.set_xlabel("compute cost (ms / sim, log scale)")
    ax.set_ylabel("far-tail CVaR$_{99.9}$ (m/s)")
    ax.set_title("Deployability: tail vs compute")
    ax.set_xlim(0.8, 160)
    ax.set_ylim(115, 178)
    fig.tight_layout()
    fl.save(fig, "fig_classical_vs_nn")


if __name__ == "__main__":
    main()
