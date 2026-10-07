"""fig_robustness -- off-nominal stress at depth (#176).

The deployed v4 champion and the four retuned classical cells (#172) on the reserved 9M stress
pool (atmosphere, density perturbation, navigation and nav filter at `high`), n = 10 000
scenarios per scheme, per-scenario noise (data/stress_depth.json, asserted per_draw), against
each scheme's nominal numbers on the 10^6 per-scenario confirmatory pool
(data/confirmatory_marginal.json, the cell named by the stress row's `run_dir`). x = capture-rate
drop (nominal minus stress, percentage points), y = CVaR95 inflation (stress minus nominal, m/s);
whiskers are the stress run's 95 % bootstrap CIs (the 10^6 nominal is exact at this scale).
Lower-left is robust. Half-column single panel, axis limits from the data.
"""

import figlib as fl
import matplotlib.pyplot as plt

# (display label, stress_depth label, color key, label offset (dx, dy) in points)
SCHEMES = [
    ("joint-FTC", "joint-FTC", "jointftc", (0, 12)),
    ("FNPAG", "FNPAG", "fnpag", (-8, -2)),
    ("PredGuid", "PredGuid", "classical", (8, -10)),
    ("NN-mamba", "NN", "mamba", (-8, -14)),
    ("FTC-fixed", "FTC-fixed", "ftc", (8, 10)),
]


def main():
    fl.style()
    stress = fl.stress_depth()
    nominal = fl.marginal()

    fig, ax = plt.subplots(figsize=fl.SIZE_HALF)
    xs, ys = [], []
    for label, key, ckey, (dx, dy) in SCHEMES:
        s = stress["cells"][key]
        n = nominal[s["run_dir"]]["pooled"]
        nominal_capture = 100.0 * n["n_captured"] / n["n"]
        xv = nominal_capture - s["capture_pct"]
        yv = s["dv_cvar95"] - n["cvar95"]
        xerr = [[s["capture_pct_ci"][1] - s["capture_pct"]], [s["capture_pct"] - s["capture_pct_ci"][0]]]  # a higher stress capture = a smaller drop
        yerr = [[s["dv_cvar95"] - s["dv_cvar95_ci"][0]], [s["dv_cvar95_ci"][1] - s["dv_cvar95"]]]
        xs.append(xv)
        ys.append(yv)
        ax.errorbar([xv], [yv], xerr=xerr, yerr=yerr, fmt="none", ecolor=fl.C[ckey], elinewidth=0.9, capsize=2, zorder=3, alpha=0.8)
        ax.scatter([xv], [yv], color=fl.C[ckey], s=90, zorder=4, edgecolor="white", linewidth=1.0)
        ax.annotate(
            f"{label}\n$-{xv:.1f}$ pts, $+{yv:.0f}$ m/s",
            (xv, yv),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8,
            color=fl.C[ckey],
            fontweight="bold",
            ha="center" if dx == 0 else ("left" if dx > 0 else "right"),
            va="bottom" if dy >= 0 else "top",
        )

    # lower-left = robust guide
    ax.annotate(
        "robust\n(small drop + small inflation)", xy=(0.03, 0.05), xycoords="axes fraction", fontsize=8, color="#555555", ha="left", va="bottom", style="italic"
    )

    ax.set_xlabel("capture-rate drop (pts)")
    ax.set_ylabel("CVaR$_{95}$ inflation (m/s)")
    ax.set_title(f"Off-nominal stress at depth (n = {stress['n_sims']:,})".replace(",", " "))
    ax.set_xlim(0, max(xs) * 1.25)
    ax.set_ylim(min(ys) * 0.5, max(ys) * 1.2)
    fig.tight_layout()
    fl.save(fig, "fig_robustness")


if __name__ == "__main__":
    main()
