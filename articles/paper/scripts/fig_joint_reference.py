"""fig_joint_reference -- Study E (joint reference recovers ref-tracking schemes).

Paired bars (fixed-ref vs joint-ref) for the three table-reading classical schemes
{FTC, energy_controller, pred_guid}, on the mission-SIZING TAIL. Co-optimizing the
constant-bank reference together with the guidance gains (joint_bank gene) collapses
the sizing tail: FTC drops from CVaR95 244.1 -> 142.9 m/s (mean 170.7 -> 126.2).
Primary metric dv_cvar95 (heavy bars), dv_mean overlaid as light hatched bars.
"""

import figlib as fl
import matplotlib.pyplot as plt
import numpy as np

# (display label, fixed-ref key, joint-ref key) ordered by joint-ref CVaR95.
SCHEMES = [
    ("FTC", "classical_baselines/ftc", "joint_reference/ftc"),
    ("Energy ctrl", "classical_baselines/energy_controller", "joint_reference/energy_controller"),
    ("PredGuid", "classical_baselines/pred_guid", "joint_reference/pred_guid"),
]


def main():
    fl.style()
    runs = fl.results()["runs"]

    labels = [s[0] for s in SCHEMES]
    fixed_cvar = np.array([runs[s[1]]["dv_cvar95"] for s in SCHEMES])
    joint_cvar = np.array([runs[s[2]]["dv_cvar95"] for s in SCHEMES])
    fixed_mean = np.array([runs[s[1]]["dv_mean"] for s in SCHEMES])
    joint_mean = np.array([runs[s[2]]["dv_mean"] for s in SCHEMES])

    x = np.arange(len(SCHEMES))
    w = 0.36
    c_fixed = fl.C["ftc"]       # grey -- the fixed-ref baseline
    c_joint = fl.C["jointftc"]  # orange -- the joint-ref winner

    fig, ax = plt.subplots(figsize=fl.SIZE1)

    # primary metric: CVaR95 (solid bars)
    ax.bar(x - w / 2, fixed_cvar, w, color=c_fixed, label="fixed ref (CVaR$_{95}$)", zorder=3)
    ax.bar(x + w / 2, joint_cvar, w, color=c_joint, label="joint ref (CVaR$_{95}$)", zorder=3)
    # secondary metric: mean (light hatched overlay, same x-slots)
    ax.bar(x - w / 2, fixed_mean, w, facecolor="none", edgecolor="white", hatch="///", lw=0, zorder=4)
    ax.bar(x + w / 2, joint_mean, w, facecolor="none", edgecolor="white", hatch="///", lw=0, zorder=4)

    for xi, (fc, jc) in enumerate(zip(fixed_cvar, joint_cvar, strict=True)):
        ax.annotate(f"{fc:.0f}", (xi - w / 2, fc + 3), ha="center", va="bottom", fontsize=8, color=c_fixed, fontweight="bold")
        ax.annotate(f"{jc:.0f}", (xi + w / 2, jc + 3), ha="center", va="bottom", fontsize=8, color=c_joint, fontweight="bold")
        ax.annotate(f"$\\Delta${fc - jc:.0f}", (xi, max(fc, jc) + 18), ha="center", va="bottom", fontsize=8.5,
                    color=c_joint, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("correction $\\Delta v$ (m/s)")
    ax.set_title("Joint reference optimization recovers ref-tracking schemes\n"
                 "(CVaR$_{95}$ bars, mean = white hatch overlay)", fontsize=10, loc="left")
    ax.set_ylim(0, max(fixed_cvar) * 1.22)
    ax.legend(loc="upper right")
    ax.margins(x=0.08)
    fig.tight_layout()
    fl.save(fig, "fig_joint_reference")


if __name__ == "__main__":
    main()
