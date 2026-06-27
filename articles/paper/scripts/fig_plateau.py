"""fig_plateau -- training is compute-bound, not overfitting-bound (methodology, §5).

Best-so-far validation RMS vs generation for the two dense headline-allocation runs
(n=2/512, 20000 gens). Both keep improving for ~15-20k gens then plateau -- the
non-stationary (adaptive-seed) objective never overfits. The 515-param net plateaus
LOWER than the 972 (GA-dimensionality: more dense params are harder for the GA).
Data: articles/paper/data/plateau.json (running-min val RMS, extracted from the full
4-segment training logs).
"""

import json

import figlib as fl
import matplotlib.pyplot as plt


def main():
    fl.style()
    curves = json.loads((fl.DATA / "plateau.json").read_text())
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    # Clip the gen-0 cold-start spike (val RMS ~1e11 before the GA learns) so the
    # slow plateau in the 1.3-2.5e6 region is visible -- that descent is the story.
    for key, ckey, label in (("dense_515", "dense", "Dense-515"), ("dense_972", "gru", "Dense-972")):
        c = [(g, r / 1e6) for g, r in curves[key] if r / 1e6 < 2.5]
        gens = [g for g, _ in c]
        rms = [r for _, r in c]
        ax.plot(gens, rms, color=fl.C[ckey], lw=2.0, label=label)
        ax.annotate(f"{rms[-1]:.3f}M", (gens[-1], rms[-1]), textcoords="offset points",
                    xytext=(6, 0), va="center", color=fl.C[ckey], fontsize=8, fontweight="bold")
    ax.set_ylim(1.25, 2.1)
    ax.set_xlabel("generation")
    ax.set_ylabel("best validation RMS cost ($\\times 10^6$)")
    ax.set_title("Compute-bound, not overfitting-bound (n=2/512)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.margins(x=0.08)
    fig.tight_layout()
    fl.save(fig, "fig_plateau")


if __name__ == "__main__":
    main()
