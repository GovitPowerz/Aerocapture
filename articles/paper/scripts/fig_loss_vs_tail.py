"""fig_loss_vs_tail -- validation loss orders seeds within a family, not across.

Scatter of best validation RMS (the training objective; all cells share the
cubed transform and the converged n=2/512 budget, so the RMS scale is
comparable) against far-tail CVaR99.9 (n=10000) for the eleven converged runs.
Within every family the seeds order identically on both axes (overall Spearman
rho ~ 0.91); the BETWEEN-family offsets are what the loss cannot see -- the
lowest-loss run (the LSTM s1, also the heat-load-infeasible one) is not the
best tail, and the dense cells sit above the Mamba at matched loss. Backs
paper section 6.3.
"""

import figlib as fl
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# (far_tail_eval label, results.json runs key, family, display note)
CELLS = [
    ("mamba_p962_long", "headline/mamba_p962", "mamba", "s1"),
    ("paper/tail_repeats/mamba962_s2", "tail_repeats/mamba962_s2", "mamba", "s2"),
    ("paper/tail_repeats/mamba962_s3", "tail_repeats/mamba962_s3", "mamba", "s3"),
    ("lstm_p1082_long", "headline/lstm_p1082", "lstm", "s1"),
    ("paper/tail_repeats/lstm1082_s2", "tail_repeats/lstm1082_s2", "lstm", "s2"),
    ("paper/tail_repeats/lstm1082_s3", "tail_repeats/lstm1082_s3", "lstm", "s3"),
    ("gru_p1014_long", "headline/gru_p1014", "gru", "s1"),
    ("dense_p515_ga_paper_best", "headline/dense_p515", "dense", "s1"),
    ("paper/tail_repeats/dense515_s2", "tail_repeats/dense515_s2", "dense", "s2"),
    ("paper/tail_repeats/dense515_s3", "tail_repeats/dense515_s3", "dense", "s3"),
    ("dense_p972_ga_paper_best", "headline/dense_p972", "dense", "972"),
]
INFEASIBLE = {"lstm_p1082_long"}  # heat-load violations on 14.4% of the sizing pool


def main():
    fl.style()
    runs = fl.results()["runs"]
    ft = fl.far_tail()

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    xs_all, ys_all = [], []
    seen = set()
    for ft_label, run_key, fam, _note in CELLS:
        rms = runs[run_key]["best_val_rms_within_transform_only"] / 1e6
        cv = ft[ft_label]["cvar999"]
        xs_all.append(rms)
        ys_all.append(cv)
        marker = "*" if ft_label in INFEASIBLE else "o"
        size = 220 if ft_label in INFEASIBLE else 70
        ax.scatter([rms], [cv], color=fl.C[fam], marker=marker, s=size, zorder=4,
                   edgecolor="white", linewidth=0.8,
                   label=fam if fam not in seen else None)
        seen.add(fam)

    # No connector lines: reviewer R1-S6 -- lines between independently trained
    # runs read as trajectories/ordered observations. Family color carries the
    # grouping; within-family ordering is stated in the annotation and caption.

    rho = spearmanr(xs_all, ys_all).statistic
    ax.annotate(f"Spearman $\\rho$ = {rho:.2f} (n = {len(xs_all)}, descriptive)\n"
                "within-family: identically ordered\nbetween families: offsets decide",
                xy=(0.02, 0.96), xycoords="axes fraction", va="top", fontsize=8.5,
                color="#444444")
    ax.annotate("infeasible\n(heat load)", xy=(1.276, 123.24), xytext=(8, -18),
                textcoords="offset points", fontsize=8, color=fl.C["lstm"], fontweight="bold")

    ax.set_xlabel("best validation RMS ($\\times 10^6$, cubed-transform cost space)")
    ax.set_ylabel("far-tail CVaR$_{99.9}$ (m/s)")
    ax.set_title("Validation loss vs the sizing tail (11 converged runs)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fl.save(fig, "fig_loss_vs_tail")


if __name__ == "__main__":
    main()
