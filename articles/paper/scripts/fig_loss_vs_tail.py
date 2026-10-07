"""fig_loss_vs_tail -- validation loss against the sizing tail for the v4 runs.

Scatter of best validation RMS (the training objective: cubed-transform cost over the reserved
validation pool, `best_val_rms_within_transform_only` of data/results.json, read from each run's
run.jsonl.gz) against far-tail CVaR99.9 on the 10^6 per-scenario confirmatory pool
(10 x 100 000 scenarios per cell, per_draw, asserted at load). Filled markers: the nine runs
trained from scratch at the headline allocation (#173 `hl_*`, GA 512 x 2, 20 000 generations);
hollow markers: the four bundled fine-tune runs (60 x 10, 2000 per-scenario generations from a
shared-path champion), the other v4 recipe. The Spearman rank correlation is quoted over the
nine same-allocation runs, pooled across families and read as descriptive (the runs are not
exchangeable across families). Every v4 cell is heat-load feasible on the pool (no starred
point). Data: articles/paper/data/results.json, confirmatory_marginal.json.
"""

import figlib as fl
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# (confirmatory / results.json label, family, recipe)
HEADLINE = [
    ("ou_marginal/hl_mamba_p962", "mamba"),
    ("ou_marginal/hl_mamba_p962_s2", "mamba"),
    ("ou_marginal/hl_mamba_p962_s3", "mamba"),
    ("ou_marginal/hl_lstm_p1082", "lstm"),
    ("ou_marginal/hl_gru_p1014", "gru"),
    ("ou_marginal/hl_dense_p515", "dense"),
    ("ou_marginal/hl_dense_p515_s2", "dense"),
    ("ou_marginal/hl_dense_p515_s3", "dense"),
    ("ou_marginal/hl_dense_p972", "dense"),
]
FINE_TUNE = [
    ("ou_marginal/ft_dense_p515", "dense"),
    ("ou_marginal/ft_dense_p515_s2", "dense"),
    ("ou_marginal/ft_dense_p515_s3", "dense"),
    ("ou_marginal/ft_gru_p1014", "gru"),
]


def main():
    fl.style()
    runs = fl.results()["runs"]
    conf = fl.marginal()

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    seen = set()
    xs, ys = [], []
    for label, fam in HEADLINE:
        assert runs[label]["noise_seeding"] == "per_draw", label
        rms = runs[label]["best_val_rms_within_transform_only"] / 1e6
        cv = conf[label]["pooled"]["cvar999"]
        xs.append(rms)
        ys.append(cv)
        ax.scatter([rms], [cv], color=fl.C[fam], marker="o", s=70, zorder=4, edgecolor="white", linewidth=0.8, label=fam if fam not in seen else None)
        seen.add(fam)
    for label, fam in FINE_TUNE:
        assert runs[label]["noise_seeding"] == "per_draw", label
        rms = runs[label]["best_val_rms_within_transform_only"] / 1e6
        cv = conf[label]["pooled"]["cvar999"]
        ax.scatter([rms], [cv], facecolor="white", edgecolor=fl.C[fam], marker="o", s=70, zorder=3, linewidth=1.4)
    ax.scatter([], [], facecolor="white", edgecolor="#444444", marker="o", s=70, linewidth=1.4, label="fine-tune recipe (60 $\\times$ 10)")

    # No connector lines (reviewer R1-S6: lines between independently trained runs read as trajectories).
    rho = spearmanr(xs, ys).statistic
    ax.annotate(
        f"Spearman $\\rho$ = {rho:.2f} (n = {len(xs)} headline-allocation runs, descriptive)",
        xy=(0.98, 0.5),
        xycoords="axes fraction",
        ha="right",
        va="center",
        fontsize=8.5,
        color="#444444",
    )

    ax.set_xlabel("best validation RMS ($\\times 10^6$, cubed-transform cost space)")
    ax.set_ylabel("far-tail CVaR$_{99.9}$ (m/s)")
    ax.set_title("Validation loss vs the sizing tail (v4 runs, $10^6$ per-scenario pool)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fl.save(fig, "fig_loss_vs_tail")


if __name__ == "__main__":
    main()
