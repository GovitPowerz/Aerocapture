"""Ablation analysis chart."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # type: ignore[import-untyped]  # noqa: E402


def chart_ablation_bar(ranked: list[dict], output_path: str) -> None:
    """Bar chart of cost delta per input, ranked by importance."""
    sns.set_theme(style="whitegrid", palette="muted", font_scale=0.9, rc={"axes.facecolor": "#f5f5f5"})

    names = [r["name"] for r in ranked]
    deltas = [r["delta"] for r in ranked]

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["#e74c3c" if d > 0 else "#3498db" for d in deltas]
    ax.barh(range(len(names)), deltas, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Cost Delta (ablated - baseline)")
    ax.set_title("NN Input Importance (Ablation Analysis)")
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
