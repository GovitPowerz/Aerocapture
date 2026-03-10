"""Plot guidance scheme comparison results.

Reads comparison_results.json (from compare_guidance.py) and produces
a multi-panel figure showing performance metrics across all schemes.

Usage:
    uv run python -m aerocapture.training.plot_comparison \
        --results training_output/comparison_results.json \
        --output guidance_comparison.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCHEME_LABELS = {
    "ftc": "FTC",
    "neural_network": "Neural Net",
    "equilibrium_glide": "Eq. Glide",
    "energy_controller": "Energy Ctrl",
    "pred_guid": "PredGuid",
    "fnpag": "FNPAG",
}

SCHEME_COLORS = {
    "ftc": "#2196F3",
    "neural_network": "#FF9800",
    "equilibrium_glide": "#4CAF50",
    "energy_controller": "#9C27B0",
    "pred_guid": "#F44336",
    "fnpag": "#795548",
}


def plot_comparison(results: dict[str, dict], output: Path) -> None:
    """Create a multi-panel comparison figure."""
    # Sort schemes by cost (best first)
    schemes = sorted(results.keys(), key=lambda s: results[s].get("cost", 1e30))

    labels = [SCHEME_LABELS.get(s, s) for s in schemes]
    colors = [SCHEME_COLORS.get(s, "#666666") for s in schemes]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Guidance Scheme Comparison — Mars Aerocapture", fontsize=16, fontweight="bold")

    # 1. Capture rate
    ax = axes[0, 0]
    capture_rates = [results[s].get("capture_rate", 0) for s in schemes]
    bars = ax.bar(labels, capture_rates, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Capture Rate (%)")
    ax.set_ylim(0, 105)
    ax.axhline(y=100, color="green", linestyle="--", alpha=0.3, linewidth=1)
    for bar, val in zip(bars, capture_rates, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_title("Capture Rate")
    ax.tick_params(axis="x", rotation=30)

    # 2. Cost (log scale)
    ax = axes[0, 1]
    costs = [results[s].get("cost", 1e30) for s in schemes]
    bars = ax.bar(labels, costs, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Cost (RMS)")
    ax.set_yscale("log")
    ax.set_title("Overall Cost (lower is better)")
    ax.tick_params(axis="x", rotation=30)

    # 3. Apoapsis error
    ax = axes[0, 2]
    apo_means = [results[s].get("apo_err_mean", float("nan")) for s in schemes]
    apo_stds = [results[s].get("apo_err_std", 0) for s in schemes]
    x = np.arange(len(schemes))
    ax.bar(x, apo_means, yerr=apo_stds, color=colors, edgecolor="white", linewidth=0.5, capsize=3, error_kw={"linewidth": 1})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.set_ylabel("Apoapsis Error (km)")
    ax.set_title("Apoapsis Error (mean +/- std)")

    # 4. Periapsis error
    ax = axes[1, 0]
    peri_means = [results[s].get("peri_err_mean", float("nan")) for s in schemes]
    peri_stds = [results[s].get("peri_err_std", 0) for s in schemes]
    ax.bar(x, peri_means, yerr=peri_stds, color=colors, edgecolor="white", linewidth=0.5, capsize=3, error_kw={"linewidth": 1})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.set_ylabel("Periapsis Error (km)")
    ax.set_title("Periapsis Error (mean +/- std)")

    # 5. Delta-V
    ax = axes[1, 1]
    dv_means = [results[s].get("dv_mean", float("nan")) for s in schemes]
    dv_stds = [results[s].get("dv_std", 0) for s in schemes]
    ax.bar(x, dv_means, yerr=dv_stds, color=colors, edgecolor="white", linewidth=0.5, capsize=3, error_kw={"linewidth": 1})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.set_ylabel("Delta-V (m/s)")
    ax.set_title("Correction Delta-V (mean +/- std)")

    # 6. Summary table
    ax = axes[1, 2]
    ax.axis("off")
    table_data = []
    col_labels = ["Scheme", "Cap %", "Cost", "Apo (km)", "Peri (km)", "dV (m/s)"]
    for s in schemes:
        m = results[s]
        row = [
            SCHEME_LABELS.get(s, s),
            f"{m.get('capture_rate', 0):.0f}",
            f"{m.get('cost', 0):.0e}",
            f"{m.get('apo_err_mean', float('nan')):.0f}",
            f"{m.get('peri_err_mean', float('nan')):.0f}",
            f"{m.get('dv_mean', float('nan')):.0f}",
        ]
        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    # Color the first column cells
    for i, s in enumerate(schemes):
        table[i + 1, 0].set_facecolor(SCHEME_COLORS.get(s, "#666666") + "30")

    # Bold header
    for j in range(len(col_labels)):
        table[0, j].set_text_props(fontweight="bold")

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot guidance comparison results")
    parser.add_argument("--results", type=str, default="training_output/comparison_results.json")
    parser.add_argument("--output", type=str, default="guidance_comparison.png")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    plot_comparison(results, Path(args.output))


if __name__ == "__main__":
    main()
