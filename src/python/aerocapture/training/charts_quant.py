"""Weight-only PTQ sweep chart: capture rate and mean cost vs bit width."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from aerocapture.training.charts import apply_theme  # noqa: E402


def chart_quant_sweep(results: dict[str, Any], output_path: str) -> None:
    """Two panels (capture rate, mean cost) vs weight bits; one line per granularity."""
    apply_theme()
    baseline = results["baseline"]
    variants = results["variants"]

    fig, (ax_cap, ax_cost) = plt.subplots(1, 2, figsize=(14, 6))
    for gran in results["granularities"]:
        rows = sorted((v for v in variants if v["granularity"] == gran), key=lambda r: r["bits"])
        bits = [r["bits"] for r in rows]
        ax_cap.plot(bits, [r["capture_rate"] for r in rows], marker="o", label=gran)
        ax_cost.plot(bits, [r["mean_cost"] for r in rows], marker="o", label=gran)

    ax_cap.axhline(baseline["capture_rate"], color="grey", ls="--", label="fp baseline")
    ax_cost.axhline(baseline["mean_cost"], color="grey", ls="--", label="fp baseline")

    ax_cap.set_xlabel("weight bits")
    ax_cap.set_ylabel("capture rate")
    ax_cap.set_title("Capture rate vs bit width")
    ax_cap.legend()
    ax_cost.set_xlabel("weight bits")
    ax_cost.set_ylabel("mean cost")
    ax_cost.set_title("Mean cost vs bit width")
    ax_cost.legend()
    ax_cap.invert_xaxis()  # high bits left, low bits right (degradation reads left-to-right)
    ax_cost.invert_xaxis()

    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
