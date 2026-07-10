"""Quantization study charts: PTQ sweep curve, LOO bars, QAT convergence overlay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from aerocapture.training.charts import apply_theme  # noqa: E402


def chart_quant_sweep(results: dict[str, Any], output_path: str) -> None:
    """Two panels (capture rate, DV CVaR95) vs weight bits; one line per (granularity, policy)."""
    apply_theme()
    baseline = results["baseline"]
    fig, (ax_cap, ax_tail) = plt.subplots(1, 2, figsize=(14, 6))
    grans = sorted({v["granularity"] for v in results["variants"]})
    policies = sorted({v["tensor_policy"] for v in results["variants"]})
    for gran in grans:
        for policy in policies:
            rows = sorted((v for v in results["variants"] if v["granularity"] == gran and v["tensor_policy"] == policy), key=lambda r: r["bits"])
            if not rows:
                continue
            bits = [r["bits"] for r in rows]
            label = f"{gran} / {policy}"
            ax_cap.plot(bits, [r["capture_rate"] for r in rows], marker="o", label=label)
            ax_tail.plot(bits, [r["dv_cvar95"] if r["dv_cvar95"] is not None else float("nan") for r in rows], marker="o", label=label)
    ax_cap.axhline(baseline["capture_rate"], color="grey", ls="--", label="fp baseline")
    if baseline["dv_cvar95"] is not None:
        ax_tail.axhline(baseline["dv_cvar95"], color="grey", ls="--", label="fp baseline")
    for ax, ylab, title in ((ax_cap, "capture rate", "Capture rate vs bit width"), (ax_tail, "DV CVaR95 [m/s]", "Sizing tail vs bit width")):
        ax.set_xlabel("weight bits")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend()
        ax.invert_xaxis()  # degradation reads left-to-right
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def chart_quant_loo(results: dict[str, Any], output_path: str) -> None:
    """Horizontal bars: CVaR95 delta vs fp baseline when ONE tensor group is quantized."""
    apply_theme()
    rows = results["loo"]
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(rows) + 2))
    names = [r["tensor"] for r in rows]
    deltas = [r["delta_dv_cvar95"] if r["delta_dv_cvar95"] is not None else float("nan") for r in rows]
    colors = ["tab:red" if (d == d and d > 0) else "tab:blue" for d in deltas]
    ax.barh(names, deltas, color=colors)
    for i, r in enumerate(rows):
        if r["capture_rate"] < results["baseline"]["capture_rate"]:
            ax.annotate(f"capture {r['capture_rate']:.1%}", (0, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
    ax.axvline(0.0, color="grey", lw=0.8)
    ax.set_xlabel(f"delta DV CVaR95 [m/s] at {rows[0]['bits']} bits (one tensor quantized, rest fp)")
    ax.set_title("Leave-one-out tensor sensitivity")
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def chart_qat_convergence(jsonl_by_label: dict[str, list[Path]], output_path: str) -> None:
    """Best-cost convergence overlay: champion vs QAT fine-tune vs QAT from-scratch.

    Each label maps to that run's ordered `run_*.jsonl` files (concatenated)."""
    apply_theme()
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, paths in jsonl_by_label.items():
        gens: list[int] = []
        best: list[float] = []
        for p in paths:
            with open(p) as fh:
                for line in fh:
                    rec = json.loads(line)
                    gens.append(int(rec["generation"]))
                    best.append(float(rec["best_cost"]))
        ax.plot(gens, best, label=label, lw=1.0)
    ax.set_xlabel("generation")
    ax.set_ylabel("best training cost")
    ax.set_yscale("log")
    ax.set_title("QAT convergence vs fp champion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
