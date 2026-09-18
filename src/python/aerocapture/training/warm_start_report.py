"""Warm-start performance snapshot: charts + Typst PDF at the end of supervised pretrain.

Triggered automatically from `train.py` after `build_warm_start_chromosome` +
the gen-0 validation baseline complete. Also runnable standalone via the CLI:

    python -m aerocapture.training.warm_start_report training_output/<scheme>

Reads four sidecar artifacts written during warm-start:
  - warm_start_loss.json       (per-epoch MSE)
  - warm_start_baseline.json   (validation-pool DV mean/RMS/capture_rate)
  - warm_start_bounds.json     (per-parameter ParamSpec bounds used to encode)
  - warm_start_selection.json  (per-supervisor selection counts + capture stats)

Produces:
  - `<save_dir>/warm_start_report/*.svg`  (charts)
  - `<save_dir>/warm_start_report.pdf`    (Typst-rendered PDF; degrades gracefully if Typst missing)

The PDF mirrors the main training report's style. It is INTENDED for fast
visual inspection between "did supervised pretrain converge?" and "is PSO
about to start from a meaningful chromosome?" -- no actual MC sims are
issued here (the validation baseline was already run by train.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from aerocapture.training.charts import apply_theme  # noqa: E402
from aerocapture.training.report_render import render_pdf  # noqa: E402

apply_theme()

_REPORT_SUBDIR = "warm_start_report"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):  # fmt: skip
        return None


def _load_artifacts(save_dir: Path) -> dict[str, Any]:
    """Load every available warm-start sidecar, tolerating missing files."""
    return {
        "loss": _load_json(save_dir / "warm_start_loss.json"),
        "baseline": _load_json(save_dir / "warm_start_baseline.json"),
        "bounds": _load_json(save_dir / "warm_start_bounds.json"),
        "selection": _load_json(save_dir / "warm_start_selection.json"),
        "cache_key": _load_json(save_dir / "warm_start_cache_key.json"),
        "eval_summary": _load_json(save_dir / "warm_start_eval_summary.json"),
        # Manifest written by warm_start_compare.render_trajectory_comparison
        # listing per-(pool, side, panel) SVG filenames. Absent when the
        # comparison didn't run (older training output, or it errored out).
        "compare_manifest": _load_json(save_dir / _REPORT_SUBDIR / "compare_manifest.json"),
    }


# ---------------------------------------------------------------------------
# Chart functions (each writes a single SVG)
# ---------------------------------------------------------------------------


def chart_supervised_mse(loss_records: list[dict], output: Path) -> None:
    """Per-epoch MSE convergence with log-y when range > 1 decade."""
    fig, ax = plt.subplots(figsize=(10, 4))
    if not loss_records:
        ax.text(0.5, 0.5, "No loss records (n_epochs=0)", transform=ax.transAxes, ha="center", va="center", fontsize=11, color="gray")
        ax.set_axis_off()
    else:
        epochs = [r["epoch"] + 1 for r in loss_records]
        mses = [float(r["mean_mse"]) for r in loss_records]
        ax.plot(epochs, mses, marker="o", linewidth=1.5, markersize=4, color="#1f77b4")
        if mses and max(mses) > 0.0 and min(mses) > 0.0 and (max(mses) / min(mses)) > 10.0:
            ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean MSE (chunked BPTT)")
        ax.set_title("Supervised pretrain convergence")
        # Annotate first / last
        if len(mses) >= 1:
            ax.annotate(f"{mses[0]:.3e}", (epochs[0], mses[0]), textcoords="offset points", xytext=(6, 6), fontsize=8)
        if len(mses) >= 2:
            ax.annotate(f"{mses[-1]:.3e}", (epochs[-1], mses[-1]), textcoords="offset points", xytext=(6, -10), fontsize=8)
    fig.tight_layout()
    fig.savefig(output, format="svg")
    plt.close(fig)


def chart_supervisor_selection(selection: dict, output: Path) -> None:
    """Per-supervisor: bars for capture rate (on warm-start seeds) and
    selection share (which scheme won per-seed)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    per_scheme: dict[str, dict] = selection.get("per_scheme", {}) if selection else {}
    if not per_scheme:
        ax.text(0.5, 0.5, "No supervisor selection data", transform=ax.transAxes, ha="center", va="center", fontsize=11, color="gray")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output, format="svg")
        plt.close(fig)
        return
    schemes = list(per_scheme.keys())
    n_supervised = max(s["n_supervised"] for s in per_scheme.values()) or 1
    n_selected_total = max(selection.get("n_selected_total", 1), 1)
    capture_rates = [per_scheme[s]["capture_rate"] for s in schemes]
    selection_shares = [per_scheme[s]["n_selected"] / n_selected_total for s in schemes]
    x = np.arange(len(schemes))
    w = 0.38
    bars_cap = ax.bar(x - w / 2, capture_rates, w, label="Capture rate (per scheme over n_warm_seeds)", color="#2ca02c")
    bars_sel = ax.bar(x + w / 2, selection_shares, w, label="Selection share (winner of best-of)", color="#1f77b4")
    for bars in (bars_cap, bars_sel):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.0%}", (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(schemes, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction")
    ax.set_title(f"Supervisor pool ({n_supervised} seeds × {len(schemes)} schemes; {selection.get('n_selected_total', 0)} captured-winners)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, format="svg")
    plt.close(fig)


def chart_bound_widening(bounds: list[dict], output: Path, top_n: int = 12) -> None:
    """Per-layer-slab bound widening factor relative to slab median.

    Adaptive bounds collapse to a single (p_min, p_max) per layer slab;
    we plot the slab half-width grouped by parameter-name prefix. The
    parameter naming convention encodes the layer (e.g. `w0_*` for layer 0
    dense weights, `w_ih1_*` for layer 1 GRU input-hidden, etc.), so we
    group by everything before the first digit / underscore-digit and
    report the slab half-width.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    if not bounds:
        ax.text(0.5, 0.5, "No bounds artifact (warm-start not yet run?)", transform=ax.transAxes, ha="center", va="center", fontsize=11, color="gray")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output, format="svg")
        plt.close(fig)
        return

    # Group consecutive ParamSpec entries that share the same (p_min, p_max);
    # adaptive_bounds writes one symmetric range per layer slab, so this
    # collapses to one entry per slab. For non-adaptive (static Xavier),
    # consecutive specs still typically share bounds within a sub-block.
    groups: list[tuple[str, int, float]] = []  # (group_name, count, half_width)
    last_key: tuple[float, float] | None = None
    cur_name = ""
    cur_count = 0
    for spec in bounds:
        key = (float(spec["p_min"]), float(spec["p_max"]))
        if key != last_key:
            if last_key is not None:
                groups.append((cur_name, cur_count, 0.5 * (last_key[1] - last_key[0])))
            last_key = key
            cur_name = _slab_label(str(spec["name"]))
            cur_count = 1
        else:
            cur_count += 1
    if last_key is not None:
        groups.append((cur_name, cur_count, 0.5 * (last_key[1] - last_key[0])))

    # Sort by half-width desc; show top_n.
    groups.sort(key=lambda g: g[2], reverse=True)
    shown = groups[:top_n]
    names = [g[0] for g in shown]
    widths = [g[2] for g in shown]
    counts = [g[1] for g in shown]

    bars = ax.barh(range(len(shown)), widths, color="#ff7f0e")
    for i, (b, n) in enumerate(zip(bars, counts, strict=True)):
        ax.annotate(f"{b.get_width():.2e}  (n={n})", (b.get_width(), i), textcoords="offset points", xytext=(4, 0), va="center", fontsize=8)
    ax.set_yticks(range(len(shown)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Slab bound half-width (|p_max - p_min| / 2)")
    suffix = f" — top {top_n} of {len(groups)} slabs" if len(groups) > top_n else ""
    ax.set_title(f"PSO/GA/DE search-space bounds per layer slab{suffix}")
    fig.tight_layout()
    fig.savefig(output, format="svg")
    plt.close(fig)


def _slab_label(name: str) -> str:
    """Strip the trailing per-element indices from a ParamSpec name.

    Naming convention from encoding.py: `{root}{layer_idx}_{i}[_{j}[_{k}]]`,
    where `layer_idx` is suffixed onto the root with no underscore (e.g.
    `w0`, `bias1`, `w_ih0`, `a_log1`). The trailing `_i_j_k` are per-element
    indices. The slab label is everything UP TO AND INCLUDING the part that
    ends with the layer-index digit.

    e.g. `w0_5_3` -> `w0`, `bias1_2` -> `bias1`, `w_ih0_5` -> `w_ih0`,
    `a_log1_3_5` -> `a_log1`.
    """
    parts = name.split("_")
    out: list[str] = []
    for p in parts:
        out.append(p)
        # Stop after the first part that ends with a digit (the layer-index
        # baked into the name root). Pure-digit parts (per-element indices)
        # come AFTER this and are dropped.
        if p and p[-1].isdigit() and not p.isdigit():
            break
    return "_".join(out)


# ---------------------------------------------------------------------------
# Metadata + Typst render
# ---------------------------------------------------------------------------


def _format_value(x: float | int | None, fmt: str = "{:.4e}") -> str:
    if x is None:
        return "n/a"
    try:
        return fmt.format(float(x))
    except (TypeError, ValueError):  # fmt: skip
        return str(x)


def _build_metadata(artifacts: dict[str, Any], save_dir: Path) -> dict[str, Any]:
    """Assemble the metadata dict Typst reads to render the report."""
    cache = artifacts.get("cache_key") or {}
    baseline = artifacts.get("baseline") or {}
    loss: list[dict] = artifacts.get("loss") or []
    selection = artifacts.get("selection") or {}

    # Architecture summary (compact string)
    arch = cache.get("architecture") or []
    arch_summary = " -> ".join(_layer_summary(layer) for layer in arch) if arch else "n/a"

    # Per-supervisor table rows
    per_scheme = selection.get("per_scheme") or {}
    supervisor_rows = []
    for scheme, stats in per_scheme.items():
        supervisor_rows.append(
            {
                "scheme": scheme,
                "n_supervised": stats["n_supervised"],
                "n_captured": stats["n_captured"],
                "capture_rate": f"{stats['capture_rate']:.0%}",
                "n_selected": stats["n_selected"],
                "median_dv": _format_value(stats.get("median_dv_captured"), "{:.1f}"),
            }
        )

    # Loss summary
    if loss:
        first_mse = float(loss[0]["mean_mse"])
        last_mse = float(loss[-1]["mean_mse"])
        reduction = (first_mse - last_mse) / first_mse * 100.0 if first_mse > 0 else 0.0
        loss_summary = f"{first_mse:.4e} → {last_mse:.4e}  ({reduction:+.1f}%, {len(loss)} epochs)"
    else:
        loss_summary = "n_epochs=0; no supervised pretraining performed"

    return {
        "save_dir": str(save_dir),
        "scheme": save_dir.name,
        "arch_summary": arch_summary,
        "loss_summary": loss_summary,
        "n_chunks": int(loss[0]["n_chunks"]) if loss else 0,
        "config": {
            "supervisor_schemes": cache.get("supervisor_schemes", []),
            "bptt_length": cache.get("bptt_length"),
            "n_warm_seeds": cache.get("n_warm_seeds"),
            "n_epochs": cache.get("n_epochs"),
            "bound_multiplier": cache.get("bound_multiplier"),
            "adaptive_bounds": cache.get("adaptive_bounds"),
            "mode": cache.get("mode"),
            "output_parameterization": cache.get("output_parameterization"),
            "base_mc_seed": cache.get("base_mc_seed"),
        },
        "baseline": {
            "n_sims": baseline.get("n_sims"),
            "capture_rate": f"{baseline.get('capture_rate', 0):.0%}" if baseline.get("capture_rate") is not None else "n/a",
            "rms_cost": _format_value(baseline.get("rms_cost")),
            "mean_cost": _format_value(baseline.get("mean_cost")),
            "median_cost": _format_value(baseline.get("median_cost")),
            "p95_cost": _format_value(baseline.get("p95_cost")),
            "worst_cost": _format_value(baseline.get("worst_cost")),
        },
        "supervisors": supervisor_rows,
        "n_selected_total": selection.get("n_selected_total"),
        "min_corpus_required": selection.get("min_corpus_required"),
        "eval_summary_lines": _eval_summary_lines(artifacts.get("eval_summary")),
        # Trajectory-comparison manifest (warm_start_compare.py). Empty dict when
        # the comparison didn't run, so the Typst template can `if` on
        # `meta.compare.has_data` without crashing.
        "compare": _build_compare_section(artifacts.get("compare_manifest")),
    }


def _build_compare_section(manifest: dict | None) -> dict[str, Any]:
    """Shape the comparison manifest for Typst consumption.

    Output structure (Typst-friendly: lists instead of mixed dict/list nesting):
      {
        "has_data": bool,
        "primary_supervisor": str,
        "panels": [str, ...],             # 5 panel names
        "side_labels": {sup: str, nn: str},
        "rows": [                         # 4 rows (2 pools x 2 sides)
            {"pool": str, "side": str, "side_label": str, "n_sims": int,
             "n_captured": int, "capture_rate_pct": str,
             "panels": [str, ...]},       # 5 SVG filenames in the same order as `panels`
            ...
        ],
      }
    """
    if not manifest:
        return {"has_data": False, "primary_supervisor": "", "panels": [], "side_labels": {}, "rows": []}
    panels: list[str] = list(manifest.get("panels") or [])
    side_labels: dict[str, str] = dict(manifest.get("side_labels") or {})
    pools = manifest.get("pools") or {}
    rows: list[dict[str, Any]] = []
    for pool_name in ("train", "val"):
        pool_entry = pools.get(pool_name) or {}
        n_sims = int(pool_entry.get("n_sims", 0))
        for side in ("supervisor", "nn"):
            side_entry = (pool_entry.get("sides") or {}).get(side) or {}
            if side_entry.get("error"):
                # Surface the failure in the report rather than silently dropping
                # the row -- helps diagnose missing supervisor params, NN
                # write failures, etc.
                rows.append(
                    {
                        "pool": pool_name,
                        "side": side,
                        "side_label": side_labels.get(side, side),
                        "n_sims": n_sims,
                        "n_captured": 0,
                        "capture_rate_pct": "error",
                        "error": str(side_entry["error"]),
                        "panels": [],
                    }
                )
                continue
            n_captured = int(side_entry.get("n_captured", 0))
            cap_rate = side_entry.get("capture_rate", 0.0) or 0.0
            # Bare filenames; the template prefixes the staging dir.
            panel_files = side_entry.get("panels") or {}
            row_panels = [panel_files.get(p, "") for p in panels]
            rows.append(
                {
                    "pool": pool_name,
                    "side": side,
                    "side_label": side_labels.get(side, side),
                    "n_sims": n_sims,
                    "n_captured": n_captured,
                    "capture_rate_pct": f"{cap_rate:.1%}",
                    "panels": row_panels,
                }
            )
    return {
        "has_data": bool(rows),
        "primary_supervisor": str(manifest.get("primary_supervisor", "")),
        "panels": panels,
        "side_labels": side_labels,
        "rows": rows,
    }


def _eval_summary_lines(eval_summary: dict | None) -> list[str]:
    """Render the structured eval-summary dict as a list of text lines for
    the Typst template's raw-text block. Returns [] when no summary present."""
    if not eval_summary:
        return []
    from aerocapture.training.report import format_eval_summary

    return format_eval_summary(eval_summary, indent="    ")


def _layer_summary(layer: dict) -> str:
    t = layer.get("type", "?")
    if t == "dense":
        return f"Dense({layer.get('input_size', '?')}->{layer.get('output_size', '?')},{layer.get('activation', '?')})"
    if t == "gru":
        return f"GRU({layer.get('input_size', '?')}->{layer.get('hidden_size', '?')})"
    if t == "lstm":
        return f"LSTM({layer.get('input_size', '?')}->{layer.get('hidden_size', '?')})"
    if t == "window":
        return f"Window({layer.get('input_size', '?')}x{layer.get('n_steps', '?')})"
    if t == "transformer":
        return f"Transformer(d_model={layer.get('d_model', '?')},n_heads={layer.get('n_heads', '?')})"
    if t == "mamba":
        return f"Mamba(in={layer.get('input_size', '?')},d_state={layer.get('d_state', '?')})"
    return str(t)


def render_report(save_dir: Path) -> Path | None:
    """Generate the warm-start report. Returns the PDF path on success.

    On Typst missing, still writes the SVG charts + metadata.json and returns
    None (charts are usable on their own).
    """
    save_dir = Path(save_dir)
    report_dir = save_dir / _REPORT_SUBDIR
    report_dir.mkdir(parents=True, exist_ok=True)

    artifacts = _load_artifacts(save_dir)

    # Charts
    chart_supervised_mse(artifacts.get("loss") or [], report_dir / "mse_convergence.svg")
    chart_supervisor_selection(artifacts.get("selection") or {}, report_dir / "supervisor_selection.svg")
    chart_bound_widening(artifacts.get("bounds") or [], report_dir / "bound_widening.svg")

    meta = _build_metadata(artifacts, save_dir)
    (report_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    return render_pdf("warm_start_report", report_dir, save_dir / "warm_start_report.pdf", label="warm_start_report")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the warm-start performance snapshot for a scheme directory.")
    parser.add_argument("save_dir", type=Path, help="Training output dir containing warm_start_*.json sidecars.")
    args = parser.parse_args(argv)
    if not args.save_dir.is_dir():
        print(f"ERROR: not a directory: {args.save_dir}", file=sys.stderr)
        return 2
    pdf = render_report(args.save_dir)
    if pdf is not None:
        print(f"Wrote {pdf}")
    return 0


# Re-export so train.py imports stay stable across refactors.
__all__ = ["render_report", "main"]


# Public OrderedDict so callers can introspect which sidecars feed the report.
EXPECTED_SIDECARS: OrderedDict[str, str] = OrderedDict(
    [
        ("warm_start_loss.json", "Per-epoch supervised MSE"),
        ("warm_start_baseline.json", "Validation-pool DV mean/RMS/capture_rate"),
        ("warm_start_bounds.json", "Per-parameter ParamSpec bounds used to encode the chromosome"),
        ("warm_start_selection.json", "Per-supervisor selection counts + capture stats"),
        ("warm_start_cache_key.json", "Config snapshot used as the cache key"),
        ("warm_start_eval_summary.json", "Final-evaluation statistics block (DV / apoapsis / heat-flux)"),
    ]
)


if __name__ == "__main__":
    raise SystemExit(main())
