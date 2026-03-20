"""Generate self-contained Plotly HTML reports from training JSONL logs.

Usage:
    uv run python -m aerocapture.training.report training_output/equilibrium_glide/
    uv run python -m aerocapture.training.report --compare training_output/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from aerocapture.training.metrics import convergence_speed, stagnation_count


def load_run_data(scheme_dir: Path) -> tuple[list[dict], list[int]]:
    """Load all JSONL records from a scheme directory, sorted by generation.

    Returns:
        Tuple of (records, resume_generations) where resume_generations
        contains the first generation number from each JSONL file after
        the first (i.e., where training was resumed).
    """
    file_records: list[list[dict]] = []
    for jsonl_file in sorted(scheme_dir.glob("*.jsonl")):
        file_recs: list[dict] = []
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    file_recs.append(json.loads(line))
        if file_recs:
            file_records.append(file_recs)

    records: list[dict] = []
    for file_recs in file_records:
        records.extend(file_recs)
    records.sort(key=lambda r: r["generation"])

    # Deduplicate: last-writer-wins for same generation (safety net for legacy logs)
    seen: dict[int, int] = {}
    deduped: list[dict] = []
    for r in records:
        gen = r["generation"]
        if gen in seen:
            deduped[seen[gen]] = r
        else:
            seen[gen] = len(deduped)
            deduped.append(r)

    # Detect resume points: first generation of each file after the first
    resume_gens: list[int] = []
    for file_recs in file_records[1:]:
        if file_recs:
            first_gen = min(r["generation"] for r in file_recs)
            if first_gen not in resume_gens:
                resume_gens.append(first_gen)
    resume_gens.sort()

    return deduped, resume_gens


def _add_resume_markers(
    fig: object,
    resume_gens: list[int],
    n_rows: int,
    n_cols: int,
    skip_panels: set[tuple[int, int]] | None = None,
) -> None:
    """Add vertical dashed lines at resume points across all subplots.

    Args:
        skip_panels: Set of (row, col) tuples to skip (e.g., categorical x-axis panels).
    """
    skip = skip_panels or set()
    first_marker = True
    for gen in resume_gens:
        for row in range(1, n_rows + 1):
            for col in range(1, n_cols + 1):
                if (row, col) in skip:
                    continue
                kwargs: dict[str, object] = {
                    "x": gen,
                    "line_dash": "dash",
                    "line_color": "rgba(128, 128, 128, 0.5)",
                    "row": row,
                    "col": col,
                }
                if first_marker:
                    kwargs["annotation_text"] = "resumed"
                    kwargs["annotation_font_color"] = "gray"
                    first_marker = False
                fig.add_vline(**kwargs)  # type: ignore[attr-defined]


def generate_single_report(scheme_dir: Path) -> None:
    """Generate a single-run HTML report from JSONL data."""
    import plotly.graph_objects as go  # type: ignore[import-untyped]
    from plotly.subplots import make_subplots  # type: ignore[import-untyped]

    data, resume_gens = load_run_data(scheme_dir)
    if not data:
        print(f"No JSONL data found in {scheme_dir}")
        return

    gens = [r["generation"] for r in data]
    best_costs = [r["best_cost"] for r in data]
    mean_costs = [r["mean_cost"] for r in data]
    worst_costs = [r["worst_cost"] for r in data]
    cap_rates = [r["capture_rate"] * 100 for r in data]
    diversities = [r["population_diversity"] for r in data]

    scheme = data[0].get("scheme", scheme_dir.name)

    # Detect conditional panels
    has_pool_metrics = any(r.get("pool_metrics") for r in data)
    has_mc_seed = any(r.get("mc_seed") is not None for r in data)

    # Build panel list: (title, specs_dict)
    panels: list[tuple[str, dict]] = [
        ("Convergence (log scale)", {}),
        ("Population Diversity vs Best Cost", {"secondary_y": True}),
        ("Capture Rate (%)", {}),
        ("Cost Distribution", {}),
        ("Parameter Evolution", {}),
    ]
    if has_pool_metrics:
        panels.append(("Seed Pool Evolution", {"secondary_y": True}))
    if has_mc_seed:
        panels.append(("MC Seed Trace", {}))
    panels.append(("Summary", {}))

    n_cols = 2
    n_rows = (len(panels) + 1) // 2
    subplot_titles = [p[0] for p in panels]
    specs: list[list[dict]] = []
    for row_start in range(0, len(panels), n_cols):
        row_specs = [panels[i][1] if i < len(panels) else {} for i in range(row_start, row_start + n_cols)]
        specs.append(row_specs)

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles, specs=specs)

    # Panel position lookup: maps 0-indexed panel to (row, col)
    panel_positions = [(i // n_cols + 1, i % n_cols + 1) for i in range(len(panels))]

    # 1. Convergence (row=1, col=1)
    fig.add_trace(go.Scatter(x=gens, y=best_costs, name="Best", line={"color": "#2196F3"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=gens, y=mean_costs, name="Mean", line={"color": "#FF9800", "dash": "dash"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=gens, y=worst_costs, name="Worst", line={"color": "#F44336", "dash": "dot"}), row=1, col=1)
    imp_gens = [r["generation"] for r in data if r["improvement"]]
    imp_costs = [r["best_cost"] for r in data if r["improvement"]]
    fig.add_trace(go.Scatter(x=imp_gens, y=imp_costs, mode="markers", name="Improvement", marker={"color": "#4CAF50", "size": 6}), row=1, col=1)
    fig.update_yaxes(type="log", title_text="Cost", row=1, col=1)

    # 2. Diversity + best cost overlay (row=1, col=2)
    fig.add_trace(go.Scatter(x=gens, y=diversities, name="Diversity", line={"color": "#9C27B0"}), row=1, col=2, secondary_y=False)
    fig.add_trace(go.Scatter(x=gens, y=best_costs, name="Best Cost", line={"color": "#2196F3", "dash": "dot"}), row=1, col=2, secondary_y=True)
    fig.update_yaxes(title_text="Diversity", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Best Cost", type="log", row=1, col=2, secondary_y=True)

    # 3. Capture rate (row=2, col=1)
    fig.add_trace(go.Scatter(x=gens, y=cap_rates, name="Capture %", line={"color": "#4CAF50"}, fill="tozeroy"), row=2, col=1)
    fig.update_yaxes(title_text="Capture Rate (%)", range=[0, 105], row=2, col=1)

    # 4. Cost distribution (row=2, col=2)
    n_boxes = min(10, len(data))
    step = max(1, len(data) // n_boxes)
    for i in range(0, len(data), step):
        r = data[i]
        fig.add_trace(
            go.Box(y=[r["best_cost"], r["median_cost"], r["mean_cost"], r["worst_cost"]], name=f"Gen {r['generation']}", showlegend=False),
            row=2,
            col=2,
        )
    fig.update_yaxes(type="log", title_text="Cost", row=2, col=2)

    # 5. Parameter evolution (row=3, col=1)
    first_params = data[0].get("best_params")
    if first_params is not None:
        for param_name in first_params:
            vals = [r["best_params"][param_name] for r in data if r.get("best_params")]
            param_gens = [r["generation"] for r in data if r.get("best_params")]
            fig.add_trace(go.Scatter(x=param_gens, y=vals, name=param_name), row=3, col=1)
    fig.update_yaxes(title_text="Parameter Value", row=3, col=1)

    # Conditional panels — look up positions by title
    pool_pos = next((panel_positions[i] for i, (t, _) in enumerate(panels) if t == "Seed Pool Evolution"), None)
    seed_pos = next((panel_positions[i] for i, (t, _) in enumerate(panels) if t == "MC Seed Trace"), None)
    summary_pos = next((panel_positions[i] for i, (t, _) in enumerate(panels) if t == "Summary"), None)

    if has_pool_metrics and pool_pos:
        p_row, p_col = pool_pos
        pool_gens = [r["generation"] for r in data if r.get("pool_metrics")]
        pool_sizes = [r["pool_metrics"]["pool_size"] for r in data if r.get("pool_metrics")]
        diff_mins = [r["pool_metrics"]["difficulty_min"] for r in data if r.get("pool_metrics")]
        diff_maxs = [r["pool_metrics"]["difficulty_max"] for r in data if r.get("pool_metrics")]
        fig.add_trace(go.Scatter(x=pool_gens, y=pool_sizes, name="Pool Size", line={"color": "#2196F3"}), row=p_row, col=p_col, secondary_y=False)
        fig.add_trace(
            go.Scatter(x=pool_gens, y=diff_maxs, name="Diff. Max", line={"color": "#FF9800", "dash": "dot"}, fill=None),
            row=p_row,
            col=p_col,
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(x=pool_gens, y=diff_mins, name="Diff. Min", line={"color": "#FF9800", "dash": "dot"}, fill="tonexty"),
            row=p_row,
            col=p_col,
            secondary_y=True,
        )
        fig.update_yaxes(title_text="Pool Size", row=p_row, col=p_col, secondary_y=False)
        fig.update_yaxes(title_text="Difficulty", row=p_row, col=p_col, secondary_y=True)

    if has_mc_seed and seed_pos:
        p_row, p_col = seed_pos
        seed_gens = [r["generation"] for r in data if r.get("mc_seed") is not None]
        seed_vals = [r["mc_seed"] for r in data if r.get("mc_seed") is not None]
        fig.add_trace(
            go.Scatter(x=seed_gens, y=seed_vals, name="MC Seed", mode="lines+markers", line={"color": "#795548"}, marker={"size": 4}), row=p_row, col=p_col
        )
        fig.update_yaxes(title_text="MC Seed", row=p_row, col=p_col)

    # Summary panel (always last)
    assert summary_pos is not None
    summary_row, summary_col = summary_pos

    cost_history = [r["best_cost"] for r in data]
    conv_speed = convergence_speed(cost_history)
    stag = stagnation_count(cost_history)
    config_hash = data[0].get("config_hash", "N/A")

    summary_text = (
        f"Scheme: {scheme}<br>"
        f"Final best cost: {best_costs[-1]:.4e}<br>"
        f"Total generations: {len(data)}<br>"
        f"Convergence speed (90%): gen {conv_speed}<br>"
        f"Final stagnation: {stag} gens<br>"
        f"Config hash: {config_hash}"
    )
    if resume_gens:
        summary_text += f"<br>Resume points: {len(resume_gens)}"

    fig.add_annotation(
        text=summary_text,
        xref="x domain",
        yref="y domain",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 12},
        align="left",
        row=summary_row,
        col=summary_col,
    )

    # Resume markers on all panels (skip categorical/non-numeric x-axis panels)
    skip = {(2, 2), (summary_row, summary_col)}  # Cost Distribution (boxplot) + Summary
    _add_resume_markers(fig, resume_gens, n_rows, n_cols, skip_panels=skip)

    fig.update_layout(height=max(1000, n_rows * 350), title_text=f"Training Report — {scheme}", showlegend=True)
    fig.update_xaxes(title_text="Generation", row=n_rows, col=1)

    output_path = scheme_dir / "report.html"
    fig.write_html(str(output_path), include_plotlyjs=True)
    print(f"Report saved to {output_path}")


_SCHEME_LABELS = {
    "ftc": "FTC",
    "neural_network": "Neural Net",
    "equilibrium_glide": "Eq. Glide",
    "energy_controller": "Energy Ctrl",
    "pred_guid": "PredGuid",
    "fnpag": "FNPAG",
}

_SCHEME_COLORS = {
    "ftc": "#2196F3",
    "neural_network": "#FF9800",
    "equilibrium_glide": "#4CAF50",
    "energy_controller": "#9C27B0",
    "pred_guid": "#F44336",
    "fnpag": "#795548",
}


def generate_comparison_report(
    base_dir: Path,
    schemes: list[str] | None = None,
    after: str | None = None,
) -> None:
    """Generate a cross-scheme comparison HTML report."""
    import plotly.graph_objects as go  # type: ignore[import-untyped]
    from plotly.subplots import make_subplots  # type: ignore[import-untyped]

    scheme_dirs = sorted(d for d in base_dir.iterdir() if d.is_dir() and list(d.glob("*.jsonl")))

    if schemes:
        scheme_dirs = [d for d in scheme_dirs if d.name in schemes]

    if not scheme_dirs:
        print(f"No JSONL data found in subdirectories of {base_dir}")
        return

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Cross-Scheme Convergence", "Final Metrics"),
        specs=[[{}], [{"type": "table"}]],
        row_heights=[0.65, 0.35],
    )

    summary_rows: list[list[str]] = []
    all_resume_gens: set[int] = set()

    for scheme_dir in scheme_dirs:
        scheme_name = scheme_dir.name
        data, resume_gens = load_run_data(scheme_dir)
        all_resume_gens.update(resume_gens)
        if not data:
            continue

        # Filter by date if requested
        if after:
            data = [r for r in data if r.get("timestamp", "") >= after]
            if not data:
                continue

        gens = [r["generation"] for r in data]
        best_costs = [r["best_cost"] for r in data]
        color = _SCHEME_COLORS.get(scheme_name, "#666666")
        label = _SCHEME_LABELS.get(scheme_name, scheme_name)

        fig.add_trace(go.Scatter(x=gens, y=best_costs, name=label, line={"color": color}), row=1, col=1)

        cost_history = [r["best_cost"] for r in data]
        conv = convergence_speed(cost_history)
        cap = data[-1].get("capture_rate", 0) * 100

        summary_rows.append([label, f"{best_costs[-1]:.2e}", str(len(data)), f"{cap:.0f}%", str(conv)])

    for gen in sorted(all_resume_gens):
        fig.add_vline(
            x=gen,
            line_dash="dash",
            line_color="rgba(128, 128, 128, 0.5)",
            annotation_text="resumed",
            annotation_font_color="gray",
            row=1,
            col=1,
        )

    fig.update_yaxes(type="log", title_text="Best Cost", row=1, col=1)
    fig.update_xaxes(title_text="Generation", row=1, col=1)

    # Summary table
    header = ["Scheme", "Best Cost", "Generations", "Capture %", "Conv. Speed"]
    fig.add_trace(
        go.Table(
            header={"values": header, "fill_color": "#2196F3", "font_color": "white", "align": "center"},
            cells={"values": list(zip(*summary_rows, strict=False)) if summary_rows else [[] for _ in header], "align": "center"},  # type: ignore[misc]
        ),
        row=2,
        col=1,
    )

    fig.update_layout(height=800, title_text="Training Comparison Report")

    output_path = base_dir / "comparison_report.html"
    fig.write_html(str(output_path), include_plotlyjs=True)
    print(f"Comparison report saved to {output_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate training reports from JSONL logs")
    parser.add_argument("path", type=str, help="Path to scheme directory (single) or training_output/ (comparison)")
    parser.add_argument("--compare", action="store_true", help="Generate cross-scheme comparison report")
    parser.add_argument("--schemes", nargs="*", help="Filter by scheme names (comparison mode)")
    parser.add_argument("--after", type=str, default=None, help="Filter runs after this date (YYYY-MM-DD)")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"ERROR: Path not found: {path}")
        sys.exit(1)

    if args.compare:
        generate_comparison_report(path, schemes=args.schemes, after=args.after)
    else:
        generate_single_report(path)


if __name__ == "__main__":
    main()
