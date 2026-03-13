"""Final evaluation report — statistical distributions from large-MC re-evaluation.

Usage (standalone):
    uv run python -m aerocapture.training.final_report \\
        training_output/equilibrium_glide/ \\
        --toml configs/training/msr_aller_eqglide_train.toml \\
        --n-sims 1000 --seed 42
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

# Legacy array column indices (53-column format from evaluate._parse_final_to_legacy_array)
_COL_VELOCITY = 4
_COL_FPA = 5
_COL_ENERGY = 8
_COL_ECC = 10
_COL_INCL = 11
_COL_PERI_ERR = 30
_COL_APO_ERR = 31
_COL_DV1 = 38
_COL_DV2 = 39
_COL_DV3 = 40
_COL_DV_TOTAL = 42

_PERCENTILES = [5, 25, 50, 75, 95]

# Colors consistent with report.py palette
_COLOR_PRIMARY = "#2196F3"
_COLOR_SECONDARY = "#FF9800"
_COLOR_TERTIARY = "#4CAF50"
_COLOR_DV1 = "#2196F3"
_COLOR_DV2 = "#FF9800"
_COLOR_DV3 = "#4CAF50"
_COLOR_CAPTURED = "#4CAF50"
_COLOR_HYPERBOLIC = "#F44336"
_COLOR_CDF = "#9C27B0"


def generate_final_report(
    final_array: npt.NDArray[np.float64],
    scheme: str,
    target_inclination: float,
    output_path: Path,
) -> Path:
    """Generate self-contained Plotly HTML report with statistical distributions.

    Returns path to generated HTML file.
    Handles 0% capture rate gracefully (empty distribution panels with annotation).
    """
    import plotly.graph_objects as go  # type: ignore[import-untyped]
    from plotly.subplots import make_subplots  # type: ignore[import-untyped]

    energy = final_array[:, _COL_ENERGY]
    ecc = final_array[:, _COL_ECC]
    captured = (ecc < 1.0) & (energy < 0)
    n_total = len(final_array)
    n_captured = int(captured.sum())
    capture_rate = n_captured / n_total * 100 if n_total > 0 else 0.0

    fig = make_subplots(
        rows=4,
        cols=2,
        subplot_titles=(
            "Total Delta-V Distribution",
            "Individual Correction Burns",
            "Apoapsis Error (km)",
            "Periapsis Error (km)",
            "Inclination Error (deg)",
            "Entry Conditions",
            "Delta-V vs Orbital Error",
            "Summary Statistics",
        ),
        specs=[
            [{"secondary_y": True}, {}],
            [{"secondary_y": True}, {"secondary_y": True}],
            [{"secondary_y": True}, {}],
            [{}, {"type": "table"}],
        ],
    )

    if n_captured == 0:
        # Add "No captured trajectories" annotation to all distribution panels
        for row, col in [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (4, 1)]:
            idx = (row - 1) * 2 + col
            axis_suffix = "" if idx == 1 else str(idx)
            fig.add_annotation(
                text="No captured trajectories",
                xref=f"x{axis_suffix} domain",
                yref=f"y{axis_suffix} domain",
                x=0.5,
                y=0.5,
                showarrow=False,
                font={"size": 14, "color": "#F44336"},
            )
    else:
        cap = final_array[captured]
        dv_total = cap[:, _COL_DV_TOTAL]
        dv1 = cap[:, _COL_DV1]
        dv2 = cap[:, _COL_DV2]
        dv3 = cap[:, _COL_DV3]
        apo_err = cap[:, _COL_APO_ERR]
        peri_err = cap[:, _COL_PERI_ERR]
        incl_err = cap[:, _COL_INCL] - target_inclination

        # Panel 1: Total Delta-V histogram + CDF
        _add_hist_cdf(fig, dv_total, "Delta-V (m/s)", _COLOR_PRIMARY, row=1, col=1)

        # Panel 2: Individual corrections overlaid
        fig.add_trace(go.Histogram(x=dv1, name="dv1 (incl.)", opacity=0.5, marker_color=_COLOR_DV1, nbinsx=30), row=1, col=2)
        fig.add_trace(go.Histogram(x=dv2, name="dv2 (SMA/ecc)", opacity=0.5, marker_color=_COLOR_DV2, nbinsx=30), row=1, col=2)
        fig.add_trace(go.Histogram(x=dv3, name="dv3 (RAAN)", opacity=0.5, marker_color=_COLOR_DV3, nbinsx=30), row=1, col=2)
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title_text="m/s", row=1, col=2)

        # Panel 3: Apoapsis error
        _add_hist_cdf(fig, apo_err, "km", _COLOR_PRIMARY, row=2, col=1)

        # Panel 4: Periapsis error
        _add_hist_cdf(fig, peri_err, "km", _COLOR_SECONDARY, row=2, col=2)

        # Panel 5: Inclination error
        _add_hist_cdf(fig, incl_err, "deg", _COLOR_TERTIARY, row=3, col=1)

    # Panel 6: Entry conditions scatter (all trajectories, colored by outcome)
    velocity = final_array[:, _COL_VELOCITY]
    fpa = final_array[:, _COL_FPA]
    dv_all = final_array[:, _COL_DV_TOTAL]

    if n_captured > 0:
        fig.add_trace(
            go.Scatter(
                x=velocity[captured],
                y=fpa[captured],
                mode="markers",
                name="Captured",
                marker={"color": _COLOR_CAPTURED, "size": np.clip(dv_all[captured] / 20, 3, 15), "opacity": 0.6},
            ),
            row=3,
            col=2,
        )
    hyper = ~captured
    if hyper.any():
        fig.add_trace(
            go.Scatter(
                x=velocity[hyper],
                y=fpa[hyper],
                mode="markers",
                name="Hyperbolic",
                marker={"color": _COLOR_HYPERBOLIC, "size": 5, "opacity": 0.6, "symbol": "x"},
            ),
            row=3,
            col=2,
        )
    fig.update_xaxes(title_text="Entry Velocity (m/s)", row=3, col=2)
    fig.update_yaxes(title_text="Entry FPA (deg)", row=3, col=2)

    # Panel 7: Delta-V vs orbital error scatter (captured only)
    if n_captured > 0:
        cap = final_array[captured]
        orbital_err = np.sqrt(cap[:, _COL_APO_ERR] ** 2 + cap[:, _COL_PERI_ERR] ** 2)
        fig.add_trace(
            go.Scatter(
                x=orbital_err,
                y=cap[:, _COL_DV_TOTAL],
                mode="markers",
                name="DV vs Error",
                marker={"color": _COLOR_PRIMARY, "opacity": 0.5},
            ),
            row=4,
            col=1,
        )
    fig.update_xaxes(title_text="Orbital Error (km)", row=4, col=1)
    fig.update_yaxes(title_text="Delta-V (m/s)", row=4, col=1)

    # Panel 8: Summary statistics table
    _add_summary_table(fig, final_array, captured, target_inclination, row=4, col=2)

    fig.update_layout(
        height=1600,
        title_text=f"Final Evaluation — {scheme} ({n_captured}/{n_total} captured, {capture_rate:.1f}%)",
        showlegend=True,
    )

    fig.write_html(str(output_path), include_plotlyjs=True)
    return output_path


def _add_hist_cdf(
    fig: object,
    data: npt.NDArray[np.float64],
    xaxis_label: str,
    color: str,
    row: int,
    col: int,
) -> None:
    """Add histogram + CDF overlay with percentile lines to a subplot."""
    import plotly.graph_objects as go  # type: ignore[import-untyped]

    fig.add_trace(go.Histogram(x=data, name=xaxis_label, marker_color=color, opacity=0.7, nbinsx=40, showlegend=False), row=row, col=col)  # type: ignore[union-attr]

    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    fig.add_trace(go.Scatter(x=sorted_data, y=cdf, name="CDF", line={"color": _COLOR_CDF, "width": 2}, showlegend=False), row=row, col=col, secondary_y=True)  # type: ignore[union-attr]

    # Percentile lines
    for p in _PERCENTILES:
        val = float(np.percentile(data, p))
        fig.add_vline(x=val, line_dash="dot", line_color="gray", opacity=0.5, row=row, col=col, annotation_text=f"p{p}")  # type: ignore[union-attr]

    fig.update_xaxes(title_text=xaxis_label, row=row, col=col)  # type: ignore[union-attr]
    fig.update_yaxes(title_text="Count", row=row, col=col, secondary_y=False)  # type: ignore[union-attr]
    fig.update_yaxes(title_text="CDF", row=row, col=col, secondary_y=True)  # type: ignore[union-attr]


def _add_summary_table(
    fig: object,
    final_array: npt.NDArray[np.float64],
    captured: npt.NDArray[np.bool_],
    target_inclination: float,
    row: int,
    col: int,
) -> None:
    """Add summary statistics table to a subplot."""
    import plotly.graph_objects as go  # type: ignore[import-untyped]

    n_total = len(final_array)
    n_captured = int(captured.sum())

    header = ["Metric", "Mean", "Std", "p5", "p25", "p50", "p75", "p95"]
    rows: list[list[str]] = []

    if n_captured > 0:
        cap = final_array[captured]
        metrics = {
            "Delta-V total (m/s)": cap[:, _COL_DV_TOTAL],
            "dv1 incl. (m/s)": cap[:, _COL_DV1],
            "dv2 SMA/ecc (m/s)": cap[:, _COL_DV2],
            "dv3 RAAN (m/s)": cap[:, _COL_DV3],
            "Apoapsis err (km)": cap[:, _COL_APO_ERR],
            "Periapsis err (km)": cap[:, _COL_PERI_ERR],
            "Inclination err (deg)": cap[:, _COL_INCL] - target_inclination,
        }
        for name, data in metrics.items():
            pcts = np.percentile(data, _PERCENTILES)
            rows.append([name, f"{data.mean():.2f}", f"{data.std():.2f}", *[f"{p:.2f}" for p in pcts]])

    # Add capture rate as first row
    rows.insert(0, [f"Capture rate: {n_captured}/{n_total} ({n_captured / n_total * 100:.1f}%)", "", "", "", "", "", "", ""])

    cells_transposed = list(zip(*rows, strict=False)) if rows else [[] for _ in header]
    fig.add_trace(  # type: ignore[union-attr]
        go.Table(
            header={"values": header, "fill_color": _COLOR_PRIMARY, "font_color": "white", "align": "center"},
            cells={"values": cells_transposed, "align": "center"},
        ),
        row=row,
        col=col,
    )
