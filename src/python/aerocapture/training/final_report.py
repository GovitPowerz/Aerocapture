"""Final evaluation report — statistical distributions from large-MC re-evaluation.

Usage (standalone):
    uv run python -m aerocapture.training.final_report \\
        training_output/equilibrium_glide/ \\
        --toml configs/training/msr_aller_eqglide_train.toml \\
        --n-sims 1000 --seed 42
"""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from aerocapture.training.config import TrainingConfig

# ---------------------------------------------------------------------------
# FinalEvalData: return type from run_final_evaluation()
# ---------------------------------------------------------------------------
FinalEvalData = namedtuple("FinalEvalData", ["final_array", "trajectories", "dispersions"])

# Array column indices (0-based 52-column format, no sim_number prefix)
_COL_VELOCITY = 3
_COL_FPA = 4
_COL_ENERGY = 7
_COL_ECC = 9
_COL_INCL = 10
_COL_MAX_HEAT_FLUX = 16
_COL_MAX_G_LOAD = 17
_COL_PERI_ERR = 29
_COL_APO_ERR = 30
_COL_DV1 = 37
_COL_DV2 = 38
_COL_DV3 = 39
_COL_DV_TOTAL = 41
_COL_BANK_CONSUMPTION = 45

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

# Trajectory column indices (12-column per-timestep format)
_TRAJ_COL_VELOCITY = 3
_TRAJ_COL_FPA = 4
_TRAJ_COL_ENERGY = 8
_TRAJ_COL_PDYN = 9
_TRAJ_COL_BANK = 10
_TRAJ_COL_INCL = 11

# Dispersion field labels (24 fields)
_DISPERSION_LABELS = [
    ("Entry Altitude", "m"),
    ("Entry Longitude", "rad"),
    ("Entry Latitude", "rad"),
    ("Entry Velocity", "m/s"),
    ("Entry FPA", "rad"),
    ("Entry Azimuth", "rad"),
    ("Density Error", "frac"),
    ("Drag Coeff Error", "frac"),
    ("Lift Coeff Error", "frac"),
    ("Incidence Error", "rad"),
    ("Nav Altitude Error", "m"),
    ("Nav Longitude Error", "rad"),
    ("Nav Latitude Error", "rad"),
    ("Nav Velocity Error", "m/s"),
    ("Nav FPA Error", "rad"),
    ("Nav Azimuth Error", "rad"),
    ("Nav Drag Accel Error", "m/s\u00b2"),
    ("Mass Error", "frac"),
    ("Ref Area Error", "frac"),
    ("Max Bank Rate Error", "frac"),
    ("Pilot Tau Error", "frac"),
    ("Pilot Damping Error", "frac"),
    ("Pilot Frequency Error", "frac"),
    ("Filter Gain Error", "abs"),
]


def _read_target_inclination(toml_path: Path) -> float:
    """Read target inclination from TOML [flight.target_orbit] section."""
    from aerocapture.training.toml_utils import load_toml_with_bases

    data = load_toml_with_bases(toml_path)
    return float(data.get("flight", {}).get("target_orbit", {}).get("inclination", 0.0))




def _patch_toml_for_final_eval(
    base_toml_path: Path,
    n_sims: int,
    seed: int,
) -> Path:
    """Create a temporary TOML with overridden n_sims and mc_seed."""
    import os
    import tempfile

    from aerocapture.training.toml_utils import load_toml_with_bases

    toml_data = load_toml_with_bases(base_toml_path)

    toml_data.setdefault("simulation", {})["n_sims"] = n_sims
    toml_data.setdefault("monte_carlo", {})["seed"] = seed

    from aerocapture.training.evaluate import _write_toml

    fd, path_str = tempfile.mkstemp(suffix=".toml", prefix="final_eval_")
    os.close(fd)
    output_path = Path(path_str)
    _write_toml(toml_data, output_path)
    return output_path


def run_final_evaluation(
    cfg: TrainingConfig,
    n_sims: int = 1000,
    seed: int | None = None,
    cwd: Path | None = None,
) -> FinalEvalData | None:
    """Run large-MC re-evaluation of best solution.

    Patches the TOML config to override n_sims and mc_seed, then runs
    the simulator via PyO3 ``run_mc()`` (returns all n_sims results) or
    subprocess fallback. Returns FinalEvalData with final conditions array
    (n_sims, 52), optional trajectories list, and optional dispersions
    (n_sims, 24). Returns None if the simulation fails.
    """
    from aerocapture.training.evaluate import _HAS_PYO3, _aero_rs

    if cfg.sim.toml_config is None:
        return None

    cwd_path = Path(cwd) if cwd else Path(".")
    base_toml = cwd_path / cfg.sim.toml_config

    patched_toml = _patch_toml_for_final_eval(base_toml, n_sims, 0 if seed is None else seed)
    orig_toml = cfg.sim.toml_config
    try:
        if _HAS_PYO3:
            assert _aero_rs is not None
            toml_path = str(patched_toml.resolve())
            results = _aero_rs.run_mc(toml_path=toml_path, include_trajectories=True)
            return FinalEvalData(
                final_array=results.final_records,
                trajectories=results.trajectories,
                dispersions=results.dispersions,
            )
        else:
            # Subprocess fallback: run_simulation parses all rows from CSV
            from aerocapture.training.evaluate import run_simulation

            cfg.sim.toml_config = str(patched_toml)
            arr = run_simulation(cfg, cwd=cwd)
            return FinalEvalData(final_array=arr, trajectories=None, dispersions=None) if arr is not None else None
    except Exception:
        import traceback

        traceback.print_exc()
        return None
    finally:
        cfg.sim.toml_config = orig_toml
        patched_toml.unlink(missing_ok=True)


def generate_final_report(
    eval_data: FinalEvalData,
    scheme: str,
    target_inclination: float,
    output_path: Path,
    corridor_path: Path | None = None,
) -> Path:
    """Generate self-contained Plotly HTML report with statistical distributions.

    Returns path to generated HTML file.
    Handles 0% capture rate gracefully (empty distribution panels with annotation).
    """
    import plotly.graph_objects as go  # type: ignore[import-untyped]
    from plotly.subplots import make_subplots  # type: ignore[import-untyped]

    final_array = eval_data.final_array
    trajectories = eval_data.trajectories
    dispersions = eval_data.dispersions

    energy = final_array[:, _COL_ENERGY]
    ecc = final_array[:, _COL_ECC]
    captured = (ecc < 1.0) & (energy < 0)
    n_total = len(final_array)
    n_captured = int(captured.sum())
    capture_rate = n_captured / n_total * 100 if n_total > 0 else 0.0

    # Determine whether we have trajectory data for corridor panels
    has_trajectories = trajectories is not None and len(trajectories) > 0 and any(len(t) > 0 for t in trajectories)


    # Build subplot layout
    n_rows = 5  # base rows: 2 dist + 2 dist + entry/exit + DV-vs-error/table
    row_specs: list[list[dict]] = [
        [{"secondary_y": True}, {}],  # Row 1: DV histogram+CDF, individual burns
        [{"secondary_y": True}, {"secondary_y": True}],  # Row 2: apo/peri error
        [{"secondary_y": True}, {}],  # Row 3: incl error, DV vs orbital error
        [{}, {}],  # Row 4: entry conditions, exit conditions
        [{"type": "table", "colspan": 2}, None],  # type: ignore[list-item]  # Row 5: performance table
    ]
    subplot_titles = [
        "Total Delta-V Distribution",
        "Individual Correction Burns",
        "Apoapsis Error (km)",
        "Periapsis Error (km)",
        "Inclination Error (deg)",
        "Delta-V vs Orbital Error",
        "Entry Conditions",
        "Exit Conditions",
        "Performance Summary",
        # No empty slot for the None cell in colspan row — Plotly skips it automatically
    ]

    # Corridor panels are rendered as a static matplotlib PNG (not in the Plotly figure)

    fig = make_subplots(
        rows=n_rows,
        cols=2,
        subplot_titles=subplot_titles,
        specs=row_specs,
    )

    if n_captured == 0:
        # Add "No captured trajectories" annotation to distribution panels
        for row, col in [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]:
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

        # Row 1 left: Total Delta-V histogram + CDF
        _add_hist_cdf(fig, dv_total, "Delta-V (m/s)", _COLOR_PRIMARY, row=1, col=1)

        # Row 1 right: Individual corrections overlaid
        fig.add_trace(go.Histogram(x=dv1, name="dv1 (incl.)", opacity=0.5, marker_color=_COLOR_DV1, nbinsx=30), row=1, col=2)
        fig.add_trace(go.Histogram(x=dv2, name="dv2 (SMA/ecc)", opacity=0.5, marker_color=_COLOR_DV2, nbinsx=30), row=1, col=2)
        fig.add_trace(go.Histogram(x=dv3, name="dv3 (RAAN)", opacity=0.5, marker_color=_COLOR_DV3, nbinsx=30), row=1, col=2)
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title_text="m/s", row=1, col=2)

        # Row 2 left: Apoapsis error
        _add_hist_cdf(fig, apo_err, "km", _COLOR_PRIMARY, row=2, col=1)

        # Row 2 right: Periapsis error
        _add_hist_cdf(fig, peri_err, "km", _COLOR_SECONDARY, row=2, col=2)

        # Row 3 left: Inclination error
        _add_hist_cdf(fig, incl_err, "deg", _COLOR_TERTIARY, row=3, col=1)

        # Row 3 right: Delta-V vs orbital error scatter (captured only)
        orbital_err = np.sqrt(cap[:, _COL_APO_ERR] ** 2 + cap[:, _COL_PERI_ERR] ** 2)
        fig.add_trace(
            go.Scatter(
                x=orbital_err,
                y=cap[:, _COL_DV_TOTAL],
                mode="markers",
                name="DV vs Error",
                marker={"color": _COLOR_PRIMARY, "opacity": 0.5},
            ),
            row=3,
            col=2,
        )
    fig.update_xaxes(title_text="Orbital Error (km)", row=3, col=2)
    fig.update_yaxes(title_text="Delta-V (m/s)", row=3, col=2)

    # Row 4 left: Entry Conditions (from trajectory initial state, if available)
    if has_trajectories:
        assert trajectories is not None
        entry_v = []
        entry_fpa = []
        entry_captured_mask = []
        for i, t in enumerate(trajectories):
            t_arr = np.asarray(t)
            if t_arr.ndim == 2 and t_arr.shape[0] > 0:
                entry_v.append(t_arr[0, _TRAJ_COL_VELOCITY])
                entry_fpa.append(t_arr[0, _TRAJ_COL_FPA])
                entry_captured_mask.append(bool(captured[i]))
        if entry_v:
            entry_v_arr = np.array(entry_v)
            entry_fpa_arr = np.array(entry_fpa)
            entry_cap = np.array(entry_captured_mask)
            if entry_cap.any():
                fig.add_trace(
                    go.Scatter(
                        x=entry_v_arr[entry_cap],
                        y=entry_fpa_arr[entry_cap],
                        mode="markers",
                        name="Captured (entry)",
                        marker={"color": _COLOR_CAPTURED, "size": 5, "opacity": 0.6},
                    ),
                    row=4,
                    col=1,
                )
            if (~entry_cap).any():
                fig.add_trace(
                    go.Scatter(
                        x=entry_v_arr[~entry_cap],
                        y=entry_fpa_arr[~entry_cap],
                        mode="markers",
                        name="Hyperbolic (entry)",
                        marker={"color": _COLOR_HYPERBOLIC, "size": 5, "opacity": 0.6, "symbol": "x"},
                    ),
                    row=4,
                    col=1,
                )
        fig.update_xaxes(title_text="Entry Velocity (m/s)", row=4, col=1)
        fig.update_yaxes(title_text="Entry FPA (deg)", row=4, col=1)

    # Row 4 right: Exit Conditions (final_record state, colored by outcome)
    velocity = final_array[:, _COL_VELOCITY]
    fpa = final_array[:, _COL_FPA]
    dv_all = final_array[:, _COL_DV_TOTAL]

    if n_captured > 0:
        fig.add_trace(
            go.Scatter(
                x=velocity[captured],
                y=fpa[captured],
                mode="markers",
                name="Captured (exit)",
                marker={"color": _COLOR_CAPTURED, "size": np.clip(dv_all[captured] / 20, 3, 15), "opacity": 0.6},
            ),
            row=4,
            col=2,
        )
    hyper = ~captured
    if hyper.any():
        fig.add_trace(
            go.Scatter(
                x=velocity[hyper],
                y=fpa[hyper],
                mode="markers",
                name="Hyperbolic (exit)",
                marker={"color": _COLOR_HYPERBOLIC, "size": 5, "opacity": 0.6, "symbol": "x"},
            ),
            row=4,
            col=2,
        )
    fig.update_xaxes(title_text="Exit Velocity (m/s)", row=4, col=2)
    fig.update_yaxes(title_text="Exit FPA (deg)", row=4, col=2)

    # Row 5: Performance summary table (colspan 2)
    _add_performance_table(fig, final_array, captured, target_inclination, row=5, col=1)

    # Corridor panels: static matplotlib PNG (lighter than interactive Plotly)
    if has_trajectories:
        assert trajectories is not None
        corridor_png = output_path.with_name(output_path.stem + "_corridors.png")
        dv_cap = final_array[captured, _COL_DV_TOTAL] if n_captured > 0 else None
        # Load pre-computed corridor boundaries if available
        corridor_data: dict[str, npt.NDArray[np.float64]] | None = None
        if corridor_path is not None:
            from aerocapture.training.corridor import load_corridor

            corridor_data = load_corridor(corridor_path)
            if corridor_data is not None:
                print(f"  Loaded corridor boundaries from {corridor_path}")
        _generate_corridor_png(trajectories, captured, corridor_png, dv_captured=dv_cap, corridor_data=corridor_data, final_array=final_array)
        print(f"Corridor plots saved to {corridor_png}")

    fig.update_layout(
        height=400 * n_rows,
        title_text=f"Final Evaluation — {scheme} ({n_captured}/{n_total} captured, {capture_rate:.1f}%)",
        showlegend=True,
    )

    fig.write_html(str(output_path), include_plotlyjs=True)

    # Write dispersion grid as a separate HTML file for memory efficiency
    has_dispersions = dispersions is not None and dispersions.shape[0] > 0  # type: ignore[union-attr]
    if has_dispersions and n_captured > 0:
        disp_fig = _build_dispersion_grid(dispersions, final_array, captured)  # type: ignore[arg-type]
        disp_path = output_path.with_name(output_path.stem + "_dispersions.html")
        disp_fig.write_html(str(disp_path), include_plotlyjs=True)  # type: ignore[attr-defined]
        print(f"Dispersion correlations saved to {disp_path}")

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

    fig.add_trace(go.Histogram(x=data, name=xaxis_label, marker_color=color, opacity=0.7, nbinsx=40, showlegend=False), row=row, col=col)  # type: ignore[attr-defined]

    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    fig.add_trace(go.Scatter(x=sorted_data, y=cdf, name="CDF", line={"color": _COLOR_CDF, "width": 2}, showlegend=False), row=row, col=col, secondary_y=True)  # type: ignore[attr-defined]

    # Percentile lines
    for p in _PERCENTILES:
        val = float(np.percentile(data, p))
        fig.add_vline(x=val, line_dash="dot", line_color="gray", opacity=0.5, row=row, col=col, annotation_text=f"p{p}")  # type: ignore[attr-defined]

    fig.update_xaxes(title_text=xaxis_label, row=row, col=col)  # type: ignore[attr-defined]
    fig.update_yaxes(title_text="Count", row=row, col=col, secondary_y=False)  # type: ignore[attr-defined]
    fig.update_yaxes(title_text="CDF", row=row, col=col, secondary_y=True)  # type: ignore[attr-defined]


def _add_performance_table(
    fig: object,
    final_array: npt.NDArray[np.float64],
    captured: npt.NDArray[np.bool_],
    target_inclination: float,
    row: int,
    col: int,
) -> None:
    """Add detailed performance statistics table to a subplot."""
    import plotly.graph_objects as go  # type: ignore[import-untyped]

    n_captured = int(captured.sum())

    header = ["Parameter", "Mean", "Std", "Min", "p5", "p25", "p50", "p75", "p95", "Max"]
    rows: list[list[str]] = []

    if n_captured > 0:
        cap = final_array[captured]
        metrics = {
            "Max g-load (g)": cap[:, _COL_MAX_G_LOAD],
            "Max heat flux (kW/m\u00b2)": cap[:, _COL_MAX_HEAT_FLUX],
            "Bank angle consumption (deg)": cap[:, _COL_BANK_CONSUMPTION],
            "Apoapsis error (km)": cap[:, _COL_APO_ERR],
            "Periapsis error (km)": cap[:, _COL_PERI_ERR],
            "Inclination error (deg)": cap[:, _COL_INCL] - target_inclination,
            "Correction cost \u0394V (m/s)": cap[:, _COL_DV_TOTAL],
        }
        for name, data in metrics.items():
            pcts = np.percentile(data, _PERCENTILES)
            rows.append(
                [
                    name,
                    f"{data.mean():.2f}",
                    f"{data.std():.2f}",
                    f"{data.min():.2f}",
                    *[f"{p:.2f}" for p in pcts],
                    f"{data.max():.2f}",
                ]
            )

    cells_transposed = list(zip(*rows, strict=False)) if rows else [[] for _ in header]  # type: ignore[misc]
    fig.add_trace(  # type: ignore[attr-defined]
        go.Table(
            header={"values": header, "fill_color": _COLOR_PRIMARY, "font_color": "white", "align": "center"},
            cells={"values": cells_transposed, "align": "center"},
        ),
        row=row,
        col=col,
    )


def _compute_envelope(
    trajectories: list[npt.NDArray[np.float64]],
    mask: npt.NDArray[np.bool_],
    y_col: int,
    n_bins: int = 100,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Bin trajectories by energy and compute min/max y per bin.

    Returns (bin_centers, y_lo, y_hi, valid_mask).
    """
    all_e: list[float] = []
    all_y: list[float] = []
    for i in np.where(mask)[0]:
        t = np.asarray(trajectories[i])
        if t.ndim != 2 or t.shape[0] == 0:
            continue
        all_e.extend(t[:, _TRAJ_COL_ENERGY].tolist())
        all_y.extend(t[:, y_col].tolist())
    if not all_e:
        empty = np.array([])
        return empty, empty, empty, np.array([], dtype=bool)
    e_arr = np.array(all_e)
    y_arr = np.array(all_y)
    bins = np.linspace(e_arr.min(), e_arr.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(e_arr, bins) - 1, 0, n_bins - 1)
    bc = (bins[:-1] + bins[1:]) / 2
    y_lo = np.full(n_bins, np.nan)
    y_hi = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = bin_idx == b
        if m.any():
            y_lo[b] = y_arr[m].min()
            y_hi[b] = y_arr[m].max()
    valid = ~np.isnan(y_lo)
    return bc, y_lo, y_hi, valid


_COLOR_CRASH = "#E57373"  # light red for crash zone
_COLOR_HYPERBOLIC = "#90A4AE"  # blue-grey for hyperbolic exit zone


def _draw_pdyn_zones(
    ax: Any,  # matplotlib Axes
    trajectories: list[npt.NDArray[np.float64]],
    captured: npt.NDArray[np.bool_],
    corridor_data: dict[str, npt.NDArray[np.float64]] | None,
) -> None:
    """Draw crash (upper) and hyperbolic exit (lower) zones on the pdyn panel.

    Uses corridor MC captured trajectories for the envelope when available,
    otherwise falls back to the final-evaluation MC captured envelope.
    Crash zone (above envelope) and hyperbolic zone (below) use distinct colors.
    """
    # Add 30% headroom above the data so the crash zone is clearly visible
    y_data_max = ax.get_ylim()[1]
    y_axis_max = y_data_max * 1.3
    ax.set_ylim(bottom=0, top=y_axis_max)

    # Determine which trajectory set to use for the envelope
    if corridor_data is not None and "traj_lengths" in corridor_data:
        from aerocapture.training.corridor import _unpack_trajectories

        corr_trajs = _unpack_trajectories(corridor_data)
        if corr_trajs:
            all_mask = np.ones(len(corr_trajs), dtype=bool)
            bc, y_lo, y_hi, valid = _compute_envelope(corr_trajs, all_mask, _TRAJ_COL_PDYN)
        else:
            bc, y_lo, y_hi, valid = _compute_envelope(trajectories, captured, _TRAJ_COL_PDYN)
    elif captured.any():
        bc, y_lo, y_hi, valid = _compute_envelope(trajectories, captured, _TRAJ_COL_PDYN)
    else:
        return

    if not valid.any():
        return

    x_lo_ax, x_hi_ax = ax.get_xlim()

    # Split the full plot background: crash (upper half) + hyperbolic (lower half)
    # at the envelope's mid-height, then carve out the corridor
    y_mid = (y_hi[valid].max() + y_lo[valid].min()) / 2 if valid.any() else y_axis_max / 2
    ax.axhspan(y_mid, y_axis_max, color=_COLOR_CRASH, alpha=0.4, zorder=4)
    ax.axhspan(0, y_mid, color=_COLOR_HYPERBOLIC, alpha=0.4, zorder=4)

    # Carve out the corridor: white fill then light blue
    ax.fill_between(bc[valid], y_lo[valid], y_hi[valid], color="white", zorder=4.1)
    ax.fill_between(bc[valid], y_lo[valid], y_hi[valid], color="#2196F3", alpha=0.10, zorder=4.2)

    # Annotations — on top of everything
    mid_e = (x_lo_ax + x_hi_ax) / 2
    ax.text(mid_e, y_axis_max * 0.90, "Crash", ha="center", fontsize=10, fontstyle="italic", color="#B71C1C", zorder=6)
    ax.text(mid_e, y_axis_max * 0.02, "Hyperbolic exit", ha="center", fontsize=10, fontstyle="italic", color="#37474F", zorder=6)
    ax.text(x_hi_ax * 0.9, y_axis_max * 0.02, "Entry", fontsize=8, color="#37474F", ha="right", zorder=6)
    ax.text(x_lo_ax * 0.9, y_axis_max * 0.02, "Atm. exit", fontsize=8, color="#37474F", ha="left", zorder=6)


def _generate_corridor_png(
    trajectories: list[npt.NDArray[np.float64]],
    captured: npt.NDArray[np.bool_],
    output_path: Path,
    dv_captured: npt.NDArray[np.float64] | None = None,
    corridor_data: dict[str, npt.NDArray[np.float64]] | None = None,
    final_array: npt.NDArray[np.float64] | None = None,
) -> None:
    """Generate publication-quality corridor plots as a 2×2 matplotlib PNG.

    Panels: (a) energy vs pdyn with crash/hyperbolic zones,
    (b) energy vs inclination, (c) energy vs bank angle,
    (d) correction cost distribution (histogram + CDF).

    Both the corridor nominal (optimal constant-bank, red) and the guidance
    scheme nominal (first captured MC sim, green) are overlaid on all panels.
    On panel (d) they appear as vertical dashed lines.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    opacity = max(0.02, min(0.15, 10.0 / max(len(trajectories), 1)))

    # Extract nominal trajectories
    corr_nom: npt.NDArray[np.float64] | None = None
    corr_nom_dv: float | None = None
    if corridor_data is not None:
        _nom = corridor_data.get("nominal", np.array([]))
        if _nom.size > 0 and _nom.ndim == 2:
            corr_nom = _nom
        # Corridor nominal DV stored in captured_final_records is not the nominal itself.
        # We need to store it separately. For now, check if "nominal_dv" was saved.
        _nom_dv = corridor_data.get("nominal_dv", np.array([]))
        if _nom_dv.size > 0:
            corr_nom_dv = float(_nom_dv[0])

    # Guidance nominal: first captured trajectory from final-evaluation MC
    guid_nom: npt.NDArray[np.float64] | None = None
    guid_nom_dv: float | None = None
    if captured.any():
        first_cap = int(np.where(captured)[0][0])
        t = np.asarray(trajectories[first_cap])
        if t.ndim == 2 and t.shape[0] > 0:
            guid_nom = t
        if final_array is not None:
            guid_nom_dv = float(final_array[first_cap, _COL_DV_TOTAL])

    corridor_panels = [
        (axes[0, 0], _TRAJ_COL_PDYN, "Dynamic Pressure (kPa)", "(a)"),
        (axes[0, 1], _TRAJ_COL_INCL, "Inclination (deg)", "(b)"),
        (axes[1, 0], _TRAJ_COL_BANK, "Bank Angle (deg)", "(c)"),
    ]

    for ax, y_col, y_label, panel_label in corridor_panels:
        # MC spaghetti — captured blue, hyperbolic red
        for is_cap, color in [(True, "#2196F3"), (False, "#F44336")]:
            mask = captured if is_cap else ~captured
            for i in np.where(mask)[0]:
                t_arr = np.asarray(trajectories[i])
                if t_arr.ndim != 2 or t_arr.shape[0] == 0:
                    continue
                ax.plot(t_arr[:, _TRAJ_COL_ENERGY], t_arr[:, y_col], color=color, alpha=opacity, linewidth=0.5, zorder=3)

        # Crash / hyperbolic exit zones on pdyn panel (a) — drawn AFTER spaghetti so ylim is set
        if y_col == _TRAJ_COL_PDYN:
            _draw_pdyn_zones(ax, trajectories, captured, corridor_data)
        else:
            # Captured envelope for non-pdyn panels
            if captured.any():
                bc, y_lo, y_hi, valid = _compute_envelope(trajectories, captured, y_col)
                if valid.any():
                    ax.fill_between(bc[valid], y_lo[valid], y_hi[valid], color="#2196F3", alpha=0.15, zorder=2)

        # Corridor nominal (optimal constant-bank) — orange
        if corr_nom is not None:
            ax.plot(corr_nom[:, _TRAJ_COL_ENERGY], corr_nom[:, y_col], color="#D32F2F", linewidth=2, linestyle="-", zorder=5)

        # Guidance scheme nominal — green
        if guid_nom is not None:
            ax.plot(guid_nom[:, _TRAJ_COL_ENERGY], guid_nom[:, y_col], color="#4CAF50", linewidth=2, linestyle="-", zorder=5)

        ax.set_xlabel("Orbital Energy (MJ/kg)")
        ax.set_ylabel(y_label)
        ax.set_title(panel_label)
        ax.grid(True, alpha=0.3)

    # Legend on panel (a)
    legend_elements: list[Any] = [
        Patch(facecolor="#2196F3", alpha=0.4, label="MC captured"),
        Patch(facecolor=_COLOR_CRASH, alpha=0.5, label="Crash"),
        Patch(facecolor=_COLOR_HYPERBOLIC, alpha=0.5, label="Hyperbolic exit"),
    ]
    if corr_nom is not None:
        legend_elements.append(Line2D([0], [0], color="#D32F2F", linewidth=2, label="Nominal (const. bank)"))
    if guid_nom is not None:
        legend_elements.append(Line2D([0], [0], color="#4CAF50", linewidth=2, label="Nominal (guidance)"))
    axes[0, 0].legend(handles=legend_elements, loc="upper left", fontsize=7)

    # Panel (d): Correction cost distribution
    ax_dv = axes[1, 1]
    if dv_captured is not None and len(dv_captured) > 0:
        ax_dv.hist(dv_captured, bins=30, color="#F44336", alpha=0.7, edgecolor="white", density=True)
        # CDF on secondary axis
        ax_cdf = ax_dv.twinx()
        sorted_dv = np.sort(dv_captured)
        cdf = np.arange(1, len(sorted_dv) + 1) / len(sorted_dv)
        ax_cdf.plot(sorted_dv, cdf, color="#2196F3", linewidth=2)
        ax_cdf.set_ylabel("Distribution (-)")
        ax_cdf.set_ylim(0, 1.05)

    # Vertical dashed lines for nominal DV values
    if corr_nom_dv is not None:
        ax_dv.axvline(x=corr_nom_dv, color="#D32F2F", linewidth=2, linestyle="--", label=f"Const. bank: {corr_nom_dv:.0f} m/s")
    if guid_nom_dv is not None:
        ax_dv.axvline(x=guid_nom_dv, color="#4CAF50", linewidth=2, linestyle="--", label=f"Guidance: {guid_nom_dv:.0f} m/s")
    if corr_nom_dv is not None or guid_nom_dv is not None:
        ax_dv.legend(fontsize=7, loc="center right")

    ax_dv.set_xlabel("Correction cost (m/s)")
    ax_dv.set_ylabel("Density")
    ax_dv.set_title("(d)")
    ax_dv.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_dispersion_grid(
    dispersions: npt.NDArray[np.float64],
    final_array: npt.NDArray[np.float64],
    captured: npt.NDArray[np.bool_],
) -> object:
    """Build a separate Plotly figure with dispersion-vs-DV correlation grid."""
    import plotly.graph_objects as go  # type: ignore[import-untyped]
    from plotly.subplots import make_subplots  # type: ignore[import-untyped]
    from scipy.stats import linregress  # type: ignore[import-untyped]

    n_fields = len(_DISPERSION_LABELS)
    n_cols = 4
    n_rows = (n_fields + n_cols - 1) // n_cols

    titles = [f"{label} ({unit})" for label, unit in _DISPERSION_LABELS]
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles)

    cap_disp = dispersions[captured]
    cap_dv = final_array[captured, _COL_DV_TOTAL]

    for i, (label, unit) in enumerate(_DISPERSION_LABELS):
        r = i // n_cols + 1
        c = i % n_cols + 1

        x = cap_disp[:, i]
        # Skip fields with zero variance
        if np.std(x) < 1e-15:
            fig.add_annotation(  # type: ignore[attr-defined]
                text="Zero variance",
                x=0.5,
                y=0.5,
                showarrow=False,
                font={"size": 10, "color": "gray"},
                row=r,
                col=c,
            )
            continue

        fig.add_trace(
            go.Scattergl(
                x=x,
                y=cap_dv,
                mode="markers",
                marker={"size": 3, "color": _COLOR_PRIMARY, "opacity": 0.4},
                showlegend=False,
            ),
            row=r,
            col=c,
        )

        # Regression line + R^2
        slope, intercept, r_value, p_value, _ = linregress(x, cap_dv)
        x_range = np.array([x.min(), x.max()])
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=slope * x_range + intercept,
                mode="lines",
                line={"color": "#F44336", "width": 2},
                showlegend=False,
            ),
            row=r,
            col=c,
        )

        # Annotation with R^2 and p-value
        fig.add_annotation(  # type: ignore[attr-defined]
            text=f"R\u00b2={r_value**2:.3f} p={p_value:.2e}",
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            font={"size": 9},
            bgcolor="rgba(255,255,255,0.7)",
            row=r,
            col=c,
        )

        # Tighten axes around data with a small margin
        x_margin = (x.max() - x.min()) * 0.05 or 1.0
        y_margin = (cap_dv.max() - cap_dv.min()) * 0.05 or 1.0
        fig.update_xaxes(title_text=f"{label} ({unit})", range=[x.min() - x_margin, x.max() + x_margin], row=r, col=c)
        fig.update_yaxes(title_text="\u0394V (m/s)", range=[cap_dv.min() - y_margin, cap_dv.max() + y_margin], row=r, col=c)

    fig.update_layout(
        height=300 * n_rows,
        title_text="Dispersion Correlation Grid",
        showlegend=False,
    )

    return fig


def main() -> None:
    """CLI entry point for standalone final evaluation."""
    import argparse
    import json
    import sys

    from aerocapture.training.evaluate import write_guidance_toml

    parser = argparse.ArgumentParser(description="Run final evaluation and generate report")
    parser.add_argument("scheme_dir", type=str, help="Path to scheme output directory (contains best_params.json or best_model.json)")
    parser.add_argument("--toml", type=str, required=True, help="Base TOML config path")
    parser.add_argument("--n-sims", type=int, default=1000, help="Number of MC simulations (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="MC seed for re-evaluation")
    parser.add_argument("--corridor", type=str, default=None, help="Path to pre-computed corridor boundaries (.npz)")
    args = parser.parse_args()

    scheme_dir = Path(args.scheme_dir)
    if not scheme_dir.exists():
        print(f"ERROR: Directory not found: {scheme_dir}")
        sys.exit(1)

    scheme = scheme_dir.name

    params_path = scheme_dir / "best_params.json"
    model_path = scheme_dir / "best_model.json"

    cfg = TrainingConfig()
    cfg.sim.toml_config = args.toml
    cfg.sim.executable = "src/rust/target/release/aerocapture"
    cfg.guidance_type = scheme

    if params_path.exists():
        with open(params_path) as f:
            params = json.load(f)
        opt_toml = scheme_dir / f"optimized_{scheme}.toml"
        if not opt_toml.exists():
            base_toml = Path(args.toml)
            write_guidance_toml(base_toml, scheme, params, opt_toml)
        cfg.sim.toml_config = str(opt_toml)
    elif model_path.exists():
        from aerocapture.training.toml_utils import load_toml_with_bases

        toml_data = load_toml_with_bases(Path(args.toml))
        cfg.sim.nn_param_file = toml_data.get("data", {}).get("neural_network", "data/neural_network/nn_model.json")
    else:
        print(f"ERROR: No best_params.json or best_model.json found in {scheme_dir}")
        sys.exit(1)

    target_incl = _read_target_inclination(Path(args.toml))

    print(f"Running {args.n_sims}-sim final evaluation for {scheme} (seed={args.seed})...")
    eval_data = run_final_evaluation(cfg, n_sims=args.n_sims, seed=args.seed)

    if eval_data is None:
        print("ERROR: Simulation failed")
        sys.exit(1)

    output_path = scheme_dir / "final_report.html"
    corr_path = Path(args.corridor) if args.corridor else None
    generate_final_report(eval_data, scheme, target_incl, output_path, corridor_path=corr_path)
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()
