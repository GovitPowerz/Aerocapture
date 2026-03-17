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


def _read_ref_trajectory_path(toml_path: Path) -> Path | None:
    """Read reference trajectory path from TOML [data] section."""
    from aerocapture.training.toml_utils import load_toml_with_bases

    data = load_toml_with_bases(toml_path)
    ref_path = data.get("data", {}).get("reference_trajectory")
    if ref_path is None or isinstance(ref_path, bool):
        return None
    p = Path(ref_path)
    if not p.is_absolute():
        # Resolve relative to repo root (cwd)
        p = Path(".") / p
    return p if p.exists() else None


def _load_reference_trajectory(path: Path) -> dict[str, npt.NDArray[np.float64]] | None:
    """Load reference trajectory from .dat file.

    Returns dict with keys: energy_MJkg, pdyn_kPa, inclination_deg, bank_deg.
    """
    try:
        data = np.loadtxt(path)
    except Exception:
        return None
    if data.ndim != 2 or data.shape[1] < 7:
        return None
    return {
        "energy_MJkg": data[:, 0],
        "pdyn_kPa": data[:, 1] / 1e3,  # Pa -> kPa
        "inclination_deg": np.degrees(data[:, 4]),  # rad -> deg
        "bank_deg": np.degrees(np.arccos(np.clip(data[:, 6], -1.0, 1.0))),  # cos(bank) -> deg
    }


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
    ref_trajectory_path: Path | None = None,
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

    # Load reference trajectory if path provided
    ref_traj = _load_reference_trajectory(ref_trajectory_path) if ref_trajectory_path is not None else None

    # Build subplot layout
    n_rows = 5  # base rows: 2 dist + 2 dist + entry/exit + DV-vs-error/table
    row_specs: list[list[dict]] = [
        [{"secondary_y": True}, {}],  # Row 1: DV histogram+CDF, individual burns
        [{"secondary_y": True}, {"secondary_y": True}],  # Row 2: apo/peri error
        [{"secondary_y": True}, {}],  # Row 3: incl error, DV vs orbital error
        [{}, {}],  # Row 4: entry conditions, exit conditions
        [{"type": "table", "colspan": 2}, None],  # Row 5: performance table
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
        "",
    ]

    if has_trajectories:
        n_rows += 2
        row_specs.append([{}, {}])  # Row 6: energy-pdyn, energy-incl
        row_specs.append([{}, {}])  # Row 7: energy-bank, empty
        subplot_titles.extend([
            "Energy vs Dynamic Pressure",
            "Energy vs Inclination",
            "Energy vs Bank Angle",
            "",
        ])

    fig = make_subplots(
        rows=n_rows,
        cols=2,
        subplot_titles=subplot_titles[:n_rows * 2],
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

    # Rows 6-7: Corridor panels (if trajectories available)
    if has_trajectories:
        assert trajectories is not None
        _add_corridor_panel(fig, trajectories, captured, ref_traj, "pdyn_kPa", "Dynamic Pressure (kPa)", row=6, col=1)
        _add_corridor_panel(fig, trajectories, captured, ref_traj, "inclination_deg", "Inclination (deg)", row=6, col=2)
        _add_corridor_panel(fig, trajectories, captured, ref_traj, "bank_deg", "Bank Angle (deg)", row=7, col=1)

    fig.update_layout(
        height=400 * n_rows,
        title_text=f"Final Evaluation — {scheme} ({n_captured}/{n_total} captured, {capture_rate:.1f}%)",
        showlegend=True,
    )

    # Write HTML — combine with dispersion grid if available
    has_dispersions = dispersions is not None and dispersions.shape[0] > 0  # type: ignore[union-attr]
    if has_dispersions and n_captured > 0:
        disp_fig = _build_dispersion_grid(dispersions, final_array, captured)  # type: ignore[arg-type]
        main_html = fig.to_html(include_plotlyjs=True, full_html=False)
        disp_html = disp_fig.to_html(include_plotlyjs=False, full_html=False)
        with open(str(output_path), "w") as f:
            f.write(f"<html><body>{main_html}<hr>{disp_html}</body></html>")
    else:
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

    n_total = len(final_array)
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
            rows.append([
                name,
                f"{data.mean():.2f}",
                f"{data.std():.2f}",
                f"{data.min():.2f}",
                *[f"{p:.2f}" for p in pcts],
                f"{data.max():.2f}",
            ])

    # Add capture rate as first row annotation
    rate_str = f"Capture rate: {n_captured}/{n_total} ({n_captured / n_total * 100:.1f}%)" if n_total > 0 else "No simulations"
    rows.insert(0, [rate_str, "", "", "", "", "", "", "", "", ""])

    cells_transposed = list(zip(*rows, strict=False)) if rows else [[] for _ in header]  # type: ignore[misc]
    fig.add_trace(  # type: ignore[attr-defined]
        go.Table(
            header={"values": header, "fill_color": _COLOR_PRIMARY, "font_color": "white", "align": "center"},
            cells={"values": cells_transposed, "align": "center"},
        ),
        row=row,
        col=col,
    )


def _add_corridor_panel(
    fig: object,
    trajectories: list[npt.NDArray[np.float64]],
    captured: npt.NDArray[np.bool_],
    ref_traj: dict[str, npt.NDArray[np.float64]] | None,
    y_key: str,
    y_label: str,
    row: int,
    col: int,
) -> None:
    """Add energy corridor panel with concatenated traces (NOT one trace per sim).

    Concatenates all trajectories into one Scattergl trace with None separators
    to keep HTML manageable for 1000+ sims.
    """
    import plotly.graph_objects as go  # type: ignore[import-untyped]

    # Map y_key to trajectory column index
    y_col_map = {
        "pdyn_kPa": _TRAJ_COL_PDYN,
        "inclination_deg": _TRAJ_COL_INCL,
        "bank_deg": _TRAJ_COL_BANK,
    }
    y_col = y_col_map[y_key]

    # Concatenate captured and hyperbolic trajectories separately with None separators
    for is_captured, color, name in [(True, _COLOR_CAPTURED, "Captured"), (False, _COLOR_HYPERBOLIC, "Hyperbolic")]:
        mask = captured if is_captured else ~captured
        xs: list[float | None] = []
        ys: list[float | None] = []
        for i in np.where(mask)[0]:
            t = np.asarray(trajectories[i])
            if t.ndim != 2 or t.shape[0] == 0:
                continue
            xs.extend(t[:, _TRAJ_COL_ENERGY].tolist())
            ys.extend(t[:, y_col].tolist())
            xs.append(None)
            ys.append(None)
        if xs:
            fig.add_trace(  # type: ignore[attr-defined]
                go.Scattergl(
                    x=xs,
                    y=ys,
                    mode="lines",
                    name=name,
                    line={"color": color, "width": 0.5},
                    opacity=0.3,
                    showlegend=(row == 6 and col == 1),  # legend only on first corridor panel
                ),
                row=row,
                col=col,
            )

    # Envelope: bin captured trajectories by energy, compute min/max y per bin
    if captured.any():
        all_energy: list[float] = []
        all_y: list[float] = []
        for i in np.where(captured)[0]:
            t = np.asarray(trajectories[i])
            if t.ndim != 2 or t.shape[0] == 0:
                continue
            all_energy.extend(t[:, _TRAJ_COL_ENERGY].tolist())
            all_y.extend(t[:, y_col].tolist())

        if all_energy:
            e_arr = np.array(all_energy)
            y_arr = np.array(all_y)
            bins = np.linspace(e_arr.min(), e_arr.max(), 101)
            bin_idx = np.digitize(e_arr, bins) - 1
            bin_idx = np.clip(bin_idx, 0, 99)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            y_min = np.full(100, np.nan)
            y_max = np.full(100, np.nan)
            for b in range(100):
                mask_b = bin_idx == b
                if mask_b.any():
                    y_min[b] = y_arr[mask_b].min()
                    y_max[b] = y_arr[mask_b].max()

            valid = ~np.isnan(y_min)
            if valid.any():
                bc = bin_centers[valid]
                fig.add_trace(  # type: ignore[attr-defined]
                    go.Scatter(
                        x=bc.tolist(),
                        y=y_min[valid].tolist(),
                        mode="lines",
                        line={"color": "rgba(0,0,0,0)"},
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )
                fig.add_trace(  # type: ignore[attr-defined]
                    go.Scatter(
                        x=bc.tolist(),
                        y=y_max[valid].tolist(),
                        mode="lines",
                        fill="tonexty",
                        fillcolor="rgba(33,150,243,0.15)",
                        line={"color": "rgba(0,0,0,0)"},
                        name="Envelope",
                        showlegend=(row == 6 and col == 1),
                    ),
                    row=row,
                    col=col,
                )

    # Reference trajectory overlay
    if ref_traj is not None and y_key in ref_traj:
        fig.add_trace(  # type: ignore[attr-defined]
            go.Scatter(
                x=ref_traj["energy_MJkg"].tolist(),
                y=ref_traj[y_key].tolist(),
                mode="lines",
                name="Reference",
                line={"color": "#F44336", "width": 3, "dash": "dash"},
                showlegend=(row == 6 and col == 1),
            ),
            row=row,
            col=col,
        )

    fig.update_xaxes(title_text="Orbital Energy (MJ/kg)", row=row, col=col)  # type: ignore[attr-defined]
    fig.update_yaxes(title_text=y_label, row=row, col=col)  # type: ignore[attr-defined]


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
        axis_suffix = "" if (i + 1) == 1 else str(i + 1)
        if np.std(x) < 1e-15:
            fig.add_annotation(
                text="Zero variance",
                xref=f"x{axis_suffix} domain",
                yref=f"y{axis_suffix} domain",
                x=0.5,
                y=0.5,
                showarrow=False,
                font={"size": 10, "color": "gray"},
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
        fig.add_annotation(
            text=f"R\u00b2={r_value**2:.3f} p={p_value:.2e}",
            xref=f"x{axis_suffix} domain",
            yref=f"y{axis_suffix} domain",
            x=0.02,
            y=0.98,
            showarrow=False,
            font={"size": 9},
            bgcolor="rgba(255,255,255,0.7)",
        )

        fig.update_xaxes(title_text=f"{label} ({unit})", row=r, col=c)
        fig.update_yaxes(title_text="\u0394V (m/s)", row=r, col=c)

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
    ref_traj_path = _read_ref_trajectory_path(Path(args.toml))

    print(f"Running {args.n_sims}-sim final evaluation for {scheme} (seed={args.seed})...")
    eval_data = run_final_evaluation(cfg, n_sims=args.n_sims, seed=args.seed)

    if eval_data is None:
        print("ERROR: Simulation failed")
        sys.exit(1)

    output_path = scheme_dir / "final_report.html"
    generate_final_report(eval_data, scheme, target_incl, output_path, ref_trajectory_path=ref_traj_path)
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()
