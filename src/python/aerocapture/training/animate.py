"""Generate GIF animations of corridor/trajectory evolution during GA training."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

from aerocapture.training.charts import (
    COLOR_CAPTURE,
    COLOR_CONSTRAINED,
    COLOR_HYPERBOLIC,
    COLOR_WORST,
    TRAJ_FAILED,
    _draw_spaghetti,
    classify_trajectories,
)


def _discover_checkpoints(training_dir: Path, every: int) -> list[dict]:
    """Find and load checkpoint metadata, sorted by generation.

    Args:
        training_dir: Path to scheme training output directory.
        every: Sample every Nth checkpoint. Last checkpoint is always included.

    Returns:
        List of dicts with keys: generation, best_cost, cost_history,
        json_path, npz_path, costs (population costs array).
    """
    pattern = re.compile(r"checkpoint_r\d+_g(\d+)\.json$")
    found: list[tuple[int, Path]] = []
    for p in training_dir.glob("checkpoint_r*_g*.json"):
        m = pattern.search(p.name)
        if m:
            found.append((int(m.group(1)), p))

    found.sort(key=lambda x: x[0])
    if not found:
        return []

    # Apply --every filter, always keeping the last checkpoint
    indices = list(range(0, len(found), every))
    if indices[-1] != len(found) - 1:
        indices.append(len(found) - 1)

    result: list[dict] = []
    for idx in indices:
        gen, json_path = found[idx]
        npz_path = json_path.with_suffix(".npz")
        with open(json_path) as f:
            meta = json.load(f)

        # Load population costs from npz (supports both new flat and legacy subpop formats)
        with np.load(npz_path, allow_pickle=False) as data:
            if "costs" in data:
                all_costs = data["costs"]
            else:
                costs_arrays: list[npt.NDArray[np.float64]] = []
                n_subpops = int(data["n_subpops"][0]) if "n_subpops" in data else 1
                for k in range(n_subpops):
                    key = f"costs_{k}"
                    if key in data:
                        costs_arrays.append(data[key])
                all_costs = np.concatenate(costs_arrays) if costs_arrays else np.array([])

            best_chrom = data.get("best_individual", data.get("best_chromosome", None))
            npz_data = dict(data)

        result.append(
            {
                "generation": gen,
                "best_cost": meta["best_cost"],
                "cost_history": meta.get("cost_history", []),
                "json_path": json_path,
                "npz_path": npz_path,
                "costs": all_costs,
                "best_chromosome": best_chrom,
                "npz_data": npz_data,
            }
        )

    return result


def _build_overrides(guidance_type: str, params: dict[str, float], n_sims: int) -> dict[str, object]:
    """Convert decoded parameters to dot-path TOML overrides for run_mc.

    Lateral params (prefixed ``lateral.``) go to ``guidance.lateral.*``.
    Scheme params go to ``guidance.<section>.*``.
    """
    from aerocapture.training.param_spaces import GUIDANCE_TOML_SECTIONS

    section = GUIDANCE_TOML_SECTIONS[guidance_type]
    overrides: dict[str, object] = {}
    for k, v in params.items():
        if k.startswith("lateral."):
            overrides[f"guidance.{k}"] = v
        else:
            overrides[f"guidance.{section}.{k}"] = v
    overrides["guidance.type"] = guidance_type
    overrides["simulation.n_sims"] = n_sims
    return overrides


def _compute_axis_ranges(
    trajectories: list[npt.NDArray[np.float64]],
    costs: npt.NDArray[np.float64],
    margin: float = 0.05,
) -> dict[str, float]:
    """Compute fixed axis ranges from trajectories and costs.

    Adds a relative margin to each dimension so frames don't clip data.
    """
    all_energy = np.concatenate([t[:, 8] for t in trajectories])
    all_pdyn = np.concatenate([t[:, 9] for t in trajectories])
    all_bank = np.concatenate([t[:, 10] for t in trajectories])
    all_incl = np.concatenate([t[:, 11] for t in trajectories])

    def _padded(arr: npt.NDArray[np.float64]) -> tuple[float, float]:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        span = hi - lo if hi > lo else 1.0
        return lo - margin * span, hi + margin * span

    e_min, e_max = _padded(all_energy)
    p_min, p_max = _padded(all_pdyn)
    b_min, b_max = _padded(all_bank)
    i_min, i_max = _padded(all_incl)

    return {
        "energy_min": e_min,
        "energy_max": e_max,
        "pdyn_min": p_min,
        "pdyn_max": p_max,
        "bank_min": b_min,
        "bank_max": b_max,
        "incl_min": i_min,
        "incl_max": i_max,
        "cost_max": float(np.max(costs)) * 1.1,
    }


def _render_corridor_panel(
    ax: plt.Axes,  # type: ignore[name-defined]
    trajectories: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    corridor_data: dict[str, npt.NDArray[np.float64]] | None,
    axis_ranges: dict[str, float],
) -> None:
    """Render the corridor (energy vs dynamic pressure) panel."""
    # Draw corridor zone fills if available
    if corridor_data is not None:
        e_bins = corridor_data["energy_bins"]
        crash_pdyn = corridor_data["envelope_crash_pdyn"]
        restricted_max = corridor_data["envelope_restricted_max_pdyn"]
        restricted_min = corridor_data["envelope_restricted_min_pdyn"]
        capture_pdyn = corridor_data["envelope_capture_pdyn"]

        ax.fill_between(e_bins, restricted_max, crash_pdyn, color=COLOR_WORST, alpha=0.15)
        ax.fill_between(e_bins, restricted_max, restricted_min, color="white", alpha=0.6)
        ax.fill_between(e_bins, restricted_min, capture_pdyn, color="#cccccc", alpha=0.3)
        ax.fill_between(e_bins, capture_pdyn, 0, color=COLOR_WORST, alpha=0.15)

    # col 8 = energy, col 9 = pdyn
    _draw_spaghetti(ax, trajectories, traj_class, x_col=8, y_col=9)
    ax.set_xlim(axis_ranges["energy_min"], axis_ranges["energy_max"])
    ax.set_ylim(axis_ranges["pdyn_min"], axis_ranges["pdyn_max"])
    ax.set_xlabel("Energy (MJ/kg)")
    ax.set_ylabel("Dynamic pressure (kPa)")
    ax.set_title("Corridor")


def _render_inclination_panel(
    ax: plt.Axes,  # type: ignore[name-defined]
    trajectories: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    axis_ranges: dict[str, float],
) -> None:
    """Render the inclination (energy vs inclination) panel."""
    _draw_spaghetti(ax, trajectories, traj_class, x_col=8, y_col=11)
    ax.set_xlim(axis_ranges["energy_min"], axis_ranges["energy_max"])
    ax.set_ylim(axis_ranges["incl_min"], axis_ranges["incl_max"])
    ax.set_xlabel("Energy (MJ/kg)")
    ax.set_ylabel("Inclination (deg)")
    ax.set_title("Inclination")


def _render_bank_panel(
    ax: plt.Axes,  # type: ignore[name-defined]
    trajectories: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    axis_ranges: dict[str, float],
) -> None:
    """Render the bank angle (energy vs bank) panel."""
    _draw_spaghetti(ax, trajectories, traj_class, x_col=8, y_col=10)
    ax.set_xlim(axis_ranges["energy_min"], axis_ranges["energy_max"])
    ax.set_ylim(axis_ranges["bank_min"], axis_ranges["bank_max"])
    ax.set_xlabel("Energy (MJ/kg)")
    ax.set_ylabel("Bank angle (deg)")
    ax.set_title("Bank angle")


def _render_cost_panel(
    ax: plt.Axes,  # type: ignore[name-defined]
    costs: npt.NDArray[np.float64],
    best_cost: float,
    axis_ranges: dict[str, float],
) -> None:
    """Render the cost CDF panel (histogram + cumulative)."""
    finite = costs[np.isfinite(costs)]
    if len(finite) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    ax.hist(finite, bins=30, density=True, alpha=0.5, color=COLOR_CAPTURE, label="Population")

    # ECDF overlay on twin axis
    ax2 = ax.twinx()
    sorted_costs = np.sort(finite)
    ecdf = np.arange(1, len(sorted_costs) + 1) / len(sorted_costs)
    ax2.plot(sorted_costs, ecdf, color=COLOR_CONSTRAINED, linewidth=1.5, label="ECDF")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("CDF")

    # Best cost marker
    ax.axvline(best_cost, color=COLOR_CAPTURE, linestyle="--", linewidth=1, label=f"Best: {best_cost:.1f}")

    # Mean/median annotations
    median = float(np.median(finite))
    mean = float(np.mean(finite))
    ax.axvline(median, color=COLOR_CONSTRAINED, linestyle=":", linewidth=1, label=f"Median: {median:.1f}")
    ax.axvline(mean, color=COLOR_HYPERBOLIC, linestyle="-.", linewidth=1, label=f"Mean: {mean:.1f}")

    ax.set_xlim(0, axis_ranges["cost_max"])
    ax.set_xlabel("Cost")
    ax.set_ylabel("Density")
    ax.set_title("Cost distribution")
    ax.legend(fontsize=7, loc="upper right")


def _render_frame(
    generation: int,
    best_cost: float,
    capture_rate: float,
    trajectories: list[npt.NDArray[np.float64]],
    traj_class: npt.NDArray[np.int8],
    costs: npt.NDArray[np.float64],
    corridor_data: dict[str, npt.NDArray[np.float64]] | None,
    axis_ranges: dict[str, float],
) -> Figure:
    """Render a single animation frame (2x2 grid).

    Returns:
        Matplotlib Figure ready for GIF writer.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Generation {generation}  |  Best cost: {best_cost:.1f}  |  Capture: {capture_rate:.0%}", fontsize=13, fontweight="bold")

    _render_corridor_panel(axes[0, 0], trajectories, traj_class, corridor_data, axis_ranges)
    _render_inclination_panel(axes[0, 1], trajectories, traj_class, axis_ranges)
    _render_bank_panel(axes[1, 0], trajectories, traj_class, axis_ranges)
    _render_cost_panel(axes[1, 1], costs, best_cost, axis_ranges)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def _load_pyo3():  # type: ignore[no-untyped-def]
    """Import and return the aerocapture_rs module, or raise with a clear message."""
    try:
        import aerocapture_rs  # type: ignore[import-not-found, import-untyped]

        return aerocapture_rs
    except ImportError as err:
        msg = "aerocapture_rs (PyO3) is required for animation generation.\nBuild it with: cd src/rust/aerocapture-py && maturin develop --release"
        raise RuntimeError(msg) from err


def _decode_and_build_overrides(
    best_individual: npt.NDArray,
    guidance_type: str,
    toml_data: dict,
    n_sims: int,
) -> dict[str, object]:
    """Decode a checkpoint's best individual into TOML overrides.

    Handles both neural_network (writes JSON file, returns minimal overrides)
    and guidance-parameter schemes (returns full dot-path overrides).
    Supports both new real-valued (float64) and legacy binary (int8) checkpoints.
    """
    from aerocapture.training.config import TrainingConfig
    from aerocapture.training.encoding import decode_normalized, nn_param_specs_from_architecture

    cfg = TrainingConfig()
    cfg.guidance_type = guidance_type

    net = toml_data.get("network", {})
    if "layer_sizes" in net:
        cfg.network.layer_sizes = net["layer_sizes"]
    if "activations" in net:
        cfg.network.activations = net["activations"]

    if guidance_type == "neural_network":
        from aerocapture.training.evaluate import write_nn_json

        cfg.sim.nn_param_file = toml_data.get("data", {}).get("neural_network", "data/neural_network/nn_model.json")
        specs = nn_param_specs_from_architecture(cfg.network.layer_sizes, cfg.network.activations)
        x = best_individual.astype(np.float64)
        weights = np.array([s.p_min + float(x[i]) * (s.p_max - s.p_min) for i, s in enumerate(specs)])
        nn_path = Path(cfg.sim.nn_param_file)
        write_nn_json(weights, cfg.network, nn_path)
        return {"simulation.n_sims": n_sims}

    from aerocapture.training.param_spaces import PARAM_SPACES

    specs = PARAM_SPACES[guidance_type]
    x = best_individual.astype(np.float64)
    params = decode_normalized(x, specs)
    return _build_overrides(guidance_type, params, n_sims)


def _reconstruct_corridor(npz_data: dict) -> dict[str, npt.NDArray[np.float64]] | None:
    """Reconstruct corridor data from checkpoint npz, if available."""
    if "corridor_energy_bins" not in npz_data:
        return None

    from aerocapture.training.corridor import CorridorAccumulator

    acc = CorridorAccumulator.from_checkpoint(npz_data)
    return acc.to_corridor_data()


def generate_animation(
    training_dir: Path,
    toml_path: Path,
    n_sims: int = 100,
    fps: int = 4,
    output: Path | None = None,
    every: int = 1,
    sim_timeout_secs: float | None = None,
) -> Path:
    """Generate a GIF animation of training evolution.

    Args:
        training_dir: Path to scheme training output directory.
        toml_path: Path to training TOML config.
        n_sims: Number of MC simulations per frame.
        fps: Frames per second in output GIF.
        output: Output GIF path. Defaults to training_dir/animation.gif.
        every: Use every Nth checkpoint (1 = all).
        sim_timeout_secs: Wall-clock timeout per simulation in seconds (default: no limit).

    Returns:
        Path to generated GIF file.
    """
    aero_rs = _load_pyo3()

    if output is None:
        output = training_dir / "animation.gif"

    checkpoints = _discover_checkpoints(training_dir, every)
    if not checkpoints:
        msg = f"No checkpoints found in {training_dir}"
        raise FileNotFoundError(msg)

    # Load TOML for guidance type and constraint limits
    from aerocapture.training.toml_utils import load_toml_with_bases

    toml_data = load_toml_with_bases(toml_path)
    guidance_type = toml_data.get("guidance", {}).get("type")
    if guidance_type is None:
        msg = "TOML config must contain [guidance] type"
        raise ValueError(msg)

    constraints = toml_data.get("flight", {}).get("constraints", {})
    heat_flux_limit: float | None = constraints.get("max_heat_flux")
    g_load_limit: float | None = constraints.get("max_load_factor")
    heat_load_limit: float | None = constraints.get("max_heat_load")

    toml_resolved = str(toml_path.resolve())

    # Step 1: Pre-compute axis ranges from the final checkpoint
    # NOTE: Axis ranges are computed from the final (most converged) checkpoint's trajectories.
    # Early-generation frames may have trajectories that extend beyond these limits and get clipped.
    # This is a deliberate trade-off to avoid running N extra MC evals just for range computation.
    last = checkpoints[-1]
    last_overrides = _decode_and_build_overrides(last["best_chromosome"], guidance_type, toml_data, n_sims)
    last_results = aero_rs.run_mc(toml_path=toml_resolved, overrides=last_overrides, include_trajectories=True, sim_timeout_secs=sim_timeout_secs)
    all_costs_for_range = np.concatenate([c["costs"] for c in checkpoints])
    axis_ranges = _compute_axis_ranges(last_results.trajectories, all_costs_for_range)

    # Step 2: Render frames with progress bar
    try:
        from rich.progress import Progress

        progress_ctx = Progress()
    except ImportError:
        progress_ctx = None  # type: ignore[assignment]

    matplotlib.use("Agg")
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="muted", font_scale=0.9, rc={"axes.facecolor": "#f5f5f5"})

    from matplotlib.animation import PillowWriter

    fig_placeholder = plt.figure()  # PillowWriter needs a figure to init
    writer = PillowWriter(fps=fps)
    writer.setup(fig_placeholder, str(output), dpi=100)
    plt.close(fig_placeholder)

    if progress_ctx is not None:
        progress_ctx.start()
        task_id = progress_ctx.add_task("Rendering frames", total=len(checkpoints))

    try:
        for ckpt in checkpoints:
            gen = ckpt["generation"]
            best_chrom = ckpt["best_chromosome"]
            if best_chrom is None:
                if progress_ctx is not None:
                    progress_ctx.advance(task_id)
                continue

            # Decode + run MC
            overrides = _decode_and_build_overrides(best_chrom, guidance_type, toml_data, n_sims)
            results = aero_rs.run_mc(toml_path=toml_resolved, overrides=overrides, include_trajectories=True, sim_timeout_secs=sim_timeout_secs)
            trajectories = results.trajectories
            final_records = results.final_records

            # Classify trajectories and derive capture rate (avoids magic column indices)
            traj_class = classify_trajectories(
                final_records,
                heat_flux_limit=heat_flux_limit,
                g_load_limit=g_load_limit,
                heat_load_limit=heat_load_limit,
            )
            captured = traj_class != TRAJ_FAILED
            capture_rate = float(np.mean(captured))

            # Corridor from checkpoint
            corridor_data = _reconstruct_corridor(ckpt["npz_data"])

            # Render frame
            fig = _render_frame(
                generation=gen,
                best_cost=ckpt["best_cost"],
                capture_rate=capture_rate,
                trajectories=trajectories,
                traj_class=traj_class,
                costs=ckpt["costs"],
                corridor_data=corridor_data,
                axis_ranges=axis_ranges,
            )
            writer.fig = fig  # Point writer at the actual frame
            writer.grab_frame()
            plt.close(fig)

            if progress_ctx is not None:
                progress_ctx.advance(task_id)
    finally:
        writer.finish()
        if progress_ctx is not None:
            progress_ctx.stop()

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GIF animation of training corridor/trajectory evolution",
    )
    parser.add_argument("training_dir", type=Path, help="Path to scheme training output directory")
    parser.add_argument("--toml", type=Path, required=True, help="Training TOML config path")
    parser.add_argument("--n-sims", type=int, default=100, help="MC simulations per frame (default: 100)")
    parser.add_argument("--fps", type=int, default=4, help="Frames per second (default: 4)")
    parser.add_argument("--output", type=Path, default=None, help="Output GIF path (default: <training_dir>/animation.gif)")
    parser.add_argument("--every", type=int, default=1, help="Use every Nth checkpoint (default: 1 = all)")
    parser.add_argument("--sim-timeout", type=float, default=None, help="Wall-clock timeout per simulation in seconds (default: no limit)")
    args = parser.parse_args()

    result = generate_animation(
        training_dir=args.training_dir,
        toml_path=args.toml,
        n_sims=args.n_sims,
        fps=args.fps,
        output=args.output,
        every=args.every,
        sim_timeout_secs=args.sim_timeout,
    )
    print(f"Animation saved to {result}")


if __name__ == "__main__":
    main()
