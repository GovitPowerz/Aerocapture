"""Live Rich TUI display for GA training progress.

Shows sparklines for key metrics, progress bar with ETA,
stagnation warnings, and current best parameters.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from types import TracebackType

    from rich.console import ConsoleRenderable
    from rich.live import Live

    from aerocapture.training.logger import TrainingLogger

# Sparkline characters (increasing height)
_SPARK_CHARS = " ▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], width: int = 30) -> str:
    """Render a list of floats as a Unicode sparkline string."""
    if not values:
        return " " * width
    vals = values[-width:]
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi > lo else 1.0
    return "".join(_SPARK_CHARS[min(int((v - lo) / span * 8), 8)] for v in vals)


def _format_cost(value: float) -> str:
    """Format cost in scientific notation with 4 decimals."""
    return f"{value:.4e}"


def _format_validation_summary(summary: dict, indent: str = "  ") -> list[str]:
    """Compact one-line-per-metric rendering of `compute_eval_summary` for TUI.

    Mirrors `report.format_eval_summary` but inlines the formatter to keep
    display.py free of the matplotlib-heavy report import path.
    """
    n_sims = summary.get("n_sims", 0)
    n_cap = summary.get("n_captured", 0)
    cost = summary.get("cost", {}) or {}
    lines = [
        f"Validation ({n_sims} sims)",
        f"{indent}Cost:        p50={cost.get('p50', float('nan')):.1f}  p95={cost.get('p95', float('nan')):.1f}  RMS={cost.get('rms', float('nan')):.1f}",
        f"{indent}Capture:     {n_cap}/{n_sims} ({100 * n_cap / max(n_sims, 1):.1f}%)",
    ]
    cap = summary.get("captured")
    if cap:
        dv = cap.get("dv", {})
        apo = cap.get("apoapsis", {})
        peri = cap.get("periapsis", {})
        incl = cap.get("inclination", {})
        lines.extend(
            [
                f"{indent}DV (m/s):    p50={dv.get('p50', float('nan')):.1f}  p95={dv.get('p95', float('nan')):.1f}  mean={dv.get('mean', float('nan')):.1f}",
                f"{indent}Apo (km):    p50={apo.get('p50', float('nan')):.1f}  p95={apo.get('p95', float('nan')):.1f}  mean={apo.get('mean', float('nan')):.1f}",
                f"{indent}Peri (km):   p50={peri.get('p50', float('nan')):.1f}  p95={peri.get('p95', float('nan')):.1f}  mean={peri.get('mean', float('nan')):.1f}",
                f"{indent}Incl (deg):  p50={incl.get('p50', float('nan')):.2f}  p95={incl.get('p95', float('nan')):.2f}  mean={incl.get('mean', float('nan')):.2f}",
            ]
        )
    con = summary.get("constraints", {}) or {}

    def _con_line(label: str, block: dict | None, val_fmt: str, lim_fmt: str) -> str:
        if block is None:
            return f"{indent}{label}: n/a"
        suffix = ""
        if block.get("limit") is not None and block.get("viol_pct") is not None:
            suffix = f"  {block['viol_pct']:.1f}% > {lim_fmt.format(block['limit'])}"
        return (
            f"{indent}{label}: p50={val_fmt.format(block.get('p50', float('nan')))}  "
            f"p95={val_fmt.format(block.get('p95', float('nan')))}  "
            f"max={val_fmt.format(block.get('max', float('nan')))}{suffix}"
        )

    lines.append(_con_line("Q (kW/m²) ", con.get("heat_flux"), "{:.1f}", "{:.0f}"))
    lines.append(_con_line("G (g)     ", con.get("g_load"), "{:.2f}", "{:.1f}"))
    lines.append(_con_line("HL (kJ/m²)", con.get("heat_load"), "{:.0f}", "{:.0f}"))
    return lines


def _format_duration(seconds: float) -> str:
    if not (seconds == seconds and seconds != float("inf")):  # NaN or inf
        return "--"
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class DisplayProtocol(Protocol):
    """Protocol for training display (allows NoopDisplay as substitute)."""

    def update(self, logger: TrainingLogger, current_run: int, island_records: dict[str, dict] | None = None) -> None: ...
    def stop(self) -> None: ...
    def set_start_gen(self, start_gen: int) -> None: ...
    def __enter__(self) -> DisplayProtocol: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None: ...


class NoopDisplay:
    """No-op display for non-interactive terminals or --no-tui mode."""

    def update(self, logger: TrainingLogger, current_run: int, island_records: dict[str, dict] | None = None) -> None:
        pass

    def stop(self) -> None:
        pass

    def set_start_gen(self, start_gen: int) -> None:
        pass

    def __enter__(self) -> NoopDisplay:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None:
        pass


class LiveDisplay:
    """Rich Live TUI for training progress."""

    def __init__(self, scheme: str, n_runs: int, n_generations: int) -> None:
        self._scheme = scheme
        self._n_runs = n_runs
        self._n_gens = n_generations
        self._live: Live | None = None
        self._start_time: float | None = None
        self._start_gen: int = 0

    def set_start_gen(self, start_gen: int) -> None:
        self._start_gen = start_gen

    def _build_panel(self, logger: TrainingLogger, current_run: int) -> ConsoleRenderable:
        """Build a Rich Panel from logger buffer."""
        import time

        from rich.panel import Panel
        from rich.text import Text

        buf = logger.buffer
        if not buf:
            return Panel("Waiting for first generation...", title=self._scheme)

        if self._start_time is None:
            self._start_time = time.monotonic()

        latest = buf[-1]
        gen = latest["generation"]

        best_costs = [r["best_cost"] for r in buf]
        mean_costs = [r["mean_cost"] for r in buf]
        cap_rates = [r["capture_rate"] for r in buf]
        diversities = [r["population_diversity"] for r in buf]

        lines = []
        lines.append(f"Best cost  {_format_cost(latest['best_cost']):>10s}  {_sparkline(best_costs)}")
        lines.append(f"Mean cost  {_format_cost(latest['mean_cost']):>10s}  {_sparkline(mean_costs)}")
        lines.append(f"Capture    {latest['capture_rate']:>9.0%}  {_sparkline(cap_rates)}")
        lines.append(f"Diversity  {latest['population_diversity']:>9.2f}  {_sparkline(diversities)}")
        lines.append("")

        # Progress bar
        progress = gen / self._n_gens
        bar_width = 40
        filled = int(progress * bar_width)
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
        eta_str = ""
        if self._start_time is not None and progress > 0:
            elapsed = time.monotonic() - self._start_time
            remaining = elapsed / progress * (1 - progress)
            mins, secs = divmod(int(remaining), 60)
            eta_str = f"  ETA {mins}m {secs:02d}s"
        lines.append(f"{bar}  {progress:.0%}{eta_str}")
        lines.append("")

        # Validation metrics: show the most recent validation attempt and the
        # best-ever validated candidate (permanent).
        best_val_r: dict | None = None
        last_val_r: dict | None = None
        for r in buf:
            if "validation" not in r:
                continue
            if last_val_r is None or r["generation"] >= last_val_r["generation"]:
                last_val_r = r
            rms = r["validation"].get("rms_cost")
            if rms is None:
                continue
            if best_val_r is None or rms < best_val_r["validation"].get("rms_cost", float("inf")):
                best_val_r = r
        if last_val_r is not None and last_val_r is not best_val_r:
            lv = last_val_r["validation"]
            outcome = "PROMOTED" if last_val_r.get("improvement") else "REJECTED"
            rms_str = _format_cost(lv["rms_cost"]) if lv.get("rms_cost") is not None else "n/a"
            lines.append(f"Last val  g{last_val_r['generation']}: RMS={rms_str} cap={lv['capture_rate']:.0%} -> {outcome}")
        if best_val_r is not None:
            bv = best_val_r["validation"]
            lines.append(
                f"Best val  g{best_val_r['generation']}: RMS={_format_cost(bv['rms_cost'])} "
                f"mean={_format_cost(bv['mean_cost'])} p95={_format_cost(bv['p95_cost'])} cap={bv['capture_rate']:.0%}"
            )
        # Rich validation dashboard from `compute_eval_summary`. Prefer the
        # most-recently-collected summary so the operator sees current shape
        # (DV / apoapsis / heat-flux / g-load) rather than the historical best.
        detail_src = last_val_r if last_val_r is not None and last_val_r.get("validation_summary") else best_val_r
        if detail_src is not None and detail_src.get("validation_summary"):
            lines.append("")
            lines.append(f"-- Validation detail (g{detail_src['generation']}) --")
            lines.extend(_format_validation_summary(detail_src["validation_summary"]))

        # Stagnation
        improvements = [i for i, r in enumerate(buf) if r["improvement"]]
        if improvements:
            last_imp_gen = buf[improvements[-1]]["generation"]
            stag = gen - last_imp_gen
            if stag > 0:
                lines.append(f"Stagnant for {stag} gens \u00b7 Last improvement: gen {last_imp_gen}")
        else:
            lines.append("No improvement yet")

        # Best params
        params = latest.get("best_params")
        if params is not None:
            param_str = ", ".join(f"{k}: {v:.4g}" for k, v in params.items())
            lines.append(f"Best params: {{{param_str}}}")

        title = f"{self._scheme} \u00b7 Run {current_run + 1}/{self._n_runs} \u00b7 Gen {gen}/{self._n_gens}"
        return Panel(Text("\n".join(lines)), title=title)

    def _update_islands(
        self,
        logger: TrainingLogger,
        island_records: dict[str, dict],
    ) -> None:
        """Render a richer 3-column layout with header + migration summary."""
        import time  # noqa: PLC0415

        from rich.columns import Columns  # noqa: PLC0415
        from rich.console import Group  # noqa: PLC0415
        from rich.panel import Panel  # noqa: PLC0415
        from rich.text import Text  # noqa: PLC0415

        if self._live is None:  # defensive: assert is stripped under -O
            return

        # Header: gen X/N | elapsed | rate | ETA. Uses `time.monotonic`
        # consistently with `_build_panel` so a LiveDisplay reused across
        # both paths can't mix epochs from two distinct clocks.
        gen = int(island_records.get("_gen", 0))  # type: ignore[arg-type]
        n_gen = int(island_records.get("_n_gen", self._n_gens))  # type: ignore[arg-type]
        if self._start_time is None:
            self._start_time = time.monotonic()
        elapsed = time.monotonic() - self._start_time
        rate = (gen - self._start_gen) / elapsed if elapsed > 0 and gen > self._start_gen else 0.0
        # Cap the remaining-gen count at 0 so an overshoot resume (gen >
        # n_gen) doesn't render a negative duration.
        remaining_gens = max(n_gen - gen, 0)
        remaining = remaining_gens / rate if rate > 0 else float("inf")
        header_text = f"Gen {gen}/{n_gen}  elapsed {_format_duration(elapsed)}  rate {rate:.2f} gens/s  ETA {_format_duration(remaining)}"
        header = Panel(Text(header_text, style="bold"), title="Islands training", border_style="green")

        # Migration summary panel.
        summary = island_records.get("_latest_migration_summary", {}) or {}
        latest_gen = island_records.get("_latest_migration_gen")
        total = island_records.get("_total_migrations", 0)
        if not summary:
            mig_panel = Panel(
                Text(f"No migrations yet ({total} total)", style="dim"),
                title="Migrations",
                border_style="dim",
            )
        else:
            lines: list[str] = []
            for dst_name in ("pso", "ga", "de"):
                rec = summary.get(dst_name)
                if rec is None:
                    continue
                best = rec["best"]
                worst = rec["worst"]
                lines.append(
                    f"{dst_name.upper():<4} ⬇ best:  {best['src']:<3} F={_format_cost(best['F_migrant'])} (displaced {_format_cost(best['F_displaced'])})"
                )
                lines.append(f"     ⬆ worst: {worst['src']:<3} F={_format_cost(worst['F_migrant'])} (displaced {_format_cost(worst['F_displaced'])})")
            origin_stats = island_records.get("_origin_stats", {}) or {}
            if origin_stats:
                lines.append("")
                lines.append("Origins (best-migrant wins per destination, cumulative)")
                for dst_name in ("pso", "ga", "de"):
                    src_map = origin_stats.get(dst_name, {})
                    if not src_map:
                        continue
                    # Sort sources by win count desc, then by mean_F asc.
                    sorted_srcs = sorted(
                        src_map.items(),
                        key=lambda kv: (-kv[1]["wins"], kv[1]["mean_F"]),
                    )
                    label_shown = False
                    for src, st in sorted_srcs:
                        prefix = f"  {dst_name.upper():<3} <- " if not label_shown else "       "
                        label_shown = True
                        lines.append(f"{prefix}{src:<3} : {int(st['wins']):>3d} wins  mean F={_format_cost(float(st['mean_F']))}  ({int(st['count'])} total)")
            mig_panel = Panel(
                Text("\n".join(lines)),
                title=f"Migrations (latest gen {latest_gen} · {total} total)",
                border_style="cyan",
            )

        # Per-island panels with best_val added.
        panels = []
        for name in ("pso", "ga", "de"):
            rec = island_records.get(name, {})
            best_val = rec.get("best_val", float("inf"))
            last_val = rec.get("val_rms", float("inf"))
            stag = rec.get("stagnation", 0)
            argmin = rec.get("argmin_train_cost", float("inf"))
            content = f"best_val: {_format_cost(best_val)}\nlast_val: {_format_cost(last_val)}\nargmin:   {_format_cost(argmin)}\nstag:     {stag} gens"
            panels.append(Panel(content, title=name.upper(), border_style="cyan"))

        # Rich per-island validation dashboards (DV / apoapsis / heat-flux /
        # g-load / heat-load percentiles + violation rates). Only the island(s)
        # that validated THIS gen carry a fresh `val_summary`; render whichever
        # are present so the operator can compare across PSO/GA/DE.
        detail_panels: list[Panel] = []
        for name in ("pso", "ga", "de"):
            summary = (island_records.get(name) or {}).get("val_summary")
            if not summary:
                continue
            lines = _format_validation_summary(summary)
            detail_panels.append(Panel(Text("\n".join(lines)), title=f"{name.upper()} validation", border_style="green"))

        if detail_panels:
            group = Group(header, Columns(panels), Columns(detail_panels), mig_panel)
        else:
            group = Group(header, Columns(panels), mig_panel)
        self._live.update(group)

    def update(self, logger: TrainingLogger, current_run: int, island_records: dict[str, dict] | None = None) -> None:
        """Update the live display with current logger state."""
        if self._live is None:
            return
        if island_records is not None:
            self._update_islands(logger, island_records)
            return
        panel = self._build_panel(logger, current_run)
        self._live.update(panel)

    def stop(self) -> None:
        """Stop the Live display (for clean interrupt output)."""
        if self._live is not None:
            self._live.stop()

    def __enter__(self) -> LiveDisplay:
        from rich.live import Live

        self._live = Live(refresh_per_second=2)
        self._live.__enter__()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None:
        if self._live is not None:
            self._live.__exit__(exc_type, exc_val, exc_tb)
            self._live = None


def create_display(scheme: str, n_runs: int, n_generations: int, *, enabled: bool = True) -> LiveDisplay | NoopDisplay:
    """Factory: returns LiveDisplay if enabled and terminal is interactive, else NoopDisplay."""
    if not enabled or not sys.stdout.isatty():
        return NoopDisplay()
    return LiveDisplay(scheme=scheme, n_runs=n_runs, n_generations=n_generations)
