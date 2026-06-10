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
    from rich.text import Text

    from aerocapture.training.logger import TrainingLogger

# Sparkline characters (increasing height)
_SPARK_CHARS = " ▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], width: int = 30) -> str:
    """Render a list of floats as a Unicode sparkline string."""
    if not values:
        return " " * width
    vals = values[-width:]
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return "▄" * len(vals)  # flat series: midline, not blanks
    span = hi - lo
    return "".join(_SPARK_CHARS[min(int((v - lo) / span * 8), 8)] for v in vals)


def _format_cost(value: float) -> str:
    """Format cost in scientific notation with 4 decimals."""
    return f"{value:.4e}"


def _cost_histogram(all_costs: list[float], bins: int = 16) -> tuple[str, str]:
    """Log-binned histogram of a population's costs as (glyphs, dim caption).

    Empty bins render as a middle dot so gaps in the distribution stay
    visible; non-finite entries (inf/NaN sim failures) are counted in the
    caption rather than binned.
    """
    import math  # noqa: PLC0415

    finite = sorted(c for c in all_costs if math.isfinite(c) and c > 0.0)
    n_nonfinite = sum(1 for c in all_costs if not math.isfinite(c))
    inf_suffix = f"  ∞×{n_nonfinite}" if n_nonfinite else ""
    if not finite:
        return "", f"no finite costs{inf_suffix}".strip()
    lo, hi = finite[0], finite[-1]
    if hi <= lo:
        return "█" + "·" * (bins - 1), f"{lo:.0e} log{inf_suffix}"
    log_lo, log_hi = math.log10(lo), math.log10(hi)
    counts = [0] * bins
    for c in finite:
        idx = min(int((math.log10(c) - log_lo) / (log_hi - log_lo) * bins), bins - 1)
        counts[idx] += 1
    peak = max(counts)
    glyphs = "".join("·" if n == 0 else _SPARK_CHARS[max(1, min(int(n / peak * 8), 8))] for n in counts)
    return glyphs, f"{lo:.0e}→{hi:.0e} log{inf_suffix}"


def _rate_and_eta(gen: int, start_gen: int, n_gen: int, elapsed: float) -> tuple[float, float]:
    """(gens/sec, remaining seconds) — resume-aware, mirrors the islands header math."""
    rate = (gen - start_gen) / elapsed if elapsed > 0 and gen > start_gen else 0.0
    remaining_gens = max(n_gen - gen, 0)
    remaining = remaining_gens / rate if rate > 0 else float("inf")
    return rate, remaining


def _validation_summary_rows(summary: dict) -> list[tuple[str, list[str], str]]:
    """Shape a `compute_eval_summary` payload into (label, cells, style) rows.

    Consumed by the single-algo Validation panel (as a Table.grid) and the
    islands per-island detail panels (as text lines). Style is a row-level
    Rich style hint: "" | "dim" | "red" | "yellow" | "green".
    """
    nan = float("nan")
    n_sims = summary.get("n_sims", 0)
    n_cap = summary.get("n_captured", 0)
    pct = 100.0 * n_cap / max(n_sims, 1)
    rows: list[tuple[str, list[str], str]] = []
    cap_style = "red" if n_cap == 0 else ("green" if pct >= 95.0 else "")
    rows.append(("Cap", [f"{n_cap}/{n_sims} ({pct:.1f}%)"], cap_style))
    rows.append(("", ["min", "p50", "p95", "max"], "dim"))

    def _grid(block: dict, fmt: str = "{:.1f}") -> list[str]:
        return [fmt.format(block.get(k, nan)) for k in ("min", "p50", "p95", "max")]

    cost = summary.get("cost", {}) or {}
    cost_style = "yellow" if cost.get("max", 0.0) > 10.0 * cost.get("p95", float("inf")) else ""
    rows.append(("Cost", _grid(cost), cost_style))
    cap_block = summary.get("captured")
    if cap_block:
        rows.append(("DV", _grid(cap_block.get("dv", {})), ""))
        for i in (1, 2, 3):
            rows.append((f"DV{i}", _grid(cap_block.get(f"dv{i}", {})), "dim"))
        apo = cap_block.get("apoapsis", {})
        rows.append(("Apo", [f"p50 {apo.get('p50', nan):.1f} · p95 {apo.get('p95', nan):.1f} km"], ""))
    else:
        rows.append(("DV", ["—"], "dim"))
    con = summary.get("constraints", {}) or {}
    for label, key, val_fmt, lim_fmt in (
        ("Q", "heat_flux", "{:.1f}", "{:.0f}"),
        ("G", "g_load", "{:.2f}", "{:.1f}"),
        ("HL", "heat_load", "{:.0f}", "{:.0f}"),
    ):
        block = con.get(key)
        if block is None:
            rows.append((label, ["n/a"], "dim"))
            continue
        cells = [f"max {val_fmt.format(block.get('max', nan))}"]
        style = ""
        if block.get("limit") is not None and block.get("viol_pct") is not None:
            cells.append(f"{block['viol_pct']:.1f}% > {lim_fmt.format(block['limit'])}")
            style = "red" if block["viol_pct"] > 0 else "dim"
        rows.append((label, cells, style))
    return rows


def _rows_to_text(summary: dict) -> Text:
    """Render summary rows as styled text lines (the islands detail panels)."""
    from rich.text import Text  # noqa: PLC0415

    text = Text(f"Validation ({summary.get('n_sims', 0)} sims)\n")
    for label, cells, style in _validation_summary_rows(summary):
        # 4-cell grid rows (min/p50/p95/max header + value rows) get right-aligned
        # fixed-width cells so columns line up; non-grid rows keep loose spacing.
        body = "  ".join(f"{c:>8}" for c in cells) if len(cells) == 4 else "   ".join(cells)
        text.append(f"  {label:<5} " + body + "\n", style=style or None)
    return text


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
            lines.extend(str(_rows_to_text(detail_src["validation_summary"])).splitlines())

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
        rate, remaining = _rate_and_eta(gen, self._start_gen, n_gen, elapsed)
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
            island_summary: dict | None = (island_records.get(name) or {}).get("val_summary")
            if not island_summary:
                continue
            detail_panels.append(Panel(_rows_to_text(island_summary), title=f"{name.upper()} validation", border_style="green"))

        group = Group(header, Columns(panels), Columns(detail_panels), mig_panel) if detail_panels else Group(header, Columns(panels), mig_panel)
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
