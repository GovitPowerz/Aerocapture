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
    from rich.panel import Panel
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


def _progress_line(gen: int, n_gen: int, width: int = 50) -> Text:
    """Styled ━/╸ progress bar line (blue filled, dim remainder, bold percent)."""
    from rich.text import Text  # noqa: PLC0415

    progress = min(max(gen / n_gen, 0.0), 1.0) if n_gen > 0 else 0.0
    filled = int(progress * width)
    t = Text()
    if filled > 0:
        t.append("━" * (filled - 1) + "╸", style="blue")
    t.append("━" * (width - filled), style="dim")
    t.append(f" {progress:.0%}", style="bold")
    return t


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


def _summary_renderables(summary: dict) -> list[ConsoleRenderable]:
    """Render a validation summary for a dashboard panel. Split the rows: the
    4-cell numeric rows (min/p50/p95/max header + Cost/DV/DV1-3) go into a tight
    right-aligned Table.grid; the 1-2 cell rows (Cap/Apo/Q/G/HL) render as styled
    Text lines above/below so they don't inflate the grid's first numeric column.
    """
    from rich.table import Table  # noqa: PLC0415
    from rich.text import Text  # noqa: PLC0415

    parts: list[ConsoleRenderable] = []
    rows = _validation_summary_rows(summary)
    grid = Table.grid(padding=(0, 2))
    grid.add_column()
    for _ in range(4):
        grid.add_column(justify="right")
    grid_added = False
    for label, cells, row_style in rows:
        if len(cells) != 4:
            continue
        grid.add_row(*(Text(c, style=row_style or "") for c in [label, *cells]))
        grid_added = True
    for label, cells, row_style in rows:
        if len(cells) == 4:
            if grid_added:
                parts.append(grid)
                grid_added = False
            continue
        parts.append(Text(f"{label:<5} " + "   ".join(cells), style=row_style or ""))
    if grid_added:  # no 1-2 cell rows after the grid (degenerate summary)
        parts.append(grid)
    return parts


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

    is_live: bool

    def update(self, logger: TrainingLogger, current_run: int, island_records: dict[str, dict] | None = None) -> None: ...
    def stop(self) -> None: ...
    def set_start_gen(self, start_gen: int) -> None: ...
    def __enter__(self) -> DisplayProtocol: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None: ...


class NoopDisplay:
    """No-op display for non-interactive terminals or --no-tui mode.

    `is_live = False` tells the training loop to fall back to plain per-gen
    heartbeat prints (the TUI dashboard otherwise leaves headless runs silent).
    """

    is_live = False

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

    is_live = True

    def __init__(self, scheme: str, n_runs: int, n_generations: int, algorithm: str = "", seed_strategy: str = "", training_n_sims: int | None = None) -> None:
        self._scheme = scheme
        self._algorithm = algorithm
        self._n_runs = n_runs
        self._n_gens = n_generations
        self._seed_strategy = seed_strategy
        self._training_n_sims = training_n_sims
        self._live: Live | None = None
        self._start_time: float | None = None
        self._start_gen: int = 0

    def set_start_gen(self, start_gen: int) -> None:
        self._start_gen = start_gen

    def _build_header(self, gen: int, pop: int | None, elapsed: float) -> Panel:
        from rich.console import Group  # noqa: PLC0415
        from rich.panel import Panel  # noqa: PLC0415
        from rich.text import Text  # noqa: PLC0415

        rate, remaining = _rate_and_eta(gen, self._start_gen, self._n_gens, elapsed)
        t = Text()
        t.append(self._scheme, style="bold")
        if self._algorithm:
            t.append(" \u00b7 ", style="dim")
            t.append(self._algorithm, style="bold")
        t.append("  \u2502  Gen ", style="dim")
        t.append(str(gen), style="bold")
        t.append(f"/{self._n_gens}", style="dim")
        if pop is not None:
            t.append("  \u2502  pop ", style="dim")
            t.append(str(pop), style="bold")
        if gen > self._start_gen:
            t.append(f"  \u2502  elapsed {_format_duration(elapsed)}  \u2502  {rate:.2f} gen/s  \u2502  ETA ", style="dim")
            t.append(_format_duration(remaining), style="bold")
        return Panel(Group(t, _progress_line(gen, self._n_gens)), border_style="green")

    def _build_optimization_panel(self, buf: list[dict]) -> Panel:
        from rich.console import Group  # noqa: PLC0415
        from rich.panel import Panel  # noqa: PLC0415
        from rich.table import Table  # noqa: PLC0415
        from rich.text import Text  # noqa: PLC0415

        latest = buf[-1]
        grid = Table.grid(padding=(0, 2))
        grid.add_column()
        grid.add_column(justify="right")
        grid.add_column()
        grid.add_row("Best", Text(_format_cost(latest["best_cost"]), style="bold"), Text(_sparkline([r["best_cost"] for r in buf]), style="cyan"))
        grid.add_row("Mean", _format_cost(latest["mean_cost"]), Text(_sparkline([r["mean_cost"] for r in buf]), style="cyan"))
        if latest.get("worst_cost") is not None:
            grid.add_row("Worst", _format_cost(latest["worst_cost"]), Text(f"\u03c3 {latest.get('std_cost', float('nan')):.1e}", style="dim"))
        grid.add_row("Capture", Text(f"{latest['capture_rate']:.0%}", style="green"), Text(_sparkline([r["capture_rate"] for r in buf]), style="green"))
        grid.add_row("Divers", f"{latest['population_diversity']:.2f}", Text(_sparkline([r["population_diversity"] for r in buf]), style="magenta"))
        all_costs = latest.get("all_costs")
        if all_costs:
            glyphs, caption = _cost_histogram(all_costs)
            grid.add_row("Pop cost", Text(glyphs, style="blue"), Text(caption, style="dim"))
        pool = latest.get("pool_metrics") or {}
        if self._seed_strategy:
            n_prefix = f"n {self._training_n_sims} \u00b7 " if self._training_n_sims is not None else ""
            if self._seed_strategy == "adaptive":
                detail = f"{n_prefix}refreshed g{pool['last_curation_gen']}" if pool.get("last_curation_gen") is not None else f"{n_prefix}no curation yet"
            elif self._seed_strategy == "rotating":
                detail = f"{n_prefix}fresh every gen"
            else:
                detail = f"{n_prefix}deterministic"
            grid.add_row("Seeds", self._seed_strategy, Text(detail, style="dim"))
        bits = []
        if latest.get("gen_elapsed_s") is not None:
            bits.append(f"gen wall {latest['gen_elapsed_s']:.2f}s")
        if not self._seed_strategy and pool.get("last_curation_gen") is not None:
            bits.append(f"pool refresh g{pool['last_curation_gen']}")
        body: ConsoleRenderable = Group(grid, Text(" \u00b7 ".join(bits), style="dim")) if bits else grid
        return Panel(body, title="Optimization", border_style="cyan")

    def _build_validation_panels(self, buf: list[dict]) -> list[Panel]:
        """[Last validation, Best validation] panels \u2014 the Last panel is framed
        green when it became the new best (promoted) and red otherwise; the Best
        panel is a grey-framed reminder of the best stats seen so far."""
        from rich.panel import Panel  # noqa: PLC0415
        from rich.text import Text  # noqa: PLC0415

        best_val_r, last_val_r = self._scan_validation_records(buf)
        if last_val_r is None and best_val_r is None:
            placeholder = Text("waiting for first validation\u2026\n", style="dim")
            placeholder.append("(gate fires when the gen-best individual changes)", style="dim")
            return [Panel(placeholder, title="Validation", border_style="green")]

        panels: list[Panel] = []
        if last_val_r is not None:
            promoted = bool(last_val_r.get("improvement"))
            headline = Text(f"RMS {_format_cost(last_val_r['validation']['rms_cost'])}  ")
            headline.append("PROMOTED" if promoted else "REJECTED", style="green" if promoted else "yellow")
            headline.append(f"  g{last_val_r['generation']}", style="dim")
            panels.append(
                self._validation_detail_panel(
                    last_val_r,
                    headline=headline,
                    title_prefix="Last validation",
                    border_style="green" if promoted else "red",
                )
            )
        if best_val_r is not None:
            headline = Text("")
            headline.append(f"RMS {_format_cost(best_val_r['validation']['rms_cost'])}", style="green")
            headline.append(f"  g{best_val_r['generation']}", style="dim")
            panels.append(
                self._validation_detail_panel(
                    best_val_r,
                    headline=headline,
                    title_prefix="Best validation",
                    border_style="grey50",
                )
            )
        return panels

    @staticmethod
    def _validation_detail_panel(record: dict, *, headline: Text, title_prefix: str, border_style: str) -> Panel:
        from rich.console import Group  # noqa: PLC0415
        from rich.panel import Panel  # noqa: PLC0415

        # parts: list[ConsoleRenderable] avoids mypy issues with Group(*parts) when
        # the union is Text | Table (both satisfy ConsoleRenderable).
        parts: list[ConsoleRenderable] = [headline]
        title = f"{title_prefix} (g{record['generation']})"
        summary = record.get("validation_summary")
        if summary:
            title = f"{title_prefix} ({summary.get('n_sims', 0)} sims \u00b7 g{record['generation']})"
            parts.extend(_summary_renderables(summary))
        return Panel(Group(*parts), title=title, border_style=border_style)

    @staticmethod
    def _scan_validation_records(buf: list[dict]) -> tuple[dict | None, dict | None]:
        """(best_val_record, last_val_record) by min rms / max generation."""
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
        return best_val_r, last_val_r

    def _build_footer(self, buf: list[dict]) -> Text:
        from rich.text import Text  # noqa: PLC0415

        latest = buf[-1]
        gen = latest["generation"]
        t = Text(" ")
        improvements = [r["generation"] for r in buf if r.get("improvement")]
        if improvements:
            last_imp = improvements[-1]
            stag = gen - last_imp
            if stag > 0:
                t.append(f"Stagnant {stag} gens", style="yellow")
            else:
                t.append("Improved this gen", style="green")
            t.append(f" \u00b7 improved g{last_imp}", style="dim")
        else:
            t.append("No improvement yet", style="yellow")
        params = latest.get("best_params")
        if params:
            if self._scheme == "neural_network":
                t.append(f" \u00b7 {len(params)} NN params (best_model.json)", style="dim")
            else:
                items = list(params.items())
                preview = ", ".join(f"{k} {v:.4g}" for k, v in items[:3])
                more = f" (+{len(items) - 3} more)" if len(items) > 3 else ""
                t.append(f" \u00b7 best: {preview}{more}", style="dim")
        return t

    def _build_dashboard(self, logger: TrainingLogger, current_run: int) -> ConsoleRenderable:
        import time  # noqa: PLC0415

        from rich.console import Group  # noqa: PLC0415
        from rich.table import Table  # noqa: PLC0415
        from rich.text import Text  # noqa: PLC0415

        if self._start_time is None:
            self._start_time = time.monotonic()
        buf = logger.buffer
        if not buf:
            return Group(self._build_header(gen=self._start_gen, pop=None, elapsed=0.0), Text(" Waiting for first generation\u2026", style="dim"))
        latest = buf[-1]
        gen = latest["generation"]
        pop = len(latest["all_costs"]) if latest.get("all_costs") else None
        elapsed = time.monotonic() - self._start_time
        # Header and Optimization span the full console width; the two
        # validation panels share that same width 50/50 in a row below
        # (an expand=True grid \u2014 Columns would shrink-wrap instead).
        panels = self._build_validation_panels(buf)
        validation_row: ConsoleRenderable
        if len(panels) == 2:
            row = Table.grid(expand=True)
            row.add_column(ratio=1)
            row.add_column(ratio=1)
            row.add_row(*panels)
            validation_row = row
        else:
            validation_row = panels[0]
        return Group(
            self._build_header(gen, pop, elapsed),
            self._build_optimization_panel(buf),
            validation_row,
            self._build_footer(buf),
        )

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
        # consistently with `_build_dashboard` so a LiveDisplay reused across
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
        self._live.update(self._build_dashboard(logger, current_run))

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


def create_display(
    scheme: str,
    n_runs: int,
    n_generations: int,
    *,
    enabled: bool = True,
    algorithm: str = "",
    seed_strategy: str = "",
    training_n_sims: int | None = None,
) -> LiveDisplay | NoopDisplay:
    """Factory: returns LiveDisplay if enabled and terminal is interactive, else NoopDisplay."""
    if not enabled or not sys.stdout.isatty():
        return NoopDisplay()
    return LiveDisplay(
        scheme=scheme, n_runs=n_runs, n_generations=n_generations, algorithm=algorithm, seed_strategy=seed_strategy, training_n_sims=training_n_sims
    )
