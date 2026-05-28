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


class DisplayProtocol(Protocol):
    """Protocol for training display (allows NoopDisplay as substitute)."""

    def update(self, logger: TrainingLogger, current_run: int, island_records: dict[str, dict] | None = None) -> None: ...
    def stop(self) -> None: ...
    def __enter__(self) -> DisplayProtocol: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None: ...


class NoopDisplay:
    """No-op display for non-interactive terminals or --no-tui mode."""

    def update(self, logger: TrainingLogger, current_run: int, island_records: dict[str, dict] | None = None) -> None:
        pass

    def stop(self) -> None:
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
        """Render a 3-column layout, one Panel per island."""
        from rich.columns import Columns  # noqa: PLC0415
        from rich.panel import Panel  # noqa: PLC0415

        panels = []
        for name in ("pso", "ga", "de"):
            rec = island_records.get(name, {})
            best = rec.get("best_overall_cost", float("inf"))
            val_rms = rec.get("val_rms", float("inf"))
            stag = rec.get("stagnation", 0)
            argmin = rec.get("argmin_train_cost", float("inf"))
            content = (
                f"best: {_format_cost(best)}\n"
                f"last_val: {_format_cost(val_rms)}\n"
                f"stag: {stag} gens\n"
                f"argmin: {_format_cost(argmin)}"
            )
            panels.append(Panel(content, title=name.upper(), border_style="cyan"))
        self._live.update(Columns(panels))

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
