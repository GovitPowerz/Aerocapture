"""Rich TUI for RL training. Matches GA LiveDisplay interface (update, close)."""

from __future__ import annotations

from typing import Any

_RICH_AVAILABLE = False
try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass


class NoopDisplay:
    def update(self, record: dict[str, Any]) -> None:  # noqa: D401
        pass

    def close(self) -> None:
        pass


class RLLiveDisplay:
    def __init__(self, total_env_steps: int) -> None:
        self._total = total_env_steps
        self._console = Console()
        self._live = Live(self._render({}), console=self._console, refresh_per_second=2)
        self._live.start()

    def _render(self, r: dict[str, Any]) -> Table:
        t = Table(title=f"RL training — {r.get('env_steps', 0)} / {self._total} env steps")
        t.add_column("metric")
        t.add_column("value")
        for k in (
            "episodic_return_mean",
            "episodic_dv_m_s_mean",
            "episodic_capture_rate",
            "entropy",
            "policy_loss",
            "value_loss",
            "best_val_cost",
        ):
            v = r.get(k)
            t.add_row(k, f"{v:.4g}" if isinstance(v, (int, float)) else "—")
        return t

    def update(self, record: dict[str, Any]) -> None:
        self._live.update(self._render(record))

    def close(self) -> None:
        self._live.stop()


def make_display(total_env_steps: int, enabled: bool) -> NoopDisplay | RLLiveDisplay:
    if not enabled or not _RICH_AVAILABLE:
        return NoopDisplay()
    return RLLiveDisplay(total_env_steps)
