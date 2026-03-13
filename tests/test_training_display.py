"""Smoke tests for LiveDisplay — verify no crash, not visual correctness."""

from __future__ import annotations

from unittest.mock import MagicMock

from aerocapture.training.display import LiveDisplay, NoopDisplay, create_display


class TestLiveDisplay:
    def test_create_display_returns_noop_in_non_tty(self) -> None:
        """In non-interactive environments (CI), create_display returns NoopDisplay."""
        display = create_display(scheme="equilibrium_glide", n_runs=1, n_generations=50, enabled=False)
        assert isinstance(display, NoopDisplay)

    def test_build_panel_does_not_crash(self) -> None:
        """Test panel building logic without opening a Live context."""
        display = LiveDisplay(scheme="equilibrium_glide", n_runs=1, n_generations=50)
        logger = MagicMock()
        logger.buffer = [
            {
                "generation": 1,
                "best_cost": 1000.0,
                "mean_cost": 5000.0,
                "capture_rate": 0.8,
                "population_diversity": 0.45,
                "improvement": True,
                "best_params": {"k_hdot_scale": 0.3},
            },
        ]
        panel = display._build_panel(logger, current_run=0)
        assert panel is not None


class TestNoopDisplay:
    def test_noop_context_manager(self) -> None:
        display = NoopDisplay()
        with display:
            display.update(MagicMock(), current_run=0)
