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


class TestDisplayPrimitives:
    def test_sparkline_flat_series_renders_midline(self) -> None:
        from aerocapture.training.display import _sparkline

        s = _sparkline([5.0, 5.0, 5.0, 5.0])
        assert s == "▄" * 4  # ▄ midline, not blanks

    def test_sparkline_empty_and_varying(self) -> None:
        from aerocapture.training.display import _sparkline

        assert _sparkline([]) == " " * 30
        s = _sparkline([0.0, 1.0])
        assert s[0] == " " and s[-1] == "█"

    def test_cost_histogram_bins_and_caption(self) -> None:
        from aerocapture.training.display import _cost_histogram

        costs = [10.0] * 50 + [100.0] * 10 + [10000.0] * 2
        glyphs, caption = _cost_histogram(costs, bins=8)
        assert len(glyphs) == 8
        assert glyphs[0] == "█"  # densest bin -> full block
        assert "·" in glyphs  # empty bins as middle dot
        assert "log" in caption and "1e+01" in caption

    def test_cost_histogram_nonfinite_counted(self) -> None:
        from aerocapture.training.display import _cost_histogram

        glyphs, caption = _cost_histogram([10.0, 20.0, float("inf"), float("nan")], bins=4)
        assert "∞×2" in caption  # ∞×2

    def test_cost_histogram_all_nonfinite(self) -> None:
        from aerocapture.training.display import _cost_histogram

        glyphs, caption = _cost_histogram([float("inf"), float("nan")], bins=4)
        assert glyphs == ""
        assert "no finite costs" in caption and "∞×2" in caption

    def test_cost_histogram_flat(self) -> None:
        from aerocapture.training.display import _cost_histogram

        glyphs, caption = _cost_histogram([42.0, 42.0, 42.0], bins=8)
        assert glyphs[0] == "█" and set(glyphs[1:]) == {"·"}

    def test_rate_and_eta_resume_aware(self) -> None:
        from aerocapture.training.display import _rate_and_eta

        rate, remaining = _rate_and_eta(gen=740, start_gen=700, n_gen=2000, elapsed=40.0)
        assert rate == 1.0  # (740-700)/40 -- NOT 740/40
        assert remaining == 1260.0

    def test_rate_and_eta_no_progress(self) -> None:
        from aerocapture.training.display import _rate_and_eta

        rate, remaining = _rate_and_eta(gen=700, start_gen=700, n_gen=2000, elapsed=10.0)
        assert rate == 0.0 and remaining == float("inf")

    def test_rate_and_eta_overshoot_resume(self) -> None:
        from aerocapture.training.display import _rate_and_eta

        rate, remaining = _rate_and_eta(gen=2100, start_gen=2000, n_gen=2000, elapsed=50.0)
        assert remaining == 0.0  # gen > n_gen never yields negative ETA
