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


def _summary_fixture() -> dict:
    block = {"min": 1.0, "p50": 2.0, "p95": 3.0, "mean": 2.2, "max": 4.0}
    return {
        "n_sims": 1000,
        "n_captured": 968,
        "capture_rate": 0.968,
        "cost": {"min": 38.1, "p50": 112.4, "p95": 387.2, "rms": 181.2, "max": 12000.0},
        "captured": {
            "dv": {"min": 62.0, "p50": 118.2, "p95": 342.0, "mean": 141.7, "max": 980.4},
            "dv1": dict(block),
            "dv2": dict(block),
            "dv3": dict(block),
            "apoapsis": {"p50": 41.2, "p95": 96.0, "mean": 50.0},
            "periapsis": {"p50": 5.0, "p95": 9.0, "mean": 6.0},
            "inclination": {"p50": 0.1, "p95": 0.3, "mean": 0.15},
        },
        "constraints": {
            "heat_flux": {"p50": 142.1, "p95": 188.4, "max": 204.9, "limit": 200.0, "viol_pct": 2.1},
            "g_load": {"p50": 6.0, "p95": 9.8, "max": 11.2, "limit": 15.0, "viol_pct": 0.0},
            "heat_load": {"p50": 9000.0, "p95": 14000.0, "max": 16000.0, "limit": None, "viol_pct": None},
        },
    }


class TestValidationSummaryRows:
    def test_full_summary_rows(self) -> None:
        from aerocapture.training.display import _validation_summary_rows

        rows = _validation_summary_rows(_summary_fixture())
        labels = [r[0] for r in rows]
        assert labels[:3] == ["Cap", "", "Cost"]
        assert "DV" in labels and "DV1" in labels and "DV2" in labels and "DV3" in labels
        assert "Apo" in labels and "Q" in labels and "G" in labels and "HL" in labels
        grid_header = next(r for r in rows if r[0] == "")
        assert grid_header[1] == ["min", "p50", "p95", "max"] and grid_header[2] == "dim"
        q_row = next(r for r in rows if r[0] == "Q")
        assert q_row[2] == "red"  # 2.1% violation
        assert any("2.1% > 200" in c for c in q_row[1])
        g_row = next(r for r in rows if r[0] == "G")
        assert g_row[2] == "dim"  # zero violations
        cost_row = next(r for r in rows if r[0] == "Cost")
        assert cost_row[2] == "yellow"  # max 12000 > 10x p95 387.2 -> outlier hint
        dv1_row = next(r for r in rows if r[0] == "DV1")
        assert dv1_row[1] == ["1.0", "2.0", "3.0", "4.0"]

    def test_captured_none_renders_placeholder(self) -> None:
        from aerocapture.training.display import _validation_summary_rows

        s = _summary_fixture()
        s["captured"] = None
        s["n_captured"] = 0
        rows = _validation_summary_rows(s)
        cap_row = rows[0]
        assert cap_row[0] == "Cap" and cap_row[2] == "red"
        dv_row = next(r for r in rows if r[0] == "DV")
        assert dv_row[1] == ["—"] and dv_row[2] == "dim"

    def test_missing_limits_render_na_style(self) -> None:
        from aerocapture.training.display import _validation_summary_rows

        rows = _validation_summary_rows(_summary_fixture())
        hl_row = next(r for r in rows if r[0] == "HL")
        assert hl_row[2] == "" and not any(">" in c for c in hl_row[1])

    def test_old_formatter_gone(self) -> None:
        import aerocapture.training.display as d

        assert not hasattr(d, "_format_validation_summary")

    def test_grid_columns_aligned(self) -> None:
        from aerocapture.training.display import _rows_to_text

        lines = str(_rows_to_text(_summary_fixture())).splitlines()
        header = next(line for line in lines if line.lstrip().startswith("min "))
        # `12000.0` is the widest value cell; the Cost row exercises the worst case.
        cost = next(line for line in lines if line.lstrip().startswith("Cost "))
        # Right-aligned grid columns: the 4 value cells' right edges must coincide
        # with the 4 header cells' right edges. Detect right edges directly from
        # the rendered strings (no layout math in the assertion) and compare the
        # trailing 4 (the value row carries a leading label token the header lacks).
        assert _grid_right_edges(header)[-4:] == _grid_right_edges(cost)[-4:]
        # Sanity: `max`/`12000.0` share the last (worst-case-width) edge.
        assert header.rstrip().endswith("max") and cost.rstrip().endswith("12000.0")


def _grid_right_edges(line: str) -> list[int]:
    """Right-edge column indices of whitespace-separated tokens in a line."""
    import re

    return [m.end() for m in re.finditer(r"\S+", line)]
