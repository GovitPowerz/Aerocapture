"""Smoke tests for LiveDisplay — verify no crash, not visual correctness."""

from __future__ import annotations

from unittest.mock import MagicMock

from aerocapture.training.display import LiveDisplay, NoopDisplay, create_display


class TestNoopDisplay:
    def test_noop_context_manager(self) -> None:
        display = NoopDisplay()
        with display:
            display.update(MagicMock(), current_run=0)

    def test_create_display_returns_noop_in_non_tty(self) -> None:
        display = create_display(scheme="equilibrium_glide", n_runs=1, n_generations=50, enabled=False, algorithm="ga")
        assert isinstance(display, NoopDisplay)

    def test_is_live_flag(self) -> None:
        """train() gates the headless per-gen heartbeat print on `not display.is_live`."""
        assert NoopDisplay().is_live is False
        assert LiveDisplay(scheme="s", n_runs=1, n_generations=1).is_live is True


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
    block = {"min": 1.0, "p50": 2.0, "p95": 3.0, "s3sigma": 3.5, "mean": 2.2, "max": 4.0}
    return {
        "n_sims": 1000,
        "n_captured": 968,
        "capture_rate": 0.968,
        "cost": {"min": 38.1, "p50": 112.4, "p95": 387.2, "s3sigma": 9000.0, "rms": 181.2, "max": 12000.0},
        "captured": {
            "dv": {"min": 62.0, "p50": 118.2, "p95": 342.0, "s3sigma": 700.0, "mean": 141.7, "max": 980.4},
            "dv1": dict(block),
            "dv2": dict(block),
            "dv3": dict(block),
            "apoapsis": {"p50": 41.2, "p95": 96.0, "mean": 50.0},
            "periapsis": {"p50": 5.0, "p95": 9.0, "mean": 6.0},
            "inclination": {"p50": 0.1, "p95": 0.3, "mean": 0.15},
        },
        "constraints": {
            "heat_flux": {"p50": 142.1, "p95": 188.4, "s3sigma": 200.0, "max": 204.9, "limit": 200.0, "viol_pct": 2.1},
            "g_load": {"p50": 6.0, "p95": 9.8, "s3sigma": 10.9, "max": 11.2, "limit": 15.0, "viol_pct": 0.0},
            "heat_load": {"p50": 9000.0, "p95": 14000.0, "s3sigma": 15800.0, "max": 16000.0, "limit": None, "viol_pct": None},
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
        assert grid_header[1] == ["min", "p50", "p95", "3σ", "max"] and grid_header[2] == "dim"
        q_row = next(r for r in rows if r[0] == "Q")
        assert q_row[2] == "red"  # 2.1% violation
        assert any("2.1% > 200" in c for c in q_row[1])
        g_row = next(r for r in rows if r[0] == "G")
        assert g_row[2] == "dim"  # zero violations
        cost_row = next(r for r in rows if r[0] == "Cost")
        assert cost_row[2] == "yellow"  # max 12000 > 10x p95 387.2 -> outlier hint
        dv1_row = next(r for r in rows if r[0] == "DV1")
        assert dv1_row[1] == ["1.0", "2.0", "3.0", "3.5", "4.0"]

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
        # Right-aligned grid columns: the 5 value cells' right edges must coincide
        # with the 5 header cells' right edges. Detect right edges directly from
        # the rendered strings (no layout math in the assertion) and compare the
        # trailing 5 (the value row carries a leading label token the header lacks).
        assert _grid_right_edges(header)[-5:] == _grid_right_edges(cost)[-5:]
        # Sanity: `max`/`12000.0` share the last (worst-case-width) edge.
        assert header.rstrip().endswith("max") and cost.rstrip().endswith("12000.0")


def _record(gen: int, **over: object) -> dict:
    rec: dict = {
        "generation": gen,
        "best_cost": 334.5,
        "mean_cost": 480.07,
        "worst_cost": 21300.0,
        "std_cost": 3100.0,
        "capture_rate": 1.0,
        "population_diversity": 0.42,
        "improvement": False,
        "best_params": {"gain": 1.24, "tau": 8.31, "thr": 2.05, "k4": 0.1, "k5": 0.2},
        "all_costs": [300.0 + 10.0 * i for i in range(64)],
        "gen_elapsed_s": 0.83,
        "pool_metrics": {"pool_size": 20, "last_curation_gen": 720},
    }
    rec.update(over)
    return rec


def _val_record(gen: int, promoted: bool, rms: float = 181.2) -> dict:
    return _record(
        gen,
        improvement=promoted,
        validation={
            "rms_cost": rms,
            "mean_cost": 142.9,
            "median_cost": 112.0,
            "std_cost": 80.0,
            "p95_cost": 355.7,
            "worst_cost": 900.0,
            "capture_rate": 0.968,
            "n_sims": 1000,
        },
        validation_summary=_summary_fixture(),
    )


def _logger_with(records: list[dict]) -> MagicMock:
    logger = MagicMock()
    logger.buffer = records
    return logger


def _render(renderable: object, width: int = 120) -> str:
    from rich.console import Console

    console = Console(record=True, width=width, force_terminal=True)
    console.print(renderable)
    return console.export_text()


class TestDashboard:
    def _display(self) -> LiveDisplay:
        d = LiveDisplay(scheme="ftc", n_runs=1, n_generations=2000, algorithm="qpso")
        d.set_start_gen(700)
        return d

    def test_dashboard_renders_key_fragments(self) -> None:
        # Two validation records: the PROMOTED one (lower rms) feeds the grey
        # "Best validation" panel, the later REJECTED one the red-framed
        # "Last validation" panel.
        records = [_record(g) for g in range(701, 733)] + [
            _val_record(704, promoted=True, rms=168.4),
            _val_record(733, promoted=False, rms=181.2),
            _record(740),
        ]
        out = _render(self._display()._build_dashboard(_logger_with(records), current_run=0), width=130)
        assert "ftc" in out and "qpso" in out
        assert "pop 64" in out
        assert "Optimization" in out
        assert "Last validation" in out and "Best validation" in out
        assert "REJECTED" in out
        assert "DV2" in out and "DV3" in out
        assert "2.1% > 200" in out
        assert "pool refresh g720" in out
        assert "gen wall 0.83s" in out
        assert "Run 1/1" not in out  # vestigial fragment removed

    def test_validation_panels_side_by_side_below_optimization(self) -> None:
        # Layout: header and Optimization span the full width; the two
        # validation panels sit side by side BELOW the optimization panel,
        # splitting the same width 50/50.
        records = [_record(g) for g in range(701, 733)] + [
            _val_record(704, promoted=True, rms=168.4),
            _val_record(733, promoted=False, rms=181.2),
        ]
        out = _render(self._display()._build_dashboard(_logger_with(records), current_run=0), width=130)
        lines = out.splitlines()
        opt_line = next(i for i, line in enumerate(lines) if "Optimization" in line)
        val_line = next(i for i, line in enumerate(lines) if "Last validation" in line)
        assert val_line > opt_line  # validation row below the optimization panel
        assert "Best validation" in lines[val_line]  # side by side, not stacked
        # all three frames share the full console width: their right borders align
        opt_width = len(lines[opt_line].rstrip())
        val_width = len(lines[val_line].rstrip())
        assert opt_width == val_width == 130

    def test_footer_truncates_params(self) -> None:
        out = _render(self._display()._build_footer([_record(740)]))
        assert "(+2 more)" in out  # 5 params -> 3 shown
        assert "k5" not in out

    def test_footer_suppresses_nn_params(self) -> None:
        d = LiveDisplay(scheme="neural_network", n_runs=1, n_generations=100, algorithm="pso")
        rec = _record(5, best_params={f"w_{i}": 0.1 for i in range(515)})
        out = _render(d._build_footer([rec]))
        assert "515 NN params" in out
        assert "w_0" not in out

    def test_validation_panel_placeholder_before_first_validation(self) -> None:
        panels = self._display()._build_validation_panels([_record(701)])
        assert len(panels) == 1
        assert _render(panels[0]).find("waiting for first validation") != -1

    def test_validation_panels_last_rejected_framed_red(self) -> None:
        records = [_val_record(704, promoted=True, rms=168.4), _val_record(733, promoted=False, rms=181.2)]
        last, best = self._display()._build_validation_panels(records)
        assert last.border_style == "red"
        out = _render(last)
        assert "Last validation" in out and "REJECTED" in out and "1.8120e+02" in out
        assert best.border_style == "grey50"
        out = _render(best)
        assert "Best validation" in out and "1.6840e+02" in out and "g704" in out

    def test_validation_panels_promoted_framed_green(self) -> None:
        last, best = self._display()._build_validation_panels([_val_record(733, promoted=True)])
        assert last.border_style == "green"
        assert "PROMOTED" in _render(last)
        # single validation: best mirrors the same record, still framed grey
        assert best.border_style == "grey50"
        assert "g733" in _render(best)

    def test_validation_panels_both_show_stats_grid(self) -> None:
        records = [_val_record(704, promoted=True, rms=168.4), _val_record(733, promoted=False, rms=181.2)]
        for panel in self._display()._build_validation_panels(records):
            out = _render(panel)
            assert "DV2" in out and "DV3" in out and "1000 sims" in out

    def test_empty_buffer_renders_waiting(self) -> None:
        out = _render(self._display()._build_dashboard(_logger_with([]), current_run=0))
        assert "Waiting for first generation" in out

    def test_zero_captures_grid_placeholder(self) -> None:
        rec = _val_record(710, promoted=False)
        rec["validation_summary"]["captured"] = None
        rec["validation_summary"]["n_captured"] = 0
        out = _render(self._display()._build_validation_panels([rec])[0])
        assert "0/1000" in out

    def test_update_dispatches_dashboard(self) -> None:
        d = self._display()
        d._live = MagicMock()
        d.update(_logger_with([_record(1)]), current_run=0, island_records=None)
        assert d._live.update.called


def _grid_right_edges(line: str) -> list[int]:
    """Right-edge column indices of whitespace-separated tokens in a line."""
    import re

    return [m.end() for m in re.finditer(r"\S+", line)]


class TestSeedsRow:
    def _display(self, strategy: str = "adaptive", n: int | None = 20) -> LiveDisplay:
        return LiveDisplay(scheme="ftc", n_runs=1, n_generations=2000, algorithm="qpso", seed_strategy=strategy, training_n_sims=n)

    def test_adaptive_with_curation(self) -> None:
        out = _render(self._display()._build_optimization_panel([_record(740)]))
        assert "Seeds" in out
        assert "adaptive" in out
        assert "n 20 · refreshed g720" in out
        assert "pool refresh" not in out  # moved out of the status row

    def test_adaptive_before_first_curation(self) -> None:
        rec = _record(5)
        del rec["pool_metrics"]
        out = _render(self._display()._build_optimization_panel([rec]))
        assert "n 20 · no curation yet" in out

    def test_rotating_and_fixed(self) -> None:
        out_r = _render(self._display(strategy="rotating")._build_optimization_panel([_record(10)]))
        assert "n 20 · fresh every gen" in out_r
        out_f = _render(self._display(strategy="fixed")._build_optimization_panel([_record(10)]))
        assert "n 20 · deterministic" in out_f

    def test_no_strategy_omits_row(self) -> None:
        d = LiveDisplay(scheme="ftc", n_runs=1, n_generations=2000, algorithm="qpso")
        out = _render(d._build_optimization_panel([_record(10)]))
        assert "Seeds" not in out
        assert "pool refresh g720" in out  # legacy status-row fragment kept when no Seeds row

    def test_no_n_sims_drops_prefix(self) -> None:
        out = _render(self._display(n=None)._build_optimization_panel([_record(740)]))
        assert "refreshed g720" in out
        assert "n 20" not in out
