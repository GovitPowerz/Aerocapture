"""Regression tests for deploy-path routing in report.py and warm_start.py.

Covers all 5 prefix cases, the lateral.max_reversals integer coercion,
the shaping.* -> guidance.command_shaping.enabled side-effect,
and the unprefixed-key behavior (routed for warm_start; skipped for report).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from aerocapture.training.report import _load_nn_scaffolding_overrides  # noqa: PLC2701  (private but stable)
from aerocapture.training.warm_start import _build_overrides_for_source  # noqa: PLC2701

# ---------------------------------------------------------------------------
# warm_start._build_overrides_for_source
# ---------------------------------------------------------------------------


class TestBuildOverridesForSource:
    def test_all_prefixes_plus_unprefixed(self) -> None:
        params: dict[str, float] = {
            "lateral.tau": 20.0,
            "lateral.max_reversals": 4.2,  # coerced to int -> 4
            "exit.exit_velocity_threshold": 4000.0,
            "nav.density_filter_gain": 0.9,
            "thermal.heat_flux_activation": 0.8,
            "shaping.max_bank_acceleration": 7.0,
            "k_hdot_scale": 0.25,  # unprefixed -> guidance.{scheme}.*
        }
        result = _build_overrides_for_source(params, "equilibrium_glide")
        assert result == {
            "guidance.lateral.tau": 20.0,
            "guidance.lateral.max_reversals": 4,  # int coercion
            "guidance.ftc.exit_velocity_threshold": 4000.0,
            "navigation.density_filter_gain": 0.9,
            "guidance.thermal_limiter.heat_flux_activation": 0.8,
            "guidance.command_shaping.max_bank_acceleration": 7.0,
            "guidance.command_shaping.enabled": True,  # shaping side-effect
            "guidance.equilibrium_glide.k_hdot_scale": 0.25,
        }

    def test_shaping_enabled_side_effect(self) -> None:
        result = _build_overrides_for_source({"shaping.max_bank_acceleration": 5.0}, "ftc")
        assert result["guidance.command_shaping.enabled"] is True

    def test_unprefixed_routes_to_scheme(self) -> None:
        result = _build_overrides_for_source({"gain": 1e-6}, "energy_controller")
        assert "guidance.energy_controller.gain" in result
        assert result["guidance.energy_controller.gain"] == pytest.approx(1e-6)

    def test_lateral_max_reversals_int_coercion(self) -> None:
        result = _build_overrides_for_source({"lateral.max_reversals": 5.2}, "ftc")
        v = result["guidance.lateral.max_reversals"]
        assert isinstance(v, int)
        assert v == 5


# ---------------------------------------------------------------------------
# report._load_nn_scaffolding_overrides  (via a tmp directory)
# ---------------------------------------------------------------------------


class TestLoadNnScaffoldingOverrides:
    def _make_scheme_dir(self, tmp_path: Path, params: dict[str, object]) -> Path:
        scheme_dir = tmp_path / "scheme"
        scheme_dir.mkdir()
        (scheme_dir / "best_params.json").write_text(json.dumps(params))
        return scheme_dir

    def test_all_prefixes_routed(self, tmp_path: Path) -> None:
        params: dict[str, object] = {
            "lateral.tau": 20.0,
            "lateral.max_reversals": 4.2,  # coerced to int -> 4
            "exit.exit_velocity_threshold": 4000.0,
            "nav.density_filter_gain": 0.9,
            "thermal.heat_flux_activation": 0.8,
            "shaping.max_bank_acceleration": 7.0,
        }
        scheme_dir = self._make_scheme_dir(tmp_path, params)
        result = _load_nn_scaffolding_overrides(scheme_dir, scheme_dir / "optimized_nn.toml")
        assert result == {
            "guidance.lateral.tau": 20.0,
            "guidance.lateral.max_reversals": 4,  # int coercion
            "guidance.ftc.exit_velocity_threshold": 4000.0,
            "navigation.density_filter_gain": 0.9,
            "guidance.thermal_limiter.heat_flux_activation": 0.8,
            "guidance.command_shaping.max_bank_acceleration": 7.0,
            "guidance.command_shaping.enabled": True,
        }

    def test_unprefixed_key_skipped(self, tmp_path: Path) -> None:
        """Keys without a recognized prefix must be absent (not routed to guidance.{scheme}.*)."""
        params: dict[str, object] = {
            "lateral.tau": 15.0,
            "some_unknown_key": 99.0,  # no prefix -> must be skipped
        }
        scheme_dir = self._make_scheme_dir(tmp_path, params)
        result = _load_nn_scaffolding_overrides(scheme_dir, scheme_dir / "optimized_nn.toml")
        assert "some_unknown_key" not in result
        # Only the prefixed key appears
        assert set(result.keys()) == {"guidance.lateral.tau"}

    def test_shaping_enabled_side_effect(self, tmp_path: Path) -> None:
        params: dict[str, object] = {"shaping.max_bank_acceleration": 5.0}
        scheme_dir = self._make_scheme_dir(tmp_path, params)
        result = _load_nn_scaffolding_overrides(scheme_dir, scheme_dir / "optimized_nn.toml")
        assert result["guidance.command_shaping.enabled"] is True

    def test_optimized_toml_present_returns_empty(self, tmp_path: Path) -> None:
        scheme_dir = tmp_path / "scheme"
        scheme_dir.mkdir()
        optimized = scheme_dir / "optimized_nn.toml"
        optimized.write_text("[guidance]\ntype = 'neural_network'\n")
        (scheme_dir / "best_params.json").write_text(json.dumps({"nav.density_filter_gain": 0.8}))
        result = _load_nn_scaffolding_overrides(scheme_dir, optimized)
        assert result == {}

    def test_no_best_params_returns_empty(self, tmp_path: Path) -> None:
        scheme_dir = tmp_path / "scheme"
        scheme_dir.mkdir()
        result = _load_nn_scaffolding_overrides(scheme_dir, scheme_dir / "optimized_nn.toml")
        assert result == {}

    def test_lateral_max_reversals_int_coercion(self, tmp_path: Path) -> None:
        params: dict[str, object] = {"lateral.max_reversals": 5.2}
        scheme_dir = self._make_scheme_dir(tmp_path, params)
        result = _load_nn_scaffolding_overrides(scheme_dir, scheme_dir / "optimized_nn.toml")
        v = result["guidance.lateral.max_reversals"]
        assert isinstance(v, int)
        assert v == 5
