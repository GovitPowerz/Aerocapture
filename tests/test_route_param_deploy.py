"""Regression tests for the deploy-path routing rule (deploy_overrides).

Covers all 5 prefix cases, the lateral.max_reversals integer coercion,
the shaping.* -> guidance.command_shaping.enabled side-effect,
and the unprefixed-key behavior (routed for warm_start; skipped for report).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from aerocapture.training.deploy_overrides import load_scaffolding_overrides, overrides_from_params

# ---------------------------------------------------------------------------
# overrides_from_params (the warm-start supervisor path)
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
        result = overrides_from_params(params, "equilibrium_glide")
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
        result = overrides_from_params({"shaping.max_bank_acceleration": 5.0}, "ftc")
        assert result["guidance.command_shaping.enabled"] is True

    def test_unprefixed_routes_to_scheme(self) -> None:
        result = overrides_from_params({"gain": 1e-6}, "energy_controller")
        assert "guidance.energy_controller.gain" in result
        assert result["guidance.energy_controller.gain"] == pytest.approx(1e-6)

    def test_lateral_max_reversals_int_coercion(self) -> None:
        result = overrides_from_params({"lateral.max_reversals": 5.2}, "ftc")
        v = result["guidance.lateral.max_reversals"]
        assert isinstance(v, int)
        assert v == 5


# ---------------------------------------------------------------------------
# load_scaffolding_overrides  (via a tmp directory)
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
        result = load_scaffolding_overrides(scheme_dir)
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
        result = load_scaffolding_overrides(scheme_dir)
        assert "some_unknown_key" not in result
        # Only the prefixed key appears
        assert set(result.keys()) == {"guidance.lateral.tau"}

    def test_shaping_enabled_side_effect(self, tmp_path: Path) -> None:
        params: dict[str, object] = {"shaping.max_bank_acceleration": 5.0}
        scheme_dir = self._make_scheme_dir(tmp_path, params)
        result = load_scaffolding_overrides(scheme_dir)
        assert result["guidance.command_shaping.enabled"] is True

    def test_optimized_toml_present_wins_in_resolve(self, tmp_path: Path) -> None:
        # The 'optimized TOML wins, no overrides' decision lives in resolve_eval_toml;
        # the loader itself just reads best_params.json.
        from aerocapture.training.deploy_overrides import resolve_eval_toml

        scheme_dir = tmp_path / "scheme"
        scheme_dir.mkdir()
        optimized = scheme_dir / "optimized_scheme.toml"
        optimized.write_text("[guidance]\ntype = 'ftc'\n")
        (scheme_dir / "best_params.json").write_text(json.dumps({"nav.density_filter_gain": 0.8}))
        eval_toml, result = resolve_eval_toml(tmp_path / "base.toml", scheme_dir)
        assert eval_toml == optimized
        assert result == {}
        assert load_scaffolding_overrides(scheme_dir) == {"navigation.density_filter_gain": 0.8}

    def test_no_best_params_returns_empty(self, tmp_path: Path) -> None:
        scheme_dir = tmp_path / "scheme"
        scheme_dir.mkdir()
        result = load_scaffolding_overrides(scheme_dir)
        assert result == {}

    def test_lateral_max_reversals_int_coercion(self, tmp_path: Path) -> None:
        params: dict[str, object] = {"lateral.max_reversals": 5.2}
        scheme_dir = self._make_scheme_dir(tmp_path, params)
        result = load_scaffolding_overrides(scheme_dir)
        v = result["guidance.lateral.max_reversals"]
        assert isinstance(v, int)
        assert v == 5


def test_warm_start_pure_helpers_importable_without_pyo3() -> None:
    """warm_start's pure-Python helpers must import without the aerocapture_rs PyO3 build,
    so the no-PyO3 CI 'Python' job can collect this file (it imports a warm_start helper at
    module scope, with no importorskip guard). Regression: a module-level hard
    `import aerocapture_rs` that raised on absence broke collection of this whole file. Run
    in a subprocess so blocking aerocapture_rs can't pollute the session's module cache."""
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import sys
        sys.modules["aerocapture_rs"] = None  # force ImportError on `import aerocapture_rs`
        from aerocapture.training.deploy_overrides import overrides_from_params
        out = overrides_from_params({"lateral.tau": 20.0}, "ftc")
        assert out["guidance.lateral.tau"] == 20.0, out
        print("WARM_START_IMPORT_OK")
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"warm_start not importable without aerocapture_rs:\n{result.stderr}"
    assert "WARM_START_IMPORT_OK" in result.stdout
