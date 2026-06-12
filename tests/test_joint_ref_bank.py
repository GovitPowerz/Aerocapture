"""Joint reference optimization: `ref_bank` as a chromosome gene.

Each individual carries its own constant-bank reference angle; the evaluation
layer generates its reference table (undispersed 1-segment nominal) and injects
a per-individual `data.reference_trajectory` override. The gene must never be
routed into the guidance TOML (Rust would silently drop the unknown key)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from aerocapture.training.param_spaces import JOINT_REF_BANK_SCHEMES, PARAM_SPACES
from aerocapture.training.problem import AerocaptureProblem
from aerocapture.training.train import _setup_param_specs, build_training_config_from_toml

FTC_TOML = "configs/training/msr_aller_ftc_train.toml"


def test_joint_ref_schemes_are_the_table_readers() -> None:
    # fnpag never reads the reference table; it must NOT get a dead gene.
    assert {"ftc", "energy_controller", "pred_guid"} == JOINT_REF_BANK_SCHEMES


class TestSpecInjection:
    def test_ref_bank_appended_for_ftc(self) -> None:
        cfg, toml = build_training_config_from_toml(FTC_TOML)
        toml["reference"] = {"joint_bank": True}
        specs, n = _setup_param_specs(cfg, toml, verbose=False)
        assert specs[-1].name == "ref_bank"
        assert n == len(PARAM_SPACES["ftc"]) + 1
        assert specs[-1].p_min == 55.0 and specs[-1].p_max == 80.0

    def test_custom_bounds(self) -> None:
        cfg, toml = build_training_config_from_toml(FTC_TOML)
        toml["reference"] = {"joint_bank": True, "bank_low": 60.0, "bank_high": 75.0}
        specs, _ = _setup_param_specs(cfg, toml, verbose=False)
        assert specs[-1].p_min == 60.0 and specs[-1].p_max == 75.0

    def test_absent_knob_leaves_specs_unchanged(self) -> None:
        cfg, toml = build_training_config_from_toml(FTC_TOML)
        toml.pop("reference", None)
        specs, _ = _setup_param_specs(cfg, toml, verbose=False)
        assert all(s.name != "ref_bank" for s in specs)

    def test_non_tracking_scheme_rejected(self) -> None:
        cfg, toml = build_training_config_from_toml("configs/training/msr_aller_piecewise_constant_train.toml")
        toml["reference"] = {"joint_bank": True}
        with pytest.raises(SystemExit):
            _setup_param_specs(cfg, toml, verbose=False)


class TestProblemIntegration:
    def _problem(self) -> AerocaptureProblem:
        cfg, toml = build_training_config_from_toml(FTC_TOML)
        toml["reference"] = {"joint_bank": True}
        specs, _ = _setup_param_specs(cfg, toml, verbose=False)
        return AerocaptureProblem(param_specs=specs, toml_path=FTC_TOML, seeds=[1, 2], cost_kwargs={}, scheme="ftc")

    def test_ref_bank_not_routed_to_guidance_toml(self) -> None:
        ov = self._problem()._build_grid_overrides({"ref_bank": 68.0, "capture_damping": 0.05})
        assert not any("ref_bank" in k for k in ov)
        assert ov["guidance.ftc.capture_damping"] == 0.05

    def test_evaluate_injects_per_individual_reference(self) -> None:
        problem = self._problem()
        x = np.full(problem.n_var, 0.5)
        costs = problem.evaluate_individual_per_seed(x, [11, 12])
        assert np.all(np.isfinite(costs))
        assert problem._ref_table_dir is not None
        table = problem._ref_table_dir / "ref_bank_0000.dat"
        assert table.exists()
        d = np.loadtxt(table)
        assert np.abs(d[:, 0]).max() < 100.0  # MJ/kg contract

    def test_joint_ref_disabled_without_gene(self) -> None:
        problem = AerocaptureProblem(param_specs=PARAM_SPACES["ftc"], toml_path=FTC_TOML, seeds=[1], cost_kwargs={}, scheme="ftc")
        assert problem._joint_ref_bank is False


def test_mission_name_resolves_through_nested_bases() -> None:
    # The joint config inherits the mission via msr_aller_ftc_train.toml; the
    # shallow scan saw no missions/ entry and derived a bogus mission name.
    from aerocapture.training.toml_utils import find_mission_name

    assert find_mission_name(Path("configs/training/msr_aller_ftc_joint_ref_train.toml")) == "mars"
    assert find_mission_name(Path("configs/training/msr_aller_ftc_train.toml")) == "mars"


def test_generate_constant_bank_tables_batched(tmp_path: Path) -> None:
    from aerocapture.training.reference import generate_constant_bank_tables
    from aerocapture.training.toml_utils import load_toml_with_bases

    mc = load_toml_with_bases(Path(FTC_TOML)).get("monte_carlo", {})
    paths = generate_constant_bank_tables(FTC_TOML, [68.0, 64.0], mc, tmp_path)
    assert [p.name for p in paths] == ["ref_bank_0000.dat", "ref_bank_0001.dat"]
    d68, d64 = np.loadtxt(paths[0]), np.loadtxt(paths[1])
    # steeper bank digs deeper: more energy shed on the nominal
    assert d68[:, 0].min() < d64[:, 0].min()
    # constant commanded-cos feedforward
    assert np.allclose(d68[:, 6], np.cos(np.radians(68.0)))


class TestDeployOptimizedArtifacts:
    """deploy_optimized_artifacts: optimized TOML + joint-ref table deploy (shared
    by train.py main() and the final_select CLI -- the CLI previously left a stale
    reference table behind a re-selected ref_bank)."""

    @staticmethod
    def _fake_tables(content: str) -> Callable[..., list[Path]]:
        def fake(toml_path: str, banks_deg: list[float], mc_config: dict, out_dir: Path, sim_timeout_secs: float | None = None) -> list[Path]:
            p = Path(out_dir) / "ref_bank_0000.dat"
            p.write_text(content)
            return [p]

        return fake

    def test_empty_reference_table_refused(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A winner whose constant-bank nominal produced no trajectory must not be
        deployed: the Rust loader silently accepts a 0-point table and interpolates 0.0."""
        from aerocapture.training import reference as ref_mod
        from aerocapture.training.train import deploy_optimized_artifacts

        cfg, toml = build_training_config_from_toml(FTC_TOML)
        monkeypatch.setattr(ref_mod, "generate_constant_bank_tables", self._fake_tables(""))
        with pytest.raises(RuntimeError, match="empty reference table"):
            deploy_optimized_artifacts({"gain": 1.0, "ref_bank": 67.0}, cfg, toml, tmp_path, Path(FTC_TOML), verbose=False)

    def test_deploys_table_and_rewires_toml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import tomllib

        from aerocapture.training import reference as ref_mod
        from aerocapture.training.train import deploy_optimized_artifacts

        cfg, toml = build_training_config_from_toml(FTC_TOML)
        row = "  1.0 2.0 3.0 3.0 0.5 10.0 0.42\n"
        monkeypatch.setattr(ref_mod, "generate_constant_bank_tables", self._fake_tables(row))
        deploy_optimized_artifacts({"gain": 1.0, "ref_bank": 67.0}, cfg, toml, tmp_path, Path(FTC_TOML), verbose=False)

        deploy_ref = tmp_path / "ref_trajectory.dat"
        assert deploy_ref.exists() and deploy_ref.read_text() == row
        opt = tomllib.loads((tmp_path / "optimized_ftc.toml").read_text())
        assert opt["data"]["reference_trajectory"] == str(deploy_ref)
        # the gene must never reach a guidance TOML key (Rust drops unknown keys)
        assert "ref_bank" not in opt.get("guidance", {}).get("ftc", {})


def test_run_grid_honors_reference_trajectory_override(tmp_path: Path) -> None:
    """run_grid must honor per-individual `data.reference_trajectory` overrides.

    This is the joint ref_bank gene's delivery mechanism: without it the gene is
    DEAD -- every individual trains/validates against the base mission table while
    the deploy/report/compare paths use the winner's own table (the documented
    gains<->reference co-adaptation trap)."""
    import aerocapture_rs
    from aerocapture.training.reference import generate_constant_bank_tables
    from aerocapture.training.toml_utils import load_toml_with_bases

    mc = load_toml_with_bases(Path(FTC_TOML)).get("monte_carlo", {})
    [tbl] = generate_constant_bank_tables(FTC_TOML, [72.0], mc, tmp_path)
    assert tbl.stat().st_size > 0

    base = np.asarray(aerocapture_rs.run_grid(FTC_TOML, [{}], [123]))
    injected = np.asarray(aerocapture_rs.run_grid(FTC_TOML, [{"data.reference_trajectory": str(tbl)}], [123]))
    assert base.shape == injected.shape
    assert not np.allclose(base[0, 0], injected[0, 0]), "run_grid silently ignored the per-individual reference-trajectory override"
