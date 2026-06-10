"""Unit tests for end-of-training final selection (pure rule, no Rust)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from aerocapture.training.final_select import (
    KnownCandidate,
    SelectionResult,
    format_selection_summary,
    select_final_individual,
    write_final_selection_json,
)


class _MockProblem:
    """Per-seed cost = sum(x) + 0.001 * seed. Records every evaluated row."""

    def __init__(self) -> None:
        self.evaluated: list[np.ndarray] = []

    def evaluate_individual_per_seed(self, x: np.ndarray, seeds: list[int]) -> np.ndarray:
        self.evaluated.append(x.copy())
        return np.array([float(np.sum(x)) + 0.001 * s for s in seeds], dtype=np.float64)


def _rms(problem_free_x: np.ndarray, seeds: list[int]) -> float:
    costs = np.array([float(np.sum(problem_free_x)) + 0.001 * s for s in seeds])
    return float(np.sqrt(np.mean(costs**2)))


SEEDS = [1, 2, 3, 4]


class TestSelectionRule:
    def test_fresh_candidate_promotes_on_strict_improvement(self) -> None:
        problem = _MockProblem()
        champ = np.full(4, 0.5)
        cands = np.vstack([np.full(4, 0.4), np.full(4, 0.6)])
        known = [KnownCandidate(x=champ, provenance="champion", val_rms=_rms(champ, SEEDS))]
        sel = select_final_individual(problem, cands, ["last_gen[0]", "last_gen[1]"], known, SEEDS)
        assert sel.promoted
        assert sel.provenance == "last_gen[0]"
        assert sel.winner_index == 0
        assert np.array_equal(sel.individual, cands[0])
        assert sel.val_rms == _rms(cands[0], SEEDS)

    def test_champion_kept_when_no_candidate_beats_it(self) -> None:
        problem = _MockProblem()
        champ = np.full(4, 0.1)
        cands = np.vstack([np.full(4, 0.4), np.full(4, 0.6)])
        known = [KnownCandidate(x=champ, provenance="champion", val_rms=_rms(champ, SEEDS))]
        sel = select_final_individual(problem, cands, ["last_gen[0]", "last_gen[1]"], known, SEEDS)
        assert not sel.promoted
        assert sel.provenance == "champion"
        assert sel.winner_index is None
        assert np.array_equal(sel.individual, champ)

    def test_tie_keeps_champion(self) -> None:
        problem = _MockProblem()
        champ = np.full(4, 0.4)
        cands = np.vstack([np.full(4, 0.4) + np.array([0.1, -0.1, 0.0, 0.0])])  # same sum -> same rms
        known = [KnownCandidate(x=champ, provenance="champion", val_rms=_rms(champ, SEEDS))]
        sel = select_final_individual(problem, cands, ["last_gen[0]"], known, SEEDS)
        assert not sel.promoted
        assert sel.provenance == "champion"

    def test_champion_never_resimulated_and_duplicates_deduped(self) -> None:
        problem = _MockProblem()
        champ = np.full(4, 0.5)
        cands = np.vstack([champ, np.full(4, 0.4), np.full(4, 0.4), np.full(4, 0.6)])
        known = [KnownCandidate(x=champ, provenance="champion", val_rms=_rms(champ, SEEDS))]
        sel = select_final_individual(problem, cands, ["last_gen[0]", "last_gen[1]", "last_gen[2]", "last_gen[3]"], known, SEEDS)
        # champ row + one duplicate skipped: only 2 unique fresh rows simulated
        assert len(problem.evaluated) == 2
        assert sel.n_candidates == 4
        assert sel.n_deduped == 2
        assert sel.promoted and sel.provenance == "last_gen[1]"

    def test_all_nonfinite_candidates_keep_champion(self) -> None:
        class _NaNProblem(_MockProblem):
            def evaluate_individual_per_seed(self, x: np.ndarray, seeds: list[int]) -> np.ndarray:
                self.evaluated.append(x.copy())
                return np.full(len(seeds), np.nan)

        problem = _NaNProblem()
        champ = np.full(4, 0.5)
        known = [KnownCandidate(x=champ, provenance="champion", val_rms=_rms(champ, SEEDS))]
        sel = select_final_individual(problem, np.vstack([np.full(4, 0.4)]), ["last_gen[0]"], known, SEEDS)
        assert not sel.promoted
        assert sel.provenance == "champion"
        # non-finite candidates are visible in the sidecar records as null val_rms
        assert any(e["val_rms"] is None for e in sel.candidate_rms)
        assert np.array_equal(sel.individual, champ)

    def test_no_known_candidates_promotes_best_fresh(self) -> None:
        problem = _MockProblem()
        cands = np.vstack([np.full(4, 0.6), np.full(4, 0.4)])
        sel = select_final_individual(problem, cands, ["last_gen[0]", "last_gen[1]"], [], SEEDS)
        assert sel.promoted
        assert sel.provenance == "last_gen[1]"

    def test_no_known_and_no_finite_raises(self) -> None:
        class _NaNProblem(_MockProblem):
            def evaluate_individual_per_seed(self, x: np.ndarray, seeds: list[int]) -> np.ndarray:
                return np.full(len(seeds), np.nan)

        with pytest.raises(ValueError, match="no finite candidate"):
            select_final_individual(_NaNProblem(), np.vstack([np.full(4, 0.4)]), ["last_gen[0]"], [], SEEDS)

    def test_islands_known_multiple_champions_incumbent_is_lowest(self) -> None:
        problem = _MockProblem()
        c_pso = np.full(4, 0.2)
        c_ga = np.full(4, 0.3)
        known = [
            KnownCandidate(x=c_ga, provenance="ga:champion", val_rms=_rms(c_ga, SEEDS)),
            KnownCandidate(x=c_pso, provenance="pso:champion", val_rms=_rms(c_pso, SEEDS)),
        ]
        cands = np.vstack([np.full(4, 0.25)])  # beats ga champion, not pso champion
        sel = select_final_individual(problem, cands, ["de:last_gen[0]"], known, SEEDS)
        assert not sel.promoted
        assert sel.provenance == "pso:champion"

    def test_candidate_rms_records_everyone(self) -> None:
        problem = _MockProblem()
        champ = np.full(4, 0.5)
        cands = np.vstack([np.full(4, 0.4)])
        known = [KnownCandidate(x=champ, provenance="champion", val_rms=_rms(champ, SEEDS))]
        sel = select_final_individual(problem, cands, ["last_gen[0]"], known, SEEDS)
        provs = {e["provenance"] for e in sel.candidate_rms}
        assert provs == {"champion", "last_gen[0]"}
        # ordering contract: known candidates first, then fresh in candidate order
        assert [e["provenance"] for e in sel.candidate_rms] == ["champion", "last_gen[0]"]


class TestSidecarAndSummary:
    def _result(self) -> SelectionResult:
        problem = _MockProblem()
        champ = np.full(4, 0.5)
        cands = np.vstack([np.full(4, 0.4)])
        known = [KnownCandidate(x=champ, provenance="champion", val_rms=_rms(champ, SEEDS))]
        return select_final_individual(problem, cands, ["last_gen[0]"], known, SEEDS)

    def test_sidecar_schema(self, tmp_path: Path) -> None:
        sel = self._result()
        write_final_selection_json(tmp_path, sel, n_val_seeds=len(SEEDS))
        data = json.loads((tmp_path / "final_selection.json").read_text())
        assert data["winner"]["provenance"] == "last_gen[0]"
        assert data["winner"]["promoted"] is True
        assert data["validation_n_sims"] == 4
        assert data["n_candidates"] == 1
        assert isinstance(data["candidate_rms"], list) and len(data["candidate_rms"]) == 2

    def test_summary_mentions_winner_and_delta(self) -> None:
        sel = self._result()
        text = format_selection_summary(sel)
        assert "last_gen[0]" in text
        assert "Final selection" in text


class TestWarmStartBoundsLoader:
    def test_loads_specs_from_sidecar(self, tmp_path: Path) -> None:
        from aerocapture.training.warm_start import load_warm_start_bounds

        (tmp_path / "warm_start_bounds.json").write_text(
            json.dumps(
                [
                    {"name": "w_0", "p_min": -2.0, "p_max": 2.0, "default": 0.0},
                    {"name": "w_1", "p_min": -0.5, "p_max": 0.5, "default": 0.1, "log_scale": False, "is_integer": False},
                ]
            )
        )
        specs = load_warm_start_bounds(tmp_path)
        assert specs is not None
        assert [s.name for s in specs] == ["w_0", "w_1"]
        assert specs[0].p_min == -2.0 and specs[1].p_max == 0.5

    def test_returns_none_when_absent(self, tmp_path: Path) -> None:
        from aerocapture.training.warm_start import load_warm_start_bounds

        assert load_warm_start_bounds(tmp_path) is None


class TestSharedTrainHelpers:
    def test_build_cost_kwargs_defaults_and_overrides(self) -> None:
        from aerocapture.training.train import build_cost_kwargs

        kw = build_cost_kwargs({})
        assert kw["dv_threshold"] == 1000.0
        assert kw["cost_transform"] == "linear"
        kw2 = build_cost_kwargs(
            {
                "cost_function": {"dv_threshold": 500.0, "cost_transform": "log"},
                "flight": {"constraints": {"max_load_factor": 4.0}},
            }
        )
        assert kw2["dv_threshold"] == 500.0
        assert kw2["g_load_limit"] == 4.0
        assert kw2["cost_transform"] == "log"

    def test_write_best_artifacts_non_nn(self, tmp_path: Path) -> None:
        from aerocapture.training.config import TrainingConfig
        from aerocapture.training.param_spaces import ParamSpec
        from aerocapture.training.train import write_best_artifacts

        cfg = TrainingConfig()
        cfg.guidance_type = "equilibrium_glide"
        specs = [
            ParamSpec(name="gain", p_min=0.0, p_max=2.0, default=1.0),
            ParamSpec(name="bias", p_min=-1.0, p_max=1.0, default=0.0),
        ]
        write_best_artifacts(np.array([0.5, 0.75]), cfg, specs, tmp_path, cwd=None)
        params = json.loads((tmp_path / "best_params.json").read_text())
        assert params["gain"] == 1.0  # 0.5 of [0, 2]
        assert params["bias"] == 0.5  # 0.75 of [-1, 1]
