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


class TestCheckpointIO:
    def _make_single_algo_ckpt(self, d: Path) -> None:
        meta = {"generation": 7, "best_cost": 1.5, "best_val_cost": 2.5, "cost_history": [], "rng_state": None}
        (d / "checkpoint_g00007.json").write_text(json.dumps(meta))
        np.savez(
            d / "checkpoint_g00007.npz",
            population=np.full((4, 3), 0.5),
            costs=np.array([1.0, 2.0, 3.0, 4.0]),
            best_individual=np.full(3, 0.9),
        )

    def test_load_single_algo(self, tmp_path: Path) -> None:
        from aerocapture.training.final_select import load_selection_state

        self._make_single_algo_ckpt(tmp_path)
        state = load_selection_state(tmp_path)
        assert state.kind == "single"
        assert state.population.shape == (4, 3)
        assert len(state.known) == 1
        assert state.known[0].provenance == "champion"
        assert state.known[0].val_rms == 2.5

    def test_patch_single_algo(self, tmp_path: Path) -> None:
        from aerocapture.training.final_select import load_selection_state, patch_checkpoint

        self._make_single_algo_ckpt(tmp_path)
        state = load_selection_state(tmp_path)
        new_best = np.full(3, 0.1)
        patch_checkpoint(state, new_best, new_val_rms=1.25)
        data = np.load(tmp_path / "checkpoint_g00007.npz")
        assert np.array_equal(data["best_individual"], new_best)
        assert np.array_equal(data["population"], np.full((4, 3), 0.5))  # untouched
        meta = json.loads((tmp_path / "checkpoint_g00007.json").read_text())
        assert meta["best_val_cost"] == 1.25
        assert meta["generation"] == 7  # untouched

    def _make_islands_ckpt(self, d: Path) -> None:
        import pickle

        states = [
            {
                "name": "pso",
                "pop_X": np.full((3, 3), 0.4),
                "pop_F": np.array([[1.0], [2.0], [3.0]]),
                "best_overall_individual": np.full(3, 0.45),
                "best_val_cost": 3.0,
            },
            {
                "name": "ga",
                "pop_X": np.full((3, 3), 0.6),
                "pop_F": np.array([[1.0], [2.0], [3.0]]),
                "best_overall_individual": np.full(3, 0.65),
                "best_val_cost": 2.0,
            },
        ]
        np.savez_compressed(
            d / "checkpoint_g00005.npz",
            version=2,
            generation=5,
            base_mc_seed=42,
            cost_transform="linear",
            island_states=np.array(pickle.dumps(states), dtype=object),
        )

    def test_load_islands(self, tmp_path: Path) -> None:
        from aerocapture.training.final_select import load_selection_state

        self._make_islands_ckpt(tmp_path)
        state = load_selection_state(tmp_path)
        assert state.kind == "islands"
        assert state.population.shape == (6, 3)  # union of both pops
        assert {k.provenance for k in state.known} == {"pso:champion", "ga:champion"}
        assert state.base_mc_seed == 42

    def test_patch_islands_winning_island(self, tmp_path: Path) -> None:
        import pickle

        from aerocapture.training.final_select import load_selection_state, patch_checkpoint

        self._make_islands_ckpt(tmp_path)
        state = load_selection_state(tmp_path)
        new_best = np.full(3, 0.2)
        patch_checkpoint(state, new_best, new_val_rms=0.5, island_name="ga")
        data = np.load(tmp_path / "checkpoint_g00005.npz", allow_pickle=True)
        states = pickle.loads(data["island_states"].item())
        ga = next(s for s in states if s["name"] == "ga")
        pso = next(s for s in states if s["name"] == "pso")
        assert np.array_equal(ga["best_overall_individual"], new_best)
        assert ga["best_val_cost"] == 0.5
        assert pso["best_val_cost"] == 3.0  # untouched
        assert np.array_equal(pso["pop_X"], np.full((3, 3), 0.4))  # untouched

    def test_no_checkpoint_raises(self, tmp_path: Path) -> None:
        from aerocapture.training.final_select import load_selection_state

        with pytest.raises(FileNotFoundError):
            load_selection_state(tmp_path)

    def test_crashed_patch_temp_never_shadows_checkpoint(self, tmp_path: Path) -> None:
        """A leftover patch temp must not match the resume globs (it would sort
        after the real checkpoint and shadow it with possibly-torn content)."""
        from aerocapture.training.final_select import load_selection_state

        self._make_single_algo_ckpt(tmp_path)
        # simulate a crashed patch: temp files left behind
        (tmp_path / ".tmp_checkpoint_g00007.json").write_text("{torn")
        np.savez(tmp_path / ".tmp_checkpoint_g00007.npz", junk=np.zeros(1))
        state = load_selection_state(tmp_path)  # must still load the REAL pair
        assert state.npz_path.name == "checkpoint_g00007.npz"
        assert not any(p.name.startswith("checkpoint_g") and ".tmp" in p.name for p in tmp_path.iterdir())


class TestRunFinalSelect:
    def _setup_dir(self, d: Path) -> None:
        meta = {"generation": 7, "best_cost": 1.5, "best_val_cost": _rms(np.full(2, 0.9), [1000001, 1000002]), "cost_history": [], "rng_state": None}
        (d / "checkpoint_g00007.json").write_text(json.dumps(meta))
        np.savez(
            d / "checkpoint_g00007.npz",
            population=np.vstack([np.full(2, 0.2), np.full(2, 0.8)]),
            costs=np.array([1.0, 2.0]),
            best_individual=np.full(2, 0.9),
        )

    def test_round_trip_promotes_and_patches(self, tmp_path: Path) -> None:
        from aerocapture.training.config import TrainingConfig
        from aerocapture.training.final_select import run_final_select
        from aerocapture.training.param_spaces import ParamSpec

        self._setup_dir(tmp_path)
        cfg = TrainingConfig()
        cfg.guidance_type = "equilibrium_glide"
        specs = [ParamSpec(name="a", p_min=0.0, p_max=1.0, default=0.5), ParamSpec(name="b", p_min=0.0, p_max=1.0, default=0.5)]

        sel = run_final_select(
            training_dir=tmp_path,
            config=cfg,
            param_specs=specs,
            problem=_MockProblem(),
            val_seeds=[1000001, 1000002],
            patch=True,
        )
        assert sel.promoted and sel.provenance == "last_gen[0]"
        # artifacts rewritten
        params = json.loads((tmp_path / "best_params.json").read_text())
        assert params["a"] == 0.2
        # checkpoint patched
        data = np.load(tmp_path / "checkpoint_g00007.npz")
        assert np.array_equal(data["best_individual"], np.full(2, 0.2))
        # sidecar present
        assert (tmp_path / "final_selection.json").exists()

    def test_no_patch_leaves_checkpoint(self, tmp_path: Path) -> None:
        from aerocapture.training.config import TrainingConfig
        from aerocapture.training.final_select import run_final_select
        from aerocapture.training.param_spaces import ParamSpec

        self._setup_dir(tmp_path)
        cfg = TrainingConfig()
        cfg.guidance_type = "equilibrium_glide"
        specs = [ParamSpec(name="a", p_min=0.0, p_max=1.0, default=0.5), ParamSpec(name="b", p_min=0.0, p_max=1.0, default=0.5)]
        run_final_select(training_dir=tmp_path, config=cfg, param_specs=specs, problem=_MockProblem(), val_seeds=[1000001, 1000002], patch=False)
        data = np.load(tmp_path / "checkpoint_g00007.npz")
        assert np.array_equal(data["best_individual"], np.full(2, 0.9))  # untouched
