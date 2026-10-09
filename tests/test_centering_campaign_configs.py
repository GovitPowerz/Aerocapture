"""The #177 objective-centering cells under per-scenario noise: the three centered-Mamba seeds and the five
dense lever cells differ from their Section 7.3 parents (`configs/training/paper/objective_centering/`) only
in regime, allocation, deploy path and checkpoint retention; the campaign runner takes their subdirectory
job names; the `--v4` modes of `centered_depth_eval.py` and `objective_centering_eval.py` score exactly those
runs under per_draw alone. The high-regime joint-FTC baseline is a `classical/` cell
(`tests/test_classical_campaign_configs.py`). Pure Python (no bindings): `evaluate_cell` is stubbed.
"""

from __future__ import annotations

import json
import re
import subprocess
import tomllib
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from aerocapture.training import cell_eval, charts
from aerocapture.training.cell_eval import FR_DV_TOTAL, FR_ECC, FR_IFINAL, CellResult
from aerocapture.training.parquet_output import FINAL_RECORD_LEN
from aerocapture.training.toml_utils import load_toml_with_bases

REPO = Path(__file__).resolve().parents[1]
CELL_DIR = REPO / "configs/training/ou_marginal/centered"
PARENT_DIR = REPO / "configs/training/paper/objective_centering"
JOBS_DIR = REPO / "experiments/ou_marginal"
SCRIPTS = REPO / "articles/paper/scripts"
RUNNER = REPO / "experiments/paper/14_objective_centering.sh"
MAKEFILE = REPO / "articles/paper/Makefile"

SEEDS = ("mamba_centered_s1", "mamba_centered_s2", "mamba_centered_s3")
BASELINES = ("jointFTC-medium", "jointFTC-high")
# dense lever cell -> (cost transform, curation bucket, training_n_sims, target gens): one lever flipped at a
# time from the stacked control to centered, iso-compute at B = 256 x n_sims x gens = 8.19 M sims.
DENSE = {
    "dense_stacked": ("cubed", "max", 2, 16000),
    "dense_plus_sims": ("cubed", "max", 16, 2000),
    "dense_plus_bucket": ("cubed", "middle", 2, 16000),
    "dense_plus_transform": ("linear", "max", 2, 16000),
    "dense_centered": ("linear", "middle", 16, 2000),
}
BUDGET = 256 * 2 * 16000
CENTERING_JOBS = [(f"centered/{s}", 4000, i, "") for i, s in enumerate(SEEDS, 1)]
DENSE_JOBS = [(f"centered/{cell}", target, 1, "") for cell, (_, _, _, target) in DENSE.items()]
HIGH_DOMAINS = ("atmosphere", "density_perturbation", "navigation", "nav_filter")


def read_jobs(path: Path) -> list[tuple[str, int, int, str]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, target, seed, src = line.split("|")
        rows.append((name, int(target), int(seed), src))
    return rows


def _strip_leaf_keys(cfg: dict[str, Any]) -> None:
    cfg["monte_carlo"].pop("noise_seeding", None)
    cfg["optimizer"].pop("n_pop")
    cfg["optimizer"].pop("training_n_sims")
    cfg["checkpoints"].pop("keep_last")
    cfg["data"].pop("neural_network")


def test_jobs_files_are_the_registered_lists() -> None:
    assert read_jobs(JOBS_DIR / "jobs_centering.txt") == CENTERING_JOBS
    assert read_jobs(JOBS_DIR / "jobs_centering_dense.txt") == DENSE_JOBS


def test_cell_dir_holds_exactly_the_cells() -> None:
    assert sorted(p.stem for p in CELL_DIR.glob("*.toml")) == sorted((*SEEDS, *DENSE))


def test_campaign_runner_takes_the_subdirectory_job_names() -> None:
    """campaign.sh validates every line against job_re before training; one subdirectory level is allowed."""
    match = re.search(r"^job_re='([^']+)'", (JOBS_DIR / "campaign.sh").read_text(), re.MULTILINE)
    assert match is not None
    job_re = match.group(1)

    def accepted(line: str) -> bool:
        return subprocess.run(["bash", "-c", '[[ "$1" =~ $2 ]]', "_", line, job_re], check=False).returncode == 0

    for name, target, seed, src in CENTERING_JOBS + DENSE_JOBS:
        assert accepted(f"{name}|{target}|{seed}|{src}"), name
    assert accepted("hl_mamba_p962|20000|1|")
    assert not accepted("centered/deeper/mamba_centered_s1|4000|1|")
    assert not accepted("../centered/mamba_centered_s1|4000|1|")


def test_seed_1_leaf_states_regime_and_allocation() -> None:
    leaf = tomllib.loads((CELL_DIR / "mamba_centered_s1.toml").read_text())
    assert leaf == {
        "base": ["../../paper/objective_centering/mamba_centered_high.toml"],
        "monte_carlo": {"noise_seeding": "per_draw"},
        "optimizer": {"n_pop": 256, "training_n_sims": 16},
        "data": {"neural_network": "training_output/ou_marginal/centered/mamba_centered_s1/best_model.json"},
        "checkpoints": {"keep_last": 3},
    }


def test_seed_1_resolves_to_the_section_7_3_recipe_otherwise() -> None:
    resolved = load_toml_with_bases(CELL_DIR / "mamba_centered_s1.toml")
    parent = load_toml_with_bases(PARENT_DIR / "mamba_centered_high.toml")
    o = resolved["optimizer"]
    assert (o["algorithm"], o["seed_strategy"], o["curation_bucket_selection"], o["n_pop"], o["training_n_sims"]) == ("ga", "adaptive", "middle", 256, 16)
    assert resolved["cost_function"]["cost_transform"] == "linear"
    assert all(resolved["monte_carlo"][d]["level"] == "high" for d in HIGH_DOMAINS)
    assert [layer["type"] for layer in resolved["network"]["architecture"]] == ["dense", "mamba", "dense"]
    for cfg in (resolved, parent):
        _strip_leaf_keys(cfg)
    assert resolved == parent


@pytest.mark.parametrize("name", SEEDS[1:])
def test_repeat_differs_from_seed_1_only_in_deploy_path(name: str) -> None:
    seed = load_toml_with_bases(CELL_DIR / f"{name}.toml")
    s1 = load_toml_with_bases(CELL_DIR / "mamba_centered_s1.toml")
    assert seed["data"].pop("neural_network") == f"training_output/ou_marginal/centered/{name}/best_model.json"
    s1["data"].pop("neural_network")
    assert seed == s1


@pytest.mark.parametrize("cell", DENSE)
def test_dense_leaf_states_regime_and_iso_compute_allocation(cell: str) -> None:
    _, _, n_sims, target = DENSE[cell]
    leaf = tomllib.loads((CELL_DIR / f"{cell}.toml").read_text())
    assert leaf == {
        "base": [f"../../paper/objective_centering/{cell}_high.toml"],
        "monte_carlo": {"noise_seeding": "per_draw"},
        "optimizer": {"n_pop": 256, "training_n_sims": n_sims},
        "data": {"neural_network": f"training_output/ou_marginal/centered/{cell}/best_model.json"},
        "checkpoints": {"keep_last": 3},
    }
    assert 256 * n_sims * target == BUDGET


@pytest.mark.parametrize("cell", DENSE)
def test_dense_resolves_to_its_lever_parent_otherwise(cell: str) -> None:
    transform, bucket, n_sims, _ = DENSE[cell]
    resolved = load_toml_with_bases(CELL_DIR / f"{cell}.toml")
    parent = load_toml_with_bases(PARENT_DIR / f"{cell}_high.toml")
    o = resolved["optimizer"]
    assert (o["algorithm"], o["seed_strategy"], o["curation_bucket_selection"], o["training_n_sims"]) == ("ga", "adaptive", bucket, n_sims)
    assert resolved["cost_function"]["cost_transform"] == transform
    assert all(resolved["monte_carlo"][d]["level"] == "high" for d in HIGH_DOMAINS)
    assert all(layer["type"] == "dense" for layer in resolved["network"]["architecture"])
    for cfg in (resolved, parent):
        _strip_leaf_keys(cfg)
    assert resolved == parent


# ---- the two scorers' --v4 modes, with the simulator stubbed ----


class ToySim:
    """Stand-in for `evaluate_cell`: records the overrides of every flight; the same pool flies the same scenarios."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, _cell_dir: Path | None, _toml: Path, seeds: list[int] | None = None, *, pool: tuple[int, int] | None = None, **kw: Any) -> CellResult:
        self.calls.append(dict(kw.get("extra_overrides") or {}))
        seed_list = list(seeds) if seeds is not None else list(range(pool[0], pool[0] + pool[1]))  # type: ignore[index]
        rng = np.random.default_rng(seed_list[0] + len(self.calls))
        n = len(seed_list)
        fr = np.zeros((n, FINAL_RECORD_LEN))
        fr[:, FR_IFINAL] = np.where(rng.random(n) < 0.9, 3.0, 1.0)
        fr[:, FR_ECC] = 0.5
        fr[:, FR_DV_TOTAL] = rng.gamma(4.0, 30.0, n)
        fr[:, charts._FR_MAX_HEAT_FLUX] = np.where(rng.random(n) < 0.25, 250.0, 150.0)  # kW/m2 against the 200 limit
        return CellResult(final_records=fr, dispersions=np.zeros((n, 26)), trajectories=None, seeds=seed_list, toml_path=Path("toy.toml"), overrides={})


@pytest.fixture
def sim(monkeypatch: pytest.MonkeyPatch) -> ToySim:
    s = ToySim()
    monkeypatch.setattr(cell_eval, "evaluate_cell", s)
    return s


@pytest.fixture
def cde(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import centered_depth_eval  # type: ignore[import-not-found]

    yield centered_depth_eval


@pytest.fixture
def oce(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import objective_centering_eval  # type: ignore[import-not-found]

    yield objective_centering_eval


def test_depth_scorer_v4_cells_are_the_per_scenario_runs(cde: ModuleType) -> None:
    assert [label for label, _, _ in cde.V4_SEEDS] == [label for label, _, _ in cde.SEEDS] == list(SEEDS)  # the paper's labels
    assert [run for _, run, _ in cde.V4_SEEDS] == [f"ou_marginal/centered/{s}" for s in SEEDS]
    assert [(label, run) for label, run, _ in cde.V4_BASELINES] == [
        ("jointFTC-medium", "ou_marginal/classical/ftc_joint"),
        ("jointFTC-high", "ou_marginal/classical/ftc_joint_high"),
    ]
    for _, run, toml in cde.V4_SEEDS + cde.V4_BASELINES:
        assert (REPO / toml).is_file(), toml
        assert load_toml_with_bases(REPO / toml)["monte_carlo"]["noise_seeding"] == "per_draw", run
    assert cde.V4_REGIMES == {"per_draw": {"monte_carlo.noise_seeding": "per_draw"}}
    assert cde.V4_OUT == REPO / "articles/paper/data/centered_depth_v4.json"
    assert "centered_depth_eval.py --v4" in MAKEFILE.read_text()


def test_depth_scorer_v4_pairs_every_seed_with_both_baselines_under_per_draw(
    tmp_path: Path, sim: ToySim, cde: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i, (_, run_dir, _) in enumerate(cde.V4_SEEDS + cde.V4_BASELINES):
        d = tmp_path / "training_output" / run_dir
        d.mkdir(parents=True)
        (d / "best_params.json").write_text("{}")
        (d / "best_model.json").write_text(json.dumps({"run": i}))  # one model per run dir
    monkeypatch.setattr(cde, "REPO", tmp_path)
    monkeypatch.setattr(cde, "V4_OUT", tmp_path / "centered_depth_v4.json")
    monkeypatch.setattr(cde, "N_BOOT", 50)
    cde.main(["--v4", "--n-sims", "40"])
    out = json.loads((tmp_path / "centered_depth_v4.json").read_text())
    assert out["regimes"] == {"per_draw": {"monte_carlo.noise_seeding": "per_draw"}} and out["n_sims"] == 40
    assert list(out["cells"]) == ["per_draw"] and list(out["paired"]) == ["per_draw"]
    assert [c["label"] for c in out["cells"]["per_draw"]] == [*SEEDS, *BASELINES]
    for c in out["cells"]["per_draw"]:  # every cell's TOML sets all three limits; only the toy heat flux exceeds its own
        v = c["violation_pct"]
        assert set(v) == {"heat_flux", "g_load", "heat_load"} and 0 < v["heat_flux"] < 100 and v["g_load"] == v["heat_load"] == 0, c["label"]
    assert [(p["a"], p["b"]) for p in out["paired"]["per_draw"]] == [(s, b) for s in SEEDS for b in BASELINES]
    for p in out["paired"]["per_draw"]:
        assert {"delta_capture_pts", "delta_capture_pts_ci", "delta_cvar95", "delta_cvar95_ci"} <= set(p)
    assert sim.calls == [{"monte_carlo.noise_seeding": "per_draw", **cde.STRESS_OVERRIDES}] * 5


def test_depth_scorer_v4_requires_every_cell_deployed(tmp_path: Path, cde: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cde, "REPO", tmp_path)
    with pytest.raises(SystemExit, match="not deployed"):
        cde.main(["--v4", "--n-sims", "40"])


def test_depth_scorer_default_mode_is_the_arxiv_v3_study(cde: ModuleType) -> None:
    assert list(cde.REGIMES) == ["per_draw", "legacy"]
    assert cde.OUT == REPO / "articles/paper/data/centered_depth.json"
    assert all(run.startswith("paper/") for _, run, _ in cde.SEEDS + cde.BASELINES)


def test_lever_scorer_v4_cells_are_the_per_scenario_runs(oce: ModuleType) -> None:
    expected = [(c.removeprefix("dense_"), f"ou_marginal/centered/{c}", n) for c, (_, _, n, _) in DENSE.items()]
    assert [(label, run, n) for label, run, _, n in oce.V4_CELLS] == expected
    for _, _, toml, _ in oce.V4_CELLS:
        assert (REPO / toml).is_file(), toml
    assert oce.V4_OUT == REPO / "articles/paper/data/objective_centering_v4.json"
    assert "objective_centering_eval.py --v4" in MAKEFILE.read_text()


def test_lever_scorer_v4_flies_per_draw_and_states_it(tmp_path: Path, sim: ToySim, oce: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    for _, run_dir, _, _ in oce.V4_CELLS:
        d = tmp_path / "training_output" / run_dir
        d.mkdir(parents=True)
        (d / "final_eval.parquet").write_bytes(b"")
    monkeypatch.setattr(oce, "REPO", tmp_path)
    monkeypatch.setattr(oce, "V4_OUT", tmp_path / "objective_centering_v4.json")
    oce.main(["--v4", "--n-sims", "40"])
    out = json.loads((tmp_path / "objective_centering_v4.json").read_text())
    assert out["noise_seeding"] == "per_draw" and out["n_sims_eval"] == 40
    for c in out["cells"]:  # every lever TOML sets all three limits; only the toy heat flux exceeds its own
        v = c["violation_pct"]
        assert set(v) == {"heat_flux", "g_load", "heat_load"} and 0 < v["heat_flux"] < 100 and v["g_load"] == v["heat_load"] == 0, c["label"]
    assert [c["label"] for c in out["cells"]] == [c.removeprefix("dense_") for c in DENSE]
    assert out["convergence"] == {c.removeprefix("dense_"): [] for c in DENSE}  # no run_*.jsonl in the toy dirs
    assert sim.calls == [{"monte_carlo.noise_seeding": "per_draw", **oce.STRESS_OVERRIDES}] * 5


def test_lever_scorer_v4_requires_every_cell_deployed(tmp_path: Path, sim: ToySim, oce: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    d = tmp_path / "training_output" / oce.V4_CELLS[0][1]
    d.mkdir(parents=True)
    (d / "final_eval.parquet").write_bytes(b"")
    monkeypatch.setattr(oce, "REPO", tmp_path)
    monkeypatch.setattr(oce, "V4_OUT", tmp_path / "objective_centering_v4.json")
    with pytest.raises(SystemExit, match="not deployed"):
        oce.main(["--v4", "--n-sims", "40"])
    assert not (tmp_path / "objective_centering_v4.json").exists()


def test_lever_scorer_convergence_spans_every_launch_of_a_resumed_run(tmp_path: Path, oce: ModuleType) -> None:
    """A resumed run writes one run_*.jsonl per launch; the later one re-logs from its resume generation on."""

    def frag(name: str, gens: range, cap: float) -> None:
        recs = [{"generation": g, "all_costs": [0.0] * 4, "validation": {"capture_rate": cap + g}} for g in gens]
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in recs))

    frag("run_000_20261001T000000.jsonl", range(1, 6), 0.0)  # crashed after g5, last checkpoint g3
    frag("run_000_20261002T000000.jsonl", range(3, 8), 100.0)
    (tmp_path / "run_000_20261003T000000.jsonl").write_text("")  # a zero-generation resume (final selection only)
    records = oce._read_run_log(sorted(str(p) for p in tmp_path.glob("run_*.jsonl")))
    assert oce._derive_n_pop(records, 256) == 4
    assert oce.extract_convergence(records, 4, 2) == [[8 * g, (0.0 if g < 3 else 100.0) + g] for g in range(1, 8)]


def test_lever_scorer_default_mode_flies_the_shared_path(tmp_path: Path, sim: ToySim, oce: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    from aerocapture.training.deploy_overrides import LEGACY_NOISE_REGIME

    d = tmp_path / "training_output" / oce.CELLS[0][1]
    d.mkdir(parents=True)
    (d / "final_eval.parquet").write_bytes(b"")
    monkeypatch.setattr(oce, "REPO", tmp_path)
    monkeypatch.setattr(oce, "OUT", tmp_path / "objective_centering.json")
    oce.main(["--n-sims", "40"])
    out = json.loads((tmp_path / "objective_centering.json").read_text())
    assert "noise_seeding" not in out and all("violation_pct" not in c for c in out["cells"])  # the committed arxiv-v3 file's shape
    assert sim.calls == [{**LEGACY_NOISE_REGIME, **oce.STRESS_OVERRIDES}]


def test_runner_v4_modes_chain_the_campaigns_and_the_scorers() -> None:
    text = RUNNER.read_text()
    v4 = text[text.index("v4() {") : text.index("v4_dense() {")]
    assert v4.index("classical_campaign.sh classical") < v4.index("jobs_centering.txt") < v4.index("centered_depth_eval.py --v4")
    dense = text[text.index("v4_dense() {") : text.index("shared() {")]
    assert dense.index("jobs_centering_dense.txt") < dense.index("objective_centering_eval.py --v4")
    assert subprocess.run(["bash", "-n", str(RUNNER)], capture_output=True, text=True).returncode == 0
