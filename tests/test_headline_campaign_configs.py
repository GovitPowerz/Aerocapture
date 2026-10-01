"""The #173 headline-allocation jobs train what the issue registers, and every campaign jobs file is runnable.

`experiments/ou_marginal/campaign.sh <jobs file>` flies `configs/training/ou_marginal/<name>.toml`
into `training_output/ou_marginal/<name>/`. An `hl_*` cell is its family config under per_draw at
the headline allocation (GA 512 x 2, stated in TOML: a config that base-inherits it does not
inherit CLI flags); the dense fine-tune seeds are `ft_dense_p515` with their own deploy path.
Pure Python (no bindings).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from aerocapture.training.toml_utils import load_toml_with_bases

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs/training/ou_marginal"
JOBS_DIR = REPO / "experiments/ou_marginal"
DENSE_515_SOURCE = "training_output/dense_p515_ga_paper_best"

# hl cell -> family config (relative to configs/training)
FAMILIES = {
    "mamba_p962": "sweep/mamba_p962.toml",
    "dense_p515": "paper/dense_p515_ga.toml",
    "lstm_p1082": "sweep/lstm_p1082.toml",
    "gru_p1014": "sweep/gru_p1014.toml",
    "dense_p972": "sweep/dense_p972.toml",
}
HEADLINE_JOBS = [
    ("hl_mamba_p962", 20000, 1, ""),
    ("hl_dense_p515", 20000, 1, ""),
    ("hl_mamba_p962_s2", 20000, 2, ""),
    ("hl_dense_p515_s2", 20000, 2, ""),
    ("hl_mamba_p962_s3", 20000, 3, ""),
    ("hl_dense_p515_s3", 20000, 3, ""),
    ("ft_dense_p515_s2", 22000, 2, DENSE_515_SOURCE),
    ("ft_dense_p515_s3", 22000, 3, DENSE_515_SOURCE),
    ("hl_lstm_p1082", 20000, 1, ""),
    ("hl_gru_p1014", 20000, 1, ""),
    ("hl_dense_p972", 20000, 1, ""),
]
HL_JOBS = [name for name, *_ in HEADLINE_JOBS if name.startswith("hl_")]
JOBS_FILES = sorted(JOBS_DIR.glob("jobs_*.txt"))


def read_jobs(path: Path) -> list[tuple[str, int, int, str]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, target, seed, src = line.split("|")
        rows.append((name, int(target), int(seed), src))
    return rows


def hl_cell(name: str) -> str:
    return re.sub(r"_s\d$", "", name.removeprefix("hl_"))


def test_headline_jobs_file_is_the_registered_list() -> None:
    assert read_jobs(JOBS_DIR / "jobs_headline.txt") == HEADLINE_JOBS


@pytest.mark.parametrize("jobs", JOBS_FILES, ids=lambda p: p.name)
def test_every_job_is_runnable(jobs: Path) -> None:
    for name, target, seed, src in read_jobs(jobs):
        cfg = load_toml_with_bases(CONFIG_DIR / f"{name}.toml")
        assert cfg["monte_carlo"]["noise_seeding"] == "per_draw", name
        # campaign.sh's allocation check reads the checkpoint's seed curator
        assert cfg["optimizer"]["seed_strategy"] == "adaptive", name
        assert cfg["data"]["neural_network"] == f"training_output/ou_marginal/{name}/best_model.json", name
        suffix = re.search(r"_s(\d)$", name)
        assert seed == (int(suffix.group(1)) if suffix else 1), name
        assert (src != "") == name.startswith(("ft_", "hs_")), name  # fine-tunes: ft_* (#173), hs_* (#192)
        assert target == (22000 if src else 20000), name


@pytest.mark.parametrize("cell", FAMILIES)
def test_hl_leaf_states_regime_and_allocation(cell: str) -> None:
    leaf = tomllib.loads((CONFIG_DIR / f"hl_{cell}.toml").read_text())
    assert leaf["base"] == [f"../{FAMILIES[cell]}"]
    assert leaf["monte_carlo"] == {"noise_seeding": "per_draw"}
    assert leaf["optimizer"] == {"n_pop": 512, "training_n_sims": 2}
    assert leaf["checkpoints"] == {"keep_last": 3}


@pytest.mark.parametrize("name", HL_JOBS)
def test_hl_resolves_to_its_family_otherwise(name: str) -> None:
    resolved = load_toml_with_bases(CONFIG_DIR / f"{name}.toml")
    family = load_toml_with_bases(REPO / "configs/training" / FAMILIES[hl_cell(name)])
    o = resolved["optimizer"]
    assert (o["algorithm"], o["seed_strategy"], o["n_pop"], o["training_n_sims"]) == ("ga", "adaptive", 512, 2)
    assert resolved["cost_function"]["cost_transform"] == "cubed"
    for cfg in (resolved, family):
        cfg["monte_carlo"].pop("noise_seeding", None)
        cfg["optimizer"].pop("n_pop")
        cfg["optimizer"].pop("training_n_sims")
        cfg["checkpoints"].pop("keep_last")
        cfg["data"].pop("neural_network")
    assert resolved == family


@pytest.mark.parametrize("name", ["ft_dense_p515_s2", "ft_dense_p515_s3"])
def test_ft_seed_differs_from_ft_dense_p515_only_in_deploy_path(name: str) -> None:
    seed = load_toml_with_bases(CONFIG_DIR / f"{name}.toml")
    s1 = load_toml_with_bases(CONFIG_DIR / "ft_dense_p515.toml")
    seed["data"].pop("neural_network")
    s1["data"].pop("neural_network")
    assert seed == s1
