"""The committed demo cells are byte copies of their bundle runs, and the demo's flags pick them (issue #182).

The arxiv-v3 fine-tune `ft_mamba_962` has no bundle run (its source is `training_output/ou_marginal/ft_mamba_p962/`), so only the
other two cells are pinned here.
"""

from pathlib import Path

import pytest
from aerocapture import demo

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "models/demo"
BUNDLE = REPO / "articles/paper/data/runs"


@pytest.mark.parametrize(("cell", "run"), [("hl_mamba_962", "ou_marginal/hl_mamba_p962"), ("mamba_962_legacy", "headline/mamba_p962")])
@pytest.mark.parametrize("name", ["best_model.json", "best_params.json"])
def test_demo_cell_is_a_byte_copy_of_its_bundle_run(cell: str, run: str, name: str) -> None:
    assert (DEMO / cell / name).read_bytes() == (BUNDLE / run / name).read_bytes()


def test_default_flies_the_deployed_seed_under_per_scenario_noise() -> None:
    model_dir, mode, _ = demo.select_model()
    assert model_dir == DEMO / "hl_mamba_962"
    assert mode == "per_draw"


def test_finetune_flag_flies_the_arxiv_v3_fine_tune_under_per_scenario_noise() -> None:
    model_dir, mode, _ = demo.select_model(finetune=True)
    assert model_dir == DEMO / "ft_mamba_962"
    assert mode == "per_draw"


def test_legacy_flag_flies_the_shared_path_champion_under_the_shared_path() -> None:
    model_dir, mode, _ = demo.select_model(legacy=True)
    assert model_dir == DEMO / "mamba_962_legacy"
    assert mode == "legacy"


def test_the_two_flags_are_exclusive() -> None:
    with pytest.raises(ValueError):
        demo.select_model(finetune=True, legacy=True)
