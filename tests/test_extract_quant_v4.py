"""articles/paper/scripts/extract_quant_v4.py: the #178 rows and their paired replicate deltas."""

from __future__ import annotations

import importlib.util
import json
import math
import statistics
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "articles/paper/scripts/extract_quant_v4.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("extract_quant_v4", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cell(label: str, toml: str, tails: list[float]) -> dict:
    reps = [{"replicate": i, "cvar95": t - 50, "cvar999": t, "p999": t - 1, "max": t + 100, "n": 100, "n_captured": 100} for i, t in enumerate(tails)]
    return {
        "label": label,
        "toml": toml,
        "eval_commit": "abc",
        "replicates": reps,
        "pooled": {"n": 300, "n_captured": 300, "cvar95": 1.0, "cvar999": 2.0, "max": 3.0, "viol_pct": 0.0, "heat_load_viol_n": 0, "p999": 1.5},
        "replicate_stats": {"cvar95": {"se": 0.1}, "cvar999": {"se": 0.2}},
    }


@pytest.fixture
def mod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    m = _load()
    manifest = tmp_path / "quant_cells_v4.txt"
    manifest.write_text(
        "# ref first\nchamp|a.toml|training_output/champ\nquant_v4/ft|b.toml|training_output/quant_v4/ft\nquant_v4/sc|c.toml|training_output/quant_v4/sc\n"
    )
    monkeypatch.setattr(m, "REPO", tmp_path)
    monkeypatch.setattr(m, "MANIFEST", manifest)
    monkeypatch.setattr(m, "SRC", tmp_path / "confirmatory_marginal.json")
    monkeypatch.setattr(m, "OUT", tmp_path / "out" / "confirmatory_marginal.json")
    return m


def _write_src(m: ModuleType, cells: list[dict]) -> None:
    m.SRC.write_text(json.dumps({"noise_seeding": "per_draw", "freeze_commit": "f", "n_replicates": 3, "n_per_replicate": 100, "cells": cells}))


def test_build_pairs_every_finalist_against_the_first_manifest_row(mod: ModuleType) -> None:
    _write_src(
        mod,
        [
            _cell("champ", "a.toml", [100.0, 110.0, 120.0]),
            _cell("quant_v4/ft", "b.toml", [101.0, 109.0, 123.0]),
            _cell("quant_v4/sc", "c.toml", [130.0, 140.0, 150.0]),
        ],
    )
    out = mod.build()
    assert out["noise_seeding"] == "per_draw" and out["reference"] == "champ"
    assert [c["label"] for c in out["cells"]] == ["champ", "quant_v4/ft", "quant_v4/sc"]
    assert set(out["paired"]) == {"quant_v4/ft", "quant_v4/sc"}
    deltas = [1.0, -1.0, 3.0]
    ft = out["paired"]["quant_v4/ft"]["delta_cvar999"]
    se = statistics.stdev(deltas) / math.sqrt(3)
    assert ft["mean"] == pytest.approx(statistics.mean(deltas)) and ft["se"] == pytest.approx(se, abs=0.01)
    assert ft["ci95"][0] < ft["mean"] < ft["ci95"][1]
    assert out["paired"]["quant_v4/sc"]["delta_cvar999"]["mean"] == pytest.approx(30.0)
    assert out["paired"]["quant_v4/sc"]["delta_max"]["mean"] == pytest.approx(30.0)
    assert "replicates" not in out["cells"][0]  # the bundle carries the pooled statistics, not 10 x 100k records


def test_check_is_pending_until_every_manifest_cell_is_scored(mod: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    _write_src(mod, [_cell("champ", "a.toml", [100.0, 110.0, 120.0])])
    mod.check()  # no copy yet, cells missing: pending, not a failure
    assert "pending" in capsys.readouterr().out
    mod.write()
    assert not mod.OUT.exists()
    mod.OUT.parent.mkdir()
    mod.OUT.write_text("{}")
    with pytest.raises(SystemExit, match="lacks"):
        mod.check()  # a committed copy whose source lost rows is an error


def test_check_fails_on_a_stale_copy_and_passes_on_a_current_one(mod: ModuleType) -> None:
    cells = [_cell("champ", "a.toml", [100.0, 110.0, 120.0]), _cell("quant_v4/ft", "b.toml", [1.0, 2.0, 3.0]), _cell("quant_v4/sc", "c.toml", [1.0, 2.0, 3.0])]
    _write_src(mod, cells)
    with pytest.raises(SystemExit, match="not what"):
        mod.check()
    mod.write()
    mod.check()
    mod.OUT.write_text(mod.OUT.read_text() + "\n")
    with pytest.raises(SystemExit, match="not what"):
        mod.check()


def test_build_refuses_a_non_per_draw_source(mod: ModuleType) -> None:
    mod.SRC.write_text(json.dumps({"noise_seeding": "legacy", "cells": []}))
    with pytest.raises(SystemExit, match="per_draw"):
        mod.build()


def test_committed_manifest_leads_with_the_champion() -> None:
    import sys

    sys.path.insert(0, str(REPO / "experiments/ou_marginal"))
    from confirmatory_marginal import read_manifest  # type: ignore[import-not-found]

    rows = read_manifest(REPO / "experiments/ou_marginal/quant_cells_v4.txt")
    assert rows[0][0] == "ou_marginal/hl_mamba_p962"
    assert [r[0] for r in rows[1:]] == ["quant_v4/ptq4_verdict", "quant_v4/qat4_finetune", "quant_v4/qat4_scratch"]
    for _, toml, _ in rows:
        assert (REPO / toml).is_file(), toml


def test_build_refuses_a_replicate_count_without_a_t_entry(mod: ModuleType) -> None:
    _write_src(mod, [_cell("champ", "a.toml", [1.0, 2.0]), _cell("quant_v4/ft", "b.toml", [1.0, 2.0]), _cell("quant_v4/sc", "c.toml", [1.0, 2.0])])
    with pytest.raises(SystemExit, match="df = 1"):
        mod.build()
