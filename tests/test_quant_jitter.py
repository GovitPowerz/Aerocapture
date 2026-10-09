"""experiments/quant_jitter/quant_jitter.py: the cells it flies and the record it writes (simulator stubbed)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
from aerocapture.training import quantize

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "experiments/quant_jitter/quant_jitter.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("quant_jitter", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_flies_every_eps_for_both_granularities_on_the_sweep_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = {
        "architecture": [{"type": "dense", "input_size": 3, "output_size": 2, "activation": "swish"}],
        "weights": {"layer_0": {"w": [[0.9, -0.31, 0.12], [0.44, 0.27, -0.6]], "b": [0.1, -0.2]}},
    }
    champ = tmp_path / "champ"
    champ.mkdir()
    (champ / "best_model.json").write_text(json.dumps(model))
    (champ / "best_params.json").write_text(json.dumps({"nav.density_filter_gain": 0.5}))
    flown: list[dict] = []

    def fake_score(toml: str, model_path: Path, seeds: list[int], cost_kwargs: dict, extra: dict, timeout: float | None, noise_seeding: str) -> dict:
        flown.append({"model": json.loads(Path(model_path).read_text()), "seeds": seeds, "noise_seeding": noise_seeding, "extra": extra})
        return {
            "capture_rate": 1.0 if len(flown) % 2 else 0.99,
            "dv_cvar95": float(len(flown)),
            "dv_p50": 1.0,
            "dv_p95": 1.0,
            "dv_p99": 1.0,
            "viol_pct": 0.0,
            "rms_cost": 1.0,
        }

    monkeypatch.setattr(quantize, "_score_variant", fake_score)
    monkeypatch.setattr(quantize, "_resolve_pool", lambda toml, offset, n: ([7, 8, 9], {"base_mc_seed": 42, "offset": offset, "n": n}))
    monkeypatch.setattr("aerocapture.training.ablation._load_cost_kwargs", lambda toml, cost_transform: {})
    out = _load().run(champ, "unused.toml", "per_draw", 4, "proj_only", 3, [0.0, 0.05])

    assert [(r["granularity"], r["eps"]) for r in out["rows"]] == [("per_tensor", 0.0), ("per_tensor", 0.05), ("per_channel", 0.0), ("per_channel", 0.05)]
    assert len(flown) == 5 and flown[0]["model"] == model  # fp baseline first
    assert all(f["seeds"] == [7, 8, 9] and f["noise_seeding"] == "per_draw" and f["extra"] == quantize._scaffolding_overrides(champ) != {} for f in flown)
    assert flown[1]["model"] == quantize.quantize_model_weights(model, 4, "per_tensor", "proj_only")  # eps 0 is the PTQ sweep's own cell
    assert flown[2]["model"] == quantize.quantize_model_weights(model, 4, "per_tensor", "proj_only", scale_factor=1.05)
    assert out["tensors"] == ["layer_0.w"] and len(out["model_sha256"]) == 64
    assert out["spread"]["per_tensor"]["n_cells"] == 2 and out["spread"]["per_channel"]["n_full_capture"] == 1
