"""Committee builder and runtime gates (#232): the builder's contract checks, the
torch mirror's refusal, and (PyO3) a one-member committee flying bit-identically
to its member under the raw_sum and medoid rules."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from aerocapture.training.committee import RULES, build_committee, sha256_of, write_committee

_MASK = [0, 2, 3, 5]


def _dense_v2(path: Path, seed: int, *, mask: list[int] | None = _MASK, output_param: str = "atan2_signed", version: int = 2) -> Path:
    rng = np.random.default_rng(seed)
    n_in = len(mask) if mask is not None else 16
    path.write_text(
        json.dumps(
            {
                "format_version": version,
                "architecture": [
                    {"type": "dense", "input_size": n_in, "output_size": 3, "activation": "tanh"},
                    {"type": "dense", "input_size": 3, "output_size": 2, "activation": "asinh"},
                ],
                "weights": {
                    "layer_0": {"w": rng.normal(size=(3, n_in)).round(3).tolist(), "b": [0.1, -0.2, 0.3]},
                    "layer_1": {"w": rng.normal(size=(2, 3)).round(3).tolist(), "b": [0.0, 0.5]},
                },
                "input_mask": mask,
                "ablated_input": None,
                "ablated_value": 0.0,
                "output_param": output_param,
                "scaled_pi_n": 1.0,
                "delta_max": 0.35,
                "normalization": None,
            }
        )
    )
    return path


def test_build_committee_nests_members_and_records_sha256(tmp_path: Path) -> None:
    a, b = _dense_v2(tmp_path / "a.json", 1), _dense_v2(tmp_path / "b.json", 2)
    model = build_committee([a, b], "circular_mean")
    (layer,) = model["architecture"]
    assert layer["type"] == "committee" and layer["rule"] == "circular_mean"
    assert layer["members"] == [json.loads(a.read_text())["architecture"], json.loads(b.read_text())["architecture"]]
    assert layer["member_sha256"] == [sha256_of(a), sha256_of(b)]
    assert model["weights"] == {"layer_0": {"members": [json.loads(a.read_text())["weights"], json.loads(b.read_text())["weights"]]}}
    assert model["input_mask"] == _MASK and model["output_param"] == "atan2_signed"
    out = write_committee([a, b], "medoid", tmp_path / "cell")
    assert out == tmp_path / "cell" / "best_model.json"
    assert json.loads(out.read_text())["architecture"][0]["rule"] == "medoid"


@pytest.mark.parametrize("rule", RULES)
def test_rules_are_accepted(tmp_path: Path, rule: str) -> None:
    a = _dense_v2(tmp_path / "a.json", 1)
    assert build_committee([a], rule)["architecture"][0]["rule"] == rule


def test_build_committee_rejections(tmp_path: Path) -> None:
    a = _dense_v2(tmp_path / "a.json", 1)
    with pytest.raises(ValueError, match="rule must be"):
        build_committee([a], "mean")
    with pytest.raises(ValueError, match="at least one member"):
        build_committee([], "raw_sum")
    with pytest.raises(ValueError, match="input_mask differs"):
        build_committee([a, _dense_v2(tmp_path / "m.json", 2, mask=[0, 1, 2, 3])], "raw_sum")
    with pytest.raises(ValueError, match="format_version must be 2"):
        build_committee([a, _dense_v2(tmp_path / "v1.json", 3, version=1)], "raw_sum")
    with pytest.raises(ValueError, match="atan2_signed"):
        build_committee([_dense_v2(tmp_path / "sp.json", 4, output_param="scaled_pi")], "raw_sum")
    nested = tmp_path / "nested.json"
    nested.write_text(json.dumps(build_committee([a], "raw_sum")))
    with pytest.raises(ValueError, match="cannot itself be a committee"):
        build_committee([a, nested], "raw_sum")


def test_write_committee_copies_member_0_scaffolding(tmp_path: Path) -> None:
    lead, other = tmp_path / "s1", tmp_path / "s2"
    lead.mkdir()
    other.mkdir()
    a, b = _dense_v2(lead / "best_model.json", 1), _dense_v2(other / "best_model.json", 2)
    (lead / "best_params.json").write_text('{"nav.density_filter_gain": 0.5}')
    (other / "best_params.json").write_text('{"nav.density_filter_gain": 0.4}')
    cell = tmp_path / "cell"
    write_committee([a, b], "circular_mean", cell)
    assert (cell / "best_params.json").read_text() == '{"nav.density_filter_gain": 0.5}'
    assert json.loads((cell / "committee.json").read_text())["scaffolding"] == str(lead / "best_params.json")
    bare = tmp_path / "bare"
    write_committee([_dense_v2(tmp_path / "c.json", 3)], "raw_sum", bare)
    assert not (bare / "best_params.json").exists()


def test_torch_mirror_refuses_a_committee(tmp_path: Path) -> None:
    torch_mirror = pytest.importorskip("aerocapture.training.model_io")
    out = write_committee([_dense_v2(tmp_path / "a.json", 1)], "raw_sum", tmp_path / "cell")
    with pytest.raises(NotImplementedError, match="committee"):
        torch_mirror.load_policy_from_json(str(out))


@pytest.mark.slow
@pytest.mark.parametrize("rule", ["raw_sum", "medoid"])
def test_one_member_committee_flies_bit_identically(tmp_path: Path, rule: str) -> None:
    """The runtime gate: a committee of one, under a rule that returns a raw head, is the
    member. Flown through the golden NN config (16 inputs, no mask) on its three dispersed draws."""
    aero = pytest.importorskip("aerocapture_rs")
    member = _dense_v2(tmp_path / "member.json", 7, mask=None)
    committee = write_committee([member], rule, tmp_path / "cell")
    toml = "configs/test/test_neural_golden.toml"  # n_sims = 3 dispersed draws
    alone = aero.run_mc(toml, overrides={"data.neural_network": str(member)})
    together = aero.run_mc(toml, overrides={"data.neural_network": str(committee)})
    assert alone.final_records.shape[0] == 3
    assert np.array_equal(alone.final_records, together.final_records)
    assert np.array_equal(alone.captured, together.captured)


@pytest.mark.slow
def test_three_member_committee_flies(tmp_path: Path) -> None:
    aero = pytest.importorskip("aerocapture_rs")
    members = [_dense_v2(tmp_path / f"m{i}.json", 10 + i, mask=None) for i in range(3)]
    committee = write_committee(members, "circular_mean", tmp_path / "cell")
    res = aero.run_mc("configs/test/test_neural_golden.toml", overrides={"data.neural_network": str(committee)})
    assert res.final_records.shape[0] == 3
    assert np.all(np.isfinite(res.final_records))
