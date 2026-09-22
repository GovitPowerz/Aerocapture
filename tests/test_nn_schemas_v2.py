import json

import pytest
from aerocapture.training.torch_mirror.schemas import ArchitectureV2, DenseSpec
from pydantic import ValidationError


def test_v2_dense_json_roundtrip() -> None:
    raw = {
        "format_version": 2,
        "architecture": [{"type": "dense", "input_size": 3, "output_size": 2, "activation": "linear"}],
        "weights": {"layer_0": {"w": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "b": [0.01, 0.02]}},
    }
    model = ArchitectureV2.model_validate(raw)
    assert len(model.architecture) == 1
    assert isinstance(model.architecture[0], DenseSpec)
    assert model.architecture[0].input_size == 3
    roundtrip = model.model_dump(exclude_none=True)
    assert json.dumps(roundtrip, sort_keys=True) == json.dumps(raw, sort_keys=True)


def test_v2_accepts_legacy_output_interpretation_and_never_dumps_it() -> None:
    """The one legacy key still loads (paper-bundle RL model) but is excluded from every dump."""
    raw = {
        "format_version": 2,
        "architecture": [{"type": "dense", "input_size": 3, "output_size": 2, "activation": "linear"}],
        "weights": {"layer_0": {"w": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "b": [0.01, 0.02]}},
        "output_interpretation": "atan2",
    }
    model = ArchitectureV2.model_validate(raw)
    assert len(model.architecture) == 1
    assert "output_interpretation" not in model.model_dump()


def test_v2_rejects_unknown_top_level_key() -> None:
    """A misspelled knob (`scaled_pi_N`) used to load and silently revert to its default (#128)."""
    raw = {
        "format_version": 2,
        "architecture": [{"type": "dense", "input_size": 3, "output_size": 2, "activation": "linear"}],
        "weights": {"layer_0": {"w": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "b": [0.01, 0.02]}},
        "scaled_pi_N": 2.0,
    }
    with pytest.raises(ValidationError, match="scaled_pi_N"):
        ArchitectureV2.model_validate(raw)


def test_v2_rejects_unknown_layer_type() -> None:
    raw = {
        "format_version": 2,
        "architecture": [{"type": "mystery", "foo": 42}],
        "weights": {},
    }
    with pytest.raises(ValidationError):
        ArchitectureV2.model_validate(raw)


def test_v2_rejects_wrong_format_version() -> None:
    raw = {
        "format_version": 3,
        "architecture": [],
        "weights": {},
    }
    with pytest.raises(ValidationError):
        ArchitectureV2.model_validate(raw)
