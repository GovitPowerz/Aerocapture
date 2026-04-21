import json

import pytest
from aerocapture.training.rl.schemas import ArchitectureV2, DenseSpec
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


def test_v2_ignores_legacy_output_interpretation() -> None:
    """Legacy JSON files carry `output_interpretation`; we silently ignore it."""
    raw = {
        "format_version": 2,
        "architecture": [{"type": "dense", "input_size": 3, "output_size": 2, "activation": "linear"}],
        "weights": {"layer_0": {"w": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "b": [0.01, 0.02]}},
        "output_interpretation": "atan2",
    }
    model = ArchitectureV2.model_validate(raw)
    assert len(model.architecture) == 1
    assert not hasattr(model, "output_interpretation")


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
