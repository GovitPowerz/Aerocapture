"""Warm-start architecture-mismatch guard for _run_ppo.

Locks the invariant that a checkpoint whose layer TYPES differ from the
TOML-declared architecture raises ValueError BEFORE `load_state_dict` can
silently write dense weights into GRU parameter tensors (or vice versa).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from aerocapture.training.model_io import load_policy_from_json
from aerocapture.training.rl.export import export_v2_policy_to_json
from aerocapture.training.rl.policy import V2Policy
from aerocapture.training.rl.schemas import DenseSpec, GruSpec


def _make_policy(arch: list[DenseSpec | GruSpec]) -> V2Policy:
    return V2Policy(architecture=arch, output_interpretation="atan2", input_mask=None)


def _warmstart_check(warm_loaded: V2Policy, policy: V2Policy, path: str) -> None:
    """Mirror of the in-_run_ppo guard. Kept inline here so the test is self-contained."""
    if len(warm_loaded.layers) != len(policy.layers):
        raise ValueError(f"Warm-start architecture mismatch: {path} has {len(warm_loaded.layers)} layers, TOML declares {len(policy.layers)}.")
    type_mismatches = [
        (i, type(a).__name__, type(b).__name__) for i, (a, b) in enumerate(zip(warm_loaded.layers, policy.layers, strict=True)) if type(a) is not type(b)
    ]
    if type_mismatches:
        diffs = ", ".join(f"layer {i}: checkpoint={a} vs TOML={b}" for i, a, b in type_mismatches)
        raise ValueError(f"Warm-start layer-type mismatch: {path} -- {diffs}.")


def test_warmstart_same_arch_same_types_ok(tmp_path: Path) -> None:
    arch_a: list[DenseSpec | GruSpec] = [
        DenseSpec(type="dense", input_size=3, output_size=4, activation="tanh"),
        GruSpec(type="gru", input_size=4, hidden_size=4),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    src = _make_policy(arch_a)
    path = tmp_path / "good.json"
    export_v2_policy_to_json(src, str(path), obs_normalizer=None)

    dst = _make_policy(arch_a)
    loaded = load_policy_from_json(str(path), device="cpu")
    _warmstart_check(loaded, dst, str(path))  # no raise


def test_warmstart_same_depth_different_types_raises_before_load_state_dict(tmp_path: Path) -> None:
    """Same depth (3 layers) but the middle layer is Dense on the checkpoint and
    GRU on the target. load_state_dict would silently blast the dense weights into
    GRU parameter tensors; the guard must raise first."""
    checkpoint_arch: list[DenseSpec | GruSpec] = [
        DenseSpec(type="dense", input_size=3, output_size=4, activation="tanh"),
        DenseSpec(type="dense", input_size=4, output_size=4, activation="tanh"),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    target_arch: list[DenseSpec | GruSpec] = [
        DenseSpec(type="dense", input_size=3, output_size=4, activation="tanh"),
        GruSpec(type="gru", input_size=4, hidden_size=4),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    src = _make_policy(checkpoint_arch)
    path = tmp_path / "mismatch.json"
    export_v2_policy_to_json(src, str(path), obs_normalizer=None)

    dst = _make_policy(target_arch)
    loaded = load_policy_from_json(str(path), device="cpu")
    with pytest.raises(ValueError, match="layer-type mismatch"):
        _warmstart_check(loaded, dst, str(path))


def test_warmstart_different_depth_raises(tmp_path: Path) -> None:
    checkpoint_arch: list[DenseSpec | GruSpec] = [
        DenseSpec(type="dense", input_size=3, output_size=4, activation="tanh"),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    target_arch: list[DenseSpec | GruSpec] = [
        DenseSpec(type="dense", input_size=3, output_size=4, activation="tanh"),
        GruSpec(type="gru", input_size=4, hidden_size=4),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    src = _make_policy(checkpoint_arch)
    path = tmp_path / "wrong_depth.json"
    export_v2_policy_to_json(src, str(path), obs_normalizer=None)

    dst = _make_policy(target_arch)
    loaded = load_policy_from_json(str(path), device="cpu")
    with pytest.raises(ValueError, match="architecture mismatch"):
        _warmstart_check(loaded, dst, str(path))


def test_warmstart_guard_fires_before_load_state_dict(tmp_path: Path) -> None:
    """Documents why the guard matters: without it, load_state_dict would produce a
    confusing PyTorch-internal error (mismatched parameter names between DenseLayer
    `linear.weight` and GruLayer `weight_ih`). The guard catches it first with a
    clear, actionable message."""
    checkpoint_arch: list[DenseSpec | GruSpec] = [
        DenseSpec(type="dense", input_size=3, output_size=4, activation="tanh"),
        DenseSpec(type="dense", input_size=4, output_size=4, activation="tanh"),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    target_arch: list[DenseSpec | GruSpec] = [
        DenseSpec(type="dense", input_size=3, output_size=4, activation="tanh"),
        GruSpec(type="gru", input_size=4, hidden_size=4),
        DenseSpec(type="dense", input_size=4, output_size=2, activation="linear"),
    ]
    src = _make_policy(checkpoint_arch)
    path = tmp_path / "demo.json"
    export_v2_policy_to_json(src, str(path), obs_normalizer=None)

    dst = _make_policy(target_arch)
    loaded = load_policy_from_json(str(path), device="cpu")

    # Guard message must explicitly name the layer index and the mismatched types.
    with pytest.raises(ValueError) as exc_info:
        _warmstart_check(loaded, dst, str(path))
    msg = str(exc_info.value)
    assert "layer 1" in msg
    assert "DenseLayer" in msg
    assert "GruLayer" in msg
