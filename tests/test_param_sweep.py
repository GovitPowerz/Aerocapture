"""Tests for the architecture parameter-budget sweep driver."""

from __future__ import annotations

import pytest
from aerocapture.training import param_sweep as ps
from aerocapture.training.config import NetworkConfig, _layer_input_size, _layer_output_size


@pytest.mark.parametrize("arch", ps.ARCHS)
def test_candidates_sorted_and_param_counts_match(arch: str) -> None:
    cands = ps.candidates(arch)
    assert cands, f"{arch} produced no candidates"
    params = [p for p, _ in cands]
    assert params == sorted(params), "candidates must be ascending in params"
    assert params == sorted(set(params)), "param counts must be unique (dedup)"
    for p, a in cands:
        # The reported count must equal the authoritative NetworkConfig counter.
        assert NetworkConfig(architecture=a, input_mask=list(range(ps.INPUT_DIM))).n_base_coef == p


@pytest.mark.parametrize("arch", ps.ARCHS)
def test_chain_is_valid_and_outputs_two(arch: str) -> None:
    """Every generated stack must consume INPUT_DIM, chain consistently, emit 2."""
    for _params, a in ps.candidates(arch):
        prev = ps.INPUT_DIM
        for layer in a:
            assert _layer_input_size(layer) == prev, f"{arch} chain break: {layer}"
            prev = _layer_output_size(layer)
        assert prev == 2, f"{arch} must end at output_size 2, got {prev}"


@pytest.mark.parametrize("arch", ps.ARCHS)
def test_select_for_budgets_picks_nearest(arch: str) -> None:
    budgets = (500, 1000, 2000, 4000)
    picked = ps.select_for_budgets(arch, budgets)
    assert picked, f"{arch} selected nothing"
    cands = ps.candidates(arch)
    for b in budgets:
        nearest = min(p for p, _ in cands)
        best = min((abs(p - b), p) for p, _ in cands)[1]
        assert any(p == best for p, _ in picked), f"{arch} missed nearest-to-{b} (={best})"
        assert nearest  # sanity


def test_transformer_d_model_divisible_by_heads() -> None:
    for _p, a in ps.candidates("transformer"):
        t = next(layer for layer in a if layer["type"] == "transformer")
        assert t["d_model"] % t["n_heads"] == 0


def test_generated_config_text_parses() -> None:
    """A generated config must load through base inheritance + NetworkConfig.

    Writes a UNIQUELY-NAMED throwaway config into the real sweep dir (base
    inheritance needs the sibling base file one level up) WITHOUT calling
    generate() -- so it never touches manifest.json or collides with the real
    swept configs.
    """
    from aerocapture.training.toml_utils import load_toml_with_bases

    params, arch = ps.select_for_budgets("gru", (1000,))[0]
    ps.SWEEP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = ps.SWEEP_CONFIG_DIR / "_pytest_throwaway.toml"
    cfg.write_text(ps._config_text("gru", params, arch))
    try:
        resolved = load_toml_with_bases(cfg)
        net = NetworkConfig(architecture=resolved["network"]["architecture"], input_mask=resolved["network"]["input_mask"])
        assert net.n_base_coef == params
        assert net.input_mask is not None
        assert len(net.input_mask) == ps.INPUT_DIM
    finally:
        cfg.unlink(missing_ok=True)
