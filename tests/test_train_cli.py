"""Tests for train.py CLI argument parsing."""

from __future__ import annotations

import argparse


def _build_parser() -> argparse.ArgumentParser:
    """Replicate the train.py argument parser for testing without running main."""
    parser = argparse.ArgumentParser()
    parser.add_argument("toml", type=str)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-gen", type=int, default=None)
    parser.add_argument("--n-pop", type=int, default=None)
    parser.add_argument("--final-n-sims", type=int, default=1000)
    parser.add_argument("--algorithm", type=str, default=None)
    parser.add_argument("--adaptive-seeds", action="store_true")
    parser.add_argument("--seed-pool-cap", type=int, default=None)
    parser.add_argument("--cost-alpha", type=float, default=None)
    parser.add_argument("--sim-timeout", type=float, default=None)
    return parser


def test_algorithm_default_none() -> None:
    parser = _build_parser()
    args = parser.parse_args(["dummy.toml"])
    assert args.algorithm is None


def test_algorithm_override() -> None:
    parser = _build_parser()
    args = parser.parse_args(["dummy.toml", "--algorithm", "de"])
    assert args.algorithm == "de"


def test_n_gen_default_none() -> None:
    parser = _build_parser()
    args = parser.parse_args(["dummy.toml"])
    assert args.n_gen is None


def test_n_pop_default_none() -> None:
    parser = _build_parser()
    args = parser.parse_args(["dummy.toml"])
    assert args.n_pop is None


def test_seed_pool_cap_default_none() -> None:
    parser = _build_parser()
    args = parser.parse_args(["dummy.toml"])
    assert args.seed_pool_cap is None


def test_adaptive_seeds_default_false() -> None:
    parser = _build_parser()
    args = parser.parse_args(["dummy.toml"])
    assert args.adaptive_seeds is False
