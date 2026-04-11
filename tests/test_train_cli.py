"""Tests for train.py CLI argument parsing."""

from __future__ import annotations

import argparse


def _build_parser() -> argparse.ArgumentParser:
    """Replicate the train.py argument parser for testing without running main."""
    parser = argparse.ArgumentParser()
    parser.add_argument("toml", type=str)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-gen", type=int, default=100)
    parser.add_argument("--n-pop", type=int, default=50)
    parser.add_argument("--mutation-rate", type=float, default=None)
    parser.add_argument("--train-n-sims", type=int, default=None)
    parser.add_argument("--final-n-sims", type=int, default=1000)
    return parser


def test_mutation_rate_default_none() -> None:
    parser = _build_parser()
    args = parser.parse_args(["dummy.toml"])
    assert args.mutation_rate is None


def test_mutation_rate_override() -> None:
    parser = _build_parser()
    args = parser.parse_args(["dummy.toml", "--mutation-rate", "0.05"])
    assert args.mutation_rate == 0.05


def test_train_n_sims_default_none() -> None:
    parser = _build_parser()
    args = parser.parse_args(["dummy.toml"])
    assert args.train_n_sims is None


def test_train_n_sims_override() -> None:
    parser = _build_parser()
    args = parser.parse_args(["dummy.toml", "--train-n-sims", "300"])
    assert args.train_n_sims == 300
