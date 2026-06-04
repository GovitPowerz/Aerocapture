"""_describe_rl_architecture must report canonical per-layer param counts (D5).

Uses the same TOML-fixture pattern as test_rl_parse_network_v2.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("aerocapture_rs")

from aerocapture.training import config as cfg_mod  # noqa: E402
from aerocapture.training.rl.config import RLConfig  # noqa: E402
from aerocapture.training.rl.schemas import DenseSpec, LstmSpec  # noqa: E402


def _make_lstm_toml(tmp_path: Path) -> Path:
    """Minimal TOML with Dense -> LSTM -> Dense architecture."""
    toml = tmp_path / "lstm_arch.toml"
    toml.write_text(
        """
[rl]
n_envs = 2

[network]
input_mask = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

[[network.architecture]]
type = "dense"
input_size = 16
output_size = 32
activation = "tanh"

[[network.architecture]]
type = "lstm"
input_size = 32
hidden_size = 32

[[network.architecture]]
type = "dense"
input_size = 32
output_size = 2
activation = "linear"
""".lstrip()
    )
    return toml


def test_describe_uses_canonical_lstm_param_count(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The 3-gate GRU formula gives a lower count than the correct 4-gate LSTM formula."""
    from aerocapture.training.rl.train import _describe_rl_architecture

    cfg = RLConfig.from_toml(_make_lstm_toml(tmp_path))

    # Canonical counts via config helpers (source of truth).
    arch = [
        DenseSpec(type="dense", input_size=16, output_size=32, activation="tanh"),
        LstmSpec(type="lstm", input_size=32, hidden_size=32),
        DenseSpec(type="dense", input_size=32, output_size=2, activation="linear"),
    ]
    expected_total = sum(cfg_mod._layer_n_params(s) for s in arch)

    # Sanity: the buggy 3-gate formula gives a different (smaller) total.
    buggy_total = 0
    for s in arch:
        if isinstance(s, DenseSpec):
            buggy_total += s.input_size * s.output_size + s.output_size
        else:
            assert isinstance(s, LstmSpec)
            buggy_total += 3 * s.hidden_size * s.input_size + 3 * s.hidden_size * s.hidden_size + 6 * s.hidden_size
    assert buggy_total != expected_total, "sanity: 3-gate and 4-gate formulas must differ for LSTM"

    _describe_rl_architecture(cfg)

    captured = capsys.readouterr()
    assert f"({expected_total} params)" in captured.err, (
        f"Expected '({expected_total} params)' in stderr.\nBuggy 3-gate total would be {buggy_total}.\nGot stderr:\n{captured.err}"
    )
