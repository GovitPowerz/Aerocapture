"""Leaf-ness of the trainer seam: `trainer.py` and `final_select.py` never import `train.py`.

`train.py` is the CLI / orchestration entry, not a library: the checkpoint,
artifact, population, corridor and config-resolution helpers live in leaf
modules that both the adapters and the retro CLI import directly. Importing
the trainer module in a fresh interpreter must not load `train`.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

TRAINING_DIR = Path(__file__).resolve().parents[1] / "src" / "python" / "aerocapture" / "training"
_TRAIN_IMPORT = re.compile(
    r"from aerocapture\.training\.train import|import aerocapture\.training\.train\b|from aerocapture\.training import (\w+, )*train\b|from \.train import"
)


@pytest.mark.parametrize("module", ["trainer", "final_select"])
def test_importing_the_seam_does_not_load_train(module: str) -> None:
    code = f"import sys\nimport aerocapture.training.{module}\nprint('aerocapture.training.train' in sys.modules)\n"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_importing_train_succeeds() -> None:
    result = subprocess.run([sys.executable, "-c", "import aerocapture.training.train"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_no_module_imports_train_as_a_library() -> None:
    offenders = [p.name for p in sorted(TRAINING_DIR.rglob("*.py")) if p.name != "train.py" and _TRAIN_IMPORT.search(p.read_text())]
    assert offenders == [], f"modules importing train.py: {offenders}"
