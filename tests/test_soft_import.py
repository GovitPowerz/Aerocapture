"""The soft-import contract: the training package imports without the PyO3 extension.

CI builds the extension for its one test job, so nothing structural exercises the
`aerocapture_rs` soft-import (`_aero_rs = None` fallbacks, `except ImportError`
branches, the pure-Python `candidate_input_names` fallback the drift tests exist
for). This test is that guard: every module under `aerocapture.training` is
imported in one subprocess with the extension blocked, so a module-level hard
`import aerocapture_rs` fails here instead of breaking collection on a machine
without the build.

Three modules need the simulator to mean anything and hard-import it by design;
each is only ever imported lazily, behind the call that needs it.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

HARD_IMPORTERS = {
    "aerocapture.training.warm_start_compare",  # raises a clear ImportError; imported lazily by train.py
    "aerocapture.training.rl.env",  # the RL environment IS the extension
    "aerocapture.training.rl.train",  # imports rl.env at module level
}

_WALK = r"""
import importlib, pkgutil, sys
sys.modules["aerocapture_rs"] = None  # force ImportError on `import aerocapture_rs`
import aerocapture.training as pkg
skip = set(sys.argv[1].split(","))
failed = []
for info in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
    if info.name in skip:
        continue
    try:
        importlib.import_module(info.name)
    except Exception as exc:  # noqa: BLE001
        failed.append(f"{info.name}: {type(exc).__name__}: {exc}")
print("\n".join(failed) if failed else "SOFT_IMPORT_OK")
sys.exit(1 if failed else 0)
"""


def test_training_package_imports_without_the_extension() -> None:
    result = subprocess.run([sys.executable, "-c", _WALK, ",".join(sorted(HARD_IMPORTERS))], capture_output=True, text=True)
    assert result.returncode == 0, f"modules that do not import without aerocapture_rs:\n{result.stdout}\n{result.stderr}"
    assert "SOFT_IMPORT_OK" in result.stdout


def test_candidate_input_names_fallback_matches_the_extension() -> None:
    """With the extension blocked, `candidate_input_names` takes its pure-Python
    fallback; that list must equal the Rust-exported contract (the drift test pins
    the same equality with the extension present, this pins the branch it cannot reach)."""
    aero = pytest.importorskip("aerocapture_rs")
    code = (
        "import sys\nsys.modules['aerocapture_rs'] = None\n"
        "from aerocapture.training.config import candidate_input_names\nprint('|'.join(candidate_input_names()))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().split("|") == [e["name"] for e in aero.candidate_inputs()]
