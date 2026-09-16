"""The soft-import contract: the training modules import without the PyO3 extension.

CI builds the extension for its one test job, so nothing structural exercises the
`aerocapture_rs` soft-import (`_aero_rs = None` fallbacks, `except ImportError`
branches, the pure-Python `candidate_input_names` fallback the drift tests exist
for). This test is that guard: each module is imported in a subprocess with the
extension blocked, so a module-level hard `import aerocapture_rs` fails here
instead of breaking collection on a machine without the build.

`warm_start_compare` is deliberately absent: it raises a clear ImportError by
design (it cannot do anything without the simulator) and is only imported lazily.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

SOFT_IMPORT_MODULES = [
    "aerocapture.training.config",
    "aerocapture.training.layer_schema",
    "aerocapture.training.evaluate",
    "aerocapture.training.problem",
    "aerocapture.training.warm_start",
    "aerocapture.training.sensitivity",
    "aerocapture.training.ablation",
    "aerocapture.training.nn_input_report",
    "aerocapture.training.calibrate_inputs",
    "aerocapture.training.report",
    "aerocapture.training.animate",
    "aerocapture.training.deploy_overrides",
    "aerocapture.training.final_select",
    "aerocapture.training.trainer",
]


@pytest.mark.parametrize("module", SOFT_IMPORT_MODULES)
def test_module_imports_without_the_extension(module: str) -> None:
    code = f"import sys\nsys.modules['aerocapture_rs'] = None\nimport {module}\nprint('SOFT_IMPORT_OK')\n"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"{module} does not import without aerocapture_rs:\n{result.stderr}"
    assert "SOFT_IMPORT_OK" in result.stdout


def test_candidate_input_names_fallback_matches_the_extension() -> None:
    """The pure-Python fallback list must equal the Rust-exported contract; the drift
    test asserts this with the extension present, this asserts the fallback path is
    the one taken when it is absent."""
    aero = pytest.importorskip("aerocapture_rs")
    code = (
        "import sys\nsys.modules['aerocapture_rs'] = None\n"
        "from aerocapture.training.config import candidate_input_names\nprint('|'.join(candidate_input_names()))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().split("|") == list(aero.NN_INPUT_NAMES)
