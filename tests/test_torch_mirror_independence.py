"""The torch mirror never imports the PPO/SAC trainer (issue #100).

`aerocapture.training.torch_mirror` is load-bearing for the shipped population
path; `aerocapture.training.rl` is the shelved trainer that depends on it. Every
mirror module is imported in a subprocess with the `rl` package blocked, so a
module-level (or lazily triggered at import) `from aerocapture.training.rl ...`
fails here instead of silently re-coupling the two."""

from __future__ import annotations

import subprocess
import sys

_WALK = r"""
import importlib, pkgutil, sys
sys.modules["aerocapture.training.rl"] = None  # any `import aerocapture.training.rl...` raises ImportError
import aerocapture.training.torch_mirror as pkg
failed = []
for info in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
    try:
        importlib.import_module(info.name)
    except Exception as exc:  # noqa: BLE001
        failed.append(f"{info.name}: {type(exc).__name__}: {exc}")
print("\n".join(failed) if failed else "MIRROR_INDEPENDENT")
sys.exit(1 if failed else 0)
"""


def test_torch_mirror_imports_without_the_rl_package() -> None:
    result = subprocess.run([sys.executable, "-c", _WALK], capture_output=True, text=True)
    assert result.returncode == 0, f"mirror modules that import the rl trainer:\n{result.stdout}\n{result.stderr}"
    assert "MIRROR_INDEPENDENT" in result.stdout
