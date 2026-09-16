"""Write articles/paper/data/provenance.json (make -C articles/paper provenance).

What the paper's data and figures were built from, so a reader can verify that the
checkout in hand is the one behind the PDF: the HEAD commit (and whether the tree
was dirty), the last commit that touched any paper input, the Release tag holding
the run logs, the simulator crate version, the toolchain versions, and a SHA-256 of
every campaign TOML. Pure stdlib + matplotlib (for its version).
"""

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "articles/paper/data/provenance.json"
RELEASE_TAG = "arxiv-v2"
RUN_LOGS_ASSET = f"https://github.com/GovitPowerz/Aerocapture/releases/download/{RELEASE_TAG}/paper_run_logs.tar"
# Everything the figures, results.json and the PDF are built from (provenance.json itself excluded).
PAPER_INPUTS = (
    "articles/paper/paper.typ",
    "articles/paper/appendix.typ",
    "articles/paper/refs.bib",
    "articles/paper/Makefile",
    "articles/paper/scripts",
    "articles/paper/fonts",
    "articles/paper/figures",
    "articles/paper/data",
    ":(exclude)articles/paper/data/provenance.json",
)
CONFIG_GLOBS = ("configs/training/common.toml", "configs/training/paper/**/*.toml", "configs/training/sweep/*.toml", "configs/training/quant/*.toml")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def _tool_version(cmd: list[str]) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()


def _crate_version() -> str:
    for line in (REPO / "src/rust/Cargo.toml").read_text().splitlines():
        if line.startswith("version"):
            return line.split("=", 1)[1].strip().strip('"')
    raise ValueError("no version line in src/rust/Cargo.toml")


def main() -> None:
    import matplotlib

    configs = sorted(p for g in CONFIG_GLOBS for p in REPO.glob(g))
    out = {
        "head_commit": _git("rev-parse", "HEAD"),
        "head_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "paper_inputs_commit": _git("log", "-1", "--format=%H", "--", *PAPER_INPUTS),
        "paper_inputs": [p for p in PAPER_INPUTS if not p.startswith(":")],
        "release_tag": RELEASE_TAG,
        "run_logs_asset": RUN_LOGS_ASSET,
        "noise_regime": "legacy (every committed cell; ADR-0003, ADR-0006)",
        "simulator_crate_version": _crate_version(),
        "typst_version": _tool_version(["typst", "--version"]),
        "matplotlib_version": matplotlib.__version__,
        "python_version": platform.python_version(),
        "config_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in configs},
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"wrote {OUT.relative_to(REPO)}: head {out['head_commit'][:10]}{' (dirty)' if out['head_dirty'] else ''}, "
        f"inputs {out['paper_inputs_commit'][:10]}, {len(configs)} config hashes",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
