"""Write (or --check) articles/paper/data/provenance.json.

What the paper's data and figures were built from, so a reader can verify that the
checkout in hand is the one behind the PDF: a SHA-256 over every tracked paper input
(content-addressed, so it survives squash-merges and rebases where a commit id would
not, and it is computed from the working tree so it commits together with the inputs
it describes), the Release tag holding the run logs, the simulator crate version, the
two toolchain versions the figure/PDF bytes depend on, and a SHA-256 of every campaign
TOML. Only per-input facts, so the file changes only when an input does; `make check`
runs `--check`, which fails when the committed file is not what this tree yields.
Pure stdlib + matplotlib (for its version).
"""

import hashlib
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "articles/paper/data/provenance.json"
RELEASE_TAG = "arxiv-v2"
RUN_LOGS_ASSET = f"https://github.com/GovitPowerz/Aerocapture/releases/download/{RELEASE_TAG}/paper_run_logs.tar"
# Everything the figures, results.json and the PDF are built from.
PAPER_INPUTS = (
    "articles/paper/paper.typ",
    "articles/paper/appendix.typ",
    "articles/paper/results.typ",
    "articles/paper/refs.bib",
    "articles/paper/Makefile",
    "articles/paper/scripts",
    "articles/paper/fonts",
    "articles/paper/figures",
    "articles/paper/data",
)
# provenance.json cannot name the commit that commits it, so it is not an input of itself.
INPUTS_PATHSPEC = (*PAPER_INPUTS, f":(exclude){OUT.relative_to(REPO)}")
CONFIG_GLOBS = ("configs/training/common.toml", "configs/training/paper/**/*.toml", "configs/training/sweep/*.toml", "configs/training/quant/*.toml")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def _inputs_sha256() -> str:
    """One digest over (path, content) of every paper input (tracked or untracked-unignored), working-tree bytes."""
    h = hashlib.sha256()
    for f in sorted(filter(None, _git("ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *INPUTS_PATHSPEC).split("\0"))):
        h.update(f.encode())
        h.update(b"\0")
        h.update(hashlib.sha256((REPO / f).read_bytes()).digest())
    return h.hexdigest()


def build() -> dict:
    import matplotlib

    typst = None
    if shutil.which("typst"):
        typst = subprocess.run(["typst", "--version"], check=True, capture_output=True, text=True).stdout.split()[1]  # "typst 0.15.1 (...)"
    configs = sorted(p for g in CONFIG_GLOBS for p in REPO.glob(g))
    return {
        "paper_inputs_sha256": _inputs_sha256(),
        "paper_inputs": list(PAPER_INPUTS),
        "release_tag": RELEASE_TAG,
        "run_logs_asset": RUN_LOGS_ASSET,
        "noise_regime": (
            "legacy for every committed cell except rl/* and ou_marginal/* (per_draw, issue #101) and every cell of "
            "confirmatory_marginal.json (per_draw, issue #137); per run in results.json (ADR-0003, ADR-0006); "
            "quote_marginal.json pins legacy for both of its regimes and re-seeds simulation.random_seed per scenario "
            "for the marginal one (issue #157); centered_depth.json carries both regimes, each cell labelled (issue #156)"
        ),
        "simulator_crate_version": tomllib.loads((REPO / "src/rust/Cargo.toml").read_text())["package"]["version"],
        "typst_version": typst,
        "matplotlib_version": matplotlib.__version__,
        "config_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in configs},
    }


def main() -> None:
    text = json.dumps(build(), indent=2) + "\n"
    if sys.argv[1:] == ["--check"]:
        if not OUT.exists() or OUT.read_text() != text:
            sys.exit(f"{OUT.relative_to(REPO)} is stale: run `make -C articles/paper provenance` and commit it")
        print(f"{OUT.relative_to(REPO)}: current")
        return
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(REPO)}", file=sys.stderr)


if __name__ == "__main__":
    main()
