"""The one render spine behind every Typst report.

Every report driver (report.py's single-scheme + comparison reports,
rl/report_rl.py, warm_start_report.py, nn_input_report.py) writes its SVG/JSON
assets into a staging directory and hands off here. This module owns what
used to be re-implemented per driver: where the templates live
(`TEMPLATES_DIR`), the staging directory lifetime (`staged_assets`), the
typst-absent degrade and the compile (`render_pdf`). It is the only place
`check_typst` / `compile_typst` are called from.

Every template is compiled with `--root / --input dir=<assets>` and reads its
assets via `sys.inputs.at("dir")`, so the staging directory can live anywhere:
a `staged_assets` temp dir, or the persistent report dir the warm-start /
nn-input drivers create themselves and hand straight to `render_pdf`.

Leaf module: no simulator or plotting imports (CI's extension-free,
typst-free job imports it).
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from aerocapture.training.typst_utils import check_typst, compile_typst

# parents[3] is src/ (this file lives at src/python/aerocapture/training/).
TEMPLATES_DIR = Path(__file__).resolve().parents[3] / "typst"


@contextmanager
def staged_assets(*, keep: bool = False, prefix: str = "aerocapture_report_") -> Iterator[Path]:
    """Yield a temp directory to stage a report's assets in; removed on exit unless `keep`."""
    tmp = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield tmp
    finally:
        if keep:
            print(f"Chart artifacts kept at: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


def render_pdf(template: str, assets: Path, out_pdf: Path, *, label: str) -> Path | None:
    """Compile `TEMPLATES_DIR/<template>.typ` over `assets` into `out_pdf`.

    Returns the PDF path, or None when typst is absent or the compile fails;
    the assets are left untouched either way.
    """
    if not check_typst():
        print(f"  [{label}] typst not installed; PDF skipped. Install with `brew install typst`.")
        return None
    # Absolute dir: with --root / a relative sys.inputs dir would resolve
    # against src/typst/, not the cwd (nn_input_report's out_dir can be relative).
    ok = compile_typst(
        TEMPLATES_DIR / f"{template}.typ",
        out_pdf,
        extra_args=["--root", "/", "--input", f"dir={assets.resolve()}"],
        label=label,
    )
    if not ok:
        return None
    print(f"  [{label}] saved to {out_pdf}")
    return out_pdf
