"""The README's relative links and in-page anchors resolve (issue #110)."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
LINK = re.compile(r"\]\(([^)\s]+)\)")


def _slug(heading: str) -> str:
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def _links() -> list[str]:
    return LINK.findall(README.read_text(encoding="utf-8"))


def test_relative_links_exist() -> None:
    missing = [t for t in _links() if "://" not in t and not t.startswith(("#", "mailto:")) and not (REPO / t.split("#")[0]).exists()]
    assert missing == []


def test_anchors_match_headings() -> None:
    text = re.sub(r"```.*?```", "", README.read_text(encoding="utf-8"), flags=re.DOTALL)
    slugs = {_slug(m) for m in re.findall(r"^#+ (.+)$", text, flags=re.MULTILINE)}
    dangling = [t for t in _links() if t.startswith("#") and t[1:] not in slugs]
    assert dangling == []
