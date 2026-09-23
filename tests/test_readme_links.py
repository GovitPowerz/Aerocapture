"""The README's relative links and in-page anchors resolve (issue #110)."""

import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
# `](target)` or `](target "title")`.
LINK = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HEADING = re.compile(r"^#+ (.+)$", flags=re.MULTILINE)
FENCE = re.compile(r"```.*?```", flags=re.DOTALL)


def _slug(heading: str) -> str:
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def _prose() -> str:
    return FENCE.sub("", README.read_text(encoding="utf-8"))


def _anchors(prose: str) -> set[str]:
    # GitHub suffixes repeated headings: foo, foo-1, foo-2.
    seen: Counter[str] = Counter()
    anchors: set[str] = set()
    for heading in HEADING.findall(prose):
        slug = _slug(heading)
        anchors.add(slug if seen[slug] == 0 else f"{slug}-{seen[slug]}")
        seen[slug] += 1
    return anchors


def test_relative_links_exist() -> None:
    links = LINK.findall(_prose())
    missing = [t for t in links if "://" not in t and not t.startswith(("#", "mailto:")) and not (REPO / t.split("#")[0]).exists()]
    assert missing == []


def test_anchors_match_headings() -> None:
    prose = _prose()
    anchors = _anchors(prose)
    dangling = [t for t in LINK.findall(prose) if t.startswith("#") and t[1:] not in anchors]
    assert dangling == []
