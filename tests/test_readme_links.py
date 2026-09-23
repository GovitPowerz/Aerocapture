"""Relative links and in-page anchors resolve in every tracked Markdown file (issue #110)."""

import posixpath
import re
import subprocess
from collections import Counter
from functools import cache
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
# `](target)` or `](target "title")`.
LINK = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
# ATX heading, up to six `#`; an optional closing `#` run is not part of the text.
HEADING = re.compile(r"^#{1,6} +(.+?)(?: +#+)? *$", flags=re.MULTILINE)
FENCE = re.compile(r"```.*?```", flags=re.DOTALL)
# git is the oracle, not the filesystem: macOS matches paths case-insensitively and an
# untracked file does not exist on GitHub.
TRACKED = set(subprocess.run(["git", "-C", str(REPO), "ls-files", "-z"], check=True, capture_output=True, text=True).stdout.split("\0")) - {""}
DIRS: set[str] = set()
for _path in TRACKED:
    while _path := posixpath.dirname(_path):
        DIRS.add(_path)
MARKDOWN = sorted(p for p in TRACKED if p.endswith(".md"))


def _slug(heading: str) -> str:
    text = re.sub(r"[^\w\- ]", "", heading.lower())
    return text.replace(" ", "-")


@cache
def _prose(path: str) -> str:
    return FENCE.sub("", (REPO / path).read_text(encoding="utf-8"))


@cache
def _anchors(path: str) -> frozenset[str]:
    # GitHub suffixes repeated headings: foo, foo-1, foo-2.
    seen: Counter[str] = Counter()
    anchors: set[str] = set()
    for heading in HEADING.findall(_prose(path)):
        slug = _slug(heading)
        anchors.add(slug if seen[slug] == 0 else f"{slug}-{seen[slug]}")
        seen[slug] += 1
    return frozenset(anchors)


def _links(path: str) -> list[str]:
    return [t for t in LINK.findall(_prose(path)) if "://" not in t and not t.startswith("mailto:")]


def _resolve(path: str, target: str) -> str:
    return path if not target else posixpath.normpath(posixpath.join(posixpath.dirname(path), target))


@pytest.mark.parametrize("path", MARKDOWN)
def test_relative_links_are_tracked(path: str) -> None:
    targets = [t for t in _links(path) if not t.startswith("#")]
    missing = [t for t in targets if _resolve(path, t.split("#")[0]) not in TRACKED | DIRS]
    assert missing == []


@pytest.mark.parametrize("path", MARKDOWN)
def test_anchors_match_headings(path: str) -> None:
    dangling = []
    for target in _links(path):
        file, sep, anchor = target.partition("#")
        rel = _resolve(path, file)
        if sep and rel.endswith(".md") and rel in TRACKED and anchor not in _anchors(rel):
            dangling.append(target)
    assert dangling == []
