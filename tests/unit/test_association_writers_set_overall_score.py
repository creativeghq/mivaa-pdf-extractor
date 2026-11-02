"""Every write to image_product_associations states an overall_score.

The column defaults to 0.0 and search multiplies each image hit by it, so a link written
without one silently removes that image from every image channel (1,172 links did on 2026-11-02).
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WRITE = re.compile(r"""table\(\s*["']image_product_associations["']\s*\)[\s\\]*\.[\s\\]*(insert|upsert)\(""")
LOOKBACK_LINES = 40


def _blank(source: str) -> str:
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from comment_budget import blank_comments
        return blank_comments(source)
    finally:
        sys.path.pop(0)


def test_every_association_write_sets_overall_score():
    offenders = []
    for path in (ROOT / "app").rglob("*.py"):
        code = _blank(path.read_text(encoding="utf-8"))
        if "image_product_associations" not in code:
            continue
        lines = code.splitlines()
        for m in WRITE.finditer(code):
            line = code[: m.start()].count("\n")
            window = "\n".join(lines[max(0, line - LOOKBACK_LINES): line + 8])
            if "overall_score" not in window:
                offenders.append(f"{path.relative_to(ROOT)}:{line + 1}")
    assert not offenders, f"association writes without an overall_score: {offenders}"
