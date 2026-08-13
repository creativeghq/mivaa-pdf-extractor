"""
Guard: no collection op over AI/operator-shaped data inside a silently-degrading handler.

This is the detector that found the 3-source meta merge bug (#347), kept as a test rather than
thrown away after one use.

THE SHAPE
---------
Three things have to line up before it bites, and each is individually reasonable:

  1. A collection operation -- ``set()``, ``sorted()``, ``min()``, ``max()``, ``sum()`` -- which
     raises on element types it cannot hash or order.
  2. Applied to data whose SHAPE we do not control: AI extraction output, operator-authored
     jsonb, chunk aggregation, a DB row.
  3. Inside a broad ``except Exception`` whose handler RETURNS a plausible value rather than
     re-raising.

Together they mean a type the author did not anticipate produces a *successful-looking* result
with data silently missing. In the case this was built for, `colors` arriving as a list of
``{"name": ..., "hex": ...}`` objects raised ``unhashable type: 'dict'`` inside
``_build_enriched_product_data``, whose except-branch falls back to un-enriched product creation
-- so the product was still created, stripped of every attribute the enrichment had found, with
one log line nobody reads.

VALIDATION
----------
Run against the file as it stood before the fix, this detector returns exactly one hit: the
offending ``set(...)`` line. Run against the tree now, it returns nothing but a known-benign
``sorted()`` over a text column. An empty result from a detector that has never been shown to
fire is worthless, so ``test_detector_finds_the_bug_it_was_built_for`` reconstructs the original
expression and asserts the scanner flags it.

SCOPE
-----
Python only, and only the ingest/enrichment tree. The same defect class in TypeScript is
supabase-js resolving instead of throwing on an RLS denial -- guarded separately by
``tests/unit/uncheckedSupabaseWrites.test.ts`` in the parent repo.
"""
import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

TARGETS = [
    "app/services/products",
    "app/services/discovery",
    "app/services/metadata",
    "app/services/facets",
    "app/api/pdf_processing",
]

COLLECTION_OPS = {"set", "sorted", "min", "max", "sum"}

#: Identifiers that carry data whose element types we do not control.
UNTRUSTED = re.compile(
    r"\b(metadata|attributes|attributes_raw|extracted|enrichment|discovered|unknown|"
    r"params|meta_fields|properties|specifications|payload|candidate|record)\b"
)

#: Known-benign: the value is a text column, so it cannot be a non-string.
ALLOWLIST = {
    ("app/api/pdf_processing/stage_4_products.py", "row['field_name']"),
}


def _handler_returns_a_value(handler: ast.ExceptHandler) -> bool:
    """A handler that re-raises is fine -- the caller sees the failure."""
    src = ast.unparse(handler)
    if re.search(r"\braise\b", src):
        return False
    return any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(handler))


def _is_broad(handler: ast.ExceptHandler) -> bool:
    return handler.type is None or (
        isinstance(handler.type, ast.Name) and handler.type.id == "Exception"
    )


def scan_source(text: str, rel_path: str = "<memory>") -> list[tuple[int, str]]:
    """Return (lineno, expression) for each risky collection op in a degrading handler."""
    hits: list[tuple[int, str]] = []
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        if not any(_is_broad(h) and _handler_returns_a_value(h) for h in node.handlers):
            continue
        for sub in ast.walk(node):
            if not (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)):
                continue
            if sub.func.id not in COLLECTION_OPS:
                continue
            expr = ast.unparse(sub)
            if not UNTRUSTED.search(expr):
                continue
            if any(rel_path == p and marker in expr for p, marker in ALLOWLIST):
                continue
            hits.append((sub.lineno, expr))
    return hits


def _scan_tree() -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    for target in TARGETS:
        for path in (ROOT / target).rglob("*.py"):
            rel = path.relative_to(ROOT).as_posix()
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for lineno, expr in scan_source(text, rel):
                out.append((rel, lineno, expr))
    return out


def test_detector_finds_the_bug_it_was_built_for():
    """
    Guard the guard. A detector that has never been observed firing proves nothing when it comes
    back empty -- this reconstructs the exact pre-fix expression and asserts it is caught.
    """
    original = (
        "def build(metadata, field_type, values):\n"
        "    try:\n"
        "        existing_values = set(\n"
        "            v.lower() if isinstance(v, str) else v for v in metadata[field_type]\n"
        "        )\n"
        "        return sorted(existing_values)\n"
        "    except Exception:\n"
        "        return fallback()\n"
    )
    hits = scan_source(original)
    assert hits, "the detector must flag the expression it was built to catch"
    assert any("metadata[field_type]" in expr for _, expr in hits)


def test_a_reraising_handler_is_not_flagged():
    """Re-raising is the correct response -- the caller sees the failure, nothing is silent."""
    safe = (
        "def build(metadata):\n"
        "    try:\n"
        "        return sorted(set(metadata['colors']))\n"
        "    except Exception:\n"
        "        logger.exception('failed')\n"
        "        raise\n"
    )
    assert scan_source(safe) == []


def test_trusted_data_is_not_flagged():
    """The risk is element types we do not control; a local literal is not that."""
    safe = (
        "def build():\n"
        "    try:\n"
        "        return sorted({'a', 'b'})\n"
        "    except Exception:\n"
        "        return []\n"
    )
    assert scan_source(safe) == []


def test_no_silent_degradation_in_the_ingest_tree():
    """
    The live assertion. A hit here means: an unexpected element type will produce a
    successful-looking result with data missing, and nothing will raise.

    If a new hit is genuinely safe (the values provably cannot be non-strings), add it to
    ALLOWLIST with the reason -- do not delete the assertion.
    """
    hits = _scan_tree()
    formatted = "\n".join(f"  {p}:{ln}  {expr[:100]}" for p, ln, expr in hits)
    assert not hits, (
        "collection op over untrusted-shaped data inside a handler that returns a degraded "
        "value -- an unanticipated element type raises, the handler swallows it, and the caller "
        f"gets a plausible result with data silently missing:\n{formatted}"
    )
