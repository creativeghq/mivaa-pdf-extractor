"""Guards for the mivaa#31 catalog-extractor fixes (M17-2, M17-3).

`M17-1` (agent-run helpers mutating by bare id) landed in bff2a48 and is guarded in
`test_audit_28_31_gates_hold.py`.

M17-2. Both extractors wrote model output into `kb_docs` as `status='published'` with no
review gate, no draft state and no write-side provenance marker. Paired with #29 M15-1
that is a complete persistent-injection path: a supplier PDF plants instructions once,
they become a published KB document, and they are replayed into every future agent turn
that retrieves them.

M17-3. Both asked for JSON in the prompt and repaired the reply by stripping markdown
fences — part of the #32 class, and the verdict here decides what becomes a KB document.

9 of the 10 cases were watched to fail against the pre-fix tree. The tenth,
`test_the_legend_extractor_is_still_on_the_ratchet_and_not_pretended_fixed`, passes both
ways by design — it exists to make the exclusion below deliberate.

NOT covered, and the reason is a decision rather than an omission:
`catalog_legend_extractor_v2._call_claude` is STILL a free-form parser. Its caller scores
candidate legend types with `sum(1 for v in r.values() if v not in (None, [], "", {}))`
— an open-ended dict whose key set is the signal. A forced schema would make that score
count only declared keys, which is arguably better but is a behaviour change to the
type-detection heuristic, not a mechanical migration. It stays on the #32 ratchet until
someone decides what the legend schema should be.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"

KNOWLEDGE = APP / "services" / "knowledge" / "catalog_knowledge_extractor.py"
LEGEND = APP / "services" / "knowledge" / "catalog_legend_extractor_v2.py"
EXTRACTORS = [KNOWLEDGE, LEGEND]


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _strip_comments(src: str) -> str:
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"^\s*#.*$", "", src, flags=re.MULTILINE)
    return src


def _node(src: str, name: str):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _body(src: str, name: str) -> str:
    return _strip_comments(ast.get_source_segment(src, _node(src, name)) or "")


# ───────────────────────────────────────────────────────────────────────────
# M17-2 — catalogue output is not born published
# ───────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path", EXTRACTORS, ids=lambda p: p.name)
def test_catalog_output_is_written_as_a_draft(path: Path):
    """`published` put model output straight into retrieval with nothing between it and
    an agent turn."""
    src = _strip_comments(_read(path))
    assert '"p_status": "draft"' in src, (
        f"{path.name} writes catalogue-derived content as published again (#31 M17-2)"
    )
    assert '"p_status": "published"' not in src


@pytest.mark.parametrize("path", EXTRACTORS, ids=lambda p: p.name)
def test_the_visibility_is_a_value_the_column_accepts(path: Path):
    """`workspace` is rejected by `kb_docs_visibility_check`, so these calls were failing
    outright — the same value 90e5a52 fixed in `job_sites_kb_sync` and missed here. The
    vocabulary is two words: `public` means the anonymous website, `private` means
    workspace and agents only."""
    src = _strip_comments(_read(path))
    assert '"p_visibility": "workspace"' not in src, (
        f"{path.name} passes 'workspace' again — not a member of the visibility "
        "vocabulary, so the upsert raises on the CHECK"
    )
    assert '"p_visibility": "private"' in src


@pytest.mark.parametrize("path", EXTRACTORS, ids=lambda p: p.name)
def test_the_provenance_marker_is_written(path: Path):
    """The finding asks for a write-side marker so retrieval can tell operator-authored
    text from catalogue-derived text. Deriving it from the other metadata keys later is
    the kind of reconstruction that goes wrong once and stays wrong."""
    src = _strip_comments(_read(path))
    assert '"source_trust": "catalog_derived"' in src, (
        f"{path.name} no longer records where this content came from"
    )
    assert '"review_required": True' in src


# ───────────────────────────────────────────────────────────────────────────
# M17-3 — the verdict that becomes a KB document
# ───────────────────────────────────────────────────────────────────────────

def test_the_knowledge_extractor_uses_a_forced_tool_call():
    body = _body(_read(KNOWLEDGE), "_call_claude_vision_knowledge")
    assert "call_with_tool(" in body, (
        "the knowledge extractor is parsing free-form text again (#31 M17-3)"
    )
    assert "```" not in body, "the markdown-fence repair is back"
    assert "json.loads" not in body


def test_the_knowledge_extractor_call_is_awaited():
    """It was the SYNC `tracked_claude_call` invoked from an `async def` caller, so every
    page blocked the event loop for a whole vision round-trip. Same defect as the agent
    handlers in 1d7edee."""
    src = _read(KNOWLEDGE)
    assert isinstance(_node(src, "_call_claude_vision_knowledge"), ast.AsyncFunctionDef), (
        "_call_claude_vision_knowledge went back to being sync"
    )
    caller = _body(src, "extract_catalog_knowledge_from_pdf")
    assert "await _call_claude_vision_knowledge(" in caller, (
        "the caller no longer awaits it, so the coroutine is never run and `data` is a "
        "coroutine object rather than a dict"
    )


def test_the_tool_schema_does_not_force_the_model_to_invent():
    """Only `page_type` is required. A catalogue page that carries nothing is a
    legitimate answer, and marking the content fields required would push the model into
    inventing text to satisfy the schema — trading a parse failure for a plausible
    fabrication, which is worse."""
    src = _read(KNOWLEDGE)
    schema = _strip_comments(src[src.index("CATALOG_KNOWLEDGE_TOOL"):src.index("CATALOG_KNOWLEDGE_TOOL") + 900])
    assert '"required": ["page_type"]' in schema, (
        "the catalogue tool schema changed which fields are mandatory"
    )


def test_the_legend_extractor_is_still_on_the_ratchet_and_not_pretended_fixed():
    """Recorded so the exclusion stays deliberate. Its caller scores candidate legend
    types by counting non-empty values in an open-ended dict, so a forced schema changes
    the heuristic rather than just the parsing."""
    body = _body(_read(LEGEND), "_call_claude")
    assert "call_with_tool(" not in body, (
        "the legend extractor was migrated — good, but then update this test and the "
        "#32 baseline in the same change, and check what the type-detection score does "
        "when only declared keys can be present"
    )
