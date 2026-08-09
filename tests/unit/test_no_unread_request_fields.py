"""
Guard: a request-model field that no code reads (#277 audit).

The defect class, stated once: **a parameter that is declared, accepted, validated, and then
never read.** The caller sets it, the API takes it, nothing happens, nothing raises. The
response is plausible — it is simply the response you would have got without the parameter.

This is not hypothetical; it is how #277 stayed broken for months. `SearchRequest.aspect` was
honored by exactly one branch of the strategy dispatch, and `visual_search` sent it to a
different branch. Pydantic validated it. The branch ignored it. Every "find a similar texture
to this image" returned plain visual similarity, confidently.

A field being unread is worse than a missing feature, because the API *documents* it. The
frontend sends `include_content` on every single search; MIVAA reads it nowhere, so a caller
asking to exclude content receives it anyway. `enable_embedding` defaults to True and is never
read, so a caller setting it False is billed for embeddings they asked to skip.

ALLOWLIST POLICY — shrink-only. Every entry is a live defect with a reason, not an exemption.
Fix one, delete its line. Adding a line requires justifying why a documented parameter that
does nothing is acceptable, which it usually is not; the honest alternative is to delete the
field from the model so the API stops advertising it.

Detection is deliberately generous — a field counts as "read" if its name appears ANYWHERE in
app/ outside its own class body, including a string key or an unrelated identifier. So this
under-reports, and a finding here is close to certain. It cannot see the reverse case (a field
read by only ONE branch of several, which is the exact #277 shape) — that half is guarded by
tests/unit/test_aspect_strategy_guard.py, which pins the refusal for the strategies that
cannot honor an aspect.
"""

import ast
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_APP = _ROOT / "app"

# Known-unread fields, each a real defect awaiting a fix. SHRINK ONLY.
KNOWN_UNREAD = {
    # (model, field): why it is still here
    ("CreateCategoryRequest", "icon"):
        "KB category icon is accepted and dropped — the category saves without it.",
    ("CreateCategoryRequest", "parent_category_id"):
        "KB sub-categories cannot actually nest; the parent is silently discarded.",
    ("DocumentUploadRequest", "enable_embedding"):
        "Defaults True and never read, so passing False does not skip embedding — the caller "
        "is billed for work they explicitly declined.",
    ("ChatRequest", "include_history"):
        "Conversation history inclusion is not togglable despite the flag.",
    ("SearchRequest", "include_content"):
        "Sent by unifiedSearchService on EVERY search and read nowhere; content is always "
        "returned regardless.",
    ("MMRSearchRequest", "diversity_threshold"):
        "MMR diversity is driven by mmr_lambda only; this second knob does nothing.",
    ("AdvancedQueryRequest", "enable_expansion"): "Query expansion toggle is inert.",
    ("AdvancedQueryRequest", "enable_rewriting"): "Query rewriting toggle is inert.",
    ("AdvancedQueryRequest", "metadata_filters"): "Filters are accepted and never applied.",
    ("AdvancedQueryRequest", "search_operator"): "AND/OR operator is accepted and never applied.",
}


def _corpus():
    return {p: p.read_text(encoding="utf-8", errors="ignore") for p in _APP.rglob("*.py")}


def _find_unread():
    corpus = _corpus()
    all_src = "\n".join(corpus.values())
    unread = []

    for path in sorted(_APP.glob("api/**/*.py")):
        try:
            tree = ast.parse(corpus[path])
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {getattr(b, "id", getattr(b, "attr", "")) for b in node.bases}
            if "BaseModel" not in bases or not node.name.endswith("Request"):
                continue

            cls_src = ast.get_source_segment(corpus[path], node) or ""
            outside = all_src.replace(cls_src, "")

            for item in node.body:
                if not (isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)):
                    continue
                field = item.target.id
                if field.startswith("_"):
                    continue
                read = re.search(rf"\.\s*{re.escape(field)}\b", outside) or re.search(
                    rf"['\"]{re.escape(field)}['\"]", outside
                )
                if not read:
                    unread.append((node.name, field, path.relative_to(_ROOT).as_posix()))
    return unread


def test_no_new_unread_request_fields():
    """A newly-added field that nothing reads fails the build."""
    unread = _find_unread()
    new = [u for u in unread if (u[0], u[1]) not in KNOWN_UNREAD]
    assert not new, (
        "Request-model field(s) declared but never read anywhere in app/:\n"
        + "\n".join(f"  {m}.{f}  ({p})" for m, f, p in new)
        + "\n\nThe API documents this parameter and silently ignores it. Either read it or "
        "delete it from the model — do NOT add it to KNOWN_UNREAD without a reason."
    )


def test_allowlist_does_not_rot():
    """
    An entry that no longer reproduces means someone fixed it — delete the line. Left behind,
    the allowlist slowly becomes a place where a real regression can hide.
    """
    unread = {(m, f) for m, f, _ in _find_unread()}
    stale = sorted(set(KNOWN_UNREAD) - unread)
    assert not stale, (
        "These are no longer unread — remove them from KNOWN_UNREAD:\n"
        + "\n".join(f"  {m}.{f}" for m, f in stale)
    )


@pytest.mark.parametrize("entry", sorted(KNOWN_UNREAD.items()))
def test_every_allowlist_entry_states_why(entry):
    """A bare exemption is indistinguishable from an oversight."""
    (model, field), reason = entry
    assert reason and len(reason) > 25, f"{model}.{field} needs a real reason, got {reason!r}"


def test_the_allowlist_only_shrinks():
    """
    A ceiling, not a target. 10 was the count when this guard was written; it may go down and
    must never go up. Raising this number is the move that turns a debt list into a dumping
    ground — the same reason the edge-typecheck baseline is ratcheted, not edited upward.
    """
    assert len(KNOWN_UNREAD) <= 10, (
        f"KNOWN_UNREAD grew to {len(KNOWN_UNREAD)}. Fix the field or delete it from the model."
    )
