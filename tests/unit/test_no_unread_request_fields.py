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

DETECTION, and the one way it lies. A field counts as "read" if its name appears ANYWHERE in
app/ outside its own class body — a string key or an unrelated identifier is enough. That is
deliberately generous, so it under-reports.

It also over-reported, once, and the fix is below. `request.dict()` / `.model_dump()` reads
every field of a model WITHOUT NAMING ANY, so a model consumed that way looked entirely unread
while working perfectly. `CreateCategoryRequest.icon` and `.parent_category_id` were reported
as dropped when `create_category` was in fact inserting them — via mass assignment, which is
its own invariant-8 bug (now fixed with an allowlisted payload). `_models_consumed_wholesale()`
now recognizes that pattern, so those fields are correctly seen as read.

Both blind spots share a cause worth remembering: this guard reasons about NAMES, and a
value can travel without its name. It cannot see the reverse case either — a field read by
only ONE branch of several, the exact #277 shape — which is guarded separately by
tests/unit/test_aspect_strategy_guard.py.
"""

import ast
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_APP = _ROOT / "app"

# Known-unread fields, each a real defect awaiting a fix. SHRINK ONLY.
#
# Emptied by the #338 sweep. All ten original entries are resolved:
#   - 6 belonged to DocumentUploadRequest / MMRSearchRequest / AdvancedQueryRequest, which no
#     route referenced at all. Deleted the models rather than implementing dead knobs.
#   - SearchRequest.include_content and ChatRequest.include_history are now honored.
#   - CreateCategoryRequest.icon / .parent_category_id were false positives (see the module
#     docstring) — stored all along via a mass assignment that is now an explicit payload.
#
# Keep it empty. An entry here is a documented parameter that does nothing.
KNOWN_UNREAD: dict = {}


def _corpus():
    return {p: p.read_text(encoding="utf-8", errors="ignore") for p in _APP.rglob("*.py")}


def _models_consumed_wholesale(all_src: str) -> set:
    """Models passed somewhere as `x.dict()` / `x.model_dump()` / `**x`.

    Such a call reads every field without naming one, so a name-based scan sees the whole
    model as dead when it is fully live. Resolved by parameter NAME (`request`, `body`, …)
    rather than by model, because the annotation is what ties a handler parameter to its
    model — so any model bound to a parameter that is dumped counts as wholesale-read.
    """
    dumped_params = set(re.findall(r"\b(\w+)\s*\.\s*(?:dict|model_dump)\s*\(", all_src))
    dumped_params |= set(re.findall(r"\*\*\s*(\w+)\b", all_src))
    if not dumped_params:
        return set()
    models = set()
    for param, model in re.findall(r"\b(\w+)\s*:\s*([A-Z]\w*Request)\b", all_src):
        if param in dumped_params:
            models.add(model)
    return models


def _find_unread():
    corpus = _corpus()
    all_src = "\n".join(corpus.values())
    wholesale = _models_consumed_wholesale(all_src)
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
            if node.name in wholesale:
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


def test_the_allowlist_stays_empty():
    """
    It started at 10 and the #338 sweep took it to 0, so the ratchet is now a floor: there is
    no longer such a thing as an acceptable unread field, and re-opening the list is how a
    debt register becomes a dumping ground.

    If you genuinely cannot honor a parameter, delete it from the model. An API that does not
    advertise a knob is honest; one that advertises a knob it ignores is not.
    """
    assert len(KNOWN_UNREAD) == 0, (
        f"KNOWN_UNREAD is back to {len(KNOWN_UNREAD)} "
        f"({', '.join(f'{m}.{f}' for m, f in KNOWN_UNREAD)}). "
        "Read the field or remove it from the model — do not re-open this list."
    )
