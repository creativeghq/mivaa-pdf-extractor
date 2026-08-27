"""A prompt-store outage is not an empty prompt list (mivaa#28 M14-5, #29 M15-5).

M14-5. Thirteen handlers across three services caught every exception and returned `[]`,
`None`, `{}` or `False`. So a database outage read as "this workspace has no prompts
configured", which is a completely different fact and calls for a completely different
response.

The platform rule is explicit that `PromptNotConfigured` ("add the row") must be
DISTINGUISHABLE from `PromptStoreUnavailable` ("the database is down"), because the six
loaders `prompt_registry` replaced returned `None` for both and no caller could react
correctly to either. That distinction exists in the registry — and every one of these
handlers erased it one layer up.

It is the same failure this codebase keeps producing in different costumes: `[]` for a
failed read (#21 M8-5), `""` and `[]` for a failed OCR (#22 M9-7), a filtered-out
`paddleocr_failed` marker (#25 M12-3), a `0.5` for a comparison that never ran (#25
M12-1). A plausible empty answer, nothing raised, nothing to see.

M15-5 is the third instance of unbounded batch input on a paid path, after #23 M10-2 and
the rechunk loop it names. Bounds are generous on purpose: a cap that fires on real usage
gets raised by the next person who hits it, and then it is not a cap.

WATCHED TO FAIL: run against the pre-fix tree, 15 of 18 cases fired. The three that
pass both ways pin PREMISES rather than fixes: that the two prompt error types are
still distinct, that `prompt_registry` itself still refuses to guess, and that
`all=True` backfill still exists — capping `doc_ids` without it would just move the
problem onto whoever has 900 docs to rechunk.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"

PROMPT_SERVICES = [
    APP / "services" / "utilities" / "unified_prompt_service.py",
    APP / "services" / "utilities" / "admin_prompt_service.py",
    APP / "services" / "utilities" / "prompt_template_service.py",
]
REGISTRY = APP / "services" / "utilities" / "prompt_registry.py"
RAG = APP / "api" / "rag_routes.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _strip_comments(src: str) -> str:
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"'''[\s\S]*?'''", "", src)
    src = re.sub(r"^\s*#.*$", "", src, flags=re.MULTILINE)
    return src


def _collapsing_handlers(src: str):
    """`except: ... return []/None/{}/False` — a caught failure turned into an answer."""
    out = []
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        last = node.body[-1]
        if not isinstance(last, ast.Return):
            continue
        v = last.value
        if (
            (isinstance(v, ast.List) and not v.elts)
            or (isinstance(v, ast.Dict) and not v.keys)
            or (isinstance(v, ast.Constant) and v.value in (None, False))
        ):
            fn = next(
                (
                    x.name
                    for x in ast.walk(tree)
                    if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and x.lineno <= node.lineno <= (x.end_lineno or 0)
                ),
                "?",
            )
            out.append(fn)
    return out


# -------------------------------------------------------------------------
# M14-5
# -------------------------------------------------------------------------

@pytest.mark.parametrize("path", PROMPT_SERVICES, ids=lambda p: p.name)
def test_no_prompt_service_turns_a_failure_into_an_answer(path: Path):
    offenders = _collapsing_handlers(_read(path))
    assert not offenders, (
        f"{path.name} collapses a caught failure into an empty result in: "
        + ", ".join(sorted(set(offenders)))
        + "\n\nRaise PromptStoreUnavailable. An empty list means 'this workspace has "
          "no prompts'; a database outage means something else entirely, and a caller "
          "that cannot tell them apart cannot do the right thing about either."
    )


@pytest.mark.parametrize("path", PROMPT_SERVICES, ids=lambda p: p.name)
def test_each_service_raises_the_store_unavailable_type(path: Path):
    src = _strip_comments(_read(path))
    assert "raise PromptStoreUnavailable" in src, (
        f"{path.name} no longer raises the typed error, so its callers are back to "
        "guessing"
    )


def test_both_error_types_still_exist_and_are_distinct():
    """Every case above is about preserving a distinction. If the two types merge, they
    are all guarding nothing."""
    src = _read(REGISTRY)
    assert "class PromptNotConfigured(PromptError)" in src
    assert "class PromptStoreUnavailable(PromptError)" in src
    assert "class PromptNotConfigured(PromptStoreUnavailable)" not in src, (
        "the two prompt failures have been collapsed into one hierarchy — the whole "
        "point is that a caller can catch either deliberately"
    )


def test_the_registry_itself_still_refuses_to_guess():
    """`prompt_registry` is the source these services sit on top of. If IT starts
    returning a default, nothing above it can be correct."""
    # Raw source, not stripped: `_collapsing_handlers` walks the AST, so comments are
    # already invisible to it — and stripping docstrings can leave a function with an
    # empty body, which is a SyntaxError rather than a finding.
    assert not _collapsing_handlers(_read(REGISTRY)), (
        "prompt_registry has grown a handler that returns an empty result — that is the "
        "defect this whole registry replaced"
    )


# -------------------------------------------------------------------------
# M15-5
# -------------------------------------------------------------------------

def _model_src(name: str) -> str:
    src = _read(RAG)
    node = next(
        n for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.ClassDef) and n.name == name
    )
    return ast.get_source_segment(src, node) or ""


@pytest.mark.parametrize(
    "field", ["top_k", "similarity_threshold", "search_types", "categories", "entity_types", "query"]
)
def test_every_knowledge_base_search_input_is_bounded(field: str):
    """This route creates a Voyage vector per call and fans out across several search
    types, so an unbounded input is paid work sized by the caller."""
    body = _model_src("KnowledgeBaseSearchRequest")
    at = body.index(f"{field}:")
    decl = body[at:at + 400]
    assert any(b in decl for b in ("le=", "max_length=")), (
        f"KnowledgeBaseSearchRequest.{field} has no upper bound (#29 M15-5)"
    )


@pytest.mark.parametrize("field", ["doc_ids", "limit", "offset"])
def test_the_rechunk_inputs_are_bounded(field: str):
    """Rechunking embeds every chunk of every named doc, and the route loops over the
    whole list — `doc_ids` and `limit` size a Voyage bill directly."""
    body = _model_src("KBRechunkRequest")
    at = body.index(f"{field}:")
    decl = body[at:at + 300]
    assert any(b in decl for b in ("le=", "ge=", "max_length=")), (
        f"KBRechunkRequest.{field} is unbounded again (#29 M15-5)"
    )


def test_the_backfill_mode_still_exists():
    """`all=True` is why an unbounded `doc_ids` is not needed — it is the sanctioned way
    to process more than one page. Capping the list without it would just move the
    problem onto whoever has 900 docs to rechunk."""
    assert "all: bool" in _model_src("KBRechunkRequest")
