"""Guards for the mivaa#20 image-processing fixes (M7-3, M7-4, M7-5).

M7-1 and M7-2 landed earlier and are held elsewhere: `test_no_undefined_attribute_access`
sweeps for the `self.supabase.client` shape that made both dead, and the workspace
predicate M7-2 needed is in `material_visual_search_service` itself.

M7-3 is the one whose own file already forbade it. Forty lines above the offending code,
a comment states the rule: the ingestion path uses real Anthropic tool_use with a forced
tool_choice — "no regex repair, no JSON-parse fallback". Underneath, when Claude returned
text instead of a tool block, the reply went through `_parse_vision_analysis_json`:
fenced-block extraction and a first-`{...}` match.

Why that mattered more than an ordinary parser: the repaired payload was persisted as
`vision_analysis` and then fed the Voyage understanding embedding. So an unvalidated
reply became a vector that is indexed and ranked exactly like a real one, and nothing
downstream can tell them apart — `_validate_vision_analysis` checks the SHAPE, and a
repaired fragment can be perfectly well-shaped. A tight token budget produces a truncated
answer; recovering JSON from it recovers a fragment and gives it the standing of a
complete analysis.

M7-4 is the same class on the path taken by the images the primary classifier was LEAST
sure about — exactly where an unvalidated verdict does the most damage.

M7-5 is the twelfth instance of the two-unchecked-ids class, and the finding scoped it
honestly: a guard could have existed in a caller, it just did not exist anywhere.

WATCHED TO FAIL: the file was run against the pre-fix source. 14 of 15 cases fired. The
one that passes both ways is `test_the_ingestion_vision_path_still_forces_its_tool`,
which pins the PREMISE rather than the fix — the vision call already forced its tool, and
every M7-3 case is guarding a contract nobody is asking for if that stops.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
IMG = APP / "services" / "images" / "image_processing_service.py"


def _read() -> str:
    return IMG.read_text(encoding="utf-8")


def _strip_comments(src: str) -> str:
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"'''[\s\S]*?'''", "", src)
    src = re.sub(r"^\s*#.*$", "", src, flags=re.MULTILINE)
    return src


def _node(src: str, name: str):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _body(name: str) -> str:
    src = _read()
    return _strip_comments(ast.get_source_segment(src, _node(src, name)) or "")


# -------------------------------------------------------------------------
# M7-3 -- a missing tool block is a failed analysis
# -------------------------------------------------------------------------

def test_the_json_repair_helper_is_gone_entirely():
    """Deleted rather than merely unused. A dead JSON-repair helper sitting next to a
    vision call is an invitation, and this one had already been accepted once."""
    src = _strip_comments(_read())
    assert "_parse_vision_analysis_json" not in src, (
        "the vision JSON-repair helper is back (#20 M7-3). A reply that is not a "
        "tool_use block is a FAILED analysis — repairing it produces a payload that "
        "validates on shape and then becomes an indexed embedding."
    )


def test_no_tool_block_stamps_the_analysis_failed():
    """The function name is deliberately not pinned — what matters is that wherever the
    tool_use search happens, the None branch marks failure instead of reading text.

    Comments are stripped BEFORE the window is measured. Measuring on raw source made
    this test fail against the fix itself, because the twenty lines of comment
    explaining why the fallback was removed pushed the assertion out of range — a guard
    that reads prose as distance is a guard that moves when someone writes a paragraph.
    """
    code = _strip_comments(_read())
    at = code.index("if tool_input is None:")
    window = code[at:at + 500]
    assert "_stamp_vision_analysis_outcome(image_id, failed=True)" in window, (
        "a missing tool_use block no longer marks the analysis failed (#20 M7-3)"
    )
    before_return = window.split("return None")[0]
    # The SHAPE, not the word: "text" appears in the log message explaining why the
    # fallback was removed, and a guard that greps for a word in a log line is a guard
    # that fires on its own explanation.
    for shape in ("text_parts", "'text'", '"text"'):
        assert shape not in before_return, (
            f"the text-block fallback is back ({shape} found) — a reply that is not a "
            "tool call is being read as prose again"
        )


def test_the_ingestion_vision_path_still_forces_its_tool():
    """Every case above assumes the call is forced. If tool_choice goes, they are all
    guarding a contract that is no longer being asked for."""
    src = _strip_comments(_read())
    assert "tool_choice" in src, "the vision path no longer forces a tool call"


# -------------------------------------------------------------------------
# M7-4 -- the least-confident images get the strictest contract
# -------------------------------------------------------------------------

def test_the_low_confidence_recheck_forces_a_tool_call():
    body = _body("validate_with_claude")
    assert "call_with_tool(" in body, (
        "the low-confidence re-check is parsing free-form JSON again (#20 M7-4). This "
        "is the path taken by the images the primary classifier was least sure about, "
        "and its verdict drives a DB write — invariant 9."
    )
    assert "json.loads" not in body, "the json.loads on the text reply is back"


def test_the_diagnostic_branches_for_an_error_that_can_no_longer_happen_are_gone():
    """`claude_empty_response` and `claude_not_json` existed to tell apart the causes of
    "Expecting value: line 1 column 1 (char 0)". A forced tool call cannot produce it —
    one typed error replaces three guesses at it."""
    body = _body("validate_with_claude")
    assert "claude_not_json" not in body
    assert "claude_empty_response" not in body


def test_a_missing_tool_block_still_returns_the_soft_dict():
    """Callers treat a soft dict as "not a material" and carry on to the next image.
    Raising here would abandon the rest of the batch over one reply."""
    body = _body("validate_with_claude")
    assert "ToolCallNotReturned" in body
    assert "'model': 'claude_no_tool_use'" in body


def test_the_classification_tool_is_one_definition():
    """Both the primary classifier and the re-check must agree on the vocabulary. A
    classifier and its own second opinion disagreeing about what MIXED means is not a
    failure anything would report."""
    src = _read()
    assert src.count('"name": "emit_classification"') == 1, (
        "a second copy of the classification tool schema exists — the two paths can now "
        "drift on the enum"
    )
    assert "CLASSIFICATION_TOOL" in src


@pytest.mark.parametrize(
    "value", ["PRODUCT_IMAGE", "TECHNICAL_DIAGRAM", "DECORATIVE", "MIXED"]
)
def test_the_classification_vocabulary_is_unchanged(value: str):
    """`is_material_classification` maps these four. A fifth would be forced on the model
    by the schema and then fall through that mapping silently."""
    src = _read()
    schema = src[src.index("CLASSIFICATION_TOOL"):src.index("CLASSIFICATION_TOOL") + 1200]
    assert f'"{value}"' in schema


# -------------------------------------------------------------------------
# M7-5 -- the document and the workspace must be related
# -------------------------------------------------------------------------

def test_the_entry_point_verifies_the_document_belongs_to_the_workspace():
    body = _body("save_images_and_generate_clips")
    assert "_assert_document_in_workspace(document_id, workspace_id)" in body, (
        "the entry point no longer checks that its two ids describe the same thing "
        "(#20 M7-5) — and MIVAA has no RLS backstop, every call is service role"
    )


def test_the_check_runs_before_any_work():
    body = _body("save_images_and_generate_clips")
    assert body.index("_assert_document_in_workspace") < body.index("icon_candidates ="), (
        "the tenancy check now runs after work has begun"
    )


def test_the_check_fails_closed():
    """A tenancy check that treats "I could not find out" as "go ahead" switches itself
    off exactly when the database is unhappy, which is when it is most needed."""
    body = _body("_assert_document_in_workspace")
    assert body.count("raise TenancyViolation") >= 3, (
        "one of the three refusal paths (missing id, lookup error, mismatch) no longer "
        "raises"
    )
    assert "except Exception" in body and "return" not in body.replace("returns", ""), (
        "the helper has grown a path that returns instead of raising"
    )


def test_the_check_reads_the_table_that_actually_exists():
    """`pdf_documents` is the STORAGE BUCKET name, not a table. Checking against it would
    have been a lookup that always missed — a tenancy check that passes because it can
    never find anything is worse than none, because it looks like one."""
    body = _body("_assert_document_in_workspace")
    assert 'table("documents")' in body, (
        "the relationship check is querying something other than `documents`, which is "
        "what document_images.document_id actually references"
    )
    assert "pdf_documents" not in body
