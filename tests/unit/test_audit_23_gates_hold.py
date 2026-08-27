"""Guards for the mivaa#23 audit fixes (M10-1 … M10-4).

One file per audit, matching `test_audit_18_gates_hold.py`.

Static, not runtime: CI installs pytest alone (`deploy.yml`) and these unit tests
import nothing from `app`, so each case parses source instead. That constrains what
can be checked — a guard here proves the SHAPE is gone, not that the replacement
behaves.

Every case below was watched to FAIL against the pre-fix source before being
committed, with one exception that earned its own lesson:
`test_health_check_is_reachable_without_a_token` originally asserted only that no
route-level dependency existed, passed both ways, and was cited as evidence for a
claim that turned out to be false in production. It now asserts the middleware
exclusion too. A guard that checks half of a claim reads exactly like one that checks
all of it.

NOT covered here, deliberately:
  * M10-4 (no pre-debit on these services) was fixed AFTER this batch, in d683113,
    and its guard lives in `test_paid_route_metering.py` — which enumerates the paid
    doors from source rather than listing them by hand, so a new door here fails on
    the day it is written. Capping the arrays bounded the amplification; it did not
    make anyone pay, and that was deliberately left as separate work rather than
    covered by a green test here.

    (This paragraph used to read "is not fixed by this batch", which was true when
    written and had become a coverage gap that no longer existed. A stale NOT-COVERED
    note is worse than none: it is the one place a reader trusts.)
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"

AI_ROUTES = APP / "api" / "ai_services_routes.py"
CLAUDE_VALIDATION = APP / "services" / "ai_validation" / "claude_validation_service.py"
STAGE_5 = APP / "api" / "pdf_processing" / "stage_5_quality.py"
IMAGE_ANALYSIS = APP / "services" / "images" / "real_image_analysis_service.py"

#: Every route in `ai_services_routes` that reaches a model, or reads a task registry.
#: `/health` is deliberately absent: the `health-check` edge function polls it and it
#: reads nothing.
GATED_ROUTE_FUNCS = [
    "classify_document",
    "classify_batch",
    "detect_boundaries",
    "group_by_product",
    "validate_product",
    "consensus_validate",
    "check_if_critical",
    "process_pdf_enhanced",
]

#: Request-model fields whose LENGTH is the bill — one provider call per element.
BOUNDED_ARRAY_FIELDS = [
    ("ClassifyBatchRequest", "contents"),
    ("ClassifyBatchRequest", "contexts"),
    ("DetectBoundariesRequest", "chunks"),
    ("ValidateProductRequest", "chunks"),
    ("ValidateProductRequest", "images"),
]

#: Free-text fields whose SIZE is the bill — length drives input tokens.
BOUNDED_TEXT_FIELDS = [
    ("ClassifyRequest", "content"),
    ("ConsensusValidateRequest", "content"),
]


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _strip_comments(src: str) -> str:
    """Drop comments and docstrings so prose about the old bug is not read as code."""
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"^\s*#.*$", "", src, flags=re.MULTILINE)
    return src


def _node(src: str, name: str):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _source_of(src: str, name: str) -> str:
    return ast.get_source_segment(src, _node(src, name)) or ""


def _field_call(src: str, model: str, field: str) -> str:
    """The `Field(...)` call source for one attribute of one pydantic model."""
    cls = _node(src, model)
    for stmt in cls.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) \
                and stmt.target.id == field:
            assert stmt.value is not None, f"{model}.{field} has no Field(...) at all"
            return ast.get_source_segment(src, stmt.value) or ""
    raise AssertionError(f"{model}.{field} not found")


# ───────────────────────────────────────────────────────────────────────────
# M10-2 / M10-3 — the arrays were the bill, and nothing bounded them
# ───────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(("model", "field"), BOUNDED_ARRAY_FIELDS, ids=lambda v: str(v))
def test_every_fan_out_array_declares_a_maximum(model: str, field: str):
    """`classify_batch` `asyncio.gather`s one Claude call per element with no
    semaphore, so an N-element array is N concurrent calls on the platform's
    Anthropic account. `chunks` on the boundary and validation doors is the same
    shape one embedding call at a time. The whole file contained zero `max_items`,
    `max_length` and `len()` guards."""
    call = _field_call(_read(AI_ROUTES), model, field)
    assert "max_length=" in call, (
        f"{model}.{field} declares no maximum length. One authenticated POST then "
        "buys an arbitrary number of provider calls (audit #23 M10-2)."
    )


@pytest.mark.parametrize(("model", "field"), BOUNDED_TEXT_FIELDS, ids=lambda v: str(v))
def test_free_text_reaching_a_model_is_bounded(model: str, field: str):
    """An array of 50 items each a megabyte is the same attack with a different
    shape: length drives input tokens."""
    call = _field_call(_read(AI_ROUTES), model, field)
    assert "max_length=" in call, (
        f"{model}.{field} is unbounded free text sent straight to a model"
    )


def test_the_batch_cap_is_a_real_number_not_a_placeholder():
    """A cap of a million is not a cap. These bound the amplification, so a change
    that raises them past what a real page or catalogue needs should be argued for
    in review rather than slipped through."""
    src = _read(AI_ROUTES)
    caps = {
        name: int(re.search(rf"^{name} = ([\d_]+)", src, re.MULTILINE).group(1).replace("_", ""))
        for name in ("MAX_BATCH_CONTENTS", "MAX_CHUNKS", "MAX_IMAGES")
        if re.search(rf"^{name} = ([\d_]+)", src, re.MULTILINE)
    }
    assert set(caps) == {"MAX_BATCH_CONTENTS", "MAX_CHUNKS", "MAX_IMAGES"}, (
        f"a fan-out cap constant went missing: {sorted(caps)}"
    )
    assert caps["MAX_BATCH_CONTENTS"] <= 200, (
        f"MAX_BATCH_CONTENTS is {caps['MAX_BATCH_CONTENTS']} — that many CONCURRENT "
        "Claude calls from one unauthenticated-shaped request is not a bound"
    )
    assert caps["MAX_CHUNKS"] <= 2000, f"MAX_CHUNKS is {caps['MAX_CHUNKS']}"


# ───────────────────────────────────────────────────────────────────────────
# #22 M9-1 — the eight routes had no gate of any kind
# ───────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("func", GATED_ROUTE_FUNCS)
def test_every_model_calling_route_gates_itself(func: str):
    """Invariant 5: a route declares its own dependency rather than relying on the
    JWT middleware alone. These eight had no decorator dependency, no router-level
    dependency and no signature dependency — and #23 confirmed no guard lived in the
    services behind them either."""
    body = _source_of(_read(AI_ROUTES), func)
    assert "Depends(get_workspace_context)" in body, (
        f"{func} no longer declares Depends(get_workspace_context)"
    )


def test_health_check_is_reachable_without_a_token():
    """Two halves, and the first shipped believing the second was already true.

    When this batch landed I wrote that /health "stays open". Only half of that was
    ever the case: no route-level dependency was added, but `/api/v1/ai-services` is
    not in `JWTAuthMiddleware.exclude_paths`, so the middleware 401'd the probe anyway
    — measured against production after the deploy. `health-check`'s
    `checkPythonEndpoint` sends no Authorization header and tests `res.ok`, so that
    dashboard row had never been green.

    So the guard now asserts BOTH halves. The route has no dependency AND the exact
    path is excluded. Asserting only the first is what let a false claim look verified.
    """
    body = _source_of(_read(AI_ROUTES), "health_check")
    assert "Depends(" not in body, (
        "health_check grew a route-level dependency — the health-check edge function "
        "polls it unauthenticated. If that is deliberate, update the edge function first."
    )

    middleware = _read(APP / "middleware" / "jwt_auth.py")
    assert '"/api/v1/ai-services/health"' in middleware, (
        "the exact-path exclusion for the ai-services liveness probe is gone, so the "
        "middleware 401s it again and the dashboard row goes back to permanently red"
    )
    assert '"/api/v1/ai-services"' not in middleware, (
        "the PREFIX /api/v1/ai-services is excluded — that hands anonymous access to "
        "all eight model-calling routes, which is the opposite of this batch's point"
    )


# ───────────────────────────────────────────────────────────────────────────
# M10-1 — cross-tenant read AND WRITE on the silver layer
# ───────────────────────────────────────────────────────────────────────────

def test_the_validation_service_knows_what_a_workspace_is():
    """The word `workspace` appeared ZERO times in this file. It reads and writes
    `document_images` — the silver layer — by caller-supplied id, over a service-role
    connection where RLS is not a backstop."""
    src = _read(CLAUDE_VALIDATION)
    assert "_require_workspace" in src, (
        "the fail-closed workspace resolver is gone; an unscoped query here spans "
        "every tenant"
    )
    assert "workspace_id" in src


@pytest.mark.parametrize(
    "method",
    ["queue_image_for_validation", "process_validation_queue", "_validate_single_image"],
)
def test_every_validation_method_takes_a_workspace(method: str):
    node = _node(_read(CLAUDE_VALIDATION), method)
    names = [a.arg for a in node.args.args] + [a.arg for a in node.args.kwonlyargs]
    assert "workspace_id" in names, (
        f"{method} no longer takes workspace_id — it is back to keying on a "
        "caller-supplied id alone"
    )


def test_no_read_or_write_reaches_document_images_by_id_alone():
    """The write is what made this the worst finding in the batch: a corrupted
    validation verdict on `document_images` propagates into embeddings, search
    ranking and product association, and no drift check would surface it."""
    src = _strip_comments(_read(CLAUDE_VALIDATION))
    for m in re.finditer(r"table\('document_images'\)(.*?)\.execute\(\)", src, re.DOTALL):
        chain = m.group(1)
        assert ".eq('workspace_id'" in chain, (
            "a document_images chain in claude_validation_service carries no "
            f"workspace predicate:\n{chain.strip()[:300]}"
        )


def test_the_queue_is_scoped_too():
    """`claude_validation_queue` gained a `workspace_id` column in the same change
    (migration `claude_validation_queue_workspace_binding`). If the predicate is
    dropped, draining the queue reaches other tenants' jobs again."""
    src = _strip_comments(_read(CLAUDE_VALIDATION))
    chains = re.findall(r"table\('claude_validation_queue'\)(.*?)\.execute\(\)", src, re.DOTALL)
    assert chains, "the queue table is no longer touched here — has the service moved?"
    for chain in chains:
        if ".insert(" in chain:
            # The insert passes a dict built above it; the binding is the stamped key.
            assert "'workspace_id': ws" in src, (
                "the enqueue no longer stamps workspace_id onto the queue row, so "
                "every later read of that row is unscoped by construction"
            )
            continue
        assert ".eq('workspace_id'" in chain, (
            f"a claude_validation_queue chain carries no workspace predicate:\n{chain.strip()[:300]}"
        )


def test_the_two_ids_are_made_to_agree_before_enqueue():
    """Storing the caller's asserted (document_id, image_id) pair is the mass-assignment
    shape — nothing downstream re-checks it. The enqueue resolves the image inside the
    workspace and rejects a document_id that is not the one it belongs to."""
    body = _strip_comments(_source_of(_read(CLAUDE_VALIDATION), "queue_image_for_validation"))
    assert "table('document_images')" in body, (
        "queue_image_for_validation no longer resolves the image before inserting"
    )
    assert "document_id" in body and "raise ValueError" in body, (
        "the mismatch between the image's real document and the claimed one is no "
        "longer rejected"
    )


def test_select_star_is_gone_from_the_validation_service():
    """`select('*')` on `document_images` returned the full row — including columns
    the validator has no use for — to whoever supplied the id."""
    src = _strip_comments(_read(CLAUDE_VALIDATION))
    assert "select('*')" not in src and 'select("*")' not in src, (
        "select('*') is back in claude_validation_service; project the columns needed"
    )


@pytest.mark.parametrize("path", [STAGE_5, IMAGE_ANALYSIS], ids=lambda p: p.name)
def test_both_call_sites_pass_the_workspace_through(path: Path):
    """The service failing closed is only half of it: if a caller has no workspace to
    pass, ingestion breaks loudly instead of writing across tenants — which is the
    intended trade. Both callers already have `workspace_id` in scope."""
    src = _strip_comments(_read(path))
    idx = src.find("ClaudeValidationService")
    assert idx != -1, f"{path.name} no longer constructs ClaudeValidationService"
    assert "workspace_id" in src[idx:idx + 800], (
        f"{path.name} calls the validation service without threading workspace_id"
    )
