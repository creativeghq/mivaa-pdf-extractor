"""Guards for the mivaa#35 ingestion-orchestrator fixes (M20-1, M20-2).

Latent — every ingestion table holds zero rows — which is the argument for fixing them
now: this is the code that goes live on the first production ingestion.

M20-1. The pre-stage validations raised BARE, outside any orchestrator handler, so a
missing file, an unstattable file or a zero-byte PDF exited the function with
`background_jobs.status` still `processing` and NO terminal `stage_history` event.
That is the state the rest of the system handles worst, and the interaction is
three-way: auto-recovery hunts jobs stuck in `processing`; `reprocess` refuses to touch
a document whose job is pending/processing (#34 M19-3), so a stranded job also BLOCKS
the obvious retry; and none of it is visible because the job simply stops. A file that
does not exist is the most ordinary failure this pipeline has.

M20-2. Third location for an unbounded PDF, after #22 M9-6 and #24 M11-6. The rule now
lives in one module — three copies is how one of them drifts, and the credit-debit
helper had seven before it was unified.

`app.utils.pdf_bounds` is pure stdlib-plus-duck-typing, so its BEHAVIOUR is checked
here, not just its shape.

On watching these fail: the 7 source-based cases were run against the pre-fix tree and
all 7 fired. The 10 behaviour cases exercise `pdf_bounds`, which is NEW, so there is no
pre-fix state for them to fail against. `test_the_bound_still_precedes_the_allocation`
also passes both ways on purpose — the ORDER was already right in catalog_routes and the
consolidation had to keep it that way.

M20-3 is now covered in HALF, deliberately and explicitly. The finding asks for two
things: validate the tuple once at entry, and thread the validated workspace through
every stage instead of re-trusting the parameter. The first is done —
`app.utils.tenancy.assert_job_tuple`, called before the credit preflight and before any
file is touched. The second is a refactor of an 1,800-line function and is not attempted.

That split is a judgement, not an oversight: rethreading changes nothing about a tuple
that has already been proven coherent, and the entry check is what makes the parameter
trustworthy in the first place. The residual gap is that a later stage could still be
handed a different workspace_id by a future edit, and nothing would catch it.
"""

import ast
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
RAG_ROUTES = APP / "api" / "rag_routes.py"
CATALOG = APP / "api" / "catalog_routes.py"


def _load():
    spec = importlib.util.spec_from_file_location("pdf_bounds", APP / "utils" / "pdf_bounds.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["pdf_bounds"] = module
    spec.loader.exec_module(module)
    return module


pb = _load()


class _Rect:
    def __init__(self, w, h):
        self.width, self.height = w, h


class _Page:
    def __init__(self, w, h):
        self.rect = _Rect(w, h)


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
# M20-2 — the bound itself
# ───────────────────────────────────────────────────────────────────────────

def test_an_ordinary_page_passes():
    """A4 at 400 DPI is ~3300x4700 — the check must not fire on real documents, or it
    gets removed rather than tuned."""
    pb.assert_renderable(_Page(595, 842), 400 / 72)
    pb.assert_page_count(400)


def test_a_page_that_would_render_enormous_is_refused():
    """THE case the compressed byte count cannot predict: the render size is
    page_rect * dpi/72, and the allocation happens inside get_pixmap long after the
    bytes were accepted."""
    with pytest.raises(pb.PdfBoundsError):
        pb.assert_renderable(_Page(200_000, 200_000), 400 / 72)


def test_a_modest_page_at_an_absurd_zoom_is_refused():
    """The page can be ordinary and the DPI the problem — both feed the same product."""
    with pytest.raises(pb.PdfBoundsError):
        pb.assert_renderable(_Page(595, 842), 1000.0)


def test_a_zero_area_page_is_refused_rather_than_rendered():
    with pytest.raises(pb.PdfBoundsError):
        pb.assert_renderable(_Page(0, 842), 1.0)


def test_a_page_with_no_geometry_is_refused():
    """Duck-typed on purpose — the callers pass a PyMuPDF page and the tests pass a
    stub — so an object with no `rect` must fail closed rather than AttributeError."""
    with pytest.raises(pb.PdfBoundsError):
        pb.assert_renderable(object(), 1.0)


@pytest.mark.parametrize("count", [0, -1, pb.MAX_PDF_PAGES + 1])
def test_an_implausible_page_count_is_refused(count: int):
    with pytest.raises(pb.PdfBoundsError):
        pb.assert_page_count(count)


def test_it_raises_a_value_error_not_an_http_exception():
    """Three callers: a route, a service and a background orchestrator. Only one can
    return a status code, so a shared helper raising HTTPException would force the
    other two to catch a web exception to do bookkeeping."""
    assert issubclass(pb.PdfBoundsError, ValueError)
    assert not any(base.__name__ == "HTTPException" for base in pb.PdfBoundsError.__mro__)


def test_the_limits_are_real_numbers():
    """A cap of a billion is not a cap."""
    assert pb.MAX_RASTER_PIXELS <= 200_000_000
    assert pb.MAX_RASTER_EDGE <= 100_000
    assert pb.MAX_PDF_PAGES <= 50_000


# ───────────────────────────────────────────────────────────────────────────
# M20-2 — one rule, not three copies
# ───────────────────────────────────────────────────────────────────────────

def test_the_rasterisation_route_uses_the_shared_bound():
    """This file carried its own copy from #24 M11-6. Two audits later the same gap was
    found in a third place, which is the argument for one module."""
    body = _body(_read(CATALOG), "_assert_renderable")
    assert "assert_renderable(page, zoom)" in body, (
        "catalog_routes grew its own bounds check again (#35 M20-2)"
    )
    assert "MAX_RASTER_EDGE" not in body, (
        "a local copy of the limits is back — that is the drift this consolidation "
        "exists to prevent"
    )


def test_the_bound_still_precedes_the_allocation():
    """The ORDER is the fix, and consolidating must not have moved it."""
    src = _strip_comments(_read(CATALOG))
    assert src.index("_assert_renderable(page") < src.index("get_pixmap("), (
        "the size check now runs after get_pixmap — the allocation it exists to "
        "prevent has already happened"
    )


def test_the_orchestrator_bounds_the_document_before_it_iterates():
    body = _body(_read(RAG_ROUTES), "process_document_with_discovery")
    assert "assert_page_count(" in body, (
        "the orchestrator no longer bounds the page count, and every stage below it is "
        "per-page (#35 M20-2)"
    )


# ───────────────────────────────────────────────────────────────────────────
# M20-1 — a job that stops must say why
# ───────────────────────────────────────────────────────────────────────────

def test_the_terminal_failure_helper_writes_both_halves():
    """Pipeline convention 9: the audit log must show why a job ended. Marking the row
    failed without the stage_history event leaves the audit log silent, which is half
    the finding."""
    body = _body(_read(RAG_ROUTES), "_fail_job_terminally")
    assert '"status": "failed"' in body, "the job row is no longer marked failed"
    assert "append_stage_history" in body, (
        "no terminal stage_history event is written, so the audit log still cannot show "
        "why the job ended"
    )


def test_the_helper_never_raises_over_its_own_bookkeeping():
    """The caller is already failing. A bookkeeping error must not replace the real
    exception — that is how the actual cause gets lost."""
    body = _body(_read(RAG_ROUTES), "_fail_job_terminally")
    assert body.count("except Exception") >= 2, (
        "one of the two writes can now raise out of the helper and mask the real failure"
    )


@pytest.mark.parametrize(
    "reason",
    ["File not found at", "Cannot stat PDF at", "is 0 bytes"],
)
def test_every_pre_stage_abort_marks_the_job_failed(reason: str):
    """These raised BARE, outside any handler, so the job stayed `processing` forever.
    A missing file is the most ordinary failure this pipeline has."""
    body = _body(_read(RAG_ROUTES), "process_document_with_discovery")
    at = body.find(reason)
    assert at != -1, f"the {reason!r} validation is gone"
    window = body[max(0, at - 600):at + 600]
    assert "_fail_job_terminally(" in window, (
        f"the {reason!r} path raises without marking the job failed, stranding it in "
        "`processing` with no terminal event (#35 M20-1)"
    )


# ───────────────────────────────────────────────────────────────────────────
# M20-4 — an ingestion must not report success it did not earn
# ───────────────────────────────────────────────────────────────────────────

def test_a_product_creation_failure_reaches_the_parent_job():
    """Only the SUB-job was marked failed, and the parent had already been persisted
    `completed` — so an ingestion whose product creation blew up reported a clean
    success with zero products. Products are the point of the pipeline."""
    body = _body(_read(RAG_ROUTES), "create_products_background")
    assert "product_creation_failed" in body, (
        "the parent job is no longer told that product creation failed (#35 M20-4)"
    )


def test_the_failure_is_recorded_in_metadata_not_as_a_new_status():
    """`background_jobs_status_check` admits seven values and
    `completed_with_failures` is not one of them. Adding it would need every reader —
    the edge runner, the admin UI, auto-recovery — to learn it, and audit #217 M7 is the
    record of what happens when they do not: writing 'running' made job-research runs
    invisible and mis-bucketed."""
    body = _body(_read(RAG_ROUTES), "create_products_background")
    assert "completed_with_failures" not in body, (
        "a status outside the CHECK vocabulary is being written — it will either raise "
        "or be unrecognised by every reader"
    )
    assert "'metadata': parent_meta" in body, (
        "the failure marker is no longer written to the parent's metadata"
    )


def test_the_parent_is_not_marked_failed_outright():
    """The document WAS processed — chunks, images and embeddings all landed. Failing
    the parent would send auto-recovery to restart a job that mostly succeeded."""
    body = _body(_read(RAG_ROUTES), "create_products_background")
    parent_writes = re.findall(r"\.eq\('id', job_id\)", body)
    assert parent_writes, "nothing updates the parent job any more"
    assert "'status': 'failed'" not in body.split("parent_meta")[0], (
        "the parent job is being marked failed, which restarts an ingestion that "
        "largely worked"
    )


def test_zero_products_from_real_candidates_is_reported():
    """Candidates found and nothing created is the silent zero at the level that matters
    most to a user, and it looks identical to a catalogue that genuinely had none. Zero
    CANDIDATES is a different, legitimate answer and is deliberately not flagged."""
    body = _body(_read(RAG_ROUTES), "create_products_background")
    assert "if candidates and not products_created:" in body, (
        "a run that found candidates and created nothing no longer reports itself"
    )
    assert "capture_message" in body or "logger.error" in body


# ───────────────────────────────────────────────────────────────────────────
# #22 M9-6 / M9-7 — the third copy of the bound, and the empty OCR result
#
# Kept in this file rather than a new one because M20-2 is the same finding: the
# PDF page bound was missing in THREE places, found by three separate audits, and
# consolidating it was the fix. A guard per audit issue would be three guards over
# one rule.
#
# Watched to fail: 3 of these 4 fired against the pre-fix tree.
# `test_the_pdf_processor_uses_the_shared_bound_not_its_own` passes both ways by
# design — it is an INVERSE assertion. There was no local copy of the limit before
# and there must not be one after; it exists to catch the tempting wrong fix
# (paste the constant in) rather than a regression of the old state.
# ───────────────────────────────────────────────────────────────────────────

PDF_PROCESSOR = APP / "services" / "pdf" / "pdf_processor.py"


def test_the_pdf_processor_bounds_the_page_count_too():
    """The third site (#22 M9-6). Every stage below it is per-page, and the compressed
    byte count cannot predict the page count."""
    src = _read(PDF_PROCESSOR)
    assert "assert_page_count(" in _strip_comments(src), (
        "pdf_processor no longer bounds the page count before per-page work (#22 M9-6)"
    )


def test_the_pdf_processor_uses_the_shared_bound_not_its_own():
    body = _strip_comments(_read(PDF_PROCESSOR))
    assert "MAX_PDF_PAGES =" not in body, (
        "a local copy of the page limit is back — three copies of this rule is how one "
        "of them drifts, which is what #22, #24 and #35 each found separately"
    )


def test_a_whole_document_ocr_failure_is_marked_rather_than_empty():
    """`return "", []` made a wholesale OCR failure identical to a document whose images
    genuinely hold no text — and the caller's `if ocr_text:` skips both."""
    src = _strip_comments(_read(PDF_PROCESSOR))
    assert '"method": "ocr_failed"' in src, (
        "the OCR failure marker is gone (#22 M9-7); an empty return is ambiguous and "
        "pipeline convention 1 exists to remove exactly that ambiguity"
    )
    assert 'return "", []' not in src


def test_the_failure_marker_is_not_attached_to_image_zero():
    """The marker list is length-1, and the positional enhancement loop below would
    otherwise hand it to the first image as though it were that image's own result."""
    src = _strip_comments(_read(PDF_PROCESSOR))
    at = src.index('ocr_results[0].get("method") == "ocr_failed"')
    window = src[at:at + 500]
    assert 'content_metrics["ocr_status"] = "failed"' in window, (
        "a wholesale OCR failure is no longer recorded on the document"
    )
    assert "for i, image_data in enumerate(extracted_images)" not in window.split("else:")[0], (
        "the per-image enhancement runs on the marker list again"
    )


# ───────────────────────────────────────────────────────────────────────────
# M20-3 — the tuple is proven coherent before anything runs
# ───────────────────────────────────────────────────────────────────────────

TENANCY = APP / "utils" / "tenancy.py"


def test_the_orchestrator_validates_its_tuple_at_entry():
    body = _body(_read(RAG_ROUTES), "process_document_with_discovery")
    assert "assert_job_tuple(" in body, (
        "the orchestrator trusts job_id / document_id / workspace_id as a coherent "
        "tuple again (#35 M20-3) — a mismatch here misroutes an entire ingestion, not "
        "one row"
    )


def test_the_tuple_check_runs_before_the_credit_preflight_and_the_file():
    """First, so a run that must not happen costs nothing — not a credit lookup, not a
    stat, and certainly not a page render."""
    body = _body(_read(RAG_ROUTES), "process_document_with_discovery")
    assert body.index("assert_job_tuple(") < body.index("CREDIT PREFLIGHT"), (
        "the tenancy check now runs after the credit preflight"
    )
    assert body.index("assert_job_tuple(") < body.index("File not found at"), (
        "the tenancy check now runs after the file validations"
    )


def test_a_tenancy_failure_marks_the_job_failed():
    """Raising bare here would strand the job in `processing` with no terminal event —
    which is M20-1, in this same function."""
    body = _body(_read(RAG_ROUTES), "process_document_with_discovery")
    at = body.index("assert_job_tuple(")
    window = body[at:at + 700]
    assert "_fail_job_terminally(" in window, (
        "a failed tenancy preflight no longer marks the job failed (#35 M20-1 + M20-3)"
    )


def test_the_tuple_check_refuses_a_job_that_does_not_exist():
    src = _read(TENANCY)
    body = _strip_comments(
        ast.get_source_segment(src, _node(src, "assert_job_tuple")) or ""
    )
    assert "does not exist" in body, (
        "a job id that resolves to nothing is accepted again — every stage below stamps "
        "progress onto that id"
    )
    assert "assert_document_in_workspace(" in body, (
        "the document/workspace pair is no longer verified, which is the half that "
        "actually binds the tenant"
    )


def test_a_null_on_the_job_row_is_logged_rather_than_read_as_agreement():
    """`background_jobs.document_id` and `.workspace_id` are nullable. Treating an absent
    value as a match would make the check weakest exactly where the data is thinnest,
    and say nothing about it."""
    src = _read(TENANCY)
    body = _strip_comments(
        ast.get_source_segment(src, _node(src, "assert_job_tuple")) or ""
    )
    assert "logger.error" in body, (
        "an absent document_id/workspace_id on the job row is now silent"
    )
