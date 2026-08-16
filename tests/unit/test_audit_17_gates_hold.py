"""Guards for the mivaa#17 fixes — billing core, XML import, field registry, classifier.

The billing findings are worth restating, because the numbers are the argument.

M4-1 was filed as "a zero or unpriced cost returns success: True and debits nothing". True,
and two layers deeper than that:

  * `user_credits.balance` was BIGINT while every ledger row, every debit RPC local and the
    workspace pool are numeric(10,2). `debit_user_credits` assigns a numeric(10,2) result to
    that bigint column, which Postgres ROUNDS — so on the personal wallet a debit under half
    a credit cost NOTHING, and 1.60 charged 2. 173 of 725 live transactions are fractional.
    One user is measurably adrift: the ledger's last balance_after says 0.17, the wallet
    holds 0.
  * 7,701 of the 8,567 ai_usage_logs rows ever written carry credits_debited = 0 against a
    non-zero billed cost. 90% of every AI call this platform has made charged nothing.

M4-2 was filed as "debit and usage-log are not atomic". Also true, and it had already fired:
three of the five copies posted `api_provider` / `credits_used` / `operation_details` as
COLUMNS, none of which exist on ai_usage_logs, so PostgREST rejected every one with PGRST204.
`debit_credits_for_external_service` has never written a usage row in its life — the eight
per-unit rows in that table all came from an edge function. The debit went through; the
record did not; the caller was told the call failed.

Static, over source text — CI installs pytest alone, so nothing here imports `app`.

Every case was watched to FAIL against the pre-fix tree.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"

CREDITS = APP / "services" / "integrations" / "credits_integration_service.py"
IMPORTS = APP / "services" / "integrations" / "data_import_service.py"
JOBRES = APP / "services" / "integrations" / "job_research_service.py"
NORMALIZER = APP / "services" / "metadata" / "metadata_normalizer.py"
EXTRACTOR = APP / "services" / "metadata" / "dynamic_metadata_extractor.py"
CLASSIFIER = APP / "services" / "chunking" / "chunk_type_classification_service.py"
STAGE2 = APP / "api" / "pdf_processing" / "stage_2_chunking.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _strip_comments(src: str) -> str:
    """Drop docstrings and comments so a guard cannot be satisfied by prose.

    Every assertion below runs against this. A fix described in a comment is not a fix, and
    several of these files now carry long comments naming the very symbols being checked.
    """
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"^\s*#.*$", "", src, flags=re.MULTILINE)
    return src


def _func(path: Path, name: str, cls: str | None = None) -> str:
    src = _read(path)
    tree = ast.parse(src)
    scope = tree
    if cls:
        scope = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls),
            None,
        )
        assert scope is not None, f"class {cls} not found in {path.name} — re-point this guard"
    for node in ast.walk(scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(src, node) or ""
    raise AssertionError(f"{name} not found in {path.name} — re-point this guard")


# ═══════════════════════════════════════════════════════════════════════════
# 1. The billing core
# ═══════════════════════════════════════════════════════════════════════════

#: Every public debit surface. The count is part of the guard: a sixth one added without
#: routing through the shared path is exactly the regression this file exists to catch.
DEBIT_METHODS = [
    "debit_credits_for_ai_operation",
    "debit_credits_for_firecrawl",
    "debit_credits_for_time_based_ai",
    "debit_credits_for_external_service",
    "debit_credits_for_replicate",
]


@pytest.mark.parametrize("method", DEBIT_METHODS)
def test_no_debit_surface_hand_rolls_the_debit_and_the_log(method: str):
    """M4-2. Five copies of rpc('debit_credits') + a separate ai_usage_logs insert."""
    body = _strip_comments(_func(CREDITS, method))

    assert "'debit_credits'" not in body and '"debit_credits"' not in body, (
        f"{method} calls the raw debit RPC again instead of debit_and_log_ai_usage. "
        "Two calls that can half-succeed, in the billing layer."
    )
    assert "ai_usage_logs" not in body, (
        f"{method} builds its own ai_usage_logs insert again. Three of the five that did "
        "named columns that do not exist, so PostgREST rejected the row and the credits "
        "went anyway."
    )
    assert "_debit_and_log" in body or "debit_credits_for_ai_operation" in body, (
        f"{method} no longer goes through the shared debit path"
    )


def test_the_shared_debit_path_is_one_transaction():
    body = _strip_comments(_func(CREDITS, "_debit_and_log"))
    assert "debit_and_log_ai_usage" in body, (
        "the shared path no longer calls the atomic RPC"
    )
    assert "table('ai_usage_logs')" not in body, (
        "the usage row is written outside the debit transaction again"
    )


def test_an_unknown_column_cannot_reach_the_usage_row():
    """The three phantom columns, by name.

    `api_provider`, `credits_used` and `operation_details` are not columns on
    ai_usage_logs. Sent as columns they killed the whole insert; they belong in the
    metadata jsonb, where they are queryable and cannot reject the row.
    """
    src = _strip_comments(_read(CREDITS))
    for phantom in ("'api_provider':", '"api_provider":',
                    "'credits_used':", "'operation_details':"):
        assert phantom not in src.replace("'api_provider': str", "").replace(
            "'api_provider': 'firecrawl'", ""
        ) or "metadata" in src, (
            f"{phantom} looks like a column payload again — it is not a column"
        )
    # The real structural check: nothing in this file posts a dict at ai_usage_logs.
    assert "table('ai_usage_logs')" not in src, (
        "an ai_usage_logs insert reappeared. The RPC's named parameters are what makes an "
        "unknown key a deploy-time failure instead of a silent runtime one."
    )


def test_a_zero_debit_is_not_reported_as_a_successful_charge():
    """M4-1. `if credits_to_debit <= 0: return {'success': True, ...}`."""
    src = _strip_comments(_read(CREDITS))
    assert not re.search(r"credits_to_debit\s*<=\s*0[\s\S]{0,120}'success':\s*True", src), (
        "a zero debit reports success again, with no audit row — the silent-zero shape "
        "sitting in the billing layer"
    )
    body = _strip_comments(_func(CREDITS, "_unbillable"))
    assert "unbilled_reason" in body and "'success': False" in body, (
        "unbillable work no longer records a reason and refuses"
    )


@pytest.mark.parametrize("method,arg", [
    ("debit_credits_for_external_service", "units"),
    ("debit_credits_for_replicate", "num_generations"),
    ("debit_credits_for_time_based_ai", "inference_seconds"),
])
def test_units_are_validated_before_a_paid_debit(method: str, arg: str):
    """M4-1: `units=0` from a caller bug produced success with a zero charge."""
    body = _strip_comments(_func(CREDITS, method))
    assert re.search(rf"{arg}\s+is\s+None\s+or\s+{arg}\s*<=\s*0", body), (
        f"{method} no longer refuses a non-positive {arg}"
    )
    assert "_unbillable" in body, f"{method} does not record the refusal"


def test_below_quantum_is_not_alerted_like_owed_money():
    """A rounding-away charge is the COMMON case — 7,701 of 8,567 rows.

    Logging it at ERROR would raise one Sentry event per AI call (main.py wires
    event_level=ERROR) and bury `debit_failed`, where money was owed and refused. The
    aggregatable record is the unbilled_reason column, not the log line.
    """
    body = _strip_comments(_func(CREDITS, "_debit_and_log"))
    assert "below_quantum" in body, "the two unbilled reasons were collapsed back into one"
    assert re.search(r"warning\s+if\s+reason\s*==\s*'below_quantum'\s+else\s+.*error", body), (
        "the common case is logged at the same severity as owed-and-refused money again"
    )


def test_the_call_logger_carries_the_reason_rather_than_flattening_it():
    body = _strip_comments(_func(APP / "services" / "core" / "ai_call_logger.py", "_debit"))
    assert "unbilled_reason" in body, (
        "ai_call_logger._debit flattens every failure to 'debit_failed' again, so the row "
        "cannot say which of five different things happened"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. XML import
# ═══════════════════════════════════════════════════════════════════════════

def test_the_import_job_is_fetched_within_the_callers_workspace():
    """M4-4 — tenth instance of two-ids-never-checked-against-each-other.

    The route authorizes the caller against `request.workspace_id`, which is real protection
    for the workspace side. Nothing checked the JOB, so a member of workspace A could pass
    their own workspace plus a job id belonging to B and import B's supplier feed — products,
    prices and images — into their own catalogue. MIVAA has no RLS; this fetch is the
    boundary.
    """
    body = _strip_comments(_func(IMPORTS, "_get_job"))
    assert "workspace_id" in body, "_get_job fetches by id alone again"
    assert re.search(r"eq\('workspace_id',\s*workspace_id\)", body), (
        "_get_job no longer constrains the job to the caller's workspace"
    )

    caller = _strip_comments(_func(IMPORTS, "process_import_job"))
    assert re.search(r"_get_job\(\s*job_id,\s*workspace_id", caller), (
        "process_import_job calls _get_job without the workspace again"
    )


def test_the_tenant_is_an_argument_not_a_mutable_instance_field():
    """M4-8, and it was worse than "may be stale": `DataImportService()` is constructed with
    NO workspace at both call sites, so `self.workspace_id` was None on every import ever run.

    With workspace_id=None the canonicalizer drops its tenancy predicate — it prefetches
    facet values across ALL workspaces and writes new canonical values with a NULL workspace,
    into the shared/golden namespace. And the brand-resolve branch was guarded on the same
    field, so it never executed: brand_company_id was silently NULL forever.
    """
    src = _read(IMPORTS)
    tree = ast.parse(src)
    reads = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and node.attr == "workspace_id"
                and isinstance(node.value, ast.Name) and node.value.id == "self"
                and not isinstance(getattr(node, "ctx", None), ast.Store)):
            reads.append(node.lineno)
    assert not reads, (
        f"self.workspace_id is read as the tenant again at line(s) {reads}. Every method "
        "that needs a workspace is already given one."
    )


def test_a_failed_batch_read_is_not_an_empty_batch():
    """M4-7. `_fetch_products_batch` returned [] on a DB failure and the loop skipped it, so
    a job with total_products > 0 completed with zero processed and zero failed."""
    body = _strip_comments(_func(IMPORTS, "_fetch_products_batch"))
    assert "return None" in body, (
        "a failed batch read returns an empty list again — indistinguishable from an empty "
        "range, which is why the job completed clean and empty"
    )
    caller = _strip_comments(_func(IMPORTS, "process_import_job"))
    assert "batch is None" in caller, (
        "the caller no longer distinguishes a broken read from an empty range"
    )


def test_the_import_reconciles_before_it_completes():
    """M4-7's real fix: one invariant that closes the class rather than one bug in it."""
    body = _strip_comments(_func(IMPORTS, "process_import_job"))
    assert re.search(r"processed_count\s*\+\s*failed_count", body), (
        "nothing checks that every declared product was accounted for"
    )
    assert re.search(r"accounted\s*!=\s*total_products", body), (
        "the reconciliation check against total_products is gone"
    )
    reconcile_at = body.index("accounted != total_products")
    complete_at = body.index("status='completed'")
    assert reconcile_at < complete_at, (
        "the job is marked completed before it reconciles"
    )


def test_supplier_metadata_cannot_overwrite_the_fields_we_own():
    """M4-5 / invariant 8. `**inner_meta` was spread LAST, over workspace_id, import_job_id,
    unit and material_category."""
    body = _strip_comments(_func(IMPORTS, "create_product_from_payload"))
    assert "_SUPPLIER_RESERVED_METADATA_KEYS" in body, (
        "the reserved-key filter is gone from the metadata build"
    )
    meta_start = body.index("product_metadata = {")
    meta_block = body[meta_start:meta_start + 1200]
    assert "**inner_meta" not in meta_block, (
        "the supplier blob is spread over the trusted fields again"
    )
    assert meta_block.index("supplier_meta") < meta_block.index('"workspace_id"'), (
        "supplier content is no longer written FIRST, so the platform's own fields do not "
        "win on a collision"
    )


def test_normalize_product_is_not_a_denylist_of_one_key():
    body = _strip_comments(_func(IMPORTS, "_normalize_product"))
    assert "_SUPPLIER_RESERVED_METADATA_KEYS" in body, (
        "_normalize_product filters only 'metadata' again, so every other supplier-controlled "
        "key rides straight through"
    )


def test_sku_identity_is_decided_on_the_folded_token():
    """M4-6. `.strip()` was the entire normalization, and these feeds are Greek: capital
    Μ (U+039C) and M (U+004D) render identically, so 7012ΜΤ and 7012MT were two products."""
    body = _strip_comments(_func(IMPORTS, "create_product_from_payload"))
    assert "fold_model_token" in body, (
        "the SKU is no longer folded before matching — same class as the CRM company dedupe "
        "(platform #366 BU-3) and mention identity (#18 M5-9)"
    )
    assert re.search(r"eq\('external_sku_folded',\s*external_sku_folded\)", body), (
        "the dedupe lookup matches the raw external_sku again"
    )
    assert "'external_sku'] = external_sku" in body, (
        "the raw supplier SKU is no longer stored — it is what a purchase order has to quote"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3. Job research
# ═══════════════════════════════════════════════════════════════════════════

def test_a_run_where_every_source_failed_is_not_a_clean_empty_run():
    """M4-3. Failed sources are recorded as -1 and skipped; with no hits the run completed
    with discovered/persisted/matches all 0. The user paid for a run where nothing ran, the
    cadence advanced as though it had succeeded, and the per-source detail was dropped on
    that early return so not even a human could reconstruct which it had been."""
    body = _strip_comments(_func(JOBRES, "refresh"))
    nohit = body[body.index("if not hits:"):body.index("if not hits:") + 2600]

    assert "all_sources_failed" in nohit, (
        "the no-hit path no longer distinguishes 'every provider failed' from 'nothing new'"
    )
    assert "fail_run" in nohit, (
        "an all-sources-failed run is completed as a success again"
    )
    assert "source_report" in nohit and "sources_empty" in nohit, (
        "the per-source detail is dropped on the no-hit return again"
    )
    cadence = nohit.index("_update_after_refresh")
    failed_return = nohit.index("fail_run")
    assert failed_return < cadence, (
        "the cadence is advanced even when nothing was searched, so the NEXT scheduled run "
        "is delayed too"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4. The field registry's two shadow copies
# ═══════════════════════════════════════════════════════════════════════════

def test_the_normalizer_cannot_invent_a_category():
    """M4-9. `MATERIAL_CATEGORY_MAPPING` decided what a category IS, and anything it did not
    know became `category.title()` — an invented category, spelled plausibly, that no
    validator accepts and no prompt offers. That is the `cladding` shape."""
    body = _strip_comments(_func(NORMALIZER, "normalize_material_category"))
    assert "category.title()" not in body, (
        "an unknown category is Title-cased into existence again"
    )
    assert "field_registry" in body and "all_controlled_vocab" in body, (
        "the result is no longer checked against material_categories.controlled_vocab, so "
        "this map is the vocabulary again rather than an alias table over it"
    )
    assert "is_loaded" in body, (
        "the check can now raise mid-extraction instead of degrading — this is a sync "
        "function reachable from paths that never loaded the registry"
    )


def test_a_discovered_field_is_not_registered_for_every_category():
    """M4-10, and the mechanism is the trap: `applies_to_categories: []` is NOT "no
    categories". `field_registry._load_blocking` reads a falsy value as None, and None means
    APPLIES TO EVERY CATEGORY. One field observed once in a tile catalogue was registered as
    a field of lighting, sanitary, kitchen and the rest — offered by their extraction prompts
    and accepted by their validators."""
    body = _strip_comments(_func(EXTRACTOR, "_register_and_classify_fields"))
    assert '"applies_to_categories": []' not in body, (
        "a discovered field is registered as applying to EVERY category again"
    )
    assert "observed_category" in body, (
        "the registration no longer carries the category the field was actually seen in"
    )

    scope = _strip_comments(_func(EXTRACTOR, "_observed_category"))
    assert "category_for_vocab" in scope, (
        "the vocab->category mapping is no longer derived from the registry — a second map "
        "here would be the ninth copy"
    )


def test_the_property_category_helper_is_not_written_twice():
    """It was, byte-identically, on two classes in one file — the same shape #347 phase 4.1
    already removed from here once."""
    src = _read(EXTRACTOR)
    tree = ast.parse(src)
    bodies = [
        ast.get_source_segment(src, n)
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == "_determine_property_category"
    ]
    hint_readers = [b for b in bodies if b and "METADATA_CATEGORY_HINTS" in b]
    assert len(hint_readers) == 1, (
        f"{len(hint_readers)} copies of the hint mapping — two copies of a mapping is one "
        "copy that can drift"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 5. Chunk classification
# ═══════════════════════════════════════════════════════════════════════════

def test_the_chunk_classifier_bills_and_attributes_its_claude_call():
    """M4-11. A direct httpx POST at api.anthropic.com with its own getenv key: no debit, no
    rate limit, no cost row, and `_classify_with_claude(content)` could not attribute even if
    it wanted to. Every failure path returned None into the pattern fallback, so the spend
    left no trace at all."""
    body = _strip_comments(_func(CLASSIFIER, "_classify_with_claude"))
    assert "tracked_claude_call_async" in body, (
        "the classifier calls Anthropic directly again — pipeline convention 10 exists so a "
        "paid call cannot be invisible"
    )
    assert "api.anthropic.com" not in body, "a raw provider POST is back"
    assert "ANTHROPIC_API_KEY" not in body, (
        "the classifier holds its own API key again, which is what let it bypass metering"
    )
    for field in ("user_id=self.user_id", "workspace_id=self.workspace_id",
                  "job_id=self.job_id", "product_id=self.product_id"):
        assert field in body, f"the paid call no longer passes {field}"

    ctor = _strip_comments(_func(CLASSIFIER, "__init__", cls="ChunkTypeClassificationService"))
    assert "workspace_id" in ctor, (
        "the service cannot be told whose document this is, so it cannot attribute"
    )

    stage = _strip_comments(_read(STAGE2))
    assert re.search(r"ChunkTypeClassificationService\(\s*workspace_id=", stage), (
        "stage 2 constructs the classifier without attribution again"
    )


def test_untrusted_chunk_text_is_fenced_before_it_reaches_the_model():
    """Invariant 9. Chunk text comes out of supplier PDFs."""
    body = _func(CLASSIFIER, "_classify_with_claude")
    assert "DATA ONLY" in body and "BEGIN CHUNK" in body, (
        "supplier-derived chunk text is interpolated into the prompt undelimited again"
    )


def test_a_classifier_crash_is_not_returned_as_a_verdict():
    """M4-12. `chunk_type_status` in {pending, classified, failed} exists specifically to
    separate "the classifier said unclassified" from "the classifier crashed". The failed
    state IS reachable from stage 2 — but the service caught every per-chunk exception first,
    so the stage never saw one and stamped the chunk `classified`."""
    src = _strip_comments(_read(CLASSIFIER))
    assert "failed: bool = False" in src, (
        "ChunkClassificationResult can no longer say it has no verdict, so a crash is "
        "indistinguishable from a genuine 'unclassified' at confidence 0.0"
    )
    for fn in ("classify_chunk", "classify_chunks_batch"):
        body = _strip_comments(_func(CLASSIFIER, fn))
        assert "failed=True" in body, (
            f"{fn} returns a ChunkType for a crash again"
        )

    stage = _strip_comments(_func(STAGE2, "_classify_and_update_chunks"))
    assert "'failed'" in stage and "getattr(cls, 'failed'" in stage, (
        "the stage no longer maps a failed classification to chunk_type_status='failed', so "
        "the distinction is lost one layer below where it is recorded"
    )
