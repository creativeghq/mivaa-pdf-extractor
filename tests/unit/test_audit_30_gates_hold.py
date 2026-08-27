"""Guards for the mivaa#30 audit fixes (M16-1 … M16-4).

One file per audit, matching `test_audit_18_gates_hold.py`.

Static, not runtime: CI installs pytest alone (`deploy.yml`) and these unit tests
import nothing from `app`, so each case parses source instead. That constrains what
can be checked — a guard here proves the SHAPE is gone, not that the replacement
behaves.

Every case below was watched to FAIL against the pre-fix source before being
committed — with one deliberate exception, named so it is not mistaken for coverage:
`test_the_rpc_verdict_is_still_read_correctly` passes both ways. It is a
stays-as-it-is guard over the audit #217 H3 fix, which was already correct; its job is
to fail on a future tidy-up, not on the pre-fix source.

NOT covered here, deliberately:
  * M16-5 (cost rollups silently no-op) and M16-6 (a successful LLM call logged with
    zero tokens) are untouched by this batch, so there is nothing to hold. Filing a
    green test over them would read as coverage that does not exist.
  * M16-7 (`FALLBACK_CATEGORY` / `SECTION_ORDER` hardcoded in `field_registry`) is
    latent until ingestion runs and is left with the ingestion work.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"

FIRECRAWL = APP / "services" / "integrations" / "firecrawl_client.py"

#: The three sibling debit helpers. They are separate files with identical contracts,
#: which is exactly how one of them drifts — so every case here is parametrized over
#: all three rather than pinned to the one the audit happened to read.
DEBIT_HELPERS = [
    APP / "services" / "integrations" / "mention_cost_logger.py",
    APP / "services" / "integrations" / "price_cost_logger.py",
    APP / "services" / "integrations" / "job_cost_logger.py",
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
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _source_of(src: str, name: str) -> str:
    return ast.get_source_segment(src, _node(src, name)) or ""


# ───────────────────────────────────────────────────────────────────────────
# M16-1 — a debit helper that reported success without charging
# ───────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path", DEBIT_HELPERS, ids=lambda p: p.name)
def test_a_non_positive_debit_is_refused_not_reported_as_success(path: Path):
    """`if amount <= 0: return True` — and its sibling spelling `return amount <= 0`
    — is the shape audit #217 H3 found ten lines below in the same file and fixed:
    a billing helper reporting success without charging.

    No caller passes a non-positive amount today, so this closes the door rather than
    repairing a leak. A route whose work is genuinely free must not call the debit
    path at all."""
    body = _strip_comments(_source_of(_read(path), "debit_credits"))

    assert "return amount <= 0" not in body, (
        f"{path.name}: `return amount <= 0` is back — a zero amount reports a "
        "successful debit having charged nothing"
    )
    assert not re.search(r"if\s+amount\s*<=\s*0\s*:\s*\n\s*return True", body), (
        f"{path.name}: a non-positive amount returns True again (#30 M16-1)"
    )
    # The refusal has to be visible, or it becomes the silent-zero shape wearing the
    # opposite hat: nothing charged, nothing said.
    assert "logger.error" in body, (
        f"{path.name}: the non-positive-amount refusal is silent — a debit that does "
        "not happen must leave a trace someone can find"
    )


@pytest.mark.parametrize("path", DEBIT_HELPERS, ids=lambda p: p.name)
def test_the_rpc_verdict_is_still_read_correctly(path: Path):
    """Recorded so the #217 H3 fix cannot be undone by a later tidy-up: the RPC
    returns `[{success: bool}]`, and an insufficient balance is a NON-EMPTY truthy
    row. `bool(data)` would read that as a successful debit."""
    body = _strip_comments(_source_of(_read(path), "debit_credits"))
    assert 'row.get("success")' in body or "row.get('success')" in body, (
        f"{path.name}: the debit no longer reads the RPC's success flag"
    )


# ───────────────────────────────────────────────────────────────────────────
# M16-2 / M16-3 — the Firecrawl client validated nothing
# ───────────────────────────────────────────────────────────────────────────

def test_every_scrape_validates_its_target_first():
    """`url: str` went straight into `body = {"url": url, ...}`: no scheme check, no
    private/loopback/link-local rejection, no DNS resolution.

    Fetching THROUGH a third party relocates the SSRF primitive rather than removing
    it — Firecrawl performs the server-side fetch on MIVAA's behalf."""
    src = _strip_comments(_read(FIRECRAWL))
    assert "assert_safe_url" in src, (
        "the Firecrawl client no longer validates its target (invariant 7, #30 M16-2)"
    )

    body = _strip_comments(_source_of(_read(FIRECRAWL), "scrape"))
    guard_at = body.find("assert_safe_url(")
    build_at = body.find("self._build_request(")
    assert guard_at != -1, "scrape() no longer calls assert_safe_url"
    assert build_at == -1 or guard_at < build_at, (
        "the SSRF check now runs AFTER the request body is built — validate before "
        "the URL can reach the provider, not after"
    )


def test_the_guard_is_per_fetch_not_only_per_write():
    """Monitoring re-fetches stored URLs on a SCHEDULE, so one write buys a recurring
    fetch and DNS can be re-pointed between the two. Firecrawl also follows redirects
    on the target page with no way for us to disable it. Validation immediately before
    each fetch is the half we control."""
    src = _read(FIRECRAWL)
    assert "assert_safe_url" in _strip_comments(_source_of(src, "scrape")), (
        "the validation moved out of the per-scrape path"
    )


def test_a_blocked_target_is_refused_without_calling_the_provider():
    body = _strip_comments(_source_of(_read(FIRECRAWL), "scrape"))
    assert "SSRFError" in body, "the blocked-target branch is gone"
    blocked_at = body.index("SSRFError")
    retry_at = body.index("_call_with_retry")
    assert blocked_at < retry_at, (
        "a blocked target now reaches _call_with_retry — the refusal must precede the "
        "provider call, not follow it"
    )


def test_the_response_is_size_bounded():
    """The request body sets no maximum content, markdown or extraction size, and the
    markdown flows on into logs, prompts and caller memory."""
    src = _strip_comments(_read(FIRECRAWL))
    assert "MAX_MARKDOWN_CHARS" in src and "MAX_EXTRACT_BYTES" in src, (
        "the Firecrawl response size caps are gone (#30 M16-3)"
    )
    body = _strip_comments(_source_of(_read(FIRECRAWL), "scrape"))
    assert "_cap_markdown(" in body, "the markdown returned by a scrape is unbounded again"
    assert "_extract_within_bounds(" in body, "the structured extract is unbounded again"


def test_the_structured_extract_is_rejected_rather_than_truncated():
    """Half a JSON object is not a smaller JSON object. Markdown truncates; the
    extract does not."""
    src = _strip_comments(_read(FIRECRAWL))
    assert "def _extract_within_bounds" in src
    body = _source_of(_read(FIRECRAWL), "_extract_within_bounds")
    assert "return len(" in _strip_comments(body), (
        "_extract_within_bounds no longer measures the encoded size"
    )


# ───────────────────────────────────────────────────────────────────────────
# M16-4 — a failed call could leave no ledger record at all
# ───────────────────────────────────────────────────────────────────────────

def test_a_failure_to_record_a_call_is_itself_recorded_loudly():
    """`_log_call` swallowed every logging error at WARNING with no detail, so a
    Firecrawl outage could produce no durable trace — making a failed monitoring run
    indistinguishable from one that never ran.

    WARNING and above is never dropped by the DB log sink, so ERROR with the call's
    identity in it is the difference between a queryable record and a shrug."""
    body = _strip_comments(_source_of(_read(FIRECRAWL), "_log_call"))
    assert "logger.error" in body, (
        "the failure to write ai_usage_logs is back to a bare warning"
    )
    assert "url" in body.split("logger.error", 1)[1][:400], (
        "the log line no longer names the call it failed to record, so it cannot be "
        "reconciled against anything"
    )
