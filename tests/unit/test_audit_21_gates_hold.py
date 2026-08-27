"""Guards for the mivaa#21 partner-key and paid-monitoring fixes (M8-1 ... M8-5).

M8-1 is the shape worth naming, because it is the INVERSE of the one that cleared about
a third of this engagement's findings. Repeatedly an application-layer finding turned out
to be bounded by a guard in SQL. Here the guard is in SQL, it is correct and well-formed,
and it is void -- because MIVAA connects as service role, which bypasses row-level
security entirely.

Four comments asserted RLS protection. The tell was in the error messages: two of the
`/sites` routes returned 404 with the text "no permission (admin-only writes)", written
by someone who believed a check was happening one layer down. Nothing was checking. Any
authenticated user could add, alter or delete rows in a PLATFORM-WIDE operator-curated
list feeding scheduled discovery defaults for everyone, delete another user's exclusion,
or mark another user's job listing.

"RLS enforces X" is never a valid justification in this codebase. There is no connection
here for a row-level policy to apply to. That is why one case below greps for the phrase
rather than checking any particular route: the assumption spreads by being read.

M8-2 is the same distinction the credit router already makes and the cron path did not.
Failing open is defensible for a USER request, where blocking a paying customer is the
greater harm. Cron is unattended, is the highest-volume caller, and nobody is waiting on
it -- so a billing-infrastructure outage there converts every scheduled run into free
provider spend, invisibly.

WATCHED TO FAIL: the whole file was run against the pre-fix tree (seven sources restored
from HEAD). 23 of 25 cases fired. The two that pass both ways do so by design and say so
in their own docstrings: `test_the_read_route_stays_open_to_any_authenticated_user` pins
a gate that must NOT be widened by reflex, and
`test_the_cost_attribution_still_comes_from_the_subject_row` pins the thing the M8-4 fix
had to avoid breaking — reading attribution from the subject was never the defect, the
subject being unvalidated was.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"

JOB_ROUTES = APP / "api" / "job_research_routes.py"
JOB_SVC = APP / "services" / "integrations" / "job_research_service.py"
OPP = APP / "services" / "integrations" / "mention_opportunity_service.py"
QUERIES = APP / "api" / "tracked_queries_routes.py"
MENTION_TRACK = APP / "api" / "mention_tracking_routes.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


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


def _body(src: str, name: str) -> str:
    return _strip_comments(ast.get_source_segment(src, _node(src, name)) or "")


def _decorators(src: str, name: str) -> str:
    node = _node(src, name)
    return " ".join(ast.unparse(d) for d in node.decorator_list)


# -------------------------------------------------------------------------
# M8-1 -- the operator-curated list is admin-only, in code
# -------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fn",
    [
        "create_job_site",
        "create_job_sites_bulk",
        "update_job_site",
        "delete_job_site",
        "resync_job_sites_kb_doc",
    ],
)
def test_every_sites_write_route_requires_an_admin(fn: str):
    """`job_research_sites` is platform-wide and operator-curated: it feeds scheduled
    discovery defaults for every workspace. The admin-only policy on it is real and is
    bypassed by this process's service-role connection."""
    src = _read(JOB_ROUTES)
    node = _node(src, fn)
    deps = " ".join(ast.unparse(a) for a in ast.walk(node) if isinstance(a, ast.Call))
    assert "require_admin" in ast.unparse(node.args), (
        f"{fn} no longer takes require_admin (#21 M8-1) — any authenticated user can "
        "edit the platform-wide job-source list, because the RLS policy that would stop "
        "them does not apply to a service-role connection"
    )
    assert "get_current_user" not in ast.unparse(node.args), (
        f"{fn} is back on get_current_user"
    )
    del deps


def test_the_read_route_stays_open_to_any_authenticated_user():
    """Deliberately NOT admin-gated: the list is operator-curated, not secret, and the
    admin page is not the only thing that reads it. Recorded so the gate is not widened
    by reflex."""
    src = _read(JOB_ROUTES)
    assert "get_current_user" in ast.unparse(_node(src, "list_job_sites").args)


def test_no_route_or_service_claims_rls_is_the_check():
    """The assumption spreads by being read. Four comments asserted it; each one was
    describing a protection that does not exist on this connection."""
    offenders = []
    for path in sorted(APP.rglob("*.py")):
        try:
            src = path.read_text(encoding="utf-8")
        except OSError:  # pragma: no cover
            continue
        for m in re.finditer(r"^\s*#(.*?)RLS (enforces|bounded|protects)", src, re.MULTILINE | re.IGNORECASE):
            before = m.group(1)
            # A QUOTED claim is history, not an assertion. Each fix records the sentence
            # it replaced, because the sentence is the whole finding — somebody read it,
            # believed a check was happening one layer down, and wrote a route that had
            # none. Quoting it is how the next reader learns that; asserting it is the
            # bug. The double quote is the discriminator, and it is a real convention in
            # these files rather than a hole punched to make this test pass.
            if '"' in before:
                continue
            offenders.append(f"{path.relative_to(ROOT)}: {m.group(0).strip()[:90]}")
    assert not offenders, (
        "these comments claim RLS is enforcing something (#21 M8-1):\n  "
        + "\n  ".join(offenders)
        + "\n\nMIVAA connects as service role, which bypasses row-level security "
          "entirely. Make the check explicit in code, or delete the claim."
    )


def test_deleting_an_exclusion_requires_owning_its_parent():
    src = _read(JOB_SVC)
    node = _node(src, "remove_exclusion")
    args = [a.arg for a in node.args.args + node.args.kwonlyargs]
    assert "owner_user_id" in args, (
        "remove_exclusion takes an id alone again (#21 M8-1) — any authenticated user "
        "who knows an exclusion id can delete another user's row"
    )
    assert not node.args.kw_defaults or all(d is None for d in node.args.kw_defaults), (
        "owner_user_id grew a default, so the check can be skipped by omission — which "
        "is exactly how it was missing in the first place"
    )
    assert "self.get(" in _body(src, "remove_exclusion")


def test_marking_a_listing_requires_owning_its_parent():
    body = _body(_read(JOB_SVC), "mark_listing")
    assert "TenancyViolation" in body or "PermissionError" in body, (
        "mark_listing no longer refuses a foreign listing (#21 M8-1)"
    )
    assert "tracked_job_id" in body and "self.get(" in body


def test_an_ownership_mismatch_answers_404_not_403():
    """A 403 confirms the id is real and belongs to somebody else, which is the
    enumeration oracle the check exists to remove (invariant 1)."""
    src = _strip_comments(_read(JOB_ROUTES))
    assert "not your listing" not in src, (
        "a 403 'not your listing' is back — it tells a caller probing ids that the "
        "object exists"
    )
    assert 'detail="site not found or no permission (admin-only writes)"' not in src, (
        "the 404 text describing a permission check that is not there is back"
    )


# -------------------------------------------------------------------------
# M8-2 -- the unattended loop fails closed
# -------------------------------------------------------------------------

def test_a_failed_owner_lookup_stops_the_cron_batch():
    body = _body(_read(JOB_ROUTES), "cron_refresh")
    assert "owner_lookup_failed" in body, (
        "the cron owner lookup no longer fails closed (#21 M8-2) — every owner would be "
        "empty, every charge would take the no-payer branch, and the whole due batch "
        "would run as free provider spend"
    )
    at = body.index("owner_lookup_failed")
    assert "return" in body[max(0, at - 200):at + 200]


def test_a_subject_with_no_payer_is_skipped_rather_than_run_free():
    body = _body(_read(JOB_ROUTES), "cron_refresh")
    assert "skipped_no_payer" in body, (
        "a due job that resolves to nobody is refreshed for free again (#21 M8-2)"
    )


def test_the_stale_fails_open_comment_is_gone():
    """It described charge_cron's behaviour incorrectly — the helper fails CLOSED on a
    failed charge — and the comment is what made the route look already-handled."""
    src = _read(JOB_ROUTES)
    # Pinned as the actual LOG LINE, not the phrase: the fix quotes the old wording in a
    # comment to explain what it did, and a bare substring check cannot tell the
    # explanation from the thing being explained.
    assert 'owner lookup failed (metering fails open)' not in src, (
        "the cron owner lookup is logging 'metering fails open' again, which means it "
        "is carrying on past a failed lookup (#21 M8-2)"
    )
    assert "# 'job-research-refresh' (3 cr); fails open, False only when out of credits." not in src, (
        "the comment describing charge_cron as failing open is back — the helper fails "
        "CLOSED on a failed charge, and that wording is what made this route look "
        "already-handled"
    )


# -------------------------------------------------------------------------
# M8-3 -- the paid doors debit before they spend
# -------------------------------------------------------------------------

@pytest.mark.parametrize(
    "path,fn",
    [
        (JOB_ROUTES, "create_tracked_job"),
        (JOB_ROUTES, "regenerate_keywords"),
        (QUERIES, "verify_tracked_query"),
        (MENTION_TRACK, "create_tracking"),
    ],
    ids=["job.track", "job.regenerate_keywords", "prices.verify", "mentions.track"],
)
def test_each_paid_door_meters_before_the_upstream_call(path: Path, fn: str):
    """All four ran paid work for free. `/job-research/track` and `/mentions/track` both
    trigger a first discovery sweep by default; `/prices/{id}/verify` says "Cost: 1
    Firecrawl credit per URL" in its own description and debited nothing."""
    body = _body(_read(path), fn)
    assert "metered_door(" in body, (
        f"{fn} no longer debits before its upstream call (#21 M8-3, invariant 10)"
    )


def test_a_paid_door_with_no_resolvable_payer_refuses():
    """`if user_id and not debit(...)` reads as metering and behaves as a free pass.
    `api_keys.user_id` is nullable, so that is a schema-permitted state rather than a
    hypothetical."""
    src = _strip_comments(_read(QUERIES))
    assert "if user_id and not debit_credits(" not in src, (
        "the free-pass metering shape is back in tracked_queries_routes (#21 M8-3)"
    )


def test_the_regenerate_keywords_cost_is_registered():
    """A cost read out of the table with a literal fallback is how a door ends up
    charging a number nobody agreed to."""
    src = _read(APP / "services" / "integrations" / "job_cost_logger.py")
    assert '"regenerate_keywords"' in src


# -------------------------------------------------------------------------
# M8-4 -- the service can check tenancy, not just the routes
# -------------------------------------------------------------------------

def test_the_opportunity_service_takes_a_caller():
    """It took no caller identity at all, so the check could not be made there even in
    principle — while it wrote `mention_ai_overview_checks` using the workspace_id and
    user_id copied OUT of the subject row the caller had named."""
    node = _node(_read(OPP), "generate")
    args = [a.arg for a in node.args.kwonlyargs]
    assert "caller_user_id" in args, (
        "mention_opportunity_service.generate no longer accepts a caller (#21 M8-4)"
    )
    assert "caller_is_admin" in args


def test_the_service_refuses_a_persisted_run_with_no_caller():
    body = _body(_read(OPP), "generate")
    assert "TenancyViolation" in body, (
        "the service no longer refuses when it cannot tell who is asking — a default of "
        "'trust the id' is one careless call site away from the cross-tenant read"
    )


# -------------------------------------------------------------------------
# M8-5 -- a failed read is not an empty result
# -------------------------------------------------------------------------

def test_a_failed_subject_read_is_not_reported_as_not_found():
    src = _read(OPP)
    assert "class SubjectLoadFailed" in src, (
        "the distinct failure type is gone (#21 M8-5) — a transient DB error is once "
        "again reported as {'subject': 'not found'} on a paid run that is not retried"
    )
    assert "subject_lookup_failed" in _body(src, "generate")


def test_a_failed_mention_read_is_not_reported_as_zero_mentions():
    assert "mention_lookup_failed" in _body(_read(OPP), "generate"), (
        "a failed mention_history read collapses back into mention_count: 0, which is a "
        "finding rather than an error"
    )


def test_a_lost_ai_overview_observation_reaches_the_caller():
    """The observation is the only thing that can ever answer 'were we in the AI
    Overview last month'. A missing one is indistinguishable from a month we were not
    there, and it cannot be reconstructed."""
    src = _read(OPP)
    body = _body(src, "_record_ai_overview_check")
    assert "return str(e)" in body, (
        "the persist failure is swallowed again (#21 M8-5) — the docstring claimed it "
        "was visible and it was not"
    )
    assert "logger.error(" in body, "still logged at WARNING for a lost, billed observation"
    assert "ai_overview_history" in _body(src, "_serp_signals")


def test_the_cost_attribution_still_comes_from_the_subject_row():
    """Passes both ways by design. It is NOT a defect that attribution is read from the
    subject — that is where the workspace lives. It was a defect only because the subject
    was unvalidated. This pins the thing the M8-4 fix must not have broken: the
    attribution path itself."""
    body = _body(_read(OPP), "generate")
    assert "CostAttribution(" in body and 'subject.get("workspace_id")' in body
