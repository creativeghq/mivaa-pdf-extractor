"""Guards for the mivaa#18 audit fixes (M5-1 … M5-9).

One file per audit, matching `test_audit_12_gates_hold.py` and
`test_audit_16_gates_hold.py`.

Static, not runtime: CI installs pytest alone (`deploy.yml`) and these unit tests
import nothing from `app`, so each case parses source instead. That constrains what
can be checked — a guard here proves the SHAPE is gone, not that the replacement
behaves. Where a case can only assert an absence, it says so rather than implying
more. The exception is `app.utils.text_fold`, which is pure stdlib and therefore
importable; those cases assert real behaviour.

Every case below was watched to FAIL against the pre-fix source before being
committed. A guard nobody has seen fire is a guard that might be asserting nothing.

NOT covered here, deliberately:
  * M5-2's product half, M5-7 and M5-10 were already fixed by the #16 batch and by
    #250 H1 before this audit was written; their guards live with those fixes.
  * The paid-door metering (M5-3/M5-4) is guarded by `test_paid_route_metering.py`,
    which now enumerates the doors from source instead of listing three by hand.
"""

import ast
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"

LOOKUP = APP / "api" / "price_lookup_routes.py"
MONITOR = APP / "services" / "tracking" / "job_monitor_service.py"
RECOVERY = APP / "services" / "tracking" / "checkpoint_recovery_service.py"
SEARCH = APP / "api" / "search.py"
BRIDGE = APP / "services" / "integrations" / "material_kai_service.py"
MENTION_SEARCH = APP / "services" / "integrations" / "mention_search_service.py"
DFS_CLIENT = APP / "services" / "integrations" / "dataforseo_unified_client.py"
MENTION_ID = APP / "services" / "integrations" / "mention_identity_service.py"
PRODUCT_ID = APP / "services" / "integrations" / "product_identity_service.py"

#: Every module that mounts `authenticate_api_key` on its own routes.
API_KEY_IMPORTERS = [
    APP / "api" / "job_tracking_routes.py",
    APP / "api" / "mention_tracking_routes.py",
    APP / "api" / "project_tracking_routes.py",
    APP / "api" / "tracked_queries_routes.py",
]


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _strip_comments(src: str) -> str:
    """Drop comments and docstrings so prose about the old bug is not read as code."""
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"^\s*#.*$", "", src, flags=re.MULTILINE)
    return src


def _func_body(src: str, name: str) -> str:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(src, node) or ""
    raise AssertionError(f"{name} not found")


# ───────────────────────────────────────────────────────────────────────────
# M5-1 — partner API keys were scoped against a module constant
# ───────────────────────────────────────────────────────────────────────────

def test_api_key_scope_checks_the_path_actually_requested():
    """The scope check compared `ENDPOINT_PATH`, a module constant, against the key's
    allowlist — so it answered "may this key call /api/v1/prices/lookup?" no matter
    which route was being served. Four other modules mount this same dependency
    (54 routes), which inverted scoping in BOTH directions: a lookup-only key passed
    every project/job/mention route, and a key scoped correctly to /api/v1/projects/*
    was rejected on all of them."""
    body = _strip_comments(_func_body(_read(LOOKUP), "authenticate_api_key"))

    assert "request.url.path" in body, (
        "authenticate_api_key does not read the real request path — the scope check "
        "cannot be about the route being served"
    )
    assert not re.search(r"_endpoint_allowed\(\s*ENDPOINT_PATH", body), (
        "the scope check compares the module constant ENDPOINT_PATH again. That is the "
        "bug: it hardcodes one route's identity into a dependency five modules mount."
    )


@pytest.mark.parametrize("path", API_KEY_IMPORTERS, ids=lambda p: p.name)
def test_every_api_key_importer_still_goes_through_the_shared_dependency(path: Path):
    """The fix is only complete while these keep sharing one dependency. If a module
    grows its own copy, the path check has to be re-proved there — so this fails and
    makes that a decision rather than an oversight."""
    src = _read(path)
    assert "from app.api.price_lookup_routes import" in src, (
        f"{path.name} no longer imports the shared api-key dependency. If that is "
        "deliberate, its replacement needs its own request.url.path scope check and "
        "its own entry in this test."
    )


# ───────────────────────────────────────────────────────────────────────────
# M5-5 — recovery killed live jobs when it correctly lost a claim race
# ───────────────────────────────────────────────────────────────────────────

def test_restart_reports_a_typed_outcome_not_a_bare_bool():
    src = _read(RECOVERY)
    assert "class RestartOutcome" in src, "the typed restart outcome is gone"
    assert "LOST_CLAIM" in src and "NO_CHECKPOINT" in src and "ERROR" in src, (
        "RestartOutcome no longer distinguishes losing the claim from failing"
    )
    body = _func_body(src, "auto_restart_stuck_job")
    assert "-> RestartOutcome" in body.split("\n")[0] or "RestartOutcome" in body[:400], (
        "auto_restart_stuck_job stopped returning RestartOutcome. A bool collapses "
        "'someone else has this job' and 'the restart broke' into one False, which is "
        "exactly what made the monitor kill live jobs."
    )
    assert "return False" not in _strip_comments(body), (
        "auto_restart_stuck_job returns a bare False again"
    )


def test_monitor_never_fails_a_job_it_merely_lost_the_race_for():
    body = _strip_comments(_func_body(_read(MONITOR), "_recover_stuck_job"))
    assert "RestartOutcome.LOST_CLAIM" in body, (
        "the monitor no longer recognises LOST_CLAIM. Losing the claim is the "
        "compare-and-swap WORKING — another tick, or the worker itself, holds the job."
    )
    # Bound the branch at the next `elif`/`else` at the same level, not by a character
    # count — the neighbouring branch is *supposed* to call _mark_job_failed.
    lost_at = body.index("RestartOutcome.LOST_CLAIM")
    rest = body[lost_at:]
    nxt = min(
        (i for i in (rest.find("\n                    elif "), rest.find("\n                    else:"))
         if i != -1),
        default=len(rest),
    )
    branch = rest[:nxt]
    assert "_mark_job_failed" not in branch, (
        "the LOST_CLAIM branch marks the job failed. That converts recovery behaving "
        "correctly into killing a job that is alive and making progress."
    )


def test_the_monitors_failure_write_is_a_compare_and_swap():
    body = _strip_comments(_func_body(_read(MONITOR), "_mark_job_failed"))
    assert "expected_updated_at" in body, (
        "_mark_job_failed no longer accepts the observation it is CASing against"
    )
    assert '.eq("status", "processing")' in body, (
        "_mark_job_failed writes without checking the job is still processing — a job "
        "that recovered between detection and this write gets failed under its own worker"
    )


# ───────────────────────────────────────────────────────────────────────────
# M5-2 — an authenticated route that was not bound to a tenant
# ───────────────────────────────────────────────────────────────────────────

def test_similar_materials_is_bound_to_the_callers_workspace():
    """Authentication is not authorization. The JWT middleware proves the caller is
    SOMEONE; without a route-level gate any valid user could pass another tenant's
    material_id into a service-role search, and MIVAA has no RLS backstop."""
    body = _func_body(_read(SEARCH), "find_similar_materials")
    assert "get_workspace_context" in body, (
        "find_similar_materials has no workspace dependency again (invariant 1/5)"
    )
    assert "workspace_id=" in body, (
        "the workspace is resolved but never passed down, so the search is not scoped by it"
    )


# ───────────────────────────────────────────────────────────────────────────
# M5-6 — the bridge sent its bearer token to an unvalidated URL
# ───────────────────────────────────────────────────────────────────────────

def test_the_bridge_validates_its_platform_url_before_it_can_carry_a_credential():
    src = _read(BRIDGE)
    assert "_validated_platform_url" in src, "the bridge URL validator is gone"
    body = _strip_comments(_func_body(src, "_validated_platform_url"))
    assert 'parsed.scheme != "https"' in body, (
        "the bridge no longer requires https — the bearer token can cross the wire in clear"
    )
    assert "assert_safe_url" in body, (
        "the bridge no longer runs its target through the shared SSRF guard, so a "
        "misconfigured platform_url makes it an authenticated probe of internal hosts"
    )
    assert "material_kai_allowed_hosts" in body, "the exact-host allowlist check is gone"

    init = _strip_comments(_func_body(src, "__init__"))
    assert "_validated_platform_url" in init, (
        "validation no longer runs at construction. Validating per-request is too late: "
        "the first credentialed call has already sent the credential."
    )


# ───────────────────────────────────────────────────────────────────────────
# M5-8 — HTTP 200 from DataForSEO is not success
# ───────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path", [MENTION_SEARCH, DFS_CLIENT], ids=lambda p: p.name)
def test_dataforseo_callers_validate_the_envelope(path: Path):
    """A rejected task returns 200 with a non-20000 body code and an empty `tasks`
    walk, which every caller recorded as a clean zero-hit search. Found and fixed
    independently four times before it became one validator."""
    src = _strip_comments(_read(path))
    assert "dataforseo_envelope" in src, (
        f"{path.name} consumes a DataForSEO response without checking the envelope. "
        "raise_for_status()/status_code only clears the HTTP layer."
    )


def test_nobody_reimplements_the_envelope_check():
    """Four copies is what created this finding. A fifth starts with someone writing
    the magic number out again."""
    offenders = []
    for path in sorted(APP.rglob("*.py")):
        if path.name == "dataforseo_envelope.py":
            continue
        src = _strip_comments(path.read_text(encoding="utf-8"))
        # Only files that actually talk to DataForSEO, and only the bare literal as a
        # whole token — `le=20000` on an unrelated Query bound and a 200000-char
        # threshold are not this defect, and a guard that flags them gets deleted.
        if "dataforseo" not in src.lower():
            continue
        if re.search(r"(?<![\d_])20000(?![\d_])", src):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, (
        "DataForSEO's 20000 success code is written out outside the shared validator: "
        + ", ".join(offenders)
        + " — import app.services.integrations.dataforseo_envelope instead."
    )


# ───────────────────────────────────────────────────────────────────────────
# M5-9 — one Greek-aware fold, not one per subsystem
# ───────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path", [MENTION_ID, PRODUCT_ID], ids=lambda p: p.name)
def test_identity_services_do_not_carry_their_own_fold(path: Path):
    src = _strip_comments(_read(path))
    assert "from app.utils.text_fold import" in src, (
        f"{path.name} no longer uses the shared fold"
    )
    assert not re.search(r'_GREEK_TO_LATIN: Dict\[str, str\] = \{', src), (
        f"{path.name} has grown its own copy of the lookalike map again. Two copies is "
        "how these drifted to different strengths — one mapped confusables and one did "
        "not, and neither folded the final sigma."
    )


def test_the_fold_handles_the_things_each_partial_copy_missed():
    """Behavioural, not structural — text_fold is pure stdlib so it can be imported.

    Each assertion below is a case one of the three pre-existing normalizers got wrong.
    """
    # Loaded by path, not imported: `app` is not an installed package in CI, and this
    # module is pure stdlib so it needs nothing from the app's runtime.
    spec = importlib.util.spec_from_file_location("_text_fold", APP / "utils" / "text_fold.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fold_for_search, fold_identity, fold_model_token = (
        mod.fold_for_search, mod.fold_identity, mod.fold_model_token,
    )

    # The final sigma. Both Python copies missed this one.
    assert fold_for_search("ΚΩΣΤΑΣ") == fold_for_search("Κωστάς")
    assert fold_for_search("ΚΩΣΤΑΣ") == "κωστασ"

    # Accents, in both alphabets.
    assert fold_for_search("Νιπτήρα") == fold_for_search("ΝΙΠΤΗΡΑ")
    assert fold_for_search("Café") == "cafe"

    # Search must NOT collapse alphabets — a Greek query matching Latin records is a
    # different (and wrong) behaviour from folding two identical-looking codepoints.
    assert fold_for_search("ΜΤ") != fold_for_search("MT")

    # Identity must. product_identity_service.normalize_text did not.
    assert fold_identity("ΜΤ") == fold_identity("MT")
    assert fold_model_token("7012ΜΤ") == fold_model_token("7012-MT") == "7012MT"

    # Empty/None stay empty rather than raising or becoming "none".
    assert fold_for_search(None) == "" and fold_identity(None) == ""
