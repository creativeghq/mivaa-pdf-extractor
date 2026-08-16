"""Guards for the six live MIVAA Sentry issues (2026-08-16).

The set divided cleanly into two kinds, and both are worth naming:

  DEFECTS — code that was wrong
    * 89 reads of `user.id` off the JWT CLAIMS DICT, across three route files that
      annotated the dependency `user: User`. FastAPI does not coerce a `Depends()`
      parameter to its annotation, so every one of those routes raised
      `AttributeError: 'dict' object has no attribute 'id'` on its first call.
    * /api/rag/health answered 500 because `services` is `Dict[str, Dict[str, Any]]`
      and one entry was the bare string "Direct Vector DB" — a health check reporting
      itself unhealthy because it could not serialise its own healthy answer.

  NOISE — correct behaviour raising alerts
    * the PostgREST retry patch logged a failed LOG-SINK flush at ERROR, which re-entered
      the sink that had just failed and (event_level=ERROR) raised a Sentry event about
      the alerting. 27 events in one day, every one a dropped log row.
    * the global HTTPException handler reported 4xx as Sentry events, against the rule
      the edge runtime already follows.

An alert that fires on the common case is not an alert; it buries the uncommon ones. Both
halves of this file exist to keep that true.

Static over source text — CI installs pytest alone, so nothing here imports `app`.

Every case was watched to FAIL against the pre-fix tree.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"

CLAIMS_DICT_ROUTES = [
    APP / "api" / "price_monitoring_routes.py",
    APP / "api" / "mention_monitoring_routes.py",
    APP / "api" / "admin.py",
]


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _strip_comments(src: str) -> str:
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"^\s*#.*$", "", src, flags=re.MULTILINE)
    return src


def _func(path: Path, name: str) -> str:
    src = _read(path)
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(src, node) or ""
    raise AssertionError(f"{name} not found in {path.name} — re-point this guard")


# ═══════════════════════════════════════════════════════════════════════════
# The defects
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("path", CLAIMS_DICT_ROUTES, ids=lambda p: p.name)
def test_no_route_reads_id_off_the_claims_dict(path: Path):
    src = _strip_comments(_read(path))
    hits = re.findall(r"\b(?:user|current_user)\.id\b", src)
    assert not hits, (
        f"{path.name} reads `.id` off the object `get_current_user` returns, which is the "
        f"raw JWT claims DICT ({len(hits)} site(s)). Every one raises AttributeError on "
        "the first call. Use current_user_id(user)."
    )


@pytest.mark.parametrize("path", CLAIMS_DICT_ROUTES, ids=lambda p: p.name)
def test_the_dependency_is_not_annotated_as_a_model_it_never_returns(path: Path):
    """The annotation is what made the bug invisible: `user: User` reads as correct, and
    FastAPI silently ignores it on a `Depends()` parameter."""
    src = _strip_comments(_read(path))
    assert not re.search(r":\s*User\s*=\s*Depends\(get_current_user\)", src), (
        f"{path.name} annotates get_current_user as `User` again. It returns "
        "Dict[str, Any]; the annotation is documentation that is wrong, and nothing "
        "validates it."
    )


def test_the_user_id_helper_accepts_both_shapes_and_refuses_neither():
    body = _strip_comments(_func(APP / "dependencies.py", "current_user_id"))
    assert '"sub"' in body or "'sub'" in body, (
        "current_user_id no longer reads the `sub` claim — that is where the JWT puts the "
        "user id"
    )
    assert "raise HTTPException" in body, (
        "current_user_id returns instead of raising when it cannot resolve an id. "
        "`str(None)` would then flow into ownership checks as the literal user id 'None'."
    )


def test_the_health_check_can_serialise_its_own_healthy_answer():
    """`services` is Dict[str, Dict[str, Any]]. Checked over the AST rather than the text,
    because the value that broke it sat one line away from a legitimately nested string."""
    src = _func(APP / "api" / "rag_routes.py", "rag_health_check")
    tree = ast.parse(src.strip())

    checked = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "HealthCheckResponse"):
            continue
        for kw in node.keywords:
            if kw.arg != "services" or not isinstance(kw.value, ast.Dict):
                continue
            checked += 1
            for key, val in zip(kw.value.keys, kw.value.values):
                name = getattr(key, "value", "?")
                assert isinstance(val, (ast.Dict, ast.Name, ast.Call)), (
                    f"services[{name!r}] is a {type(val).__name__}, not a mapping. "
                    "pydantic rejects the response, so /api/rag/health answers 500 while "
                    "its own status field says healthy — which is how a bare "
                    '"Direct Vector DB" string alerted from 2026-04-07 to 2026-08-15.'
                )
    assert checked >= 2, (
        "expected both the success and the failure HealthCheckResponse to be checked; "
        "the handler's shape changed — re-point this guard"
    )


# ═══════════════════════════════════════════════════════════════════════════
# The noise
# ═══════════════════════════════════════════════════════════════════════════


def test_a_failed_log_flush_does_not_report_itself_through_logging():
    """The sink prints to stderr precisely to avoid recursion; the retry patch underneath
    it logged at ERROR and defeated that."""
    client = _read(APP / "services" / "core" / "supabase_client.py")
    assert "def log_sink_write" in client, "the log-sink guard is gone"

    assert "_log_sink_guard" in _strip_comments(
        _func(APP / "services" / "core" / "supabase_client.py", "_install_postgrest_retry_once")
    ), "the retry patch no longer checks whether the failing request IS the log sink"

    # Scoped to the wrapper ITSELF — the enclosing installer legitimately logs about
    # library-layout problems at import time, which has nothing to do with a flush.
    wrapper = _strip_comments(
        _func(APP / "services" / "core" / "supabase_client.py", "execute_with_retry")
    )
    for level in ("error", "warning", "info"):
        assert f"logger.{level}(" not in wrapper, (
            f"the retry wrapper calls logger.{level} directly again — a failed log flush "
            "then re-enters the handler that just failed, and raises a Sentry event "
            "about the alerting"
        )
    assert "_say(" in wrapper, "the retry wrapper no longer routes its reports through _say"

    sink = _strip_comments(_read(APP / "utils" / "supabase_logging_handler.py"))
    assert "log_sink_write()" in sink, (
        "the log sink no longer marks its own write, so the retry patch cannot tell it "
        "apart from a business insert"
    )


def test_client_errors_are_never_reported_to_sentry():
    """CLAUDE.md, api-logger: 4xx are client errors, not bugs. Two runtimes, one rule."""
    body = _strip_comments(_func(APP / "main.py", "http_exception_handler"))
    assert "capture_message" not in body, (
        "the HTTPException handler reports 4xx to Sentry again — its top entry was "
        "`HTTP 400: document_id is required`, i.e. an alert that fires whenever anyone "
        "calls the API wrong"
    )
    assert re.search(r"status_code\s*>=\s*500", body), (
        "the handler no longer reports 5xx — those ARE ours and must still page"
    )
    assert not re.search(r"status_code\s*>=\s*400\b(?![\s\S]{0,40}>=\s*500)", body), (
        "a 4xx branch is back in the Sentry path"
    )


def test_the_unattributed_call_marker_stays_below_the_sentry_threshold():
    """Regression pin for the fix that quieted this once already (#16). Sentry's
    event_level is ERROR, and an unattributed AI call is the COMMON case in the ingestion
    pipeline — the measurement that matters is the unbilled_reason column, not an alert."""
    body = _strip_comments(_func(APP / "services" / "core" / "ai_call_logger.py",
                                 "_record_missing_principal"))
    assert "self.logger.warning" in body, (
        "the no-principal marker logs at ERROR again — that is one Sentry event per "
        "unattributed AI call"
    )
    assert "self.logger.error" not in body
