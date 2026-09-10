"""
Guard: a route excluded from the JWT middleware gates ITSELF.

WHY THIS EXISTS
---------------
`JWTAuthMiddleware.exclude_paths` is matched by prefix. Everything under an entry there is
outside the middleware, so its only remaining protection is whatever the route declares —
`Depends(get_workspace_context)`, `_check_secret(request)` for the x-cron-secret paths,
`_admin_user_id_from_request`, a `kai_` partner key, a Turnstile check.

CLAUDE.md invariant 5 states exactly this rule and, until now, NOTHING enforced it. The
list has grown to 39 prefixes covering 225 routes, several of which spend real money
(`/api/v1/seo-agent/*` is DataForSEO, `/api/v1/modules/*/search` is Firecrawl). The
failure mode is silent in the direction that matters: adding a route under an existing
excluded prefix inherits the exclusion automatically and announces nothing.

The exclusion list is READ FROM THE MIDDLEWARE, not restated here — a copy would drift and
this test would then be checking a list nobody uses.

Two allowlists, both deliberately small and each explaining itself:

  * `_PUBLIC_BY_DESIGN` — endpoints that are public ON PURPOSE: no tenant data, no spend.
    Mostly liveness. `/api/v1/ai-services/health` is here because the health dashboard calls
    it with NO Authorization header, and its being behind the middleware is what made that
    row read a false RED for as long as it existed; `/api/v1/public/quota` is what the public
    tools call BEFORE authenticating.
  * `_GATE_MARKERS` — the ways a route in this codebase actually authenticates. A new one
    is a one-line addition, and having to add it is the point: it makes "how does this
    authenticate" an answer someone has to write down.

Source-based: parses text, imports no app module, touches no DB. MIVAA's CI installs
pytest only, so a test that imports the app cannot run there at all.
"""

import importlib.util
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_APP = _ROOT / "app"
_MIDDLEWARE = _APP / "middleware" / "jwt_auth.py"

#: Public by design. Every entry must return no tenant data and spend nothing.
_PUBLIC_BY_DESIGN = {
    "/health/",
    "/health/detailed",
    "/health/database",
    "/health/metrics",
    "/health/circuit-breakers",
    "/health/job-monitor",
    "/api/embeddings/health",
    "/api/v1/ai-services/health",
    "/api/rag/health",
    # The public tools' own quota reader: keyed on IP for an anonymous caller, returns a
    # remaining-count and the Turnstile site key. It is what the public form calls BEFORE
    # authenticating, so gating it would break the surface it serves.
    "/api/v1/public/quota",
    # Liveness for the jobs subtree, deliberately ungated: it exists TO BE PROBED and
    # discloses two integers (active job count, history size). Every OTHER route under
    # /api/jobs carries verify_internal_access.
    "/api/jobs/health",
}

#: How a route in this codebase says "I check the caller myself".
_GATE_MARKERS = (
    "get_current_user",          # Supabase session JWT
    "get_workspace_context",     # session JWT + workspace binding
    "current_user_id",
    "require_admin",
    "_require_admin",
    "_admin_user_id_from_request",
    "verify_internal_access",    # internal service-to-service
    "_check_secret",             # x-cron-secret
    "cron_secret",
    "cron-secret",
    "_extract_bearer",           # bearer + supabase.auth.get_user
    "require_rag_resource_access",  # /api/rag/* — decorator-level dependency
    "require_trusted_service",      # /api/internal/* — service-to-service, parameter-level
    "require_deploy_token",         # /api/admin/*-for-deploy
    "api_key",
    "apikey",
    "kai_",
    "turnstile",
    "Turnstile",
    "_require_internal_kb_caller",   # /api/rag/kb-eval/* — internal caller check
    "workspace_context is None",     # the refusal itself, for routes taking the optional dep
)


def _excluded_prefixes() -> list[str]:
    """The live list, read out of the middleware's own default."""
    src = _MIDDLEWARE.read_text(encoding="utf-8")
    start = src.index("self.exclude_paths = exclude_paths or [")
    end = src.index("\n        ]", start)
    # Only the path literals — the block also carries prose in comments.
    return [s for s in re.findall(r'"([^"]+)"', src[start:end]) if s.startswith("/")]


def _load_blank_comments():
    """Load the comment blanker by path — this test imports no app module, and must not start."""
    spec = importlib.util.spec_from_file_location(
        "comment_budget_for_routes", _ROOT / "scripts" / "comment_budget.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.blank_comments


_blank_comments = _load_blank_comments()


def _handler_source(rest: str) -> str:
    """The decorated handler, whole, and nothing after it.

    A fixed character window is wrong in both directions: `upload_document` declares ~70 form
    parameters, so 4000 characters stopped short of its own refusal, and for the last route in a
    file the window ran on into unrelated helpers whose text could satisfy a marker.
    """
    own = re.search(r"\n?(?:async\s+)?def\s+\w+", rest)
    start = own.end() if own else 0
    nxt = re.search(r"\n(?:@|def\s|async\s+def\s)", rest[start:])
    return rest[: start + (nxt.start() if nxt else len(rest) - start)]


def _routes() -> list[tuple[str, str, Path, str]]:
    """(method, full path, file, handler source) for every @router decorator in app/.

    Comments and docstrings are blanked first, at the same offsets: a gate marker must be
    satisfied by CODE. `/api/jobs/health` used to pass this test because its docstring
    mentioned the `verify_internal_access` its SIBLING routes use, and it has no gate at all.
    """
    out: list[tuple[str, str, Path, str]] = []
    for path in _APP.rglob("*.py"):
        src = _blank_comments(path.read_text(encoding="utf-8", errors="replace"))
        prefix = ""
        router_has_deps = False
        m = re.search(r"APIRouter\(([^)]*)\)", src, re.S)
        if m:
            pm = re.search(r"""prefix\s*=\s*["']([^"']+)["']""", m.group(1))
            if pm:
                prefix = pm.group(1)
            router_has_deps = "dependencies" in m.group(1)
        for dm in re.finditer(
            r"""@router\.(get|post|put|patch|delete)\(\s*["']([^"']*)["']""", src
        ):
            route = (prefix + dm.group(2)) or "/"
            body = _handler_source(src[dm.end():])
            # A router-level `dependencies=[...]` gates every route it carries.
            if router_has_deps:
                body += " get_current_user"
            out.append((dm.group(1).upper(), route, path, body))
    return out


def test_the_exclusion_list_is_readable_and_non_trivial():
    """If the parse breaks, every assertion below passes over nothing."""
    prefixes = _excluded_prefixes()
    assert len(prefixes) >= 20, f"only parsed {len(prefixes)} excluded prefixes — parser broken?"
    assert "/api/v1/seo-agent" in prefixes
    assert all(p.startswith("/") for p in prefixes)


def test_no_bare_root_prefix_is_excluded():
    """CLAUDE.md invariant 5: never a prefix that swallows everything."""
    for p in _excluded_prefixes():
        assert p not in ("/", "/api", "/api/", "/api/v1", "/api/v1/"), (
            f"exclude_paths contains {p!r}, which is matched by prefix and therefore "
            "excludes far more than it names."
        )


def test_every_excluded_route_gates_itself():
    prefixes = _excluded_prefixes()
    routes = _routes()
    assert routes, "no routes parsed — the decorator scan is broken"

    covered = [r for r in routes if any(r[1].startswith(p) for p in prefixes)]
    assert len(covered) >= 50, (
        f"only {len(covered)} routes matched an excluded prefix; the scan is probably wrong"
    )

    ungated = [
        (m, route, path)
        for (m, route, path, body) in covered
        if route not in _PUBLIC_BY_DESIGN
        and not any(k in body for k in _GATE_MARKERS)
    ]

    assert not ungated, (
        "These routes sit under a JWT-middleware exclusion and declare no gate of their own, "
        "so they are reachable unauthenticated (CLAUDE.md invariant 5). Add the route's own "
        "check — or, if it is genuinely public and returns no tenant data and spends nothing, "
        "add it to _PUBLIC_BY_DESIGN with the reason:\n"
        + "\n".join(f"  {m:6} {r}   ({p.relative_to(_ROOT)})" for m, r, p in sorted(ungated, key=lambda x: x[1]))
    )


def test_public_allowlist_stays_small_and_real():
    """Every entry must still exist as a route — a stale exemption hides a real gap."""
    routes = {r[1] for r in _routes()}
    missing = sorted(p for p in _PUBLIC_BY_DESIGN if p not in routes)
    assert not missing, (
        "these are exempted as public health endpoints and no longer exist as routes; "
        f"remove them so the exemption cannot cover something else later: {missing}"
    )
    assert len(_PUBLIC_BY_DESIGN) <= 12, (
        "the public-health exemption list is growing — each entry is an unauthenticated "
        "endpoint, so it should shrink over time, not grow."
    )
