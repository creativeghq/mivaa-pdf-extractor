"""Guard: no route may use a caller-supplied `workspace_id` unbound (invariant 1, #250).

A body field or query param named `workspace_id` is caller-controlled. Used directly as
a tenancy filter it is a BOLA: the caller names the tenant whose data they get back.

This sweep fails the build when a route consumes one without any of the guards below.
It is deliberately generous about WHAT counts as a guard, because the ways this codebase
legitimately binds a workspace are genuinely varied — and a guard that only recognises
one idiom trains people to work around it:

  * `resolve_workspace_id(...)`      — the shared rule (app/auth/workspace_resolution.py)
  * `authorize_rag_workspace(...)`   — the /api/rag equivalent, predates it
  * `Depends(get_workspace_context)` — derives the workspace from the JWT instead
  * `require_admin` / `_admin_user_id_from_request` — platform admin, intentionally
    not tenant-scoped: an operator sweeping a workspace they do not belong to is the
    normal case, so a membership check there breaks maintenance
  * overwriting the field from verified claims (`request.workspace_id = user.get(...)`)

Both parameter defaults AND decorator `dependencies=[...]` are inspected. Missing the
decorator form is not hypothetical: it is exactly how this sweep's first draft reported
`save_images_to_db` and `recanonicalize` as unguarded when both already had a gate, which
nearly produced two "fixes" that would have 401'd a cron path and broken admin sweeps.
"""

import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "app"
METHODS = {"get", "post", "put", "patch", "delete"}
CONTEXT_TYPES = {"WorkspaceContext", "ApiKeyContext", "Request", "SupabaseClient"}

GUARD_CALLS = {
    "resolve_workspace_id",
    "authorize_rag_workspace",
    "_validate_workspace_access",
    "assert_workspace_member",
    "_admin_user_id_from_request",
    "_run_aspect_search",  # resolves for all four /search/by-<aspect> routes
}
GUARD_DEPS = {
    "get_workspace_context",
    "get_current_workspace_context",
    "get_optional_workspace_context",
    "require_admin",
    "verify_internal_access",
}

# Routes that legitimately read a caller-supplied workspace_id with none of the above.
# Every entry states why. Shrink-only: adding one requires the same justification.
ALLOWED = {
    # Self-guards inline on x-cron-secret OR the service-role bearer, and is not
    # user-reachable, so workspace_id is a backfill SCOPE from a trusted internal
    # caller. Adding Depends(get_current_user) here 401s the cron path, which sends
    # no bearer at all.
    ("app/api/rag_routes.py", "kb_docs_rechunk"),
    # Same shape: x-cron-secret gated (checked inline, 401 on mismatch), internal-only
    # (called by the seo-research edge function), not user-reachable. There is no
    # tracked-subject row to read an owner off, so the body workspace_id is the trusted
    # cost-attribution SCOPE the edge function passes for its own session — binding it to
    # a JWT is impossible here because the caller sends no bearer.
    ("app/api/mention_monitoring_routes.py", "opportunities_stateless"),
}


def _defaults(fn):
    d = {}
    pos = fn.args.args
    for a, v in zip(pos[len(pos) - len(fn.args.defaults):], fn.args.defaults):
        d[a.arg] = v
    for a, v in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
        if v is not None:
            d[a.arg] = v
    return d


def _dep_name(node):
    if isinstance(node, ast.Call) and (getattr(node.func, "id", "") or getattr(node.func, "attr", "")) == "Depends":
        if node.args:
            return getattr(node.args[0], "id", None) or getattr(node.args[0], "attr", None)
    return None


def _annotation_name(ann):
    if isinstance(ann, ast.Name):
        return ann.id
    if isinstance(ann, ast.Attribute):
        return ann.attr
    if isinstance(ann, ast.Subscript):
        return getattr(ann.value, "id", None) or getattr(ann.value, "attr", None)
    return None


def _models_with_workspace_id():
    models = set()
    for path in APP.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name not in CONTEXT_TYPES:
                for stmt in node.body:
                    if (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
                            and stmt.target.id == "workspace_id"):
                        models.add(node.name)
    return models


def test_every_route_binds_its_caller_supplied_workspace_id():
    models = _models_with_workspace_id()
    offenders = []

    for path in sorted(APP.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        rel = path.relative_to(APP.parent).as_posix()

        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorators = [d for d in fn.decorator_list]
            is_route = any(
                isinstance((d.func if isinstance(d, ast.Call) else d), ast.Attribute)
                and (d.func if isinstance(d, ast.Call) else d).attr in METHODS
                for d in decorators
            )
            if not is_route or (rel, fn.name) in ALLOWED:
                continue

            params = fn.args.args + fn.args.kwonlyargs
            defaults = _defaults(fn)

            takes_ws_param = any(
                a.arg == "workspace_id" and _dep_name(defaults.get(a.arg)) is None
                for a in params
            )
            takes_ws_model = any(_annotation_name(a.annotation) in models for a in params)
            if not (takes_ws_param or takes_ws_model):
                continue

            deps = {_dep_name(v) for v in defaults.values()} - {None}
            for d in decorators:
                if isinstance(d, ast.Call):
                    for kw in d.keywords:
                        if kw.arg == "dependencies":
                            for el in getattr(kw.value, "elts", []):
                                name = _dep_name(el)
                                if name:
                                    deps.add(name)

            calls = {
                getattr(c.func, "id", "") or getattr(c.func, "attr", "")
                for c in ast.walk(fn) if isinstance(c, ast.Call)
            }
            # `request.workspace_id = <from claims>` — overwritten, so never caller-set.
            overwrites = any(
                isinstance(s, ast.Assign)
                and any(isinstance(t, ast.Attribute) and t.attr == "workspace_id" for t in s.targets)
                for s in ast.walk(fn)
            )

            if calls & GUARD_CALLS or deps & GUARD_DEPS or overwrites:
                continue

            offenders.append(f"{rel}:{fn.lineno} {fn.name}")

    assert not offenders, (
        "Route(s) use a caller-supplied `workspace_id` without binding it to the "
        "authenticated caller. Any client can then name another tenant's workspace and "
        "receive its data. Call `resolve_workspace_id(claims, <value>)` (see "
        "app/auth/workspace_resolution.py), or derive the workspace from the JWT with "
        "Depends(get_workspace_context):\n  " + "\n  ".join(offenders)
    )
