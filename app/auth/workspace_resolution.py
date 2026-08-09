"""Resolve which workspace a request may operate on (BOLA — invariant 1, #250).

The problem this solves
-----------------------
A request-supplied `workspace_id` — a body field or a query param — is caller
controlled, and routes were using it directly as their tenancy filter. The
middleware already solves exactly this for the `X-Workspace-Id` header: it calls
`_validate_workspace_access` and honors the header only for an ACTIVE member. A body
field goes around that.

Two things make this non-obvious, and both are why the request value cannot simply be
replaced by the token's workspace:

1. The token's workspace is a DEFAULT, not an assertion of intent.
   `_validate_supabase_jwt` falls back to `material_kai_workspace_id` when a user's
   token carries no workspace metadata — and with no custom access-token hook
   configured, that is every user. Pinning routes to the token value would scope real
   tenants to the operator's workspace.

2. On the prefixes in `JWTAuthMiddleware.exclude_paths` — `/api/rag`, `/api/internal`,
   `/api/interior` — the middleware does not run at all, so by the time a route reads
   its body NOTHING has membership-checked anything. That is where the live exposure
   is: `/api/rag/search` accepts any valid Supabase user token and then filters by
   whatever `workspace_id` the body asked for.

The rule
--------
Service key -> the requested value is authoritative. Edge functions authenticate as
the platform identity, having already derived the workspace from the caller's verified
Supabase JWT on their side. This process cannot re-derive it, and the service identity
is not tenant-scoped, so refusing the request value would break every edge caller.

End user -> the requested value is honored only if they are an active member of it.
Otherwise 404 rather than 403: a 403 confirms the workspace exists, which turns the
endpoint into an id-enumeration oracle.

Kept free of service imports on purpose — `app.dependencies` pulls Supabase, the RAG
service and the settings bootstrap, none of which are installed in CI. Membership is
injected as `is_member`, so this module is unit-testable against a stub and the tests
run in a second with nothing but fastapi + pytest.
"""

from typing import Any, Awaitable, Callable, Dict, Optional

from fastapi import HTTPException, status

#: Subject minted by `JWTAuthMiddleware._validate_simple_api_key` for the `mk_` key.
SERVICE_SUBJECT = "material-kai-platform"

#: `(user_id, workspace_id) -> is an active member`
MembershipCheck = Callable[[str, str], Awaitable[bool]]


def is_service_caller(claims: Dict[str, Any]) -> bool:
    """True when the caller is the platform service key rather than an end user.

    `service` is set only by `_validate_simple_api_key`. A Supabase user token is
    signed by Supabase and cannot carry a top-level `service` claim, so a browser
    cannot forge its way into the trusted branch.
    """
    if not claims:
        return False
    return claims.get("service") == "mivaa" or claims.get("sub") == SERVICE_SUBJECT


async def resolve_workspace_id(
    claims: Dict[str, Any],
    requested_workspace_id: Optional[str],
    *,
    is_member: MembershipCheck,
    validated_workspace_id: Optional[str] = None,
) -> Optional[str]:
    """Return the workspace id this request may filter on, or raise 404.

    Args:
        claims: Validated token claims.
        requested_workspace_id: The caller-supplied value (body field / query param).
        is_member: Async `(user_id, workspace_id) -> bool` active-membership check.
        validated_workspace_id: A workspace the middleware already validated for this
            request (`request.state.workspace_id`), which has been through
            `_validate_workspace_access` including any `X-Workspace-Id` override.
            Absent on excluded prefixes, where the middleware never ran.

    Returns:
        The workspace to scope to, or None when none was supplied and none can be
        derived — callers that treat workspace as optional keep their behaviour.
    """
    requested = str(requested_workspace_id).strip() if requested_workspace_id else None
    if requested in ("", "None", "null"):
        requested = None

    if is_service_caller(claims):
        # Trusted channel: the edge function already bound this to a verified user.
        return requested or claims.get("workspace_id")

    token_workspace = validated_workspace_id or (claims or {}).get("workspace_id")

    # Nothing asked for, or asked for exactly what has already been established.
    if requested is None or requested == str(token_workspace):
        return token_workspace

    user_id = (claims or {}).get("sub")
    if user_id and await is_member(str(user_id), requested):
        return requested

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
