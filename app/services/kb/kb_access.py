"""One derivation each for the three things every KB search path needs.

Why this module exists
----------------------
There are TWO KB search endpoints, and both are load-bearing:

    /api/kb/search                  admin UI + `kb_search` gateway action.
                                    Whole-doc vectors (`kb_docs` / `kb_match_docs`),
                                    plus full-text and hybrid modes.
    /api/rag/search/knowledge-base  the `knowledge_base_search` agent tool + price
                                    lookup. Section-level (`kb_doc_chunks` /
                                    `kb_match_doc_chunks`), agent access model.

Different corpora, different consumers, different response shapes. They should NOT be
merged. But three things inside them must give the same answer, and were computed
separately in each:

  1. How you turn query text into a KB query vector.
  2. Who the caller actually is.
  3. What that caller may read.

(1) had already drifted. `rag_routes` passed `entity_type="query"` — with a comment
naming the bug — while `knowledge_base` passed `entity_type="search"`, which falls
through `input_type = "query" if entity_type == "query" else "document"` to
`"document"`. So one endpoint embedded its query as a query and the other embedded it
as a document, against the same document-side vectors. Nothing raises: same model, same
1024 dimensions, same column, plausible ranked results, slightly worse.

This is the platform's "one derivation, many consumers" rule (`get_order_settlements`)
applied to retrieval instead of money. The endpoints keep their real differences; only
what must agree is forced to.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: The ONLY `entity_type` that yields Voyage `input_type="query"`. Spelled out here
#: rather than at each call site because the mapping is one exact-string equality, so
#: any near-miss ("search", "kb_query", "user_query") silently produces a DOCUMENT
#: vector. That is precisely how the two endpoints ended up disagreeing.
_QUERY_ENTITY_TYPE = "query"

#: Access levels, in the order they widen.
_PUBLIC_LEVELS = ["public"]
_AGENT_LEVELS = ["agent", "public"]
_ADMIN_LEVELS = ["admin", "agent", "public"]

VALID_CALLERS = ("admin", "agent", "public")


def kb_shared_workspace_id() -> str:
    """The operator root workspace that holds the shared KB.

    A function, not a module constant: a constant is evaluated at import time, which
    shadows any DEFAULT_WORKSPACE_ID env override set afterwards, and it was one of the
    11 copies of this UUID (M3-14, #16).
    """
    from app.config import get_settings

    return get_settings().default_workspace_id


async def kb_query_vector(
    query: str,
    *,
    workspace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    job_id: Optional[str] = None,
) -> Optional[List[float]]:
    """Embed query text into the KB's 1024D text space. The only way to do this.

    `input_type="query"` (via `entity_type="query"`) is decided HERE, once. Voyage
    embeds asymmetrically — a vector built in document mode sits in a slightly
    different region from one built in query mode, and the model is trained so that
    query-mode vectors score best against document-mode vectors. Getting it wrong costs
    ranking quality and nothing else, which is exactly why it survived: no exception, no
    zero, no failed probe, just quietly worse results.

    Returns None when the embedding could not be produced. None means NO VECTOR — the
    caller must skip the vector branch, never substitute one from somewhere else.
    """
    from app.services.embeddings.real_embeddings_service import RealEmbeddingsService

    if not query or not query.strip():
        return None

    try:
        service = RealEmbeddingsService()
        result = await service.generate_all_embeddings(
            entity_id="kb_search_query",
            entity_type=_QUERY_ENTITY_TYPE,
            text_content=query,
            workspace_id=workspace_id,
            user_id=user_id,
            job_id=job_id,
        )
    except Exception as e:
        logger.warning(f"KB query embedding failed: {e}")
        return None

    if not result.get("success"):
        logger.warning(f"KB query embedding unsuccessful: {result.get('error')}")
        return None

    return result.get("embeddings", {}).get("text_1024") or None


async def resolve_kb_caller(
    supabase,
    claims: Optional[Dict[str, Any]],
    requested_caller: Optional[str],
    workspace_id: Optional[str],
) -> str:
    """Decide what kind of caller this REALLY is. Never take it from the body.

    `caller` was a request-body field feeding the access scope directly
    (`request.caller or "agent"`), and `caller="admin"` grants admin access levels plus
    private docs. `PriceLookupDrawer` sends `caller: 'admin'` from the FRONTEND through
    mivaa-gateway, and the gateway forwards the end user's own JWT for `/api/rag/*`
    paths (deliberately — so MIVAA enforces ownership). So the assertion arrived on an
    ordinary user token and was honoured unchecked: any authenticated member could send
    one string and read the workspace's admin-level and private KB. Same defect as
    MV2-12 on the sibling endpoint, one door along.

    Two trusted shapes, mirroring `authorize_rag_workspace`:

      * **Service caller** (`claims['service'] == 'mivaa'`, settable only by
        `_validate_simple_api_key` — a Supabase user token cannot carry it). This is a
        deliberate platform credential: `price-tools.ts` calls MIVAA directly with
        MIVAA_API_KEY and asserts `caller: 'admin'` on purpose. Honour it. Absent an
        explicit request, default to `agent`, NOT admin — several gateway paths forward
        the service key on behalf of an ordinary user, and defaulting to admin there
        would hand every user the admin scope through the back door.

      * **A real user JWT** — derive from `workspace_members.role`. A request may always
        narrow (asking for `public` is honoured) and may never widen: asking for `admin`
        without the role clamps to `agent`.

    Fails closed: an unreadable membership table yields `agent`, never `admin`.
    """
    requested = (requested_caller or "").strip().lower() or None
    if requested and requested not in VALID_CALLERS:
        logger.warning(f"Unknown KB caller {requested!r} — treating as 'agent'")
        requested = None

    from app.auth.workspace_resolution import is_service_caller

    if is_service_caller(claims or {}):
        return requested or "agent"

    # Narrowing is always allowed and needs no lookup.
    if requested == "public":
        return "public"

    user_id = (claims or {}).get("sub") or (claims or {}).get("user_id")
    if not user_id or not workspace_id:
        return "agent"

    try:
        resp = (
            supabase.client.table("workspace_members")
            .select("role")
            .eq("user_id", str(user_id))
            .eq("workspace_id", str(workspace_id))
            .eq("status", "active")
            .execute()
        )
        is_admin = any((r.get("role") in ("owner", "admin")) for r in (resp.data or []))
    except Exception as e:
        logger.warning(f"Could not resolve KB caller role for {user_id}: {e}")
        return "agent"

    if is_admin:
        # An admin asking for a narrower scope gets it; otherwise admin.
        return requested or "admin"

    if requested == "admin":
        logger.warning(
            "Caller %s asked for KB caller='admin' in workspace %s without the role — "
            "clamped to 'agent'",
            user_id, workspace_id,
        )
    return "agent"


def resolve_kb_access_scope(
    supabase,
    workspace_id: str,
    caller: str,
    query: str,
    *,
    per_doc_agent_gate: bool,
) -> Dict[str, Any]:
    """What this caller may read from the KB: levels, category allow-list, shared scope.

    ONE definition, shared by `/api/kb/search`, `/api/rag/search/knowledge-base` and
    `/search/read-section`. A read-by-id endpoint that re-derived this slightly
    differently would be a BOLA hole (the caller supplies the kb_doc_id), which is why
    it was already shared between the last two — this just finishes the job.

    `caller` must already have been through `resolve_kb_caller`. Passing a body-supplied
    string here is the bug this module exists to prevent.

    `accessible_category_ids` is None when no post-filter is needed (admin/public).

    ── per_doc_agent_gate ────────────────────────────────────────────────────────────
    This is NOT a preference. It is a statement about which gate the CORPUS RPC applies
    downstream, and it is the one place the two endpoints genuinely differ:

      * `kb_match_doc_chunks` (the agent path) enforces category `access_level` AND
        per-doc `allowed_agents` inside the RPC. There, `visibility='private'` means
        only "not published to the public KB website" — it is not an agent gate — so an
        agent legitimately reads private docs. Pass True.
      * `kb_match_docs` (the admin path) has no per-doc agent gate. `include_private`
        is the ONLY thing standing between a non-admin and private content, so it must
        track admin-ness. Pass False.

    Passing True from a corpus with no per-doc gate silently widens a non-admin's read
    to every private doc in the workspace. That is why it is a required keyword rather
    than a default — there is no value that is right for both, so there is no default
    that is safe to forget.
    """
    query_lower = (query or "").lower()
    shared_id = kb_shared_workspace_id()
    shared_workspace_id = shared_id if workspace_id != shared_id else None

    if caller == "admin":
        # Admin sees everything — no keyword restriction.
        return {
            "allowed_access_levels": list(_ADMIN_LEVELS),
            "accessible_category_ids": None,
            "shared_workspace_id": shared_workspace_id,
            "include_private": True,
        }

    if caller == "public":
        return {
            "allowed_access_levels": list(_PUBLIC_LEVELS),
            "accessible_category_ids": None,
            # The public-website caller never reaches across workspaces.
            "shared_workspace_id": None,
            "include_private": False,
        }

    # Agent caller: public categories always accessible; agent-level categories only if
    # trigger_keyword matches (or no keyword set).
    # Include the shared root workspace's categories so root docs are gated by the SAME
    # access_level + trigger_keyword rules instead of being dropped by the post-filter
    # for not belonging to the caller's workspace.
    kb_ws_scope = [workspace_id]
    if shared_workspace_id:
        kb_ws_scope.append(shared_workspace_id)

    accessible_category_ids: List[str] = []
    try:
        cats_resp = (
            supabase.client.table("kb_categories")
            .select("id, access_level, trigger_keyword")
            .in_("workspace_id", kb_ws_scope)
            .in_("access_level", ["agent", "public"])
            .execute()
        )
        rows = cats_resp.data or []
    except Exception as e:
        # Fail CLOSED. An empty allow-list means "no agent-level categories are
        # readable", which is a narrower answer than the truth; the alternative
        # (None = no post-filter) would mean "everything", from a failed query.
        logger.warning(f"Could not resolve KB categories: {e}")
        rows = []

    for cat in rows:
        if cat.get("access_level") == "public":
            accessible_category_ids.append(cat["id"])
            continue
        kw = cat.get("trigger_keyword")
        if kw is None or str(kw).strip() == "":
            accessible_category_ids.append(cat["id"])
        elif str(kw).lower() in query_lower:
            accessible_category_ids.append(cat["id"])
        # else: keyword required but not found — skip this category

    logger.info(f"   🔑 Accessible KB categories for query: {len(accessible_category_ids)}")

    return {
        "allowed_access_levels": list(_AGENT_LEVELS),
        "accessible_category_ids": accessible_category_ids,
        "shared_workspace_id": shared_workspace_id,
        "include_private": bool(per_doc_agent_gate),
    }
