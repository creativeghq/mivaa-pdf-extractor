"""
Public Project Workspace API — /api/v1/projects/*

External integrations authenticate with an `api_keys` Bearer token (`kai_*`)
and manage Projects: a container above moodboards, quotes, rooms, tasks,
and collaborators for a single engagement.

Mirror of `mention_tracking_routes.py` / `job_tracking_routes.py` for the
Projects module. Auth dependency is identical (`authenticate_api_key`) so the
same `kai_*` key works across price / mention / job / project tracking.

Routing summary:
  api_key_id → projects.user_id (api_key's owning user) is the acting user.
              Every project created here is owned by that user, with all
              the normal Supabase RLS guarantees (other users / api_keys
              can't read or write it).

Internal flow (browser session JWT) talks to Supabase directly and is not
exposed here — see docs/projects.md for the SDK surface.

Credit cost (all reads = 0; writes itemised below):
  create_project              = 0 cr (no upstream services hit)
  invite_collaborator         = 1 cr (sends a transactional email)
  All other writes (rooms, tasks, updates, deletes, revokes) = 0 cr.

Tables / RPCs touched (see Phase 1-4 migrations):
  projects, project_rooms, project_tasks, project_events,
  project_collaborators, accept_project_invitation,
  get_project_invitation_preview.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.price_lookup_routes import ApiKeyContext, authenticate_api_key
from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/projects",
    tags=["Project Workspace (Public API)"],
    responses={
        401: {"description": "Invalid or missing API key"},
        403: {"description": "API key does not own this project"},
        404: {"description": "Project / room / task / collaborator not found"},
        429: {"description": "Rate limit exceeded"},
    },
)


# ────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ────────────────────────────────────────────────────────────────────────────

class CreateRoomInline(BaseModel):
    """Inline room definition used on project create. Lets a partner spin up
    a fully-roomed project in one POST."""
    name: str = Field(..., min_length=1, max_length=200, examples=["Master Bath"])
    room_type: Optional[str] = Field(
        None, pattern="^(bedroom|bathroom|kitchen|living|dining|office|outdoor|hallway|other)$",
        examples=["bathroom"],
    )


class CreateProjectRequest(BaseModel):
    """Body for `POST /api/v1/projects` — create a new project owned by the
    API key's user. The first refresh of dependent state (moodboard counts,
    quote totals, room rollups) happens on demand later — there is no
    background sync."""
    name: str = Field(..., min_length=1, max_length=200, examples=["Kavouri Villa Renovation"])
    description: Optional[str] = Field(None, max_length=4000)
    status: str = Field(
        "planning", pattern="^(planning|in_progress|on_hold|completed|archived)$",
    )
    client_company_id: Optional[str] = Field(
        None, description="UUID of an existing crm_companies row. XOR with client_contact_id.",
    )
    client_contact_id: Optional[str] = Field(
        None, description="UUID of an existing crm_contacts row. XOR with client_company_id.",
    )
    deadline: Optional[str] = Field(
        None, description="ISO date (YYYY-MM-DD). Optional.",
        examples=["2026-09-30"],
    )
    budget_amount: Optional[float] = Field(None, ge=0)
    budget_currency: str = Field("EUR", min_length=3, max_length=3)
    cover_image_url: Optional[str] = None
    rooms: Optional[List[CreateRoomInline]] = Field(
        None,
        description="Optional rooms to create alongside the project (one DB write, no extra calls).",
    )


class UpdateProjectRequest(BaseModel):
    """Body for `PUT /api/v1/projects/{id}`. All fields optional (PATCH-like behaviour)."""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=4000)
    status: Optional[str] = Field(None, pattern="^(planning|in_progress|on_hold|completed|archived)$")
    client_company_id: Optional[str] = None
    client_contact_id: Optional[str] = None
    deadline: Optional[str] = None
    budget_amount: Optional[float] = Field(None, ge=0)
    budget_currency: Optional[str] = Field(None, min_length=3, max_length=3)
    cover_image_url: Optional[str] = None


class CreateRoomRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    room_type: Optional[str] = Field(None, pattern="^(bedroom|bathroom|kitchen|living|dining|office|outdoor|hallway|other)$")
    budget_amount: Optional[float] = Field(None, ge=0)
    deadline: Optional[str] = None
    notes: Optional[str] = Field(None, max_length=2000)
    sort_order: int = Field(0, ge=0)


class CreateTaskRequest(BaseModel):
    """Body for `POST /api/v1/projects/{id}/tasks`. Pass `parent_task_id` to
    make a subtask (max nesting depth = 1, enforced by trigger)."""
    title: str = Field(..., min_length=1, max_length=500, examples=["Order marble samples"])
    description: Optional[str] = Field(None, max_length=4000)
    parent_task_id: Optional[str] = Field(
        None, description="When set, this becomes a subtask. The parent cannot itself be a subtask.",
    )
    room_id: Optional[str] = None
    status: str = Field("todo", pattern="^(todo|in_progress|done|blocked)$")
    due_date: Optional[str] = None
    visibility: str = Field("internal", pattern="^(internal|client_visible)$")
    sort_order: int = Field(0, ge=0)


class UpdateTaskRequest(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    description: Optional[str] = Field(None, max_length=4000)
    status: Optional[str] = Field(None, pattern="^(todo|in_progress|done|blocked)$")
    due_date: Optional[str] = None
    visibility: Optional[str] = Field(None, pattern="^(internal|client_visible)$")
    room_id: Optional[str] = None
    sort_order: Optional[int] = Field(None, ge=0)


class InviteCollaboratorRequest(BaseModel):
    """Body for `POST /api/v1/projects/{id}/collaborators`. Triggers a branded
    invite email sent by the platform's `email-api` edge function. Invitee
    signs in passwordlessly via Supabase OTP."""
    email: str = Field(..., min_length=3, max_length=200, examples=["client@example.com"])
    message: Optional[str] = Field(None, max_length=1000)
    expires_in_days: int = Field(90, ge=1, le=365, description="Days until the invitation auto-expires.")


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

CREDIT_COST = {
    # 0-cost ops are excluded (read-only or no upstream spend).
    "invite_collaborator": 1,
}


def _owned_project(sb, project_id: str, user_id: str) -> Dict[str, Any]:
    """Fetch a project row and assert it's owned by the calling api_key's user.
    Returns the row dict. Raises 404 (not 403) on miss so we don't leak existence."""
    r = sb.table("projects").select("*").eq("id", project_id).eq("user_id", user_id).maybeSingle().execute()
    if not r.data:
        raise HTTPException(status_code=404, detail="project not found")
    return r.data


def _debit(user_id: str, op: str) -> int:
    cost = CREDIT_COST.get(op, 0)
    if cost <= 0:
        return 0
    sb = get_supabase_client().client
    res = sb.rpc("debit_user_credits", {
        "p_user_id": user_id,
        "p_amount": cost,
        "p_operation_type": f"projects.{op}",
        "p_description": f"Projects API: {op}",
        "p_metadata": {"feature": "projects_api"},
    }).execute()
    row = (res.data[0] if isinstance(res.data, list) else res.data) or {}
    if not row.get("success"):
        raise HTTPException(status_code=402, detail=row.get("error_message") or "insufficient credits")
    return cost


def _refund(user_id: str, op: str, amount: int) -> None:
    if amount <= 0:
        return
    sb = get_supabase_client().client
    try:
        sb.rpc("credit_user_credits", {
            "p_user_id": user_id,
            "p_amount": amount,
            "p_operation_type": f"projects.{op}.refund",
            "p_description": f"Projects API: {op} refund",
            "p_metadata": {"feature": "projects_api", "reason": "operation_failed"},
        }).execute()
    except Exception:
        # best-effort
        logger.warning("project refund failed user=%s op=%s amount=%d", user_id, op, amount)


# ────────────────────────────────────────────────────────────────────────────
# Endpoints — projects CRUD
# ────────────────────────────────────────────────────────────────────────────

@router.post(
    "",
    summary="Create a new project (and optionally its rooms)",
)
async def create_project(
    body: CreateProjectRequest,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    """Insert one `projects` row owned by the API key's user, plus any inline
    rooms in the same call. Returns the new project + created rooms.

    Credits: 0
    """
    if not ctx.user_id:
        raise HTTPException(status_code=403, detail="API key has no associated user")
    if body.client_company_id and body.client_contact_id:
        raise HTTPException(status_code=400, detail="client_company_id and client_contact_id are mutually exclusive")

    sb = get_supabase_client().client
    proj_payload = {
        "user_id": ctx.user_id,
        "workspace_id": ctx.workspace_id,
        "name": body.name,
        "description": body.description,
        "status": body.status,
        "client_company_id": body.client_company_id,
        "client_contact_id": body.client_contact_id,
        "deadline": body.deadline,
        "budget_amount": body.budget_amount,
        "budget_currency": body.budget_currency,
        "cover_image_url": body.cover_image_url,
    }
    r = sb.table("projects").insert(proj_payload).execute()
    if not r.data:
        raise HTTPException(status_code=500, detail="failed to create project")
    project = r.data[0]

    created_rooms: List[Dict[str, Any]] = []
    if body.rooms:
        rows = [
            {
                "project_id": project["id"],
                "name": room.name,
                "room_type": room.room_type,
                "sort_order": idx,
            }
            for idx, room in enumerate(body.rooms)
        ]
        rr = sb.table("project_rooms").insert(rows).execute()
        created_rooms = rr.data or []

    return {"success": True, "data": {"project": project, "rooms": created_rooms}}


@router.get(
    "",
    summary="List the API key user's projects",
)
async def list_projects(
    ctx: ApiKeyContext = Depends(authenticate_api_key),
    include_archived: bool = Query(False),
    limit: int = Query(100, ge=1, le=500),
):
    """Returns projects owned by the API key's user, newest activity first.

    Credits: 0
    """
    if not ctx.user_id:
        raise HTTPException(status_code=403, detail="API key has no associated user")
    sb = get_supabase_client().client
    q = (
        sb.table("projects")
        .select("*")
        .eq("user_id", ctx.user_id)
        .order("last_activity_at", desc=True)
        .limit(limit)
    )
    r = q.execute()
    rows = r.data or []
    if not include_archived:
        rows = [p for p in rows if p.get("status") not in ("archived", "completed")]
    return {"success": True, "data": rows, "count": len(rows)}


@router.get(
    "/{project_id}",
    summary="Fetch one project with its denormalised counters",
)
async def get_project(
    project_id: str,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    """Credits: 0"""
    sb = get_supabase_client().client
    row = _owned_project(sb, project_id, ctx.user_id or "")
    return {"success": True, "data": row}


@router.put(
    "/{project_id}",
    summary="Update project fields (PATCH-like — only fields you send are touched)",
)
async def update_project(
    project_id: str,
    body: UpdateProjectRequest,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    """Credits: 0"""
    sb = get_supabase_client().client
    _owned_project(sb, project_id, ctx.user_id or "")
    if body.client_company_id and body.client_contact_id:
        raise HTTPException(status_code=400, detail="client_company_id and client_contact_id are mutually exclusive")
    updates = body.model_dump(exclude_unset=True)
    r = sb.table("projects").update(updates).eq("id", project_id).execute()
    return {"success": True, "data": (r.data or [None])[0]}


@router.delete(
    "/{project_id}",
    summary="Archive a project (soft delete — sets status='archived')",
)
async def archive_project(
    project_id: str,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    """Projects are NEVER hard-deleted via the API — the financial / audit
    history they reference is load-bearing. To remove a project completely,
    do it through the Supabase dashboard as the data owner.

    Credits: 0
    """
    sb = get_supabase_client().client
    _owned_project(sb, project_id, ctx.user_id or "")
    sb.table("projects").update({"status": "archived"}).eq("id", project_id).execute()
    return {"success": True, "data": {"status": "archived"}}


# ────────────────────────────────────────────────────────────────────────────
# Endpoints — rooms
# ────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{project_id}/rooms",
    summary="List rooms in a project",
)
async def list_rooms(
    project_id: str,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    """Credits: 0"""
    sb = get_supabase_client().client
    _owned_project(sb, project_id, ctx.user_id or "")
    r = sb.table("project_rooms").select("*").eq("project_id", project_id).order("sort_order").execute()
    return {"success": True, "data": r.data or []}


@router.post(
    "/{project_id}/rooms",
    summary="Add a room to a project",
)
async def create_room(
    project_id: str,
    body: CreateRoomRequest,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    """Credits: 0"""
    sb = get_supabase_client().client
    _owned_project(sb, project_id, ctx.user_id or "")
    payload = body.model_dump(exclude_unset=True)
    payload["project_id"] = project_id
    r = sb.table("project_rooms").insert(payload).execute()
    return {"success": True, "data": (r.data or [None])[0]}


@router.delete(
    "/rooms/{room_id}",
    summary="Delete a room",
)
async def delete_room(
    room_id: str,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    """Linked moodboards keep their other links — `moodboards.room_id` is set
    to NULL by the FK (ON DELETE SET NULL).

    Credits: 0
    """
    sb = get_supabase_client().client
    # Ownership check: room must belong to a project owned by the api_key user
    r = sb.table("project_rooms").select("project_id").eq("id", room_id).maybeSingle().execute()
    if not r.data:
        raise HTTPException(status_code=404, detail="room not found")
    _owned_project(sb, r.data["project_id"], ctx.user_id or "")
    sb.table("project_rooms").delete().eq("id", room_id).execute()
    return {"success": True}


# ────────────────────────────────────────────────────────────────────────────
# Endpoints — tasks (with subtasks)
# ────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{project_id}/tasks",
    summary="List tasks (parents + subtasks nested under each parent)",
)
async def list_tasks(
    project_id: str,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    """Returns a flat array of parent tasks, each with `subtasks: [...]`.
    Subtasks are sorted by `sort_order` then `created_at`.

    Credits: 0
    """
    sb = get_supabase_client().client
    _owned_project(sb, project_id, ctx.user_id or "")
    r = (
        sb.table("project_tasks")
        .select("*")
        .eq("project_id", project_id)
        .order("sort_order")
        .order("created_at")
        .execute()
    )
    rows = r.data or []
    parents = [t for t in rows if not t.get("parent_task_id")]
    subs_by_parent: Dict[str, List[Dict[str, Any]]] = {}
    for t in rows:
        if t.get("parent_task_id"):
            subs_by_parent.setdefault(t["parent_task_id"], []).append(t)
    out = []
    for p in parents:
        subs = subs_by_parent.get(p["id"], [])
        out.append({
            **p,
            "subtasks": subs,
            "subtask_total_count": len(subs),
            "subtask_done_count": sum(1 for s in subs if s.get("status") == "done"),
        })
    return {"success": True, "data": out}


@router.post(
    "/{project_id}/tasks",
    summary="Add a task (or subtask via parent_task_id)",
)
async def create_task(
    project_id: str,
    body: CreateTaskRequest,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    """Max 1 level of nesting — a task with a `parent_task_id` cannot itself
    have children (enforced by the `_project_tasks_enforce_depth_cap` trigger;
    returns 500 with a Postgres exception message if violated).

    Credits: 0
    """
    sb = get_supabase_client().client
    _owned_project(sb, project_id, ctx.user_id or "")

    # Subtasks inherit room from parent.
    room_id = body.room_id
    if body.parent_task_id:
        pr = sb.table("project_tasks").select("room_id, project_id").eq("id", body.parent_task_id).maybeSingle().execute()
        if not pr.data or pr.data.get("project_id") != project_id:
            raise HTTPException(status_code=400, detail="parent_task_id is not in this project")
        room_id = pr.data.get("room_id")

    payload = {
        "project_id": project_id,
        "parent_task_id": body.parent_task_id,
        "room_id": room_id,
        "title": body.title,
        "description": body.description,
        "status": body.status,
        "due_date": body.due_date,
        "visibility": body.visibility,
        "sort_order": body.sort_order,
        "created_by": ctx.user_id,
    }
    r = sb.table("project_tasks").insert(payload).execute()
    return {"success": True, "data": (r.data or [None])[0]}


@router.put(
    "/tasks/{task_id}",
    summary="Update task fields",
)
async def update_task(
    task_id: str,
    body: UpdateTaskRequest,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    """Status flips to/from 'done' auto-stamp `completed_at` via trigger.

    Credits: 0
    """
    sb = get_supabase_client().client
    tr = sb.table("project_tasks").select("project_id").eq("id", task_id).maybeSingle().execute()
    if not tr.data:
        raise HTTPException(status_code=404, detail="task not found")
    _owned_project(sb, tr.data["project_id"], ctx.user_id or "")
    updates = body.model_dump(exclude_unset=True)
    r = sb.table("project_tasks").update(updates).eq("id", task_id).execute()
    return {"success": True, "data": (r.data or [None])[0]}


@router.delete(
    "/tasks/{task_id}",
    summary="Delete a task (CASCADEs subtasks)",
)
async def delete_task(
    task_id: str,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    """Credits: 0"""
    sb = get_supabase_client().client
    tr = sb.table("project_tasks").select("project_id").eq("id", task_id).maybeSingle().execute()
    if not tr.data:
        raise HTTPException(status_code=404, detail="task not found")
    _owned_project(sb, tr.data["project_id"], ctx.user_id or "")
    sb.table("project_tasks").delete().eq("id", task_id).execute()
    return {"success": True}


# ────────────────────────────────────────────────────────────────────────────
# Endpoints — collaborators (passwordless email invitations)
# ────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{project_id}/collaborators",
    summary="List all invitations (pending / active / revoked / expired)",
)
async def list_collaborators(
    project_id: str,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    """Credits: 0"""
    sb = get_supabase_client().client
    _owned_project(sb, project_id, ctx.user_id or "")
    r = sb.table("project_collaborators").select("*").eq("project_id", project_id).order("invited_at", desc=True).execute()
    return {"success": True, "data": r.data or []}


@router.post(
    "/{project_id}/collaborators",
    summary="Invite a collaborator by email (sends a magic-link invite)",
)
async def invite_collaborator(
    project_id: str,
    body: InviteCollaboratorRequest,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    """Inserts a `project_collaborators` row + sends a branded HTML email
    via the platform's `email-api` edge function. The invitee opens the link,
    enters their email, signs in via Supabase OTP magic-link, and the
    `accept_project_invitation` RPC stamps their `user_id` so RLS grants
    read access to the project, rooms, moodboards, sheets, and `client_visible`
    tasks.

    Credits: 1 (covers the transactional email).
    """
    if not ctx.user_id:
        raise HTTPException(status_code=403, detail="API key has no associated user")
    sb = get_supabase_client().client
    _owned_project(sb, project_id, ctx.user_id)

    email = body.email.strip().lower()
    if "@" not in email:
        raise HTTPException(status_code=400, detail="invalid email")

    # Reject duplicate active invites for the same email
    existing = (
        sb.table("project_collaborators")
        .select("id")
        .eq("project_id", project_id)
        .ilike("email", email)
        .is_("revoked_at", "null")
        .execute()
    )
    if existing.data:
        raise HTTPException(status_code=409, detail="active invitation already exists for this email")

    debited = _debit(ctx.user_id, "invite_collaborator")
    try:
        payload = {
            "project_id": project_id,
            "email": email,
            "invited_by": ctx.user_id,
            "message": body.message,
        }
        # expires_at defaults to invited_at + 90 days at the DB layer; only override if non-default
        if body.expires_in_days != 90:
            from datetime import datetime, timedelta, timezone
            payload["expires_at"] = (datetime.now(timezone.utc) + timedelta(days=body.expires_in_days)).isoformat()
        r = sb.table("project_collaborators").insert(payload).execute()
        row = (r.data or [None])[0]
        if not row:
            raise HTTPException(status_code=500, detail="failed to create collaborator row")

        # Best-effort email dispatch via email-api edge function. On failure we
        # keep the row + share_token so the caller can copy the link manually.
        # The frontend invite-modal pattern is the same.
        try:
            _send_invite_email(
                sb=sb,
                to=email,
                project_id=project_id,
                share_token=row["share_token"],
                inviter_user_id=ctx.user_id,
                message=body.message,
            )
        except Exception as send_err:
            logger.warning("invite email send failed (non-blocking): %s", send_err)

        return {"success": True, "data": row, "partner_credits_debited": debited}
    except HTTPException:
        _refund(ctx.user_id, "invite_collaborator", debited)
        raise
    except Exception:
        _refund(ctx.user_id, "invite_collaborator", debited)
        raise


@router.delete(
    "/collaborators/{collaborator_id}",
    summary="Revoke a collaborator's access (immediate)",
)
async def revoke_collaborator(
    collaborator_id: str,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
):
    """Sets `revoked_at = now()`. RLS policies on projects / rooms / tasks /
    moodboards / sheets all check `revoked_at IS NULL`, so the next request
    from this collaborator's session returns 404.

    Credits: 0
    """
    sb = get_supabase_client().client
    cr = sb.table("project_collaborators").select("project_id").eq("id", collaborator_id).maybeSingle().execute()
    if not cr.data:
        raise HTTPException(status_code=404, detail="collaborator not found")
    _owned_project(sb, cr.data["project_id"], ctx.user_id or "")
    from datetime import datetime, timezone
    sb.table("project_collaborators").update({"revoked_at": datetime.now(timezone.utc).isoformat()}).eq("id", collaborator_id).execute()
    return {"success": True}


# ────────────────────────────────────────────────────────────────────────────
# Endpoints — timeline (read-only)
# ────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{project_id}/events",
    summary="Project event timeline (newest first)",
)
async def list_events(
    project_id: str,
    ctx: ApiKeyContext = Depends(authenticate_api_key),
    limit: int = Query(100, ge=1, le=500),
):
    """The audit log captures: project.created / status_changed / budget_changed /
    deadline_changed, room.added / removed, moodboard.attached / detached,
    quote.attached / detached / status_changed / revised, task / subtask.created /
    completed / deleted. Capped at 500 events per project (oldest pruned).

    Credits: 0
    """
    sb = get_supabase_client().client
    _owned_project(sb, project_id, ctx.user_id or "")
    r = (
        sb.table("project_events")
        .select("*")
        .eq("project_id", project_id)
        .order("occurred_at", desc=True)
        .limit(limit)
        .execute()
    )
    return {"success": True, "data": r.data or []}


# ────────────────────────────────────────────────────────────────────────────
# Helpers (email)
# ────────────────────────────────────────────────────────────────────────────

def _send_invite_email(
    *,
    sb,
    to: str,
    project_id: str,
    share_token: str,
    inviter_user_id: str,
    message: Optional[str],
) -> None:
    """Render the same HTML invite the frontend modal uses and dispatch via
    `email-api?action=send`. Idempotent at the row level (email-api logs every
    send to email_logs; replays are visible there)."""
    import os
    import httpx

    pr = sb.table("projects").select("name").eq("id", project_id).maybeSingle().execute()
    project_name = (pr.data or {}).get("name") or "a project"
    # auth.users is not exposed via PostgREST → read display name from user_profiles instead.
    pur = sb.table("user_profiles").select("email,full_name").eq("user_id", inviter_user_id).maybeSingle().execute()
    inviter_name = (pur.data or {}).get("full_name") or (pur.data or {}).get("email") or "Your collaborator"

    app_url = (os.getenv("PUBLIC_APP_URL") or "https://app.materialshub.gr").rstrip("/")
    invite_url = f"{app_url}/projects/invite/{share_token}"

    def esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    message_block = (
        f'<p style="margin:16px 0;padding:12px;background:#f5f5f5;border-left:3px solid #999;font-style:italic;">{esc(message)}</p>'
        if message else ""
    )
    html = f"""<!doctype html>
<html><body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:560px;margin:32px auto;padding:24px;color:#222;">
  <h2 style="margin:0 0 16px;font-weight:300;">You've been invited</h2>
  <p style="margin:0 0 12px;"><strong>{esc(inviter_name)}</strong> invited you to view the project <strong>"{esc(project_name)}"</strong>.</p>
  {message_block}
  <p style="margin:24px 0;">
    <a href="{esc(invite_url)}" style="display:inline-block;padding:12px 24px;background:#8a3a6b;color:#fff;text-decoration:none;border-radius:9999px;font-weight:500;">View project</a>
  </p>
  <p style="margin:24px 0 0;font-size:13px;color:#666;">No password needed — just confirm your email on the next screen. Link expires in 90 days.</p>
</body></html>"""

    supabase_url = os.environ["SUPABASE_URL"].rstrip("/")
    service_role = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    with httpx.Client(timeout=10.0) as client:
        client.post(
            f"{supabase_url}/functions/v1/email-api?action=send",
            headers={
                "Authorization": f"Bearer {service_role}",
                "apikey": service_role,
                "Content-Type": "application/json",
            },
            json={
                "to": to,
                "subject": f'{inviter_name} invited you to view "{project_name}"',
                "html": html,
                "emailType": "transactional",
            },
        )
