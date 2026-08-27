"""
Background Agent API Routes

Handles long-running agent tasks delegated from Supabase edge functions.
The edge function fires-and-forgets here when a task exceeds the 25s timeout.

Endpoints:
  POST /api/agents/run       - Receive delegated agent task, run in background
  GET  /api/agents/runs/{id} - Get status of a specific run
  GET  /api/agents/catalog   - List all available agent types

AUTH (audit #24 M11-1). `_require_internal_key` fails CLOSED. The previous form was
`if expected_key and authorization != ...`, so an unset `MIVAA_API_KEY` short-circuited
the `and` and ran no check at all — on the route that spends AI credits and writes to
`products`. A missing env var is not hypothetical here: #16 M3-4 found the Supabase
client falling back to the anon key when the service-role key is absent, and this
repository has no startup assertion that either variable is present. The correct
fail-closed form was already in `catalog_routes._check_secret` and `seo_agent_routes`.

TENANCY (audit #24 M11-2 / M11-3, schema drift #26 M13-1). Both handlers selected the
first `batch_size` products PLATFORM-WIDE and updated them by product id alone, so one
tenant's agent run could rewrite another tenant's gold layer. The workspace now comes
from the `agent_runs` row (falling back to the agent's own workspace), never from the
request body, and every read and write carries it.

Those handlers were also unreachable: they referenced `material_type`, `tags`,
`image_url` and `search_keywords`, none of which exist on `products` — they live in the
`attributes` jsonb, resolved through the facet registry. PostgREST rejected the request
before any update ran, which is why the unscoped write was latent rather than live. The
column fix and the workspace predicate land in the SAME change deliberately: correcting
the columns alone would activate the cross-tenant write in one commit.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Header
from pydantic import BaseModel, Field

from app.dependencies import get_workspace_context
from app.schemas.auth import WorkspaceContext
from app.services.core.supabase_client import get_supabase_client
from app.schemas.api_responses import AgentCatalogResponse, DataResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["Background Agents"])


# ── Request / Response models ────────────────────────────────────────────────

class AgentRunRequest(BaseModel):
    run_id:        str = Field(..., description="agent_runs.id to update when done")
    agent_id:      str = Field(..., description="background_agents.id")
    agent_type:    str = Field(..., description="Registry key, e.g. 'product-enrichment'")
    input_data:    Dict[str, Any] = Field(default_factory=dict)
    model:         Optional[str] = Field(default="claude-haiku-4-5")
    system_prompt: Optional[str] = None
    config:        Dict[str, Any] = Field(default_factory=dict)


class AgentRunResponse(BaseModel):
    success: bool
    job_id:  str
    message: str


# ── Agent type handlers ──────────────────────────────────────────────────────

AGENT_HANDLERS: Dict[str, str] = {
    "product-enrichment": "handle_product_enrichment",
    "material-tagger":    "handle_material_tagger",
}

AGENT_CATALOG = [
    {
        "agentType":    "product-enrichment",
        "name":         "Product Enrichment",
        "description":  "AI-generates descriptions, keywords and category tags for products (large batches)",
        "defaultModel": "claude-haiku-4-5",
    },
    {
        "agentType":    "material-tagger",
        "name":         "Material Tagger",
        "description":  "Auto-tags materials with type, color, finish, application",
        "defaultModel": "claude-haiku-4-5",
    },
]


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/catalog", responses={200: {"model": AgentCatalogResponse}})
async def get_catalog():
    return {"catalog": AGENT_CATALOG}


@router.get("/runs/{run_id}", responses={200: {"model": DataResponse}})
async def get_run_status(
    run_id: str,
    workspace: WorkspaceContext = Depends(get_workspace_context),
):
    """Status of one agent run, scoped to the caller's workspace (M11-3).

    404 rather than 403 on a run belonging to someone else: a 403 confirms the id
    exists, which is the enumeration oracle invariant 1 exists to remove. A run with
    no `workspace_id` is unattributed and therefore nobody's — it reads as not found.
    """
    supabase = get_supabase_client()
    result = (supabase.table("agent_runs")
              .select("id, agent_id, status, input_data, output_data, error_message, "
                      "model_used, input_tokens, output_tokens, credits_debited, "
                      "started_at, completed_at, duration_ms, workspace_id")
              .eq("id", run_id)
              .eq("workspace_id", workspace.workspace_id)
              .limit(1)
              .execute())
    row = (result.data or [None])[0]
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")
    return row


@router.post("/run", response_model=AgentRunResponse)
async def run_agent(
    req: AgentRunRequest,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(None),
):
    """
    Receive a delegated agent task from the edge function and execute it
    in the background (no 30-second timeout limit).
    """
    _require_internal_key(authorization)

    if req.agent_type not in AGENT_HANDLERS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown agent_type '{req.agent_type}'. Available: {list(AGENT_HANDLERS)}",
        )

    background_tasks.add_task(_execute_agent, req)

    return AgentRunResponse(
        success=True,
        job_id=req.run_id,
        message=f"Agent '{req.agent_type}' accepted for background execution",
    )


# ── Auth ─────────────────────────────────────────────────────────────────────

def _require_internal_key(authorization: Optional[str]) -> None:
    """Reject unless the caller presents the internal key. Fails CLOSED.

    An unset key is 503, not 401: "this deployment is misconfigured" and "your token
    is wrong" are different facts and an operator needs to tell them apart. What is
    NOT allowed is the third answer the old code gave — letting the request through.
    """
    expected_key = os.environ.get("MIVAA_API_KEY", "")
    if not expected_key:
        logger.error("MIVAA_API_KEY is unset — refusing delegated agent runs (M11-1)")
        raise HTTPException(
            status_code=503,
            detail="Agent delegation is not configured on this deployment",
        )
    if authorization != f"Bearer {expected_key}":
        raise HTTPException(status_code=401, detail="Unauthorized")


# ── Background execution ─────────────────────────────────────────────────────

def _resolve_run_workspace(supabase, req: "AgentRunRequest") -> str:
    """The workspace this run belongs to, derived from stored rows only.

    Never from the request body (invariant 1) — `AgentRunRequest` deliberately has no
    `workspace_id` field, so there is nothing to trust. `agent_runs.workspace_id` is
    the answer; `background_agents.workspace_id` is the fallback for runs created
    before the column was populated. Neither present means the run cannot be attributed,
    and an unattributed run must NOT fall back to "every workspace".
    """
    run = (supabase.table("agent_runs")
           .select("workspace_id")
           .eq("id", req.run_id)
           .limit(1)
           .execute())
    ws = ((run.data or [{}])[0] or {}).get("workspace_id")
    if not ws:
        agent = (supabase.table("background_agents")
                 .select("workspace_id")
                 .eq("id", req.agent_id)
                 .limit(1)
                 .execute())
        ws = ((agent.data or [{}])[0] or {}).get("workspace_id")
    if not ws:
        raise ValueError(
            f"agent run {req.run_id} has no workspace (agent {req.agent_id} has none "
            "either) — refusing to run platform-wide"
        )
    return str(ws)


async def _execute_agent(req: AgentRunRequest) -> None:
    supabase = get_supabase_client()
    start_time = datetime.now(timezone.utc)

    # Mark as processing
    supabase.table("agent_runs").update({
        "status":     "processing",
        "started_at": start_time.isoformat(),
        "last_heartbeat": start_time.isoformat(),
    }).eq("id", req.run_id).execute()

    try:
        handler_name = AGENT_HANDLERS[req.agent_type]
        handler = globals().get(handler_name)
        if not handler:
            raise ValueError(f"Handler function '{handler_name}' not found")

        workspace_id = _resolve_run_workspace(supabase, req)
        result = await handler(req, supabase, workspace_id)

        duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

        supabase.table("agent_runs").update({
            "status":         "completed",
            "output_data":    result.get("output", {}),
            "input_tokens":   result.get("input_tokens", 0),
            "output_tokens":  result.get("output_tokens", 0),
            "model_used":     req.model,
            "completed_at":   datetime.now(timezone.utc).isoformat(),
            "duration_ms":    duration_ms,
        }).eq("id", req.run_id).execute()

        _log(supabase, req.run_id, "info",
             f"Agent completed in {duration_ms}ms",
             {"output": result.get("output", {})})

    except AgentCancelled:
        duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
        supabase.table("agent_runs").update({
            "status":       "cancelled",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms":  duration_ms,
        }).eq("id", req.run_id).execute()
        _log(supabase, req.run_id, "warn", "Agent run cancelled by admin")

    except Exception as exc:
        logger.exception("Background agent '%s' failed: %s", req.agent_type, exc)
        duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

        supabase.table("agent_runs").update({
            "status":        "failed",
            "error_message": str(exc),
            "completed_at":  datetime.now(timezone.utc).isoformat(),
            "duration_ms":   duration_ms,
        }).eq("id", req.run_id).execute()

        _log(supabase, req.run_id, "error", f"Agent failed: {exc}")


# ── Heartbeat helper ─────────────────────────────────────────────────────────

class AgentCancelled(Exception):
    """Raised by _heartbeat when the run has been cancelled out-of-band."""
    pass


def _heartbeat(supabase, run_id: str) -> None:
    """Bump last_heartbeat; raise AgentCancelled if admin cancelled the run.

    Long batch handlers call this every few items so an admin-cancel from the
    dashboard actually stops work instead of burning credits to the end.
    """
    (supabase.table("agent_runs")
     .update({"last_heartbeat": datetime.now(timezone.utc).isoformat()})
     .eq("id", run_id)
     .execute())
    status_resp = (supabase.table("agent_runs")
                   .select("status")
                   .eq("id", run_id)
                   .single()
                   .execute())
    if (status_resp.data or {}).get("status") == "cancelled":
        raise AgentCancelled(f"Run {run_id} was cancelled")


def _log(supabase, run_id: str, level: str, message: str, data: Optional[Dict] = None) -> None:
    try:
        supabase.table("agent_run_logs").insert({
            "run_id":  run_id,
            "level":   level,
            "message": message,
            "data":    data,
        }).execute()
    except Exception as e:
        logger.warning(f"Failed to write agent run log: {e}")


# ── Agent handlers ───────────────────────────────────────────────────────────

#: What each agent is allowed to propose. A model reply is untrusted input and
#: `products.attributes` is the GOLD layer that search, faceting and the product page
#: read — so the reply is narrowed to an allowlist before anything reaches the
#: canonicalizer (invariant 8). `description` is the one real column either agent
#: writes; every other value belongs in `attributes`, per the field registry.
ENRICHMENT_FACETS = ("material_category", "keywords")
TAGGER_FACETS = ("material_type", "color", "finish", "application", "tags")

#: Forced-tool schemas (#32). These two handlers asked for JSON in the prompt and then
#: REPAIRED the reply — `.lstrip("```json").rstrip("```")` — before writing the result
#: into `products`. That is the archetype invariant 9 names: a classifier whose verdict
#: drives a database write, parsed from free-form text. With the tool forced the model
#: cannot return prose, so there is nothing to repair, and an absent tool block is a
#: typed retryable error instead of a swallowed exception that skipped the product.
ENRICHMENT_TOOL = {
    "name": "emit_product_enrichment",
    "description": "Return the enriched description, keywords and material category.",
    "input_schema": {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "Product description."},
            "keywords": {"type": "array", "items": {"type": "string"}},
            "material_category": {"type": "string"},
        },
        "required": ["description"],
    },
}

TAGGER_TOOL = {
    "name": "emit_material_tags",
    "description": "Classify the material and return its facets.",
    "input_schema": {
        "type": "object",
        "properties": {
            "material_type": {"type": "string"},
            "color": {"type": "string"},
            "finish": {"type": "string"},
            "application": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["material_type"],
    },
}

#: How many rows to look at to find `batch_size` that still need work. The
#: "not yet tagged" question is asked of `attributes` jsonb, which is why it is
#: answered here rather than in the query.
SCAN_MULTIPLIER = 5
MAX_SCAN = 500


def _allowlisted(data: Dict[str, Any], keys: tuple) -> Dict[str, Any]:
    """The subset of a model reply we are willing to store, dropping empties."""
    out: Dict[str, Any] = {}
    for key in keys:
        value = data.get(key)
        if value in (None, "", [], {}):
            continue
        out[key] = value
    return out


async def _canonicalize_into_attributes(
    supabase,
    product: Dict[str, Any],
    proposed: Dict[str, Any],
    *,
    source: str,
    workspace_id: str,
) -> Optional[Dict[str, Any]]:
    """Merge `proposed` into the product's attributes through the facet registry.

    Returns the `{attributes, attributes_raw}` update, or None when the
    canonicalizer degraded. Degraded means "the canonical map is not trustworthy",
    not "there are no facets" — writing it would be indistinguishable from a product
    that genuinely has none, which is the silent-zero shape the status flag exists to
    prevent.

    The merge is done HERE because `CanonicalizedAttributes.attributes` is built from
    this run's resolutions alone. Writing it wholesale would erase every facet the
    product already had. `attributes_raw` is already cumulative and is written as-is.
    """
    from app.services.facets import canonicalize_product_attributes

    result = await canonicalize_product_attributes(
        supabase,
        proposed,
        source=source,
        product_id=product["id"],
        workspace_id=workspace_id,
    )
    if result.status == "degraded":
        return None

    merged = dict(product.get("attributes") or {})
    for key, raw_value in proposed.items():
        # A key the registry does not canonicalize keeps the value as proposed.
        merged[key] = result.attributes.get(key, raw_value)

    return {"attributes": merged, "attributes_raw": result.attributes_raw}


def _fetch_workspace_products(
    supabase,
    workspace_id: str,
    batch_size: int,
    *,
    category_filter: Optional[str] = None,
    undescribed_only: bool = False,
) -> List[Dict[str, Any]]:
    """Candidate products, bound to one workspace.

    The predicate is the whole point of M11-2: without it these agents select the
    first N products on the PLATFORM and then update them by id alone.
    """
    query = (supabase.table("products")
             .select("id, name, description, category, attributes")
             .eq("workspace_id", workspace_id)
             .order("created_at")
             .limit(min(batch_size * SCAN_MULTIPLIER, MAX_SCAN)))
    if undescribed_only:
        query = query.is_("description", "null")
    if category_filter:
        query = query.eq("category", category_filter)
    return query.execute().data or []


async def handle_product_enrichment(
    req: AgentRunRequest, supabase, workspace_id: str,
) -> Dict[str, Any]:
    """Enrich products with AI descriptions, keywords, and category tags."""
    cfg         = {**req.config, **req.input_data}
    batch_size  = min(int(cfg.get("batch_size", 20)), 200)
    cat_filter  = cfg.get("category_filter")
    force_rewrite = bool(cfg.get("force_rewrite", False))

    _log(supabase, req.run_id, "info",
         "Product enrichment started (Python backend)",
         {"batch_size": batch_size, "category_filter": cat_filter,
          "workspace_id": workspace_id})

    products = _fetch_workspace_products(
        supabase, workspace_id, batch_size,
        category_filter=cat_filter,
        undescribed_only=not force_rewrite,
    )[:batch_size]

    if not products:
        return {"output": {"enriched": 0, "message": "No products to enrich"},
                "input_tokens": 0, "output_tokens": 0}

    _log(supabase, req.run_id, "info", f"Found {len(products)} products to enrich")

    system = req.system_prompt or (
        "You are a material product specialist. "
        "For each product, return JSON: {\"description\":\"...\",\"keywords\":[...],\"material_category\":\"...\"}. "
        "No markdown, no prose."
    )

    enriched = 0
    degraded = 0
    total_in = total_out = 0

    for i, product in enumerate(products):
        if i % 5 == 0:
            _heartbeat(supabase, req.run_id)

        try:
            from app.services.core.claude_tool_call import call_with_tool
            attributes = product.get("attributes") or {}
            result = await call_with_tool(
                task="agent_product_enrichment",
                model=req.model or "claude-haiku-4-5",
                max_tokens=512,
                system=system,
                messages=[{"role": "user", "content":
                    f"Name: {product['name']}\n"
                    f"Category: {product.get('category','unknown')}\n"
                    f"Description: {product.get('description','(none)')}\n"
                    f"Material: {attributes.get('material_type','unknown')}"}],
                tool=ENRICHMENT_TOOL,
                job_id=req.run_id,
                workspace_id=workspace_id,
                product_id=product["id"],
            )
            # Usage still accumulates. `call_with_tool` returns it precisely so a
            # migration cannot silently zero these counters.
            total_in  += result.input_tokens
            total_out += result.output_tokens
            data = result.data

            update: Dict[str, Any] = {}
            if data.get("description"):
                update["description"] = data["description"]

            proposed = _allowlisted(data, ENRICHMENT_FACETS)
            if proposed:
                attr_update = await _canonicalize_into_attributes(
                    supabase, product, proposed,
                    source="agent_product_enrichment", workspace_id=workspace_id,
                )
                if attr_update is None:
                    degraded += 1
                    _log(supabase, req.run_id, "warn",
                         f"Facet canonicalization degraded for product {product['id']}; "
                         "attributes left unchanged rather than blanked")
                else:
                    update.update(attr_update)

            if update:
                (supabase.table("products")
                 .update(update)
                 .eq("id", product["id"])
                 .eq("workspace_id", workspace_id)
                 .execute())
                enriched += 1

        except Exception as e:
            _log(supabase, req.run_id, "warn",
                 f"Failed to enrich product {product['id']}: {e}")

    _log(supabase, req.run_id, "info",
         f"Enrichment complete: {enriched}/{len(products)}",
         {"enriched": enriched, "total": len(products), "degraded": degraded})

    return {
        "output": {"enriched": enriched, "total": len(products), "degraded": degraded},
        "input_tokens": total_in,
        "output_tokens": total_out,
    }


async def handle_material_tagger(
    req: AgentRunRequest, supabase, workspace_id: str,
) -> Dict[str, Any]:
    """Auto-tag materials with type, color, finish, and application."""
    cfg        = {**req.config, **req.input_data}
    batch_size = min(int(cfg.get("batch_size", 20)), 200)
    force_rewrite = bool(cfg.get("force_rewrite", False))

    _log(supabase, req.run_id, "info",
         "Material tagging started (Python backend)",
         {"batch_size": batch_size, "workspace_id": workspace_id})

    candidates = _fetch_workspace_products(supabase, workspace_id, batch_size)

    # "Already tagged?" is a question about `attributes` jsonb, so it is answered
    # here rather than as a PostgREST predicate. The previous form asked it of
    # `material_type` and `tags` columns, which do not exist (#26 M13-1) — and
    # filtered on `image_url`, also absent, for an image the handler never sends.
    if force_rewrite:
        products = candidates[:batch_size]
    else:
        products = [
            p for p in candidates
            if not (p.get("attributes") or {}).get("material_type")
            or not (p.get("attributes") or {}).get("tags")
        ][:batch_size]

    if not products:
        return {"output": {"tagged": 0, "message": "No products to tag"},
                "input_tokens": 0, "output_tokens": 0}

    system = req.system_prompt or (
        "You are a material classification expert. "
        "Return JSON: {\"material_type\":\"...\",\"color\":\"...\",\"finish\":\"...\",\"application\":\"...\",\"tags\":[...]}. "
        "Only include applicable fields. No markdown."
    )

    tagged = 0
    degraded = 0
    total_in = total_out = 0

    for i, product in enumerate(products):
        if i % 5 == 0:
            _heartbeat(supabase, req.run_id)

        try:
            from app.services.core.claude_tool_call import call_with_tool
            result = await call_with_tool(
                task="agent_material_tagger",
                model=req.model or "claude-haiku-4-5",
                max_tokens=256,
                system=system,
                messages=[{"role": "user", "content":
                    f"Name: {product['name']}\n"
                    f"Category: {product.get('category','unknown')}\n"
                    f"Description: {product.get('description','(none)')}"}],
                tool=TAGGER_TOOL,
                job_id=req.run_id,
                workspace_id=workspace_id,
                product_id=product["id"],
            )
            total_in  += result.input_tokens
            total_out += result.output_tokens
            data = result.data

            proposed = _allowlisted(data, TAGGER_FACETS)
            if not proposed:
                continue

            attr_update = await _canonicalize_into_attributes(
                supabase, product, proposed,
                source="agent_material_tagger", workspace_id=workspace_id,
            )
            if attr_update is None:
                degraded += 1
                _log(supabase, req.run_id, "warn",
                     f"Facet canonicalization degraded for product {product['id']}; "
                     "attributes left unchanged rather than blanked")
                continue

            (supabase.table("products")
             .update(attr_update)
             .eq("id", product["id"])
             .eq("workspace_id", workspace_id)
             .execute())
            tagged += 1

        except Exception as e:
            _log(supabase, req.run_id, "warn",
                 f"Failed to tag product {product['id']}: {e}")

    _log(supabase, req.run_id, "info",
         f"Tagging complete: {tagged}/{len(products)}",
         {"tagged": tagged, "total": len(products), "degraded": degraded})

    return {
        "output": {"tagged": tagged, "total": len(products), "degraded": degraded},
        "input_tokens": total_in,
        "output_tokens": total_out,
    }
