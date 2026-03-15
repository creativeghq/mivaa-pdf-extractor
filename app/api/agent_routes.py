"""
Background Agent API Routes

Handles long-running agent tasks delegated from Supabase edge functions.
The edge function fires-and-forgets here when a task exceeds the 25s timeout.

Endpoints:
  POST /api/agents/run       - Receive delegated agent task, run in background
  GET  /api/agents/runs/{id} - Get status of a specific run
  GET  /api/agents/catalog   - List all available agent types
"""

import logging
import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import anthropic
from fastapi import APIRouter, BackgroundTasks, HTTPException, Header
from pydantic import BaseModel, Field

from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["background-agents"])


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

@router.get("/catalog")
async def get_catalog():
    return {"catalog": AGENT_CATALOG}


@router.get("/runs/{run_id}")
async def get_run_status(run_id: str):
    supabase = get_supabase_client()
    result = supabase.table("agent_runs").select("*").eq("id", run_id).single().execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Run not found")
    return result.data


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
    # Basic auth check — expect the MIVAA API key
    expected_key = os.environ.get("MIVAA_API_KEY", "")
    if expected_key and authorization != f"Bearer {expected_key}":
        raise HTTPException(status_code=401, detail="Unauthorized")

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


# ── Background execution ─────────────────────────────────────────────────────

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

        result = await handler(req, supabase)

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

def _heartbeat(supabase, run_id: str) -> None:
    supabase.table("agent_runs").update({
        "last_heartbeat": datetime.now(timezone.utc).isoformat()
    }).eq("id", run_id).execute()


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

async def handle_product_enrichment(req: AgentRunRequest, supabase) -> Dict[str, Any]:
    """Enrich products with AI descriptions, keywords, and category tags."""
    cfg         = {**req.config, **req.input_data}
    batch_size  = min(int(cfg.get("batch_size", 20)), 200)
    cat_filter  = cfg.get("category_filter")
    force_rewrite = bool(cfg.get("force_rewrite", False))

    _log(supabase, req.run_id, "info",
         f"Product enrichment started (Python backend)",
         {"batch_size": batch_size, "category_filter": cat_filter})

    # Fetch products
    query = (supabase.table("products")
             .select("id, name, description, category, tags, material_type")
             .order("created_at")
             .limit(batch_size))
    if not force_rewrite:
        query = query.is_("description", "null")
    if cat_filter:
        query = query.eq("category", cat_filter)

    result = query.execute()
    products = result.data or []

    if not products:
        return {"output": {"enriched": 0, "message": "No products to enrich"},
                "input_tokens": 0, "output_tokens": 0}

    _log(supabase, req.run_id, "info", f"Found {len(products)} products to enrich")

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    system = req.system_prompt or (
        "You are a material product specialist. "
        "For each product, return JSON: {\"description\":\"...\",\"keywords\":[...],\"material_category\":\"...\"}. "
        "No markdown, no prose."
    )

    enriched = 0
    total_in = total_out = 0

    for i, product in enumerate(products):
        if i % 5 == 0:
            _heartbeat(supabase, req.run_id)

        try:
            msg = client.messages.create(
                model=req.model or "claude-haiku-4-5",
                max_tokens=512,
                system=system,
                messages=[{"role": "user", "content":
                    f"Name: {product['name']}\n"
                    f"Category: {product.get('category','unknown')}\n"
                    f"Description: {product.get('description','(none)')}\n"
                    f"Material: {product.get('material_type','unknown')}"}],
            )
            total_in  += msg.usage.input_tokens
            total_out += msg.usage.output_tokens

            import json
            text = msg.content[0].text.strip().lstrip("```json").rstrip("```").strip()
            data = json.loads(text)

            update: Dict[str, Any] = {}
            if data.get("description"):       update["description"]      = data["description"]
            if data.get("keywords"):          update["search_keywords"]  = data["keywords"]
            if data.get("material_category"): update["material_type"]    = data["material_category"]

            if update:
                supabase.table("products").update(update).eq("id", product["id"]).execute()
                enriched += 1

        except Exception as e:
            _log(supabase, req.run_id, "warn",
                 f"Failed to enrich product {product['id']}: {e}")

    _log(supabase, req.run_id, "info",
         f"Enrichment complete: {enriched}/{len(products)}",
         {"enriched": enriched, "total": len(products)})

    return {
        "output": {"enriched": enriched, "total": len(products)},
        "input_tokens": total_in,
        "output_tokens": total_out,
    }


async def handle_material_tagger(req: AgentRunRequest, supabase) -> Dict[str, Any]:
    """Auto-tag materials with type, color, finish, and application."""
    cfg        = {**req.config, **req.input_data}
    batch_size = min(int(cfg.get("batch_size", 20)), 200)

    _log(supabase, req.run_id, "info",
         f"Material tagging started (Python backend)", {"batch_size": batch_size})

    result = (supabase.table("products")
              .select("id, name, description, material_type, tags, image_url, category")
              .not_.is_("image_url", "null")
              .or_("material_type.is.null,tags.eq.{}")
              .order("created_at")
              .limit(batch_size)
              .execute())
    products = result.data or []

    if not products:
        return {"output": {"tagged": 0, "message": "No products to tag"},
                "input_tokens": 0, "output_tokens": 0}

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    system = req.system_prompt or (
        "You are a material classification expert. "
        "Return JSON: {\"material_type\":\"...\",\"color\":\"...\",\"finish\":\"...\",\"application\":\"...\",\"tags\":[...]}. "
        "Only include applicable fields. No markdown."
    )

    tagged = 0
    total_in = total_out = 0

    for i, product in enumerate(products):
        if i % 5 == 0:
            _heartbeat(supabase, req.run_id)

        try:
            msg = client.messages.create(
                model=req.model or "claude-haiku-4-5",
                max_tokens=256,
                system=system,
                messages=[{"role": "user", "content":
                    f"Name: {product['name']}\n"
                    f"Category: {product.get('category','unknown')}\n"
                    f"Description: {product.get('description','(none)')}"}],
            )
            total_in  += msg.usage.input_tokens
            total_out += msg.usage.output_tokens

            import json
            text = msg.content[0].text.strip().lstrip("```json").rstrip("```").strip()
            data = json.loads(text)

            update: Dict[str, Any] = {}
            for field in ("material_type", "color", "finish", "application"):
                if data.get(field):
                    update[field] = data[field]
            if data.get("tags"):
                update["tags"] = data["tags"]

            if update:
                supabase.table("products").update(update).eq("id", product["id"]).execute()
                tagged += 1

        except Exception as e:
            _log(supabase, req.run_id, "warn",
                 f"Failed to tag product {product['id']}: {e}")

    _log(supabase, req.run_id, "info",
         f"Tagging complete: {tagged}/{len(products)}")

    return {
        "output": {"tagged": tagged, "total": len(products)},
        "input_tokens": total_in,
        "output_tokens": total_out,
    }
