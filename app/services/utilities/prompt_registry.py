"""
The one way to get a prompt (#347 phase 3P).

Every prompt this platform sends to a model comes from the `prompts` table. There is NO code
fallback — not a DEFAULT_ constant, not an `or "..."`, not an `except: return HARDCODED`.

WHY THAT RULE EXISTS
--------------------
A hardcoded fallback is invisible when it fires. `segmentation_service` read the DB, caught every
exception, logged at DEBUG, and returned a 9,119-character constant. An admin editing the
segmentation prompt in the UI would have seen "saved" and changed nothing, forever, with the
system reporting perfect health. The prompt in the database and the prompt being sent to the model
were different documents and nothing anywhere could tell you that.

TWO FAILURES, TWO EXCEPTIONS
----------------------------
The six loaders this replaces all collapsed both failures into `None` or `[]`:

    except Exception:
        return None          # was the prompt missing, or is Postgres down?

The caller cannot tell, so it cannot react correctly to either. They are separated here:

  * `PromptNotConfigured`   — the query succeeded and there is no such row. Somebody has to add
                              it in /admin/ai-configs. Retrying will not help.
  * `PromptStoreUnavailable` — the query itself failed. The prompt may be perfectly fine.
                              Retrying might help; running without it must not.

RESOLUTION ORDER
----------------
Most specific wins, and every step requires `is_active`:

  1. workspace + custom      (is_custom = true)   — a tenant's own override
  2. workspace + default     (is_custom = false)
  3. the global workspace                          — the seeded platform default
  4. category = 'default' for this stage           — only when a stage was given

`is_active` was NOT filtered by `dynamic_metadata_extractor._load_prompt_from_database`, so
deactivating a prompt in the admin UI did nothing at all to what the pipeline sent.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional, Tuple

from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

#: Prompts shipped with the platform live under this workspace.
GLOBAL_WORKSPACE = "00000000-0000-0000-0000-000000000000"

#: Cache lifetime. An admin edit takes effect within this window.
CACHE_TTL_SECONDS = 300



# ── Workspace scope ──────────────────────────────────────────────────────────────────────────
#
# A prompt row lives EITHER under a tenant workspace (a customisation) or under the global
# workspace (the platform default). Every loader in this service used to filter
# `.eq('workspace_id', workspace_id)` and stop there, so a platform default was invisible to
# every tenant and each workspace needed its own copy of all ~30 pipeline prompts.
#
# That is not a theoretical gap: 32 defaults were sitting under one tenant, and moving them to
# the global workspace — the correct place for a default — broke PDF ingestion outright, because
# no loader looked there. The move is only safe once every loader resolves like
# `prompt_registry.load_prompt` does: workspace first, then global.


def workspace_scope(workspace_id: Optional[str]) -> list:
    """The workspace ids a prompt lookup may match, for `.in_('workspace_id', ...)`."""
    if not workspace_id or workspace_id == GLOBAL_WORKSPACE:
        return [GLOBAL_WORKSPACE]
    return [workspace_id, GLOBAL_WORKSPACE]


def prefer_workspace(rows: list, workspace_id: Optional[str]) -> Optional[dict]:
    """Pick the tenant's own row when it exists, else the platform default.

    PostgREST cannot order by "this workspace first" in the fluent API, so the choice is made
    here rather than by hoping the database returns them in a useful order.
    """
    if not rows:
        return None
    if workspace_id:
        for row in rows:
            if row.get("workspace_id") == workspace_id:
                return row
    for row in rows:
        if row.get("workspace_id") == GLOBAL_WORKSPACE:
            return row
    return rows[0]

class PromptError(RuntimeError):
    """Base for both prompt failures, so a caller may catch either deliberately."""


class PromptNotConfigured(PromptError):
    """The store answered and has no such prompt. Add it; do not retry."""


class PromptStoreUnavailable(PromptError):
    """The store could not be reached. The prompt may be fine; the database is not."""


_CACHE: Dict[str, Tuple[str, float]] = {}
_LOCK = asyncio.Lock()


def _key(prompt_type: str, category: str, stage: Optional[str],
         workspace_id: Optional[str], subcategory: Optional[str]) -> str:
    return "|".join([prompt_type, category, stage or "-", workspace_id or "-", subcategory or "-"])


def _select(client: Any, prompt_type: str, category: str, stage: Optional[str],
            workspace_id: str, subcategory: Optional[str], is_custom: Optional[bool]):
    query = (
        client.table("prompts")
        .select("prompt_text, version, name, is_custom")
        .eq("prompt_type", prompt_type)
        .eq("category", category)
        .eq("workspace_id", workspace_id)
        .eq("is_active", True)
    )
    if stage is not None:
        query = query.eq("stage", stage)
    if subcategory is not None:
        query = query.eq("subcategory", subcategory)
    if is_custom is not None:
        query = query.eq("is_custom", is_custom)
    return query.order("version", desc=True).limit(1).execute()


def _resolve_blocking(prompt_type: str, category: str, stage: Optional[str],
                      workspace_id: Optional[str], subcategory: Optional[str]) -> str:
    client = get_supabase_client().client
    ws = workspace_id or GLOBAL_WORKSPACE

    attempts = [
        (ws, category, True, "workspace custom"),
        (ws, category, False, "workspace default"),
    ]
    if ws != GLOBAL_WORKSPACE:
        attempts.append((GLOBAL_WORKSPACE, category, None, "platform default"))
    if stage is not None and category != "default":
        attempts.append((ws, "default", None, f"stage '{stage}' fallback category"))
        if ws != GLOBAL_WORKSPACE:
            attempts.append((GLOBAL_WORKSPACE, "default", None, "platform stage fallback"))

    try:
        for target_ws, target_cat, custom, label in attempts:
            result = _select(client, prompt_type, target_cat, stage, target_ws, subcategory, custom)
            rows = result.data or []
            if rows and (rows[0].get("prompt_text") or "").strip():
                logger.info("Prompt %s/%s/%s resolved via %s (v%s)",
                            prompt_type, stage or "-", target_cat, label, rows[0].get("version"))
                return rows[0]["prompt_text"]
    except Exception as exc:  # noqa: BLE001 — re-raised as a typed error below
        # NOT swallowed into "not found". A reachability failure and a missing row need
        # different reactions, and conflating them is what let an outage look like a
        # misconfiguration for as long as nobody looked.
        raise PromptStoreUnavailable(
            f"Could not read the prompts table for {prompt_type}/{stage or '-'}/{category}: {exc}"
        ) from exc

    raise PromptNotConfigured(
        f"No active prompt for prompt_type='{prompt_type}', stage='{stage or '-'}', "
        f"category='{category}' (workspace {ws}). Add it in /admin/ai-configs. "
        f"There is no hardcoded fallback by design — see app/services/utilities/prompt_registry.py."
    )


async def load_prompt(
    prompt_type: str,
    category: str,
    *,
    stage: Optional[str] = None,
    workspace_id: Optional[str] = None,
    subcategory: Optional[str] = None,
    use_cache: bool = True,
) -> str:
    """Return the prompt text, or raise. Never returns a default.

    Raises:
        PromptNotConfigured: no active row matches.
        PromptStoreUnavailable: the prompts table could not be read.
    """
    cache_key = _key(prompt_type, category, stage, workspace_id, subcategory)
    if use_cache:
        hit = _CACHE.get(cache_key)
        if hit and (time.monotonic() - hit[1]) < CACHE_TTL_SECONDS:
            return hit[0]

    async with _LOCK:
        if use_cache:
            hit = _CACHE.get(cache_key)
            if hit and (time.monotonic() - hit[1]) < CACHE_TTL_SECONDS:
                return hit[0]
        text = await asyncio.to_thread(
            _resolve_blocking, prompt_type, category, stage, workspace_id, subcategory
        )
        _CACHE[cache_key] = (text, time.monotonic())
        return text


def get_cached(prompt_type: str, category: str, *, stage: Optional[str] = None,
               workspace_id: Optional[str] = None, subcategory: Optional[str] = None) -> str:
    """Synchronous read of an already-loaded prompt.

    Some prompt sites are sync — `_build_spec_prompt`, the legend PROMPTS_BY_TYPE map — and one
    was evaluated at IMPORT time, which can neither await nor reach a database. They call this
    after an async entry point has called `prefetch`.

    It RAISES rather than returning a default, for the same reason the field registry does: a
    prompt site that quietly degrades is exactly the failure this phase exists to delete.
    """
    hit = _CACHE.get(_key(prompt_type, category, stage, workspace_id, subcategory))
    if hit is None:
        raise PromptNotConfigured(
            f"Prompt {prompt_type}/{stage or '-'}/{category} read synchronously before it was "
            f"loaded. Call `await prefetch(...)` at this path's entry point — sync sites cannot "
            f"reach the database and must not invent a default."
        )
    return hit[0]


async def prefetch(*keys: Tuple[str, ...]) -> None:
    """Warm the cache for sync sites. Each key is (prompt_type, category[, stage])."""
    for key in keys:
        prompt_type, category = key[0], key[1]
        stage = key[2] if len(key) > 2 else None
        await load_prompt(prompt_type, category, stage=stage)


def render(template: str, **values: Any) -> str:
    """Substitute `{name}` placeholders, failing loudly if one is not in the template.

    Prompts hold JSON examples full of literal braces, so "are there braces left?" is not a
    usable check. What IS checkable — and what actually breaks — is a caller passing a value
    the template no longer has a slot for. An admin renaming `{product_name}` to `{product}`
    would otherwise ship a prompt with an unfilled hole and a value silently dropped, and the
    model would answer confidently about a product it was never told the name of.
    """
    missing = [name for name in values if "{" + name + "}" not in template]
    if missing:
        raise PromptNotConfigured(
            f"Prompt template has no placeholder for {missing}. Either the caller is passing a "
            f"value the prompt no longer uses, or a placeholder was renamed in /admin/ai-configs "
            f"without updating the caller."
        )
    for name, value in values.items():
        template = template.replace("{" + name + "}", "" if value is None else str(value))
    return template


#: Every prompt key this service loads with a literal (prompt_type, category, stage).
#:
#: No-fallback means a missing row stops the work cold — correct, but it must be discovered at
#: deploy time, not at 2am halfway through a catalog. `check_required_prompts()` verifies these
#: exist; /health reports the result (#347 phase 3P.6).
#:
#: This list is DECLARED, not derived, because a running service cannot AST-scan itself. It is
#: kept honest by tests/unit/test_prompts_come_from_the_database.py, which walks every
#: `load_prompt(...)` / `get_cached(...)` call in app/ and fails when one is missing here — so
#: the declaration cannot drift from the code the way a hand-kept list normally would.
REQUIRED_PROMPTS: Tuple[Tuple[str, str, Optional[str]], ...] = (
    ("agent", "segmentation", None),
    ("classification", "chunk_scope", "chunking"),
    # Added by phase 4.1/4.3 after this list was written. The guard test found the drift
    # on its first run, which is the point of deriving it from the call sites.
    ("classification", "field_role", "metadata_extraction"),
    ("classification", "chunk_type", "chunking"),
    ("classification", "document_page", "discovery"),
    ("classification", "product_classification", "entity_creation"),
    ("extraction", "anthropic_chunk_analysis", "entity_creation"),
    ("extraction", "anthropic_image_analysis", "image_analysis"),
    ("extraction", "catalog_knowledge", "discovery"),
    ("extraction", "fast_product_classifier_system", "discovery"),
    ("extraction", "icon_metadata", "image_analysis"),
    ("extraction", "legend_care", "discovery"),
    ("extraction", "legend_certifications", "discovery"),
    ("extraction", "legend_icons", "discovery"),
    ("extraction", "legend_installation", "discovery"),
    ("extraction", "legend_regulations", "discovery"),
    ("extraction", "legend_sustainability", "discovery"),
    ("extraction", "material_properties_system", "entity_creation"),
    ("extraction", "product_deep_analysis", "entity_creation"),
    ("extraction", "product_name_scope", "entity_creation"),
    ("extraction", "product_spec_vision", "image_analysis"),
    ("extraction", "rag_vision_analysis", "image_analysis"),
    ("extraction", "sentiment", None),
    ("generation", "product_description", None),
    ("search", "query_parser_system", None),
    ("search", "query_structuring", None),
    ("tool", "price_monitor_facets", None),
    ("tool", "price_monitor_match", None),
)


async def check_required_prompts() -> Dict[str, Any]:
    """Verify every REQUIRED_PROMPTS key resolves. Never raises — this is a health probe.

    Returns {status, required, missing, unavailable}. `missing` is a configuration problem
    someone must fix in /admin/ai-configs; `unavailable` means the store could not be read at
    all, which is a different alarm and must not be reported as 26 missing prompts.
    """
    missing: list = []
    unavailable: Optional[str] = None

    for prompt_type, category, stage in REQUIRED_PROMPTS:
        try:
            await load_prompt(prompt_type, category, stage=stage)
        except PromptNotConfigured:
            missing.append(f"{prompt_type}/{stage or '-'}/{category}")
        except PromptStoreUnavailable as exc:
            unavailable = str(exc)
            break
        except Exception as exc:  # noqa: BLE001 — a probe must not take the service down
            unavailable = f"unexpected error checking prompts: {exc}"
            break

    if unavailable:
        status = "unknown"
    elif missing:
        status = "unhealthy"
    else:
        status = "healthy"

    return {
        "status": status,
        "required": len(REQUIRED_PROMPTS),
        "missing": missing,
        "unavailable": unavailable,
    }


def clear_cache() -> None:
    """Drop the cache — for tests and for the admin save path."""
    _CACHE.clear()
