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


def clear_cache() -> None:
    """Drop the cache — for tests and for the admin save path."""
    _CACHE.clear()
