"""
The one loader for the `Material Image Analyzer` prompt.

WHY THIS EXISTS
---------------
`vision_analysis.py` says its call paths MUST stay aligned or
`vecs.image_understanding_embeddings` drifts. The schema and the serialiser were
already shared. The PROMPT was not.

Ingestion loaded `Material Image Analyzer` from the `prompts` table. The
understanding backfill — whose entire job is to make stale rows match current ones —
sent its own hardcoded sentence instead. Same tool, same schema, same collection,
different instructions, so a backfilled image and a freshly ingested one described
the same material in two regimes and nothing raised: both rows are well-formed and
both vectors rank.

Loaded once per process and cached. It is a small row read on a path that runs per
image, and the prompt does not change mid-job; an admin edit is picked up on the next
restart, same as the ingestion service's own cached copy.

There is no fallback. `PromptNotConfigured` means "add the row" and is deliberately
distinct from a DB outage — a prompt site that quietly degrades is the failure this
whole approach exists to delete.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

PROMPT_NAME = "Material Image Analyzer"

_cache: Optional[Tuple[str, Optional[str]]] = None


class MaterialAnalyzerPromptMissing(RuntimeError):
    """The `Material Image Analyzer` row is absent or inactive."""


def load_material_analyzer_prompt(*, refresh: bool = False) -> Tuple[str, Optional[str]]:
    """Return `(prompt_text, system_prompt)` for the vision analysis call.

    Raises `MaterialAnalyzerPromptMissing` rather than returning a default. A vision
    call that runs on a fallback prompt writes a vector built from instructions
    nobody chose, into a collection everything else queries.
    """
    global _cache
    if _cache is not None and not refresh:
        return _cache

    from app.services.core.supabase_client import get_supabase_client

    result = (
        get_supabase_client().client.table("prompts")
        .select("prompt_text, system_prompt")
        .eq("name", PROMPT_NAME)
        .eq("is_active", True)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    if not rows or not rows[0].get("prompt_text"):
        raise MaterialAnalyzerPromptMissing(
            f"prompts row {PROMPT_NAME!r} is missing or inactive — vision analysis "
            f"cannot run without it, and running it on a hardcoded default would put "
            f"a differently-described vector into image_understanding_embeddings."
        )

    _cache = (rows[0]["prompt_text"], rows[0].get("system_prompt"))
    logger.info(f"✅ Loaded {PROMPT_NAME} prompt from database")
    return _cache
