"""
Query understanding for /api/rag/search.

What remains of the "Unified Search Service" is the part a route actually calls:
`UnifiedSearchService._parse_query_with_ai` (Claude Haiku 4.5 parses a natural-language
material query into structured filters and picks a 7-vector weight profile; cached in
`query_understanding_cache`) and `_select_weight_profile`.

The multi-strategy `search()` machinery that used to live here was deleted on
2026-09-05. No route reached it — rag_routes only ever called `_parse_query_with_ai` —
and what it did was wrong in ways no test could see: `_search_semantic` read the FIRST
20 rows of document_chunks with no ORDER BY and ranked those in numpy (an arbitrary
sample, not a nearest-neighbour search), and `_search_hybrid` summed a cosine
similarity and a ts_rank as if they shared a scale. Product search is
`RAGService.multi_vector_search`; KB search is the `kb_hybrid_doc_chunks` RPC.
"""

import logging
from typing import Dict, Any, Optional, Tuple

from app.services.utilities.prompt_registry import load_prompt

logger = logging.getLogger(__name__)


# Dynamic weight profiles keyed by query intent type (all sum to 1.0).
# Defined ONCE in weight_profiles.py — see that module for why. Re-exported here so
# the existing `from ...unified_search_service import WEIGHT_PROFILES` callers keep
# working; new code should import from weight_profiles directly.
from app.services.search.weight_profiles import (  # noqa: E402
    WEIGHT_PROFILES,
)


class UnifiedSearchService:
    """Query understanding: parse a material search query into filters + a weight profile.

    Instantiated per request by /api/rag/search (`UnifiedSearchService()`), which then
    hands the parsed query and weights to `RAGService.multi_vector_search`.
    """

    def __init__(self, supabase_client=None):
        """`supabase_client` feeds query-term canonicalisation (resolve_query_term)."""
        self.supabase = supabase_client
        self.logger = logger
        # Diagnostics for query understanding (used by callers for observability)
        self._last_query_understanding_was_cache_hit: bool = False
        self._last_query_understanding_ms: int = 0

    def _select_weight_profile(self, parsed_data: Dict[str, Any]) -> Tuple[str, Dict[str, float]]:
        """
        Select weight profile based on parsed query fields from query understanding.

        Maps detected query intent (colors, finish, pattern, etc.) to a weight profile
        that upweights the most relevant embedding types for that query.

        Returns:
            Tuple of (profile_name, weights_dict)
        """
        # Product name search → heavy text weight
        if parsed_data.get("is_product_name") or parsed_data.get("product_name"):
            return "product_name", WEIGHT_PROFILES["product_name"]

        has_colors = bool(parsed_data.get("colors"))
        has_finish = bool(parsed_data.get("finish"))
        has_pattern = bool(parsed_data.get("pattern"))
        has_style = bool(parsed_data.get("style"))
        has_dimensions = bool(parsed_data.get("dimensions"))
        has_material = parsed_data.get("material_type_explicit", False)
        has_application = bool(parsed_data.get("application"))

        # Priority-based selection (strongest signal wins)
        if has_dimensions:
            return "specification", WEIGHT_PROFILES["specification"]
        if has_colors or has_finish:
            return "color_finish", WEIGHT_PROFILES["color_finish"]
        if has_pattern:
            return "texture_pattern", WEIGHT_PROFILES["texture_pattern"]
        if has_material:
            return "material_search", WEIGHT_PROFILES["material_search"]
        if has_style or has_application:
            return "style_aesthetic", WEIGHT_PROFILES["style_aesthetic"]

        return "balanced", WEIGHT_PROFILES["balanced"]

    async def _parse_query_with_haiku(
        self,
        query: str,
        system_prompt: str,
        workspace_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Parse a search query into structured filters using Anthropic Claude
        Haiku 4.5 (fast, cheap, structured-output friendly). Returns the
        parsed dict or raises on failure (caller falls back to GPT-4o-mini).

        Query parsing runs on Claude —
        removed from the platform but this method retained the legacy
        name + log strings, which were misleading to operators reading
        the search-side logs.

        M3-13 (#16): this was a bare `httpx.post` to the Anthropic messages
        endpoint — no debit, no cost log, no attribution at all, so every
        query-understanding call was free as far as the platform could tell.
        It now goes through `tracked_claude_call_async` (pipeline convention
        10), which logs and debits automatically.
        """
        from app.services.core.claude_tool_call import call_with_tool

        # Forced (#32). The fence-stripping this replaces handled ```json, bare ``` and a
        # trailing fence — three separate repairs, which is a good measure of how often
        # the free-form contract was not holding.
        #
        # The schema is deliberately open: the filter shape is described inside
        # `system_prompt`, which is loaded from the database and edited by admins.
        # Restating those keys here would create a second source, and because the model
        # is FORCED to satisfy the schema, an admin's edit would silently stop taking
        # effect. What forcing buys even with an open schema is the whole failure mode —
        # no prose to repair, and an absent tool block raises `ToolCallNotReturned`
        # instead of a JSONDecodeError three lines down.
        #
        # It still RAISES on failure, which is the contract this method already had: the
        # caller falls back to the other model.
        call = await call_with_tool(
            task="search_query_understanding",
            model="claude-haiku-4-5",
            max_tokens=1024,
            system=system_prompt,
            tool={
                "name": "emit_parsed_query",
                "description": (
                    "Emit the parsed search query as the JSON object described in the "
                    "instructions."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                },
            },
            messages=[
                {"role": "user", "content": f"Parse this query: {query}"},
            ],
            workspace_id=workspace_id,
        )
        return call.data

    async def _parse_query_with_ai(self, query: str, workspace_id: Optional[str] = None) -> tuple[str, Dict[str, Any], str, Dict[str, float]]:
        """
        🧠 Parse natural language query using Claude Haiku 4.5 to extract
        structured filters and select dynamic weight profile.

        This is the PREPROCESSING step that runs BEFORE multi-strategy search.

        Cached: results are stored in `query_understanding_cache` keyed on the
        normalised query hash. Cache hits skip the LLM call entirely.

        Args:
            query: Natural language query (e.g., "waterproof ceramic tiles for outdoor patio, matte finish, light beige")

        Returns:
            Tuple of (visual_query, filters, weight_profile, dynamic_weights):
            - visual_query: Core visual concept for embedding (e.g., "ceramic tiles matte")
            - filters: Extracted structured filters (e.g., {"material_type": "ceramic tiles", ...})
            - weight_profile: Name of selected weight profile (e.g., "color_finish")
            - dynamic_weights: Dict of 7-vector weights for multi-vector fusion

        Side effect: sets `self._last_query_understanding_was_cache_hit` and
        `self._last_query_understanding_ms` so callers can record timing.
        """
        import time
        start_time = time.time()

        # Reset diagnostics
        self._last_query_understanding_was_cache_hit = False
        self._last_query_understanding_ms = 0

        # ── Cache lookup (skip the LLM if we've parsed this query before) ──
        try:
            from app.services.search.query_understanding_cache import get_query_understanding_cache
            cache = get_query_understanding_cache()
            cached = await cache.lookup(query)
            if cached:
                self._last_query_understanding_was_cache_hit = True
                self._last_query_understanding_ms = int((time.time() - start_time) * 1000)
                return (
                    cached.get("visual_query") or query,
                    cached.get("filters") or {},
                    cached.get("weight_profile") or "balanced",
                    cached.get("dynamic_weights") or WEIGHT_PROFILES["balanced"],
                )
        except Exception as cache_err:
            self.logger.debug(f"Query cache lookup failed (continuing): {cache_err}")

        try:

            system_prompt = await load_prompt("search", "query_parser_system")

            # Primary: Anthropic Haiku 4.5 (cheap, structured-output friendly).
            # The variable retains its name for compatibility with downstream
            # logs / cache writes, but the actual model is `claude-haiku-4-5`.
            parsed_data = None
            model_used = "claude-haiku-4-5"
            try:
                parsed_data = await self._parse_query_with_haiku(
                    query, system_prompt, workspace_id=workspace_id
                )
                self.logger.debug("🤖 Query parsed with Haiku 4.5")
            except Exception as parse_err:
                self.logger.debug(f"Primary query parse failed ({parse_err}), falling back to Claude Haiku")

            # Fallback: Claude Haiku 4.5.
            #
            # M3-13 (#16): this used to call the SDK client directly and then
            # hand-roll a log_claude_call AFTER json.loads succeeded — so a
            # parse failure threw past the logging and the call was billed by
            # Anthropic but recorded nowhere, and neither user_id nor
            # workspace_id was ever passed. tracked_claude_call_async logs and
            # debits around the call itself, which is why the parse can now sit
            # outside it.
            if parsed_data is None:
                model_used = "claude-haiku-4-5"
                # Forced tool (#32). This site is the clearest statement of the problem
                # in the codebase: the system prompt ENDED with "Respond with ONLY a
                # JSON object, no markdown fences, no explanation" — and the code below
                # it stripped markdown fences anyway. The instruction and the repair are
                # the same admission, written twice.
                #
                # Both are gone. The model cannot return prose, so there is nothing to
                # instruct against and nothing to strip. The schema stays OPEN because
                # the filter shape is described in `system_prompt`, which is loaded from
                # the database — same reasoning as the sibling `_parse_query_with_haiku`.
                from app.services.core.claude_tool_call import (
                    ToolCallNotReturned,
                    call_with_tool,
                )

                _QUERY_TOOL = {
                    "name": "emit_parsed_query",
                    "description": (
                        "Emit the parsed search query as the JSON object described in "
                        "the instructions."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": True,
                    },
                }

                try:
                    call = await call_with_tool(
                        task="search_query_understanding_fallback",
                        model="claude-haiku-4-5",
                        max_tokens=1024,
                        system=system_prompt,
                        tool=_QUERY_TOOL,
                        messages=[
                            {"role": "user", "content": f"Parse this query: {query}"},
                        ],
                        workspace_id=workspace_id,
                    )
                except ToolCallNotReturned as e:
                    # This IS the fallback path — the other model already failed — so
                    # there is nothing further to fall back to. Raising lets the outer
                    # handler return the unparsed query, which is the honest answer.
                    logger.warning(f"query-understanding fallback returned no tool call: {e}")
                    raise
                parsed_data = call.data

            # Select dynamic weight profile based on parsed query intent
            profile_name, dynamic_weights = self._select_weight_profile(parsed_data)

            parse_latency_ms = int((time.time() - start_time) * 1000)
            self._last_query_understanding_ms = parse_latency_ms

            # Check if this is a product name search
            if parsed_data.get("is_product_name") or parsed_data.get("product_name"):
                # For product name searches, return the original query unchanged
                self.logger.info(f"🧠 Query identified as PRODUCT NAME: '{query}' → profile='{profile_name}'")
                # Store in cache (fire-and-forget)
                try:
                    from app.services.search.query_understanding_cache import get_query_understanding_cache
                    await get_query_understanding_cache().store(
                        query=query,
                        parsed_data=parsed_data,
                        visual_query=query,
                        filters={},
                        weight_profile=profile_name,
                        dynamic_weights=dynamic_weights,
                        is_product_name=True,
                        model_used=model_used,
                        parse_latency_ms=parse_latency_ms,
                    )
                except Exception as store_err:
                    self.logger.debug(f"Cache store failed: {store_err}")
                return query, {}, profile_name, dynamic_weights

            # Build visual query (core concept for embedding)
            visual_parts = []
            if parsed_data.get("material_type"):
                visual_parts.append(parsed_data["material_type"])
            if parsed_data.get("style"):
                visual_parts.append(parsed_data["style"])
            if parsed_data.get("finish"):
                visual_parts.append(parsed_data["finish"])

            visual_query = " ".join(visual_parts) if visual_parts else query

            # Build filters dictionary (remove null values and visual_query).
            #
            # field_mapping targets `products.attributes.*` for descriptive facets
            # (color, material, finish, style, application, room, material_category)
            # — these are the canonical English values written by every ingest
            # path (PDF Stage 4, XML supplier feeds, web scrape, background agents).
            # Other fields (designer, collection, factory, dimensions) stay in
            # metadata because they're identifiers, not canonicalizable facets.
            field_mapping = {
                # Canonicalized facets — target attributes.*
                "colors": "attributes.color",
                "finish": "attributes.finish",
                "application": "attributes.application",
                "style": "attributes.style",
                "material_type": "attributes.material_category",
                # Non-canonical fields stay in metadata (identifiers / freeform)
                "pattern": "appearance.pattern",
                "designer": "design.designers",
                "collection": "design.collection",
                "properties": "material_properties",
                "factory": "factory_name",
                "dimensions": "dimensions",
            }

            # Which parsed_data fields go through query-side canonicalization
            # (each value gets normalized + alias-resolved against facet_canonical_values)
            CANONICALIZABLE_QUERY_FIELDS = {
                "colors": "color",
                "finish": "finish",
                "application": "application",
                "style": "style",
                "material_type": "material_category",
            }

            # Fields that are metadata about parsing, not actual filters
            skip_fields = {"is_product_name", "product_name", "visual_query", "material_type_explicit"}

            # ── Query-side canonicalization ──────────────────────────────────
            # Translate user-typed values (any language) → canonical English so
            # the filter matches what the ingest pipelines wrote to attributes.
            # No embedding cost — alias lookup only.
            try:
                from app.services.facets import resolve_query_term  # noqa: WPS433
                for parsed_key, facet_key in CANONICALIZABLE_QUERY_FIELDS.items():
                    val = parsed_data.get(parsed_key)
                    if val is None or val == "" or val == []:
                        continue
                    if isinstance(val, list):
                        translated = []
                        for v in val:
                            if not isinstance(v, str) or not v.strip():
                                continue
                            c = await resolve_query_term(self.supabase, facet_key, v, workspace_id=workspace_id)
                            if c:
                                translated.append(c)
                        if translated:
                            parsed_data[parsed_key] = translated
                    elif isinstance(val, str):
                        c = await resolve_query_term(self.supabase, facet_key, val, workspace_id=workspace_id)
                        if c:
                            parsed_data[parsed_key] = c
            except Exception as canon_err:
                self.logger.debug(f"Query-side canonicalization skipped: {canon_err}")

            filters = {}
            for key, value in parsed_data.items():
                if key in skip_fields:
                    continue

                # Special handling for material_type - only pass if EXPLICITLY stated
                if key == "material_type":
                    if not parsed_data.get("material_type_explicit", False):
                        # Category was inferred, not explicit - skip it
                        # e.g., "wood pattern" shouldn't filter to wood category
                        continue

                if value is not None and value != [] and value != "":
                    # Map to actual DB field name
                    db_key = field_mapping.get(key, key)

                    # Handle properties as array containment
                    if key == "properties" and isinstance(value, list):
                        filters[db_key] = {"contains": value}
                    else:
                        filters[db_key] = value

            self.logger.info(f"🧠 Query parsed [{model_used}]: '{query}' → visual_query='{visual_query}', profile='{profile_name}', filters={filters}")

            # Store in cache (fire-and-forget)
            try:
                from app.services.search.query_understanding_cache import get_query_understanding_cache
                await get_query_understanding_cache().store(
                    query=query,
                    parsed_data=parsed_data,
                    visual_query=visual_query,
                    filters=filters,
                    weight_profile=profile_name,
                    dynamic_weights=dynamic_weights,
                    is_product_name=False,
                    model_used=model_used,
                    parse_latency_ms=parse_latency_ms,
                )
            except Exception as store_err:
                self.logger.debug(f"Cache store failed: {store_err}")

            return visual_query, filters, profile_name, dynamic_weights

        except Exception as e:
            # Demoted to warning — query understanding is best-effort. The fallback
            # to the original query + balanced weights is the documented behavior
            # and search continues to work. Common cause: OpenAI rate limit / quota.
            self.logger.warning(f"Query parsing failed (using original query as fallback): {e}")
            self._last_query_understanding_ms = int((time.time() - start_time) * 1000)
            return query, {}, "balanced", WEIGHT_PROFILES["balanced"]


