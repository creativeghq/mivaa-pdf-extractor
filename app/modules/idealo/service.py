"""
Idealo discovery service.

Module gating + locale routing + adapter dispatch. Mirrors the shape of
`app.modules.greek_marketplaces.service`:

    svc = get_idealo_service()
    hits = await svc.search(query="Hansgrohe Talis E", country_code="DE")

Returns a list of PriceHit (the same type the perplexity service uses) so
the orchestrator can merge them with the existing two-source pipeline.

Locale → site mapping:
  DE / AT → idealo.de
  IT → idealo.it
  UK / GB → idealo.co.uk
  ES → idealo.es
  FR → idealo.fr

The country_code argument is the user's market. If we don't have a locale
for it we return an empty list (no fallback to .de — that would surface
out-of-stock-in-country results). The greek_marketplaces module has the
same policy.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from app.modules._core.registry import is_module_enabled
from app.modules.idealo.adapters.idealo_search import scrape_idealo_search
from app.services.integrations.perplexity_price_search_service import PriceHit

logger = logging.getLogger(__name__)

MODULE_SLUG = "idealo"

_LOCALE_HOST = {
    "DE": "www.idealo.de",
    "AT": "www.idealo.de",   # idealo.at is alias of .de in practice
    "IT": "www.idealo.it",
    "UK": "www.idealo.co.uk",
    "GB": "www.idealo.co.uk",
    "ES": "www.idealo.es",
    "FR": "www.idealo.fr",
}


class IdealoService:
    """Stateless. Holds no state — every call is independent."""

    async def search(
        self,
        *,
        query: str,
        country_code: Optional[str],
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        limit: int = 5,
    ) -> List[PriceHit]:
        if not query:
            return []
        if not is_module_enabled(MODULE_SLUG):
            return []
        host = _LOCALE_HOST.get((country_code or "").upper())
        if not host:
            return []
        try:
            return await scrape_idealo_search(
                host=host,
                query=query,
                user_id=user_id,
                workspace_id=workspace_id,
                limit=limit,
            )
        except Exception as e:
            logger.warning(f"idealo: search failed for '{query}' on {host}: {e}")
            return []


_service: Optional[IdealoService] = None


def get_idealo_service() -> IdealoService:
    global _service
    if _service is None:
        _service = IdealoService()
    return _service
