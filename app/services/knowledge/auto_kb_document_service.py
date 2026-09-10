"""Auto KB Document Service — DEPRECATED / DISABLED (2026-06-22)."""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class AutoKBDocumentService:
    """Deprecated no-op. Per-product KB-doc generation was removed 2026-06-22.

    See the module docstring for where each kind of knowledge now lives.
    """

    async def create_kb_documents_from_metadata(
        self,
        product_id: str,
        product_name: str,
        workspace_id: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Intentionally creates nothing. Catalog-wide knowledge is generated
        # once per catalog by the catalog extractors; per-SKU packaging lives on
        # the product metadata. Returning the same shape callers already expect.
        logger.debug(
            "AutoKBDocumentService is disabled (no-op) — skipping per-product "
            "KB docs for '%s'", product_name,
        )
        return {"documents_created": 0, "errors": [], "skipped": "deprecated_no_op"}
