"""Greek Marketplaces module — Skroutz + Bestdeals + Shopflix discovery."""

from app.modules._core.types import ModuleDefinition
from app.modules.greek_marketplaces.manifest import manifest
from app.modules.greek_marketplaces.service import (
    GreekMarketplacesService,
    get_greek_marketplaces_service,
)

definition = ModuleDefinition(
    manifest=manifest,
    router_path="app.modules.greek_marketplaces.routes.router",
    tags=["greek-marketplaces"],
)

__all__ = [
    "GreekMarketplacesService",
    "definition",
    "get_greek_marketplaces_service",
]
