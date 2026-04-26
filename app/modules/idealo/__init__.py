"""Idealo adapter module — DACH/IT/UK price-comparison discovery."""

from app.modules._core.types import ModuleDefinition
from app.modules.idealo.manifest import manifest
from app.modules.idealo.service import IdealoService, get_idealo_service

definition = ModuleDefinition(
    manifest=manifest,
    router_path=None,
    tags=["idealo"],
)

__all__ = ["IdealoService", "definition", "get_idealo_service"]
