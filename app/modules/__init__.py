"""
App modules package.

Each subdirectory of this package (other than `_core`) is a self-contained feature
module. Each module's `__init__.py` must export `definition: ModuleDefinition`.

Module routers are mounted in `app/main.py` via
`mount_module_routers(app)` after the core routers.
"""

import importlib
import logging
from typing import TYPE_CHECKING

from app.modules._core import (
    ModuleDefinition,
    ModuleManifest,
    discover_modules,
    get_module,
    is_module_enabled,
    list_registered_modules,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Populate the registry on first import. Safe to call multiple times.
discover_modules()


def mount_module_routers(app: "FastAPI") -> None:
    """Mount every registered module's APIRouter on the FastAPI app."""
    for definition in list_registered_modules():
        if not definition.router_path:
            continue
        try:
            dotted, attr = definition.router_path.rsplit(".", 1)
            module = importlib.import_module(dotted)
            router = getattr(module, attr)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not mount router for module %s (%s): %s",
                definition.manifest.slug,
                definition.router_path,
                exc,
            )
            continue

        app.include_router(router, tags=definition.tags or [definition.manifest.slug])
        logger.info("Mounted router for module %s", definition.manifest.slug)


__all__ = [
    "ModuleDefinition",
    "ModuleManifest",
    "discover_modules",
    "get_module",
    "is_module_enabled",
    "list_registered_modules",
    "mount_module_routers",
]
