"""Module registry primitives."""

from app.modules._core.registry import (
    discover_modules,
    get_module,
    is_module_enabled,
    list_registered_modules,
)
from app.modules._core.types import ModuleDefinition, ModuleManifest

__all__ = [
    "ModuleDefinition",
    "ModuleManifest",
    "discover_modules",
    "get_module",
    "is_module_enabled",
    "list_registered_modules",
]
