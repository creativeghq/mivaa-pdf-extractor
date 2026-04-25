"""
Backend module registry.

Behavior:
  * `discover_modules()` scans `app/modules/` for subdirectories (excluding `_core`),
    imports each as a package, and collects its exported `definition: ModuleDefinition`.
  * Any `ImportError` or missing definition is logged and skipped — a broken module
    must NOT crash server startup.
  * `is_module_enabled(slug)` reads the `public.modules` table to check the DB toggle.
    Cached in-process for `_CACHE_TTL_SECONDS` to avoid hammering Supabase on every
    request (explicit `invalidate_cache()` after admin toggles).
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
import time
from pathlib import Path
from typing import Dict, List, Optional

from app.modules._core.types import ModuleDefinition

logger = logging.getLogger(__name__)

_registry: Dict[str, ModuleDefinition] = {}
_discovered: bool = False

_CACHE_TTL_SECONDS = 300.0
_enabled_cache: Dict[str, bool] = {}
_enabled_cache_expires_at: float = 0.0


def discover_modules(package: str = "app.modules") -> Dict[str, ModuleDefinition]:
    """Import every subpackage of `app.modules` (except `_core`) and collect definitions."""
    global _discovered

    if _discovered:
        return _registry

    try:
        pkg = importlib.import_module(package)
    except ImportError as exc:
        logger.warning("Could not import %s (%s); registry will be empty.", package, exc)
        _discovered = True
        return _registry

    pkg_path = getattr(pkg, "__path__", None)
    if pkg_path is None:
        logger.warning("%s has no __path__; treating as empty.", package)
        _discovered = True
        return _registry

    for _finder, name, is_pkg in pkgutil.iter_modules(pkg_path):
        if not is_pkg or name.startswith("_"):
            continue
        dotted = f"{package}.{name}"
        try:
            submodule = importlib.import_module(dotted)
        except Exception as exc:  # noqa: BLE001  — isolate any failure per module
            logger.warning("Module %s failed to import: %s — skipping.", dotted, exc)
            continue

        definition = getattr(submodule, "definition", None)
        if not isinstance(definition, ModuleDefinition):
            logger.warning(
                "Module %s does not export a ModuleDefinition as `definition` — skipping.",
                dotted,
            )
            continue

        if definition.manifest.slug in _registry:
            logger.warning(
                "Duplicate module slug %s (from %s) — keeping first, skipping this one.",
                definition.manifest.slug,
                dotted,
            )
            continue

        _registry[definition.manifest.slug] = definition
        logger.info("Registered module %s (%s)", definition.manifest.slug, dotted)

    _discovered = True
    return _registry


def list_registered_modules() -> List[ModuleDefinition]:
    """Return the list of modules discovered at startup."""
    return list(_registry.values())


def get_module(slug: str) -> Optional[ModuleDefinition]:
    return _registry.get(slug)


def _refresh_enabled_cache() -> None:
    """Pull the enabled-flag map from Supabase. Silently falls back to the last cache on error."""
    global _enabled_cache, _enabled_cache_expires_at

    try:
        from app.services.core.supabase_client import get_supabase_client

        client = get_supabase_client().client
        response = client.table("modules").select("slug, enabled").execute()
        rows = response.data or []
        _enabled_cache = {row["slug"]: bool(row.get("enabled")) for row in rows}
        _enabled_cache_expires_at = time.monotonic() + _CACHE_TTL_SECONDS
    except Exception as exc:  # noqa: BLE001
        logger.warning("modules.is_enabled: could not refresh cache (%s) — using last known state", exc)
        # Keep existing cache (stale but non-zero); extend expiry to avoid storm.
        _enabled_cache_expires_at = time.monotonic() + _CACHE_TTL_SECONDS


def is_module_enabled(slug: str) -> bool:
    """
    Return True if the module row exists in `public.modules` with `enabled=true`.

    Unknown slugs (not in the DB) → False.
    DB unreachable → returns last cached value (False if never cached).
    """
    if time.monotonic() >= _enabled_cache_expires_at:
        _refresh_enabled_cache()
    return _enabled_cache.get(slug, False)


def invalidate_cache() -> None:
    """Force the next `is_module_enabled()` call to re-fetch from Supabase."""
    global _enabled_cache_expires_at
    _enabled_cache_expires_at = 0.0
