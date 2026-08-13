"""
The field registry, read from the database (#347 phase 3.2).

This replaces `category_field_registry.py` and `facet_whitelist.py`, which held the same facts as
Python literals. There were SIX copies of "what fields exist" across the platform; the DB tables
`material_metadata_fields` and `material_categories` are now the one that counts.

WHY THIS IS LOADED EXPLICITLY AND NOT LAZILY
--------------------------------------------
`is_canonicalizable()` is called from `collect_raw_attributes()`, which is documented as the
pure-local fallback used *when canonicalization has already failed*. It cannot make a network
call. So the registry is loaded once, up front, and every accessor after that is synchronous
against the cache.

WHY THE ACCESSORS RAISE INSTEAD OF RETURNING EMPTY
--------------------------------------------------
If the cache were empty and `is_canonicalizable()` answered `False` for everything, ingestion
would keep running and quietly write zero facets forever — the exact "silent zero" shape that
dominates this platform's history (see CLAUDE.md). An empty registry is never a valid state, so
`load()` refuses a zero-row result and the accessors refuse to answer before it has run.
Loud and stopped beats quiet and wrong.

There is NO code fallback. A registry the DB cannot serve is an outage, not a reason to fall back
to a stale hardcoded copy — a hardcoded copy is what this phase exists to delete.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Sequence

from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

#: Cache lifetime. Field definitions change when an admin edits them, which is rare; five
#: minutes bounds the staleness without putting a query in front of every extraction.
CACHE_TTL_SECONDS = 300

#: Category used when the upload category is unrecognised. Mirrors the behaviour of the retired
#: `get_category_config()`, which fell back to this rather than returning nothing.
FALLBACK_CATEGORY = "general_materials"

#: Stable ordering for prompt sections — most identifying first, commercial trivia last. Sections
#: not listed here sort after these, alphabetically, so a new one appears without a code change.
SECTION_ORDER = [
    "material_properties", "dimensions", "appearance", "performance", "features",
    "electrical_specs", "thermal_performance", "water_performance", "application",
    "installation", "care", "compliance", "packaging", "commercial", "other",
]


class FieldRegistryNotLoaded(RuntimeError):
    """Raised when an accessor is called before `load()` has succeeded.

    Deliberately an exception rather than an empty default: see the module docstring.
    """


@dataclass(frozen=True)
class FieldSpec:
    name: str
    display_name: str
    description: str
    section: str
    #: None means "applies to every category".
    categories: Optional[frozenset]
    role: str            # 'identity' | 'descriptive'
    destination: str     # 'metadata' | 'column'
    canonicalize: bool
    field_type: str
    dropdown_options: Sequence[str] = ()
    #: Per-field prose telling the model where to look. Distinct from `section`.
    extraction_hint: str = ""

    def applies_to(self, category_key: str) -> bool:
        return self.categories is None or category_key in self.categories


@dataclass(frozen=True)
class CategorySpec:
    key: str
    display_name: str
    extraction_tips: Sequence[str] = ()
    skip_fields: Sequence[str] = ()
    controlled_vocab: Sequence[str] = ()


@dataclass
class _Cache:
    fields: Dict[str, FieldSpec] = dc_field(default_factory=dict)
    categories: Dict[str, CategorySpec] = dc_field(default_factory=dict)
    loaded_at: float = 0.0


class FieldRegistry:
    """Process-wide cache over `material_metadata_fields` + `material_categories`."""

    def __init__(self) -> None:
        self._cache: Optional[_Cache] = None
        self._lock = asyncio.Lock()

    # ── Loading ──────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._cache is not None

    @property
    def is_stale(self) -> bool:
        return self._cache is None or (time.monotonic() - self._cache.loaded_at) > CACHE_TTL_SECONDS

    async def ensure_loaded(self, *, force: bool = False) -> None:
        """Load if never loaded or past the TTL. Safe to call on every extraction."""
        if not force and not self.is_stale:
            return
        async with self._lock:
            # Re-check under the lock: several extractions can race here on startup and only
            # one of them should do the round trip.
            if not force and not self.is_stale:
                return
            await asyncio.to_thread(self._load_blocking)

    def _load_blocking(self) -> None:
        client = get_supabase_client().client

        field_rows = (
            client.table("material_metadata_fields")
            .select("field_name, display_name, description, section, applies_to_categories, "
                    "role, destination, canonicalize, field_type, dropdown_options, "
                    "extraction_hints")
            .eq("status", "active")
            .execute()
        ).data or []

        category_rows = (
            client.table("material_categories")
            .select("category_key, display_name, extraction_tips, skip_fields, controlled_vocab")
            .execute()
        ).data or []

        # An empty registry is never legitimate. Accepting it would mean every subsequent
        # extraction silently asks for no fields and canonicalizes nothing, with no error
        # anywhere — refuse, and keep any previously good cache rather than replacing it with
        # a known-bad one.
        if not field_rows:
            raise RuntimeError(
                "material_metadata_fields returned zero active rows. The extraction field "
                "registry cannot be empty; refusing to serve an empty registry. Check RLS on the "
                "service-role client and that phase 3.1's seed ran."
            )
        if not category_rows:
            raise RuntimeError("material_categories returned zero rows — cannot build prompts.")

        fields = {
            r["field_name"]: FieldSpec(
                name=r["field_name"],
                display_name=r.get("display_name") or r["field_name"].replace("_", " ").title(),
                description=r.get("description") or "",
                section=r.get("section") or "other",
                categories=(frozenset(r["applies_to_categories"])
                            if r.get("applies_to_categories") else None),
                role=r.get("role") or "descriptive",
                destination=r.get("destination") or "metadata",
                canonicalize=bool(r.get("canonicalize")),
                field_type=r.get("field_type") or "text",
                dropdown_options=tuple(r.get("dropdown_options") or ()),
                extraction_hint=r.get("extraction_hints") or "",
            )
            for r in field_rows
        }
        categories = {
            r["category_key"]: CategorySpec(
                key=r["category_key"],
                display_name=r.get("display_name") or r["category_key"].replace("_", " ").title(),
                extraction_tips=tuple(r.get("extraction_tips") or ()),
                skip_fields=tuple(r.get("skip_fields") or ()),
                controlled_vocab=tuple(r.get("controlled_vocab") or ()),
            )
            for r in category_rows
        }

        self._cache = _Cache(fields=fields, categories=categories, loaded_at=time.monotonic())
        logger.info(
            "📋 Field registry loaded: %d fields (%d identity, %d canonicalizable), %d categories",
            len(fields),
            sum(1 for f in fields.values() if f.role == "identity"),
            sum(1 for f in fields.values() if f.canonicalize),
            len(categories),
        )

    def _require(self) -> _Cache:
        if self._cache is None:
            raise FieldRegistryNotLoaded(
                "Field registry accessed before load(). Call `await field_registry."
                "ensure_loaded()` at the entry point of this path. It is NOT lazily loaded on "
                "purpose — see app/services/metadata/field_registry.py."
            )
        return self._cache

    # ── Field-level accessors ────────────────────────────────────────────────

    def is_canonicalizable(self, key: str) -> bool:
        if not key or key.startswith("_"):
            return False
        spec = self._require().fields.get(key)
        return bool(spec and spec.canonicalize)

    def destination_of(self, key: str) -> str:
        """'column' means this value has a real home and must NOT be written to attributes jsonb."""
        spec = self._require().fields.get(key)
        return spec.destination if spec else "metadata"

    def is_identity(self, key: str) -> bool:
        """True when two units differing only in this field are stocked and sold separately."""
        spec = self._require().fields.get(key)
        return bool(spec and spec.role == "identity")

    def identity_fields(self, category_key: Optional[str] = None) -> List[str]:
        cache = self._require()
        return sorted(
            f.name for f in cache.fields.values()
            if f.role == "identity" and (category_key is None or f.applies_to(category_key))
        )

    def fields_for_category(self, category_key: str) -> List[FieldSpec]:
        cache = self._require()
        key = self._resolve_category(category_key)
        skip = set(self.skip_fields(key))
        return sorted(
            (f for f in cache.fields.values() if f.applies_to(key) and f.name not in skip),
            key=lambda f: (self._section_rank(f.section), f.section, f.name),
        )

    # ── Category-level accessors ─────────────────────────────────────────────

    def _resolve_category(self, category_key: str) -> str:
        cache = self._require()
        key = (category_key or "").lower().strip()
        if key in cache.categories:
            return key
        logger.warning("Unknown upload category '%s' — falling back to '%s'",
                       category_key, FALLBACK_CATEGORY)
        return FALLBACK_CATEGORY

    def category(self, category_key: str) -> CategorySpec:
        return self._require().categories[self._resolve_category(category_key)]

    def skip_fields(self, category_key: str) -> List[str]:
        return list(self.category(category_key).skip_fields)

    def controlled_vocab(self, category_key: str) -> List[str]:
        return list(self.category(category_key).controlled_vocab)

    def all_controlled_vocab(self) -> frozenset:
        """Every fine-grained material_category value, across all categories.

        The single source for stage 4's classifier: what the prompt OFFERS and what the
        validator ACCEPTS must be the same set, or the pipeline either rejects correct answers
        or accepts answers nothing can produce. Both happened — see the migration
        `material_categories_complete_controlled_vocab`.
        """
        cache = self._require()
        return frozenset(v for c in cache.categories.values() for v in c.controlled_vocab)

    def controlled_vocab_by_category(self) -> List[tuple]:
        """(label, values) pairs for prompt rendering; empty categories omitted.

        A list rather than a dict keyed on display_name: two categories are allowed to share a
        display name, and a dict would silently drop one of them from the prompt while leaving
        its values in `all_controlled_vocab()` — offered by nothing, accepted by the validator.
        That asymmetry is the exact bug this whole change removes.
        """
        cache = self._require()
        return [
            (c.display_name.upper(), sorted(c.controlled_vocab))
            for c in sorted(cache.categories.values(), key=lambda c: c.key)
            if c.controlled_vocab
        ]

    # ── Prompt fragments ─────────────────────────────────────────────────────

    @staticmethod
    def _section_rank(section: str) -> int:
        try:
            return SECTION_ORDER.index(section)
        except ValueError:
            return len(SECTION_ORDER)

    def priority_fields_prompt(self, category_key: str) -> str:
        """The PRIORITY FIELDS block. Format matches the retired Python builder."""
        cat = self.category(category_key)
        specs = self.fields_for_category(category_key)

        lines: List[str] = [
            f"PRIORITY FIELDS for {cat.display_name.upper()} products:",
            "(Extract these if present — they are the most important for this category)",
            "",
        ]
        current = None
        for spec in specs:
            if spec.section != current:
                current = spec.section
                lines.append(f"**{current.replace('_', ' ').title()}:**")
            line = f"- {spec.name}: {spec.description}" if spec.description else f"- {spec.name}"
            if spec.extraction_hint:
                line += f" ({spec.extraction_hint})"
            if spec.field_type == "dropdown" and spec.dropdown_options:
                line += f" [one of: {', '.join(spec.dropdown_options)}]"
            lines.append(line)
        lines.append("")
        return "\n".join(lines)

    def extraction_tips_prompt(self, category_key: str) -> str:
        cat = self.category(category_key)
        if not cat.extraction_tips:
            return ""
        lines = [f"CATEGORY-SPECIFIC EXTRACTION TIPS for {cat.display_name}:"]
        lines.extend(f"- {tip}" for tip in cat.extraction_tips)
        return "\n".join(lines)

    def skip_fields_prompt(self, category_key: str) -> str:
        cat = self.category(category_key)
        if not cat.skip_fields:
            return ""
        return (
            f"FIELDS TO SKIP (not relevant for {cat.display_name} products — do NOT extract or "
            f"hallucinate these): {', '.join(cat.skip_fields)}"
        )

    def controlled_vocab_prompt(self, category_key: str) -> str:
        vocab = self.controlled_vocab(category_key)
        if not vocab:
            return ""
        return (
            "ALLOWED material_subtype VALUES for this category (use one of these exactly, or "
            f"omit the field if none fits): {', '.join(vocab)}"
        )

    def category_prompt_block(self, category_key: str) -> str:
        """Everything category-specific, assembled in one block."""
        parts = [
            self.priority_fields_prompt(category_key),
            self.controlled_vocab_prompt(category_key),
            self.extraction_tips_prompt(category_key),
            self.skip_fields_prompt(category_key),
        ]
        return "\n\n".join(p for p in parts if p)


#: Process-wide singleton. Import this, not the class.
field_registry = FieldRegistry()


async def ensure_loaded(*, force: bool = False) -> None:
    """Convenience wrapper — call at the entry point of any path that reads the registry."""
    await field_registry.ensure_loaded(force=force)
