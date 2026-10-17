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
import unicodedata
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Sequence

from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

#: Cache lifetime. Field definitions change when an admin edits them, which is rare; five
#: minutes bounds the staleness without putting a query in front of every extraction.
CACHE_TTL_SECONDS = 300

# `FALLBACK_CATEGORY` and `SECTION_ORDER` used to be module constants here (#30 M16-7).


#: Unit spellings folded onto the suffix the registry uses. Spelling only: `width` is never
#: resolved to `width_mm`, because the unit of a bare key is unknown.
_UNIT_SUFFIXES = (
    (("millimetres", "millimeters", "millimetre", "millimeter"), "mm"),
    (("centimetres", "centimeters", "centimetre", "centimeter"), "cm"),
    (("kilograms", "kilogram", "kgs"), "kg"),
    (("percentage", "percent", "perc"), "pct"),
    (("kilowatts", "kilowatt"), "kw"),
    (("watts", "watt"), "w"),
    (("volts", "volt"), "v"),
    (("litres", "liters", "litre", "liter", "ltr"), "l"),
    (("degrees_c", "deg_c", "celsius"), "c"),
    (("kelvin",), "k"),
    (("decibels", "decibel"), "db"),
)


def fold_field_key(name: Any) -> str:
    """Registry spelling of an extracted key: NFKD, marks stripped, lowercase, one `_` per run.

    Unicode letters survive (a Greek key stays Greek), so an alias in any script can match.
    `²`/`³` fold to `2`/`3` through NFKD.
    """
    s = unicodedata.normalize("NFKD", str(name or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = "".join(ch if ch.isalnum() else "_" for ch in s)
    return "_".join(part for part in s.split("_") if part)


def _unit_folded(key: str) -> str:
    for spellings, unit in _UNIT_SUFFIXES:
        for spelling in spellings:
            if key.endswith("_" + spelling):
                return key[: -len(spelling)] + unit
    return key


def _plural_variants(key: str) -> List[str]:
    head, _, last = key.rpartition("_")
    prefix = head + "_" if head else ""
    out: List[str] = []
    if last.endswith("ies") and len(last) > 3:
        out.append(prefix + last[:-3] + "y")
    if last.endswith("es") and len(last) > 3:
        out.append(prefix + last[:-2])
    if last.endswith("s") and len(last) > 2:
        out.append(prefix + last[:-1])
    if not last.endswith("s"):
        out.append(prefix + last + "s")
    return out


def field_key_candidates(name: Any) -> List[str]:
    """Spellings to try, most literal first: as written, folded, unit-folded, singular/plural."""
    raw = str(name or "").strip()
    seen: List[str] = []

    def add(k: str) -> None:
        if k and k not in seen:
            seen.append(k)

    add(raw)
    add(raw.lower())
    folded = fold_field_key(raw)
    add(folded)
    unit = _unit_folded(folded)
    add(unit)
    for base in (folded, unit):
        for variant in _plural_variants(base):
            add(variant)
    return seen


def _parse_aliases(raw: Any) -> List[tuple]:
    """`aliases` rows as (name, factor); `lead_time_weeks*7` converts weeks to days."""
    out: List[tuple] = []
    for item in raw or ():
        text = str(item or "").strip().lower()
        if not text:
            continue
        name, _, factor = text.partition("*")
        try:
            value = float(factor) if factor else 1.0
        except ValueError:
            logger.error("material_metadata_fields alias %r has a bad factor; ignored", item)
            continue
        out.append((name.strip(), value))
    return out


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
    #: {category_key: description}. A field can mean different things in different categories.
    description_by_category: Optional[Dict[str, str]] = None
    #: Plausibility rules the columns cannot express: min/max/unit/pattern/max_length. Enforced
    #: by `field_validation.validate_value`, which treats the key set as CLOSED — an unknown key
    #: is reported, not ignored. `field_type` + `dropdown_options` already cover enum membership
    #: and numeric-ness, so most fields need nothing here.
    validation_rules: Optional[Dict[str, Any]] = None
    #: Retired spellings, lowercased. A `name*factor` row is stored as name + alias_factors.
    aliases: Sequence[str] = ()
    sensitivity: str = "public"
    #: {category_key: 'identity'|'descriptive'} overriding `role` in that category.
    role_by_category: Optional[Dict[str, str]] = None
    alias_factors: Optional[Dict[str, float]] = None

    def applies_to(self, category_key: str) -> bool:
        return self.categories is None or category_key in self.categories

    def role_in(self, category_key: Optional[str]) -> str:
        if category_key and self.role_by_category:
            return self.role_by_category.get(category_key) or self.role
        return self.role

    def describe(self, category_key: str) -> str:
        """The description to show for THIS category.

        `body_material` is "vitreous china, fine fireclay, acrylic" under sanitary and
        "metal, glass, fabric, wood, concrete, rattan" under lighting. Serving one merged
        description to both does not just lose detail — it steers the model toward values that
        are wrong for the category, and every one of them is a plausible string no validator
        can reject.
        """
        if self.description_by_category:
            specific = self.description_by_category.get(category_key)
            if specific:
                return specific
        return self.description


@dataclass(frozen=True)
class CategorySpec:
    key: str
    display_name: str
    extraction_tips: Sequence[str] = ()
    skip_fields: Sequence[str] = ()
    controlled_vocab: Sequence[str] = ()
    vocab_aliases: Sequence[str] = ()
    is_active: bool = True
    sort_order: Optional[int] = None


@dataclass
class _Cache:
    fields: Dict[str, FieldSpec] = dc_field(default_factory=dict)
    categories: Dict[str, CategorySpec] = dc_field(default_factory=dict)
    #: section_key -> display_order, from `material_field_sections`.
    section_order: Dict[str, int] = dc_field(default_factory=dict)
    #: The category an unrecognised upload category routes to. Never a literal.
    fallback_category: str = ""
    loaded_at: float = 0.0
    #: lowercased alias -> [(field_name, factor)], every field that claims it.
    alias_index: Dict[str, List[tuple]] = dc_field(default_factory=dict)


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
                    "extraction_hints, description_by_category, validation_rules, aliases, "
                    "sensitivity, role_by_category")
            .eq("status", "active")
            .execute()
        ).data or []

        category_rows = (
            client.table("material_categories")
            .select("category_key, display_name, extraction_tips, skip_fields, "
                    "controlled_vocab, vocab_aliases, is_fallback, is_active, sort_order")
            .execute()
        ).data or []

        section_rows = (
            client.table("material_field_sections")
            .select("section_key, display_order")
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

        # Same rule as the two above, for the same reason (#30 M16-7). An empty section
        # table would silently reorder every extraction prompt; no fallback row would
        # silently route every unknown category to whatever `.get()` returned.
        if not section_rows:
            raise RuntimeError(
                "material_field_sections returned zero rows. Prompt section ordering "
                "lives there now; serving an arbitrary order would change every "
                "extraction prompt with nothing to show for it."
            )

        fallback = next(
            (r["category_key"] for r in category_rows if r.get("is_fallback")), None
        )
        if not fallback:
            raise RuntimeError(
                "No material_categories row is marked is_fallback. An unrecognised "
                "upload category must not silently become a specific one — that is how "
                "a door ends up filed under General Materials."
            )

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
                description_by_category=r.get("description_by_category") or None,
                validation_rules=r.get("validation_rules") or None,
                aliases=tuple(name for name, _f in _parse_aliases(r.get("aliases"))),
                alias_factors={n: f for n, f in _parse_aliases(r.get("aliases")) if f != 1.0} or None,
                sensitivity=r.get("sensitivity") or "public",
                role_by_category=r.get("role_by_category") or None,
            )
            for r in field_rows
        }
        alias_index: Dict[str, List[tuple]] = {}
        for spec in fields.values():
            for alias in spec.aliases:
                factor = (spec.alias_factors or {}).get(alias, 1.0)
                alias_index.setdefault(alias, []).append((spec.name, factor))
        categories = {
            r["category_key"]: CategorySpec(
                key=r["category_key"],
                display_name=r.get("display_name") or r["category_key"].replace("_", " ").title(),
                extraction_tips=tuple(r.get("extraction_tips") or ()),
                skip_fields=tuple(r.get("skip_fields") or ()),
                controlled_vocab=tuple(r.get("controlled_vocab") or ()),
                vocab_aliases=tuple(r.get("vocab_aliases") or ()),
                is_active=r.get("is_active") is not False,
                sort_order=r.get("sort_order"),
            )
            for r in category_rows
        }

        self._cache = _Cache(
            fields=fields,
            categories=categories,
            section_order={
                r["section_key"]: int(r["display_order"]) for r in section_rows
            },
            fallback_category=fallback,
            loaded_at=time.monotonic(),
            alias_index=alias_index,
        )
        logger.info(
            "📋 Field registry loaded: %d fields (%d identity, %d canonicalizable, "
            "%d enum-constrained, %d with validation rules), %d categories",
            len(fields),
            sum(1 for f in fields.values() if f.role == "identity"),
            sum(1 for f in fields.values() if f.canonicalize),
            sum(1 for f in fields.values() if f.field_type == "dropdown" and f.dropdown_options),
            sum(1 for f in fields.values() if f.validation_rules),
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

    def spec_for(self, key: str) -> Optional[FieldSpec]:
        """The full spec for a field, or None when the registry does not know the key.

        None is a real answer — the extractor emits keys the registry never claimed to describe
        (`material_category`, `unit`, `factory_name`) — so callers must treat it as "no opinion",
        never as "invalid".
        """
        if not key or key.startswith("_"):
            return None
        return self._require().fields.get(key)

    def all_field_names(self) -> List[str]:
        """Every active field name, sorted."""
        return sorted(self._require().fields)

    def is_registered(self, key: str) -> bool:
        """True only when the registry has a row for exactly this key."""
        return self.spec_for(key) is not None

    def resolve_field(self, key: Any, category_key: Optional[str] = None) -> Optional[tuple]:
        """(field_name, factor) an extracted key names, or None. Multiply the value by `factor`.

        Order: a field of that name in the category, an alias in the category, a field of that
        name anywhere, an alias exactly one field claims. So `grade` stays `grade` on insulation
        and becomes `wood_grade` on wood. Spellings come from `field_key_candidates`.
        """
        cache = self._require()
        if not key or str(key).startswith("_"):
            return None
        category = self.resolve_category(category_key) if category_key else None
        candidates = field_key_candidates(key)
        if category:
            for cand in candidates:
                spec = cache.fields.get(cand)
                if spec and spec.applies_to(category):
                    return cand, 1.0
            for cand in candidates:
                hits = {h for h in cache.alias_index.get(cand, ())
                        if cache.fields[h[0]].applies_to(category)}
                if len(hits) == 1:
                    return next(iter(hits))
        for cand in candidates:
            if cand in cache.fields:
                return cand, 1.0
        for cand in candidates:
            hits = set(cache.alias_index.get(cand, ()))
            if len(hits) == 1:
                return next(iter(hits))
        return None

    def resolve_field_key(self, key: Any, category_key: Optional[str] = None) -> Optional[str]:
        """The field name, only when the value needs no unit conversion; else None.

        A caller that renames without converting must never turn `lead_time_weeks` into
        `lead_time_days`, so a scaled alias is invisible here. Use `resolve_field` for those.
        """
        hit = self.resolve_field(key, category_key)
        return hit[0] if hit and hit[1] == 1.0 else None

    def destination_of(self, key: str) -> str:
        """'column' means this value has a real home and must NOT be written to attributes jsonb."""
        spec = self._require().fields.get(key)
        return spec.destination if spec else "metadata"

    def is_identity(self, key: str, category_key: Optional[str] = None) -> bool:
        """True when two units differing only in this field are stocked and sold separately."""
        spec = self._require().fields.get(key)
        category = (self.resolve_category(category_key) or category_key) if category_key else None
        return bool(spec and spec.role_in(category) == "identity")

    def identity_fields(self, category_key: Optional[str] = None) -> List[str]:
        cache = self._require()
        category = (self.resolve_category(category_key) or category_key) if category_key else None
        return sorted(
            f.name for f in cache.fields.values()
            if (category is None or f.applies_to(category)) and f.role_in(category) == "identity"
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
        key = self.resolve_category(category_key)
        if key:
            return key
        logger.warning("Unknown upload category '%s' — falling back to '%s'",
                       category_key, cache.fallback_category)
        return cache.fallback_category

    def resolve_category(self, value: Any) -> Optional[str]:
        """Coarse category key for a key, a controlled-vocab value or a vocab alias, else None.

        The twin of SQL `resolve_material_category_key`: lower/trim, then exact key, then
        controlled_vocab, then vocab_aliases, active rows only, lowest sort_order first.
        """
        cache = self._require()
        needle = str(value or "").strip().lower()
        if not needle:
            return None
        active = sorted(
            (c for c in cache.categories.values() if c.is_active),
            key=lambda c: (c.sort_order is None, c.sort_order or 0, c.key),
        )
        for cat in active:
            if cat.key.lower() == needle:
                return cat.key
        for terms in ("controlled_vocab", "vocab_aliases"):
            for cat in active:
                if any(needle == str(v).strip().lower() for v in getattr(cat, terms)):
                    return cat.key
        return None

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

    def category_for_vocab(self, value: str) -> Optional[str]:
        """Which category group owns this fine-grained value? `floor_tile` -> `tiles`."""
        if not value:
            return None
        needle = value.strip().lower()
        cache = self._require()
        for cat in cache.categories.values():
            if any(needle == str(v).strip().lower() for v in cat.controlled_vocab):
                return cat.key
        return None

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

    def _section_rank(self, section: str) -> int:
        """Rank from `material_field_sections`, or last for a section nobody has placed.

        An unlisted section sorting after every listed one — alphabetically, via the
        secondary sort key — is deliberate and is the behaviour the old constant had:
        a field in a brand-new section still works, it just lands at the end until
        somebody orders it. `design` is live in that state today.
        """
        order = self._require().section_order
        return order.get(section, max(order.values(), default=0) + 1)

    def priority_fields_prompt(self, category_key: str) -> str:
        """The PRIORITY FIELDS block. Format matches the retired Python builder."""
        cat = self.category(category_key)
        specs = self.fields_for_category(category_key)

        lines: List[str] = [
            f"PRIORITY FIELDS for {cat.display_name.upper()} products:",
            "(Extract these if present — they are the most important for this category)",
            "",
        ]
        key = self._resolve_category(category_key)
        current = None
        for spec in specs:
            if spec.section != current:
                current = spec.section
                lines.append(f"**{current.replace('_', ' ').title()}:**")
            description = spec.describe(key)
            line = f"- {spec.name}: {description}" if description else f"- {spec.name}"
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
            "ALLOWED material_category VALUES for this category (use one of these exactly, or "
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


def invalidate_registry_cache() -> None:
    """Mark the cache stale after a registry write; the next `ensure_loaded()` re-reads.

    The loaded rows keep answering until then, so a sync reader never meets an unloaded
    registry mid-run.
    """
    cache = field_registry._cache
    if cache is not None:
        cache.loaded_at = float("-inf")
