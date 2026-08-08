"""
Canonical search weight profiles — ONE definition, every search path imports it.

WHY THIS MODULE EXISTS
----------------------
Fusion weights were derived in four places in three different vocabularies, and the
derivations had already drifted apart:

  * `unified_search_service.WEIGHT_PROFILES` — the intent-keyed 7-ASPECT profile
    (text / visual / understanding / color / texture / style / material). The real source.
  * `rag_routes.py` and `query_routes.py` — byte-identical copies of the adapter that
    maps a 7-aspect profile onto the RAG service's 9-SOURCE vocabulary (which splits
    `text` across chunk / product / keyword). Two copies of one mapping.
  * `vecs_service.search_all_collections` — hand-copied constants (visual 0.30,
    understanding 0.20, specialized 0.50) re-deriving the "balanced" profile for the
    image-only case, with a comment admitting it was tracking a doc.

That is the platform's #1 anti-regression shape — one quantity derived in several
places — applied to search relevance. It bites in a specific way: adding an 8th vector
means renormalizing every profile in every copy, and the one you miss does not throw.
It silently scores that vector at zero on that path only, so the vector is computed,
stored and billed while contributing nothing. Computed-but-never-read is exactly the
`ops.silent_zero` signature, and a wrong weight is still a valid float, so neither a
typecheck nor an integrity probe can see it.

So: aspects and channels are named here, every profile is defined here, and the two
vocabulary mappings are functions here. Adding a vector is a change to THIS FILE.

That 8th vector has since landed (#239, `page` — a voyage-multimodal-3 embedding of the
rendered catalog page). It arrived the way the paragraph above says it must: as one edit
here, expressed as a transformation of the seven-aspect base rather than 56 retyped
constants. See _with_page().

Deliberately imports nothing from `app` — the guard test must run without booting the
application (no DB, no secrets, no network).
"""

from typing import Dict, Iterable, Sequence

# ── Vocabularies ────────────────────────────────────────────────────────────────

#: The embedding aspects a query is scored against. Adding a vector starts here.
EMBEDDING_ASPECTS: tuple = (
    "text", "visual", "understanding", "color", "texture", "style", "material",
    "page",
)

#: The per-aspect image collections (VECS `image_*_embeddings`).
SPECIALIZED_ASPECTS: tuple = ("color", "texture", "style", "material")

#: The RAG service's result-source vocabulary. NOT the same quantity as
#: EMBEDDING_ASPECTS: `text` fans out into three retrieval sources, and there is no
#: standalone text channel here. The two are bridged by profile_to_source_weights().
SOURCE_CHANNELS: tuple = (
    "visual", "chunk", "understanding", "product", "keyword",
    "color", "texture", "style", "material", "page",
)

#: How the single `text` aspect divides across the three text-bearing sources.
TEXT_SOURCE_SPLIT: Dict[str, float] = {"chunk": 0.40, "product": 0.35, "keyword": 0.25}


# ── Profiles ────────────────────────────────────────────────────────────────────

#: The seven original aspects, at the weights they carried before the page channel
#: existed. Kept verbatim as the BASE so the page channel's arrival is a visible,
#: reviewable transformation rather than 56 hand-edited constants — see _with_page().
_BASE_PROFILES: Dict[str, Dict[str, float]] = {
    "product_name": {
        "text": 0.40, "visual": 0.25, "understanding": 0.15,
        "color": 0.05, "texture": 0.05, "style": 0.05, "material": 0.05
    },
    "color_finish": {
        "text": 0.10, "visual": 0.20, "understanding": 0.15,
        "color": 0.30, "texture": 0.05, "style": 0.15, "material": 0.05
    },
    "specification": {
        "text": 0.25, "visual": 0.10, "understanding": 0.40,
        "color": 0.05, "texture": 0.05, "style": 0.05, "material": 0.10
    },
    "texture_pattern": {
        "text": 0.10, "visual": 0.25, "understanding": 0.15,
        "color": 0.05, "texture": 0.30, "style": 0.10, "material": 0.05
    },
    "style_aesthetic": {
        "text": 0.10, "visual": 0.25, "understanding": 0.15,
        "color": 0.10, "texture": 0.10, "style": 0.25, "material": 0.05
    },
    "material_search": {
        "text": 0.15, "visual": 0.15, "understanding": 0.25,
        "color": 0.05, "texture": 0.10, "style": 0.05, "material": 0.25
    },
    "balanced": {
        "text": 0.15, "visual": 0.15, "understanding": 0.20,
        "color": 0.125, "texture": 0.125, "style": 0.125, "material": 0.125
    },
}

#: The 8th vector's share, per intent (#239). `page` is a voyage-multimodal-3
#: embedding of the whole rendered catalog page — text and picture in one vector.
#: It is weighted by how much an intent depends on what a page LOOKS like as a
#: composed unit, and specifically on text the structural OCR pass never reads:
#: `Image`/`Figure`/`chart` regions are crop sources only, so a product name
#: printed inside a photo is invisible to every other channel. That is why
#: product_name gets the largest share and the aspect-specific intents (which are
#: already served by a dedicated per-aspect vector) get the smallest.
PAGE_WEIGHTS: Dict[str, float] = {
    "product_name": 0.15,   # the gap this vector exists to close
    "specification": 0.10,  # spec values baked into diagram artwork
    "style_aesthetic": 0.10,  # page composition IS style evidence
    "balanced": 0.10,
    "color_finish": 0.08,
    "texture_pattern": 0.08,
    "material_search": 0.08,
}


def _with_page(base: Dict[str, float], page_weight: float) -> Dict[str, float]:
    """Carve `page_weight` out of a 7-aspect profile PROPORTIONALLY.

    Every pre-existing aspect is scaled by the same (1 - page_weight), so the
    ratios among the original seven are untouched — the page channel takes a slice
    of the whole, it does not rob one specific aspect. Two things follow, and both
    are load-bearing:

      * Re-tuning is one number per profile (PAGE_WEIGHTS) instead of eight, so the
        "renormalize every profile and miss one" failure this module exists to
        prevent has nothing to bite on.
      * `image_only_weights` normalizes over the collections it actually queried, and
        normalizing cancels a common factor — so the image-only fan-out, which has no
        page channel, produces byte-identical weights to before this vector existed.
        `test_image_only_full_set_matches_pre_refactor_constants` still pins that.
    """
    scaled = {aspect: w * (1.0 - page_weight) for aspect, w in base.items()}
    scaled["page"] = page_weight
    return scaled


#: Dynamic weight profiles keyed by query intent type (all sum to 1.0).
WEIGHT_PROFILES: Dict[str, Dict[str, float]] = {
    name: _with_page(base, PAGE_WEIGHTS[name]) for name, base in _BASE_PROFILES.items()
}

#: Fallback when query understanding is unavailable or times out.
DEFAULT_PROFILE = "balanced"


# ── Helpers ─────────────────────────────────────────────────────────────────────

def normalize(weights: Dict[str, float]) -> Dict[str, float]:
    """Scale weights to sum to 1.0.

    Ranking-safe: a weighted sum divided by a positive constant preserves order, so
    renormalizing never reshuffles results — it only keeps scores comparable across
    profiles and across calls where some channels were skipped.
    """
    total = sum(weights.values())
    if not total:
        return dict(weights)
    return {k: v / total for k, v in weights.items()}


def get_profile(profile_name: str) -> Dict[str, float]:
    """Look up an intent profile, falling back to `balanced` for unknown names."""
    return dict(WEIGHT_PROFILES.get(profile_name) or WEIGHT_PROFILES[DEFAULT_PROFILE])


def profile_to_source_weights(aspect_weights: Dict[str, float]) -> Dict[str, float]:
    """Map an 8-aspect profile onto the RAG service's 10-source vocabulary.

    `text` has no direct counterpart — it is the retrieval budget for everything
    text-bearing, so it fans out across chunk / product / keyword per TEXT_SOURCE_SPLIT.
    Every other aspect maps 1:1, `page` included.

    Used by BOTH `/api/rag/search` and `/api/documents/query`; it lived in those two
    files as identical copies before this module existed.
    """
    balanced = WEIGHT_PROFILES[DEFAULT_PROFILE]
    text_w = aspect_weights.get("text", balanced["text"])

    out: Dict[str, float] = {
        "visual": aspect_weights.get("visual", balanced["visual"]),
        "understanding": aspect_weights.get("understanding", balanced["understanding"]),
        "page": aspect_weights.get("page", balanced["page"]),
    }
    for source, share in TEXT_SOURCE_SPLIT.items():
        out[source] = text_w * share
    for aspect in SPECIALIZED_ASPECTS:
        out[aspect] = aspect_weights.get(aspect, balanced[aspect])
    return out


def image_only_weights(
    has_understanding: bool,
    specialized_types: Sequence[str],
) -> Dict[str, float]:
    """Weights for the image-only fan-out (`vecs_service.search_all_collections`).

    There is no text channel when searching image collections, so the `text` share is
    folded into `visual` — the visual vector is what carries the query in that path.
    Channels that were not queried are dropped and the rest renormalized, so omitting
    (say) color does not dilute the vectors that ARE present.

    `page` is deliberately absent and that is NOT the silent-zero bug this module
    guards against. This fan-out ranks IMAGES; a page vector keys a page, so there is
    no page score to attach to an image row — including the channel would mean
    inventing one. The page channel is scored in the product-level fusion
    (`rag_service.multi_vector_search`), which is where a page hit has somewhere real
    to land. Because `_with_page` scales the seven remaining aspects by a common
    factor and this function renormalizes, the numbers here are unchanged by the
    page channel's existence.

    Derived from the `balanced` profile rather than restated as constants: before this,
    the equivalent numbers were hardcoded (0.30 / 0.20 / 0.50) next to a comment
    explaining which doc they were copied from, which is precisely how they drift.
    """
    balanced = WEIGHT_PROFILES[DEFAULT_PROFILE]

    weights: Dict[str, float] = {"visual": balanced["text"] + balanced["visual"]}
    if has_understanding:
        weights["understanding"] = balanced["understanding"]

    queried = [t for t in specialized_types if t in SPECIALIZED_ASPECTS]
    if queried:
        # The full specialized pool, shared evenly across the ones actually queried.
        pool = sum(balanced[a] for a in SPECIALIZED_ASPECTS)
        per_type = pool / len(queried)
        for aspect in queried:
            weights[aspect] = per_type

    return normalize(weights)


def aspect_bias_weights(aspect: str) -> Dict[str, float]:
    """Source weights for an explicit user aspect choice (#277) — that vector dominates.

    Normalized on the way out. The inline version summed to 1.025 because the chosen
    aspect's 0.025 share was overwritten by 0.55 rather than swapped, which left scores
    on this path not comparable with every other path's.
    """
    if aspect not in SPECIALIZED_ASPECTS:
        raise ValueError(f"aspect must be one of {SPECIALIZED_ASPECTS}, got {aspect!r}")

    weights: Dict[str, float] = {
        "visual": 0.10, "chunk": 0.05, "understanding": 0.15,
        "product": 0.05, "keyword": 0.05,
        "color": 0.025, "texture": 0.025, "style": 0.025, "material": 0.025,
        # Small but never zero: an explicit aspect choice narrows the query, it does
        # not mean "ignore the page". Omitting the key would score the page vector at
        # zero on this path only — computed, stored, billed, unread. That is the
        # `ops.silent_zero` shape this module's docstring is about, and this path is
        # exactly where it would have slipped through: `aspect_bias_weights` builds
        # its dict by hand rather than deriving it from a profile.
        "page": 0.05,
    }
    weights[aspect] = 0.55
    return normalize(weights)


def profile_names() -> Iterable[str]:
    """All known intent profile names."""
    return WEIGHT_PROFILES.keys()
