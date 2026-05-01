"""
VisionAnalysis schema — single source of truth for the structured material
analysis the vision LLM emits and that Voyage embeds.

Used by three call paths that MUST stay aligned or the
`vecs.image_understanding_embeddings` collection drifts:

1. Stage 3 ingestion (image_processing_service) — vision_analysis JSON
   produced by Claude Opus 4.7 via Anthropic tool use → serialised to text
   via `serialize_vision_analysis_to_text` → embedded by Voyage → stored.
2. RAG service `analyze_material_image` — same shape, same call pattern.
3. Backfill cron — re-runs the above on existing images when schema_version
   bumps so the index stays in one coherent embedding space.

`schema_version` is bumped any time the field set or serialiser changes.
Existing rows in `vecs.image_understanding_embeddings` are tagged with the
version they were embedded under, so the backfill knows what's stale.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


# Bump this when the schema or the serialiser changes. The backfill cron
# uses it to identify stale embeddings.
SCHEMA_VERSION: int = 1


class VisionAnalysis(BaseModel):
    """Structured material analysis emitted by the vision model.

    Field set is deliberately conservative — every field is needed by the
    serialiser, none are vestigial. Adding fields requires bumping
    SCHEMA_VERSION and re-embedding (handled by the backfill cron).
    """

    model_config = ConfigDict(extra="forbid")

    # Required — every PRODUCT_IMAGE has at least these.
    material_type: str = Field(
        ...,
        description=(
            "Catalog-grade specific material name. Examples: 'Calacatta marble', "
            "'herringbone white oak engineered wood', 'large-format porcelain tile', "
            "'brushed brass metal', 'frameless tempered safety glass'. Avoid generic "
            "terms like 'metal' or 'wood' — be specific."
        ),
    )

    # Optional but typically present.
    category: Optional[str] = Field(
        None,
        description="High-level grouping. Examples: 'flooring', 'wall covering', "
                    "'countertop', 'fixture', 'upholstery', 'cabinetry'.",
    )
    subcategory: Optional[str] = Field(
        None,
        description="More specific category. Examples: 'porcelain tile', "
                    "'natural stone', 'engineered wood', 'plumbing fixture'.",
    )
    colors: List[str] = Field(
        default_factory=list,
        description="Descriptive colour names (not hex). Examples: "
                    "['warm white', 'grey veining'], ['matte black'], "
                    "['warm honey-brown'].",
    )
    textures: List[str] = Field(
        default_factory=list,
        description="Surface textures observed. Examples: ['veined natural stone', "
                    "'wood grain'], ['ribbed', 'fluted'], ['woven'].",
    )
    finish: Optional[str] = Field(
        None,
        description="Single finish keyword. One of: matte, glossy, satin, "
                    "brushed, honed, polished, textured, rough, patinated, "
                    "lacquered.",
    )
    surface_pattern: Optional[str] = Field(
        None,
        description="Visible pattern/layout. Examples: 'herringbone', 'chevron', "
                    "'stacked bond', 'running bond', 'mosaic', 'grid'.",
    )
    description: Optional[str] = Field(
        None,
        description="One-sentence visual description of the material.",
    )
    applications: List[str] = Field(
        default_factory=list,
        description="Where this material is typically used. Examples: "
                    "['kitchen countertop'], ['bathroom floor', 'wet areas'], "
                    "['feature wall cladding'].",
    )
    style: Optional[str] = Field(
        None,
        description="Aesthetic category. Examples: 'luxury contemporary', "
                    "'Scandinavian', 'industrial', 'classic'.",
    )

    # Confidence the model has in its identification.
    confidence: float = Field(
        0.85,
        ge=0.0,
        le=1.0,
        description="0.0-1.0 confidence in material identification.",
    )

    # Detected text on the image (OCR-style — only what the model can read).
    detected_text: List[str] = Field(
        default_factory=list,
        description="Text visible on or near the material in the image, if any "
                    "(brand names, codes, dimensions). Empty list if none visible.",
    )

    # Schema versioning — set by the model on emit.
    schema_version: int = Field(
        SCHEMA_VERSION,
        description=f"Always emit {SCHEMA_VERSION} (the current schema version).",
    )

    @classmethod
    def anthropic_tool_input_schema(cls) -> Dict[str, Any]:
        """JSON Schema dict suitable for Anthropic tool `input_schema`.

        Strips Pydantic-specific keys ($defs, title at root) that Anthropic
        doesn't need and that occasionally trip the API's stricter parser.
        """
        schema = cls.model_json_schema()
        # Anthropic requires a top-level type=object, properties, required.
        # Pydantic emits all of those, plus harmless extras we keep.
        schema.pop("title", None)
        return schema


# Anthropic tool definition — single export point so callers stay in sync.
VISION_ANALYSIS_TOOL: Dict[str, Any] = {
    "name": "emit_vision_analysis",
    "description": (
        "Emit a structured catalog-grade material analysis for the image. "
        "Every PRODUCT_IMAGE that flows through Stage 3 calls this exactly "
        "once. The output is deterministically serialised and embedded by "
        "Voyage AI for the catalog's understanding-search dimension, so "
        "field consistency matters more than expressive freedom — when in "
        "doubt prefer the canonical material vocabulary."
    ),
    "input_schema": VisionAnalysis.anthropic_tool_input_schema(),
}


def serialize_vision_analysis_to_text(va: VisionAnalysis) -> str:
    """Deterministic text serialisation of VisionAnalysis for Voyage.

    CRITICAL: this function MUST produce byte-identical output for any two
    VisionAnalysis instances with the same field values, regardless of how
    they were constructed or in what order fields were set. Otherwise the
    Voyage embedding for the same material drifts between runs and the
    `image_understanding_embeddings` collection becomes inconsistent.

    Both ingestion and query paths call this — single source of truth.
    """
    parts: List[str] = []

    # Order is fixed and stable across calls. Lists are joined comma-space.
    parts.append(f"Material: {va.material_type}.")

    if va.category:
        cat = f"Category: {va.category}"
        if va.subcategory:
            cat += f", {va.subcategory}"
        parts.append(cat + ".")

    if va.colors:
        parts.append(f"Colors: {', '.join(va.colors)}.")

    if va.textures:
        parts.append(f"Textures: {', '.join(va.textures)}.")

    if va.finish:
        parts.append(f"Finish: {va.finish}.")

    if va.surface_pattern:
        parts.append(f"Pattern: {va.surface_pattern}.")

    if va.style:
        parts.append(f"Style: {va.style}.")

    if va.applications:
        parts.append(f"Applications: {', '.join(va.applications)}.")

    if va.description:
        parts.append(f"Description: {va.description}.")

    if va.detected_text:
        parts.append(f"Text detected: {' '.join(va.detected_text)}.")

    return " ".join(parts)


def vision_analysis_from_legacy_dict(d: Dict[str, Any]) -> Optional[VisionAnalysis]:
    """Best-effort coercion of a legacy free-form vision_analysis dict into
    the strict schema. Used by the backfill cron when re-embedding old rows
    whose stored JSON predates the schema.

    Returns None if the legacy dict can't be coerced (e.g. error payloads,
    missing material_type). Caller should re-run the vision LLM in that case.
    """
    if not isinstance(d, dict):
        return None
    if "error" in d and "material_type" not in d:
        return None

    material_type = d.get("material_type") or d.get("type")
    if not material_type:
        return None

    def _as_list(v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v if x]
        if isinstance(v, dict):
            return [f"{k}: {x}" for k, x in v.items() if x]
        return [str(v)]

    return VisionAnalysis(
        material_type=str(material_type),
        category=d.get("category"),
        subcategory=d.get("subcategory"),
        colors=_as_list(
            d.get("colors") or d.get("color_palette") or d.get("dominant_colors")
        ),
        textures=_as_list(
            d.get("textures") or d.get("texture") or d.get("surface_texture")
        ),
        finish=d.get("finish") or d.get("surface_finish"),
        surface_pattern=d.get("pattern") or d.get("pattern_type") or d.get("surface_pattern"),
        description=d.get("description") or d.get("visual_description"),
        applications=_as_list(
            d.get("applications") or d.get("suitable_for") or d.get("usage")
        ),
        style=d.get("style") or d.get("aesthetic"),
        confidence=float(d.get("confidence", 0.85)),
        detected_text=_as_list(
            d.get("ocr_text") or d.get("detected_text") or d.get("text_content")
        ),
    )
