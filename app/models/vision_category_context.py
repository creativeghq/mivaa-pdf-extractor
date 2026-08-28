"""
Category context handed to the vision model (issue #393 Step 3).

STDLIB ONLY — same contract as `ocr_context.py`. MIVAA CI installs pytest and
nothing else, so keeping the formatting importable by path is what lets the guard
assert on behaviour instead of grepping source.

Why this exists: `Material Image Analyzer` is ONE prompt row, so a tile, a tap and a
pendant lamp were all asked the same generic question. Meanwhile
`material_categories` and `material_metadata_fields` already carry per-category
extraction tips, a controlled vocabulary, and an explicit do-not-extract list — the
registry knows that `body_material` means vitreous china in sanitary and rattan in
lighting, and the vision prompt was not asking.

WHAT IS DELIBERATELY NOT INCLUDED
---------------------------------
`FieldRegistry.priority_fields_prompt()` exists and is NOT used here. It enumerates
the PRODUCT extraction schema (`material_subtype`, `physical_properties`, …), which is
a different schema from `VisionAnalysis`. Feeding it to a call that is
`strict`-constrained to the vision tool would instruct the model to emit fields its
tool cannot accept — a contradiction the model resolves by guessing. Stage 4 is where
those fields belong.

What DOES transfer is the category's judgement, not its field list: what to look for,
what vocabulary to prefer, and what is not present on this kind of product.
"""

from typing import List, Optional, Sequence

#: Fenced for the same reason the OCR block is: category rows are admin-editable and
#: vocabulary values arrive from supplier imports, so this is data the model reads,
#: not instructions it obeys.
CATEGORY_BLOCK_OPEN = "<category_context>"
CATEGORY_BLOCK_CLOSE = "</category_context>"

#: Said when the image's category is unknown. Stated explicitly rather than rendering
#: an empty block: an empty block reads as "this category has no guidance", which is a
#: different and misleading claim.
_UNKNOWN = (
    "The category of this product is not known. Analyze the image on its own terms "
    "and do not assume any category-specific convention applies."
)


def build_category_context_block(
    *,
    category_key: Optional[str],
    display_name: Optional[str] = None,
    description: Optional[str] = None,
    extraction_tips: Optional[Sequence[str]] = None,
    controlled_vocab: Optional[Sequence[str]] = None,
    skip_fields: Optional[Sequence[str]] = None,
) -> str:
    """Render the category guidance block, including the unknown-category case."""
    if not category_key:
        return f"{CATEGORY_BLOCK_OPEN}\n{_UNKNOWN}\n{CATEGORY_BLOCK_CLOSE}"

    name = display_name or category_key
    lines: List[str] = [f"This image is catalogued under: {name}."]

    if description:
        lines.append(description.strip())

    if extraction_tips:
        lines.append("")
        lines.append(f"What to look for on {name} products:")
        lines.extend(f"- {t}" for t in extraction_tips if t and str(t).strip())

    if controlled_vocab:
        lines.append("")
        lines.append(
            "Preferred vocabulary for this category — use one of these exactly when "
            "one of them fits, so the same material is described the same way every "
            "time: " + ", ".join(str(v) for v in controlled_vocab if v)
        )

    if skip_fields:
        # Phrased as "not present", not "do not output". These names are product-schema
        # fields that VisionAnalysis has no slot for anyway; the useful signal is that
        # this category genuinely lacks those PROPERTIES, which is a reason not to
        # invent them in `description` either.
        lines.append("")
        lines.append(
            f"{name} products do not have: "
            + ", ".join(str(s) for s in skip_fields if s)
            + ". Do not infer or describe these."
        )

    body = "\n".join(lines).strip()
    return f"{CATEGORY_BLOCK_OPEN}\n{body}\n{CATEGORY_BLOCK_CLOSE}"
