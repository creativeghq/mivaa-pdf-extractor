"""
Enhanced Material Property Extractor for MIVAA

Extracts 60+ structured material functional properties across 9 categories
using a single Claude Haiku call per product. Outputs match the frontend
FunctionalMetadata interface (src/types/materials.ts).

Integration: Called from stage_4_products.py during Stage 4.7 enrichment.
The extracted properties are merged into products.metadata.functional_properties.
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from app.services.core.supabase_client import get_supabase_client

from app.services.utilities.prompt_registry import PromptNotConfigured, load_prompt

logger = logging.getLogger(__name__)


class PropertyCategory(Enum):
    """Property categories matching frontend FunctionalMetadata interface."""
    SLIP_SAFETY = "slipSafetyRatings"
    GLOSS_REFLECTIVITY = "surfaceGlossReflectivity"
    MECHANICAL = "mechanicalPropertiesExtended"
    THERMAL = "thermalProperties"
    WATER_MOISTURE = "waterMoistureResistance"
    CHEMICAL_HYGIENE = "chemicalHygieneResistance"
    ACOUSTIC_ELECTRICAL = "acousticElectricalProperties"
    SUSTAINABILITY = "environmentalSustainability"
    DIMENSIONAL_AESTHETIC = "dimensionalAesthetic"


@dataclass
class PropertyExtractionResult:
    """Result of enhanced material property extraction."""
    properties: Dict[str, Any]  # keyed by PropertyCategory.value
    confidence: float
    coverage_pct: float
    processing_time_ms: int
    method: str  # "claude_haiku" or "rule_based"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "properties": self.properties,
            "confidence": self.confidence,
            "coverage_pct": self.coverage_pct,
            "processing_time_ms": self.processing_time_ms,
            "method": self.method,
        }


# ─── Single-call extraction prompt ───────────────────────────────────────────

# Prompt: extraction / entity_creation / material_properties_system
# (#347 phase 3P — was hardcoded here AND used as the `or` fallback below).



class EnhancedMaterialPropertyExtractor:
    """
    Single-call LLM property extractor using Claude Haiku.

    Makes ONE Claude call per product to extract all 9 property categories,
    with rule-based fallback if Claude is unavailable.
    """

    def __init__(self, confidence_threshold: float = 0.5, workspace_id: str = None):
        from app.config import get_settings
        self.confidence_threshold = confidence_threshold
        self.workspace_id = workspace_id or get_settings().default_workspace_id
        self.supabase = get_supabase_client()
        self._db_prompt: Optional[str] = None
        # NOT loaded here: _load_db_prompt is async now (it must be, to reach the prompt store),
        # and calling it from __init__ would leave an un-awaited coroutine and a None prompt.
        # extract() loads it.

    async def _load_db_prompt(self) -> None:
        """Load the extraction system prompt. Raises if it is not configured.

        This used to swallow every exception and fall through to a built-in constant via
        `self._db_prompt or EXTRACTION_SYSTEM_PROMPT`, so a workspace whose prompt failed to
        load silently ran on the built-in while the admin UI showed their edit saved.
        """
        self._db_prompt = await load_prompt(
            "extraction", "material_properties_system",
            stage="entity_creation", workspace_id=self.workspace_id,
        )

    @property
    def system_prompt(self) -> str:
        if not self._db_prompt:
            raise PromptNotConfigured(
                "system_prompt read before _load_db_prompt() — there is no built-in fallback "
                "(#347 phase 3P)."
            )
        return self._db_prompt

    async def extract(
        self,
        analysis_text: str,
        product_name: str = "",
        document_context: str = "",
        job_id: str = None,
        product_id: Optional[str] = None,
    ) -> PropertyExtractionResult:
        """
        Extract all material properties in a single LLM call.

        Args:
            analysis_text: Combined chunk text + vision analysis for the product
            product_name: Product name for context
            document_context: Additional document-level context
            job_id: For logging

        Returns:
            PropertyExtractionResult with structured properties
        """
        if self._db_prompt is None:
            await self._load_db_prompt()
        start_ms = int(time.time() * 1000)

        # Build user message
        user_content = f"Product: {product_name}\n\n" if product_name else ""
        user_content += analysis_text
        if document_context:
            user_content += f"\n\nAdditional context:\n{document_context}"

        # Truncate to ~12k chars to stay within Haiku limits
        if len(user_content) > 12000:
            user_content = user_content[:12000] + "\n...[truncated]"

        # Try Claude Haiku first
        properties = await self._extract_with_claude(
            user_content, job_id, product_id=product_id,
        )
        method = "claude_haiku"

        # Fallback to rule-based if Claude fails
        if properties is None:
            properties = self._rule_based_extraction(analysis_text)
            method = "rule_based"

        elapsed_ms = int(time.time() * 1000) - start_ms

        # Calculate coverage and confidence
        filled_categories = sum(
            1 for cat in PropertyCategory
            if cat.value in properties and properties[cat.value]
        )
        total_categories = len(PropertyCategory)
        coverage_pct = round((filled_categories / total_categories) * 100, 1)

        # Average per-category confidence
        confidences = []
        for cat in PropertyCategory:
            cat_data = properties.get(cat.value)
            if isinstance(cat_data, dict):
                conf = cat_data.pop("confidence", None)
                if conf is not None:
                    confidences.append(float(conf))
                # Remove empty categories
                if not cat_data:
                    properties.pop(cat.value, None)

        avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

        logger.info(
            f"Property extraction for '{product_name}': "
            f"{filled_categories}/{total_categories} categories, "
            f"{coverage_pct}% coverage, {avg_confidence:.2f} confidence, "
            f"{elapsed_ms}ms ({method})"
        )

        return PropertyExtractionResult(
            properties=properties,
            confidence=avg_confidence,
            coverage_pct=coverage_pct,
            processing_time_ms=elapsed_ms,
            method=method,
        )

    async def _extract_with_claude(
        self,
        user_content: str,
        job_id: str = None,
        product_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Single Claude Haiku call to extract all property categories.

        Routes through `tracked_claude_call_async` so the call is logged to
        `ai_usage_logs` with `job_id` + `product_id` for per-product cost
        attribution. Previously bypassed AICallLogger via raw client.messages.create
        — this hid all property-extraction spend from the dashboard and broke
        the per-product cost rollup at job completion.
        """
        try:
            from app.services.core.claude_helper import tracked_claude_call_async

            response = await tracked_claude_call_async(
                task="material_property_extraction",
                model="claude-haiku-4-5",
                max_tokens=2048,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_content}],
                confidence_score=0.8,
                confidence_breakdown={
                    "model_confidence": 0.85,
                    "completeness": 0.80,
                },
                job_id=job_id,
                workspace_id=self.workspace_id,
                product_id=product_id,
            )

            text = response.content[0].text if response.content else ""

            # Parse JSON from response
            return self._parse_json_response(text)

        except Exception as e:
            logger.warning(f"Claude property extraction failed: {e}")
            return None

    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from Claude response, handling markdown fences."""
        # Strip markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fallback: find first { ... } block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse JSON from Claude response")
        return None

    def _rule_based_extraction(self, text: str) -> Dict[str, Any]:
        """Regex-based fallback extraction when Claude is unavailable."""
        text_lower = text.lower()
        properties: Dict[str, Any] = {}

        # Slip safety
        slip = {}
        r_matches = re.findall(r"\b(R(?:9|10|11|12|13))\b", text, re.IGNORECASE)
        if r_matches:
            slip["rValue"] = list({m.upper() for m in r_matches})
        dcof = re.findall(r"dcof[:\s]*([0-9]+\.?[0-9]*)", text_lower)
        if dcof:
            vals = [float(v) for v in dcof if 0 <= float(v) <= 1]
            if vals:
                slip["dcofRange"] = [min(vals), max(vals)]
        if slip:
            slip["confidence"] = 0.65
            properties[PropertyCategory.SLIP_SAFETY.value] = slip

        # Mechanical
        mech = {}
        mohs = re.findall(r"mohs[:\s]+(?:hardness[:\s]+)?([0-9]+\.?[0-9]*)", text_lower)
        if mohs:
            vals = [float(v) for v in mohs if 1 <= float(v) <= 10]
            if vals:
                mech["mohsHardnessRange"] = [min(vals), max(vals)]
        pei = re.findall(r"pei[:\s]+(?:rating[:\s]+)?(?:class[:\s]+)?([0-5])", text_lower)
        if pei:
            vals = [int(v) for v in pei]
            if vals:
                mech["peiRating"] = sorted(set(vals))
        if mech:
            mech["confidence"] = 0.6
            properties[PropertyCategory.MECHANICAL.value] = mech

        # Water absorption
        water = {}
        abs_match = re.findall(r"water\s+absorption[:\s]*([0-9]+\.?[0-9]*)\s*%", text_lower)
        if abs_match:
            vals = [float(v) for v in abs_match if 0 <= float(v) <= 100]
            if vals:
                water["waterAbsorptionRange"] = [min(vals), max(vals)]
        if "frost" in text_lower and ("resist" in text_lower or "proof" in text_lower):
            water["frostResistance"] = True
        if water:
            water["confidence"] = 0.6
            properties[PropertyCategory.WATER_MOISTURE.value] = water

        # Gloss
        gloss = {}
        gloss_keywords = ["super-polished", "polished", "satin", "semi-polished", "matte", "velvet", "anti-glare"]
        found_gloss = [g for g in gloss_keywords if g in text_lower]
        if found_gloss:
            gloss["glossLevel"] = found_gloss
            gloss["confidence"] = 0.7
            properties[PropertyCategory.GLOSS_REFLECTIVITY.value] = gloss

        # Shade variation
        aesthetic = {}
        shade = re.findall(r"\b(V[1-4])\b", text)
        if shade:
            aesthetic["shadeVariation"] = shade[0]
        if "rectified" in text_lower:
            aesthetic["rectifiedEdges"] = True
        if aesthetic:
            aesthetic["confidence"] = 0.7
            properties[PropertyCategory.DIMENSIONAL_AESTHETIC.value] = aesthetic

        # Sustainability
        sustainability = {}
        if "greenguard" in text_lower:
            if "gold" in text_lower:
                sustainability["greenguardLevel"] = "gold"
            else:
                sustainability["greenguardLevel"] = "certified"
        recycled = re.findall(r"(\d+)\s*%\s*recycled", text_lower)
        if recycled:
            vals = [int(v) for v in recycled if 0 <= int(v) <= 100]
            if vals:
                sustainability["totalRecycledContentRange"] = [min(vals), max(vals)]
        if sustainability:
            sustainability["confidence"] = 0.6
            properties[PropertyCategory.SUSTAINABILITY.value] = sustainability

        return properties


async def extract_functional_properties(
    analysis_text: str,
    product_name: str = "",
    document_context: str = "",
    job_id: str = None,
    workspace_id: str = None,
    product_id: Optional[str] = None,
) -> PropertyExtractionResult:
    """
    Convenience function for stage_4_products.py integration.

    Usage in stage_4_products.py:
        from app.services.products.enhanced_material_property_extractor import (
            extract_functional_properties,
        )
        result = await extract_functional_properties(
            analysis_text=combined_chunk_text,
            product_name=product_name,
            job_id=job_id,
        )
        if result.coverage_pct > 0:
            new_metadata["functional_properties"] = result.properties
    """
    extractor = EnhancedMaterialPropertyExtractor(
        workspace_id=workspace_id,
    )
    return await extractor.extract(
        analysis_text=analysis_text,
        product_name=product_name,
        document_context=document_context,
        job_id=job_id,
        product_id=product_id,
    )
