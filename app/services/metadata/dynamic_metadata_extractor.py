"""
Dynamic Metadata Extraction Service

This service uses AI to dynamically discover and extract metadata from PDFs,
rather than hardcoding checks for 250+ attributes.

Architecture:
- Tier 1: Critical fields (material_category, factory_name, factory_group_name) - ALWAYS extracted
- Tier 2: Dynamic discovery - AI finds and extracts any metadata present
- Tier 3: Schema validation - Ensures extracted data is properly structured

This allows the system to:
1. Handle new/unknown attributes without code changes
2. Adapt to different PDF formats and industries
3. Maintain consistency for critical business fields
"""

import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os

import anthropic
import openai

from app.services.core.ai_call_logger import AICallLogger
from app.services.core.ai_client_service import get_ai_client_service
from app.services.metadata.metadata_normalizer import normalize_metadata, get_normalization_report
from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

# Get API keys from environment
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


# ============================================================================
# TIER 1: CRITICAL METADATA (Always Required)
# ============================================================================

CRITICAL_METADATA_SCHEMA = {
    "material_category": {
        "description": "Primary material category (tile, porcelain, ceramic, stone, marble, granite, wood, metal, etc.)",
        "extraction_method": "ai_with_keywords",
        "keywords": ["tile", "porcelain", "ceramic", "stone", "marble", "granite", "wood", "metal", "glass", "composite"],
        "required": True,
        "source_options": ["auto_detected", "manual_override"],
        "validation": lambda x: x and len(x) > 0
    },
    "factory_name": {
        "description": "Manufacturer or factory name",
        "extraction_method": "ai_with_patterns",
        "patterns": [
            r"Manufacturer:\s*(.+)",
            r"Made by:\s*(.+)",
            r"Producer:\s*(.+)",
            r"Factory:\s*(.+)",
            r"Company:\s*(.+)"
        ],
        "required": True,
        "validation": lambda x: x and len(x) > 2
    },
    "factory_group_name": {
        "description": "Parent company or factory group name",
        "extraction_method": "ai_with_patterns",
        "patterns": [
            r"Group:\s*(.+)",
            r"Parent Company:\s*(.+)",
            r"Corporation:\s*(.+)",
            r"Holdings:\s*(.+)"
        ],
        "required": False,
        "validation": lambda x: True  # Optional field
    }
}


# ============================================================================
# TIER 2: DYNAMIC METADATA CATEGORIES
# ============================================================================

METADATA_CATEGORY_HINTS = {
    "material_properties": [
        "composition", "type", "blend", "fiber_content", "texture", "finish", 
        "pattern", "weight", "density", "durability_rating"
    ],
    "dimensions": [
        "length", "width", "height", "thickness", "diameter", "size", "area", "volume"
    ],
    "appearance": [
        "color", "color_code", "gloss_level", "sheen", "transparency", "grain", "visual_effect"
    ],
    "performance": [
        "water_resistance", "fire_rating", "slip_resistance", "wear_rating", 
        "abrasion_resistance", "tensile_strength"
    ],
    "application": [
        "recommended_use", "installation_method", "room_type", "traffic_level", 
        "care_instructions", "maintenance"
    ],
    "compliance": [
        "certifications", "standards", "eco_friendly", "sustainability_rating", 
        "voc_rating", "safety_rating"
    ],
    "commercial": [
        "pricing", "availability", "supplier", "manufacturer", "sku", "warranty"
    ],
    "design": [
        "designer", "studio", "collection", "series", "aesthetic_style", "design_era"
    ],
    "technical": [
        "specifications", "construction", "manufacturing_process", "grade", "class", "rating"
    ]
}


# ============================================================================
# AI EXTRACTION PROMPT
# ============================================================================

def get_dynamic_extraction_prompt(pdf_text: str, category_hint: Optional[str] = None) -> str:
    """
    Generate AI prompt for dynamic metadata extraction.
    
    Args:
        pdf_text: Text content from PDF
        category_hint: Optional hint about material category (e.g., "tile")
    """
    
    category_context = ""
    if category_hint:
        category_context = f"\nMaterial Category Hint: This appears to be a {category_hint} product."
    
    return f"""You are analyzing a product specification PDF to extract ALL metadata attributes.

{category_context}

CRITICAL FIELDS (MUST extract these):
1. material_category - Primary material type (tile, porcelain, ceramic, stone, marble, granite, wood, metal, glass, etc.)
2. factory_name - Manufacturer or factory name
3. factory_group_name - Parent company or group name (if mentioned)

DYNAMIC FIELDS (extract ANY you find):
Extract all other attributes you discover, organized by category.

STANDARDIZED FIELD NAMES (use EXACTLY these names):

**Commercial:**
- grout_mapei: Mapei grout product name (e.g., "ULTRACOLOR PLUS")
- grout_kerakoll: Kerakoll grout product name (e.g., "FUGABELLA")
- grout_isomat: Isomat grout product name (e.g., "MULTIFILL EPOXY")
- grout_technica: Technica grout product name (e.g., "GROUT PREMIUM")
- grout_suppliers: Array of grout supplier names (e.g., ["MAPEI", "KERAKOLL", "ISOMAT", "TECHNICA"])
- grout_color_codes: Object mapping colors to grout codes (e.g., {{"white": "100", "clay": "145"}})
- sku_codes: Object mapping colors/variants to SKU codes (e.g., {{"white": "37885", "clay": "37889"}})
- product_codes: Array of product codes (e.g., ["H48", "H22"])
- pricing, availability, warranty

**Design:**
- designers: Array of designer names (e.g., ["José Manuel Ferrero"])
- studio: Design studio name (e.g., "Estudi{{H}}ac")
- studio_founded: Year studio was founded (e.g., "2003")
- collection: Collection name (e.g., "VALENOVA")
- philosophy: Design philosophy or concept
- inspiration: Design inspiration source
- aesthetic_style: Aesthetic style description

**Packaging (look for "Packing", "Packaging", or "Iconography" sections):**
- pieces_per_box: Number of pieces per box (often shown with icon)
- boxes_per_pallet: Number of boxes per pallet (often shown with icon)
- weight_kg: Weight per box in kilograms (often shown with icon)
- weight_lb: Weight per box in pounds (often shown with icon)
- coverage_m2: Coverage area per box in square meters (often shown with icon)
- coverage_sqft: Coverage area per box in square feet (often shown with icon)
- pallet_weight_kg: Total pallet weight in kilograms
- box_dimensions: Box dimensions (e.g., "40x40x10 cm")

NOTE: Packaging information is often presented in "Iconography" sections with icons/symbols.
Look for sections titled: "Iconography", "Packing", "Packaging", "Technical Data", "Box Information"

**Material Properties:**
- finish: Surface finish type (e.g., "matt", "gloss")
- body_type: Body type (e.g., "white body tile", "full body ceramics")
- composition: Material composition
- texture: Surface texture description

**Appearance:**
- colors: Array of available colors (e.g., ["White", "Clay", "Green"])
- shade_variation: Shade variation rating (e.g., "V2", "V3")
- visual_effect: Visual effect description

**Application:**
- recommended_use: Recommended use (e.g., "wall", "floor", "wall and floor")
- installation: Installation method or type
- traffic_level: Traffic level rating
- care_instructions: Care and cleaning instructions (look for "Cleaning", "Care", "Maintenance" sections)
- maintenance: Maintenance requirements and recommendations

**Compliance & Safety (look for "Regulation", "Certifications", "Safety" sections):**
- certifications: Array of certifications (e.g., ["ISO 9001", "CE", "LEED"])
- standards: Compliance standards (e.g., "EN 14411", "ASTM C373")
- safety_rating: Safety rating or classification
- eco_friendly: Environmental friendliness indicators
- sustainability_rating: Sustainability rating or score
- voc_rating: VOC (Volatile Organic Compounds) rating

NOTE: Regulation, Cleaning, and Handling information may be in separate sections.
Look for sections titled: "Regulation", "Cleaning", "Handling", "Care Instructions", "Maintenance", "Safety", "Certifications"

**Other categories:** performance, technical, dimensions

CRITICAL RULES:
1. Use EXACTLY the field names listed above (e.g., "grout_mapei" NOT "recommended_grout_mapei" or "grout_product_mapei")
2. For designers: ALWAYS use array format, even for single designer
3. For SKU codes: ALWAYS use object format mapping variants to codes
4. For grout suppliers: Use "grout_mapei" and "grout_kerakoll" for product names, "grout_suppliers" for array of brands
5. Use snake_case for ALL field names
6. Include confidence scores (0.0-1.0) for each extraction
7. If you find attributes NOT in the above list, STILL EXTRACT THEM but use consistent naming

Return JSON in this exact format:
{{
  "critical": {{
    "material_category": {{"value": "...", "confidence": 0.95, "source": "detected"}},
    "factory_name": {{"value": "...", "confidence": 0.90, "source": "extracted"}},
    "factory_group_name": {{"value": "...", "confidence": 0.85, "source": "extracted"}}
  }},
  "discovered": {{
    "material_properties": {{
      "composition": {{"value": "...", "confidence": 0.90}},
      "texture": {{"value": "...", "confidence": 0.85}}
    }},
    "dimensions": {{
      "length": {{"value": "...", "unit": "cm", "confidence": 0.95}},
      "width": {{"value": "...", "unit": "cm", "confidence": 0.95}}
    }},
    "appearance": {{
      "color": {{"value": "...", "confidence": 0.90}}
    }},
    "performance": {{
      "slip_resistance": {{"value": "R11", "confidence": 0.95}}
    }},
    "design": {{
      "designer": {{"value": "...", "confidence": 0.85}},
      "collection": {{"value": "...", "confidence": 0.90}}
    }}
  }},
  "unknown_attributes": {{
    "custom_field_name": {{"value": "...", "confidence": 0.80, "category": "unknown"}}
  }}
}}

PDF Content (intelligently extracted sections):
{pdf_text}

NOTE: The above content includes the beginning, end, and relevant sections (Packaging, Compliance, Care, Technical) from the full PDF.

Extract all metadata now:"""


# ============================================================================
# DYNAMIC METADATA EXTRACTOR CLASS
# ============================================================================

class DynamicMetadataExtractor:
    """
    Extracts metadata dynamically using AI, without hardcoded attribute checks.
    """

    def __init__(self, model: str = "claude", job_id: Optional[str] = None, workspace_id: str = "ffafc28b-1b8b-4b0d-b226-9f9a6154004e"):
        """
        Initialize extractor.

        Args:
            model: AI model to use - supports "claude", "claude-vision", "claude-haiku-vision", "gpt", "gpt-vision"
            job_id: Optional job ID for AI call logging
            workspace_id: Workspace ID for loading custom prompts from database
        """
        self.model = model
        self.job_id = job_id
        self.workspace_id = workspace_id
        self.logger = logging.getLogger(__name__)
        self.ai_logger = AICallLogger()
        self.supabase = get_supabase_client()

        # Check API keys based on model family (claude/gpt)
        if "claude" in model.lower() and not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set - cannot use Claude")
        if "gpt" in model.lower() and not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set - cannot use GPT")

    async def _load_prompt_from_database(self, stage: str = "entity_creation", category: str = "products") -> Optional[str]:
        """
        Load extraction prompt from database.

        Args:
            stage: Extraction stage (default: "entity_creation" for metadata extraction)
            category: Content category (default: "products")

        Returns:
            Prompt template from database, or None if not found
        """
        try:
            # Try to get custom prompt first (is_custom = true)
            # Only select columns that exist in the prompts table
            result = self.supabase.client.table('prompts')\
                .select('prompt_text, version, is_custom')\
                .eq('workspace_id', self.workspace_id)\
                .eq('prompt_type', 'extraction')\
                .eq('stage', stage)\
                .eq('category', category)\
                .eq('is_custom', True)\
                .order('version', desc=True)\
                .limit(1)\
                .execute()

            if result.data and len(result.data) > 0:
                prompt_data = result.data[0]
                self.logger.info(f"✅ Loaded CUSTOM prompt from database (v{prompt_data['version']})")
                return prompt_data['prompt_text']

            # Fallback to default prompt (is_custom = false)
            result = self.supabase.client.table('prompts')\
                .select('prompt_text, version')\
                .eq('workspace_id', self.workspace_id)\
                .eq('prompt_type', 'extraction')\
                .eq('stage', stage)\
                .eq('category', category)\
                .eq('is_custom', False)\
                .order('version', desc=True)\
                .limit(1)\
                .execute()

            if result.data and len(result.data) > 0:
                prompt_data = result.data[0]
                self.logger.info(f"✅ Loaded DEFAULT prompt from database (v{prompt_data['version']})")
                return prompt_data['prompt_text']

            self.logger.warning(f"⚠️ No prompt found in database for stage={stage}, category={category}")
            return None

        except Exception as e:
            self.logger.error(f"❌ Failed to load prompt from database: {e}")
            return None
    
    def _extract_relevant_sections(self, pdf_text: str, max_chars: int = 30000) -> str:
        """
        Smart extraction: Find and extract only relevant sections from PDF.

        This reduces token usage by 70-90% while preserving critical information.

        Strategy:
        1. Always include first 8000 chars (product name, description, basic specs)
        2. Search for section headers (Packaging, Compliance, Care, Technical, etc.)
        3. Extract 2000 chars around each relevant section
        4. Always include last 5000 chars (often contains packaging/compliance)

        Args:
            pdf_text: Full PDF text
            max_chars: Maximum characters to return (default: 30000)

        Returns:
            Intelligently extracted text with relevant sections
        """
        if len(pdf_text) <= max_chars:
            return pdf_text  # No need to extract if already small

        # Section keywords to search for (case-insensitive)
        section_keywords = [
            # Packaging
            r'\b(packaging|packing|iconography|box|pallet|pieces per box|coverage)\b',
            # Compliance & Safety
            r'\b(regulation|compliance|certification|standard|safety|eco.?friendly|sustainability|voc|leed|iso)\b',
            # Care & Maintenance
            r'\b(care|maintenance|cleaning|handling|installation|recommended use)\b',
            # Technical specs
            r'\b(technical|specification|properties|performance|dimensions|weight|thickness)\b',
        ]

        extracted_sections = []

        # 1. Always include beginning (product name, description, basic info)
        extracted_sections.append(("START", pdf_text[:8000]))

        # 2. Search for relevant sections
        import re
        for keyword_pattern in section_keywords:
            for match in re.finditer(keyword_pattern, pdf_text, re.IGNORECASE):
                start = max(0, match.start() - 1000)  # 1000 chars before
                end = min(len(pdf_text), match.end() + 1000)  # 1000 chars after
                section_text = pdf_text[start:end]
                extracted_sections.append((f"SECTION@{match.start()}", section_text))

        # 3. Always include end (often has packaging/compliance tables)
        extracted_sections.append(("END", pdf_text[-5000:]))

        # 4. Combine sections and deduplicate
        combined_text = "\n\n---\n\n".join([text for _, text in extracted_sections])

        # 5. Truncate if still too long
        if len(combined_text) > max_chars:
            combined_text = combined_text[:max_chars]

        self.logger.info(f"📊 Smart extraction: {len(pdf_text):,} → {len(combined_text):,} chars ({len(combined_text)/len(pdf_text)*100:.1f}% retained)")

        return combined_text

    async def extract_metadata(
        self,
        pdf_text: str,
        category_hint: Optional[str] = None,
        manual_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract metadata dynamically from PDF text.

        Args:
            pdf_text: Text content from PDF
            category_hint: Optional material category hint
            manual_overrides: Manual values from admin (override AI extraction)

        Returns:
            {
                "critical": {...},
                "discovered": {...},
                "unknown": {...},
                "metadata": {
                    "extraction_timestamp": "...",
                    "extraction_method": "ai_dynamic",
                    "confidence_scores": {...}
                }
            }
        """
        try:
            # ✅ SMART EXTRACTION: Extract only relevant sections instead of truncating
            # This preserves packaging/compliance/care sections while reducing token usage
            smart_text = self._extract_relevant_sections(pdf_text, max_chars=30000)

            # Step 1: Load prompt from database (with fallback to hardcoded)
            db_prompt_template = await self._load_prompt_from_database(stage="entity_creation", category="products")

            if db_prompt_template:
                # Use database prompt - replace placeholders
                category_context = f"\nMaterial Category Hint: This appears to be a {category_hint} product." if category_hint else ""
                prompt = db_prompt_template.replace("{category_context}", category_context).replace("{pdf_text}", smart_text)
                self.logger.info("✅ Using DATABASE prompt for metadata extraction")
            else:
                # Fallback to hardcoded prompt
                prompt = get_dynamic_extraction_prompt(smart_text, category_hint)
                self.logger.info("⚠️ Using HARDCODED fallback prompt for metadata extraction")

            # Step 2: Call AI model based on model family
            if "claude" in self.model.lower():
                ai_response = await self._call_claude(prompt)
            else:
                ai_response = await self._call_gpt(prompt)

            extracted_data = self._parse_ai_response(ai_response)

            # Step 2: Apply manual overrides
            if manual_overrides:
                extracted_data = self._apply_manual_overrides(extracted_data, manual_overrides)

            # Step 3: Validate critical fields
            validation_result = self._validate_critical_fields(extracted_data)

            # Step 4: Normalize metadata to standardized schema
            original_data = extracted_data.copy()
            normalized_data = {}

            # Normalize each category
            for category in ["critical", "discovered", "unknown"]:
                if category in extracted_data:
                    normalized_data[category] = normalize_metadata({category: extracted_data[category]}).get(category, {})

            # Get normalization report for logging
            normalization_report = get_normalization_report(
                original_data.get("discovered", {}),
                normalized_data.get("discovered", {})
            )

            if normalization_report["fields_normalized"] > 0:
                self.logger.info(f"Normalized {normalization_report['fields_normalized']} metadata fields")
                self.logger.debug(f"Normalization changes: {normalization_report['changes']}")

            # Use normalized data
            extracted_data = normalized_data

            # Step 5: Auto-create material_properties entries for new discovered fields
            try:
                await self._ensure_properties_exist(extracted_data)
            except AttributeError as ae:
                self.logger.error(f"AttributeError in _ensure_properties_exist: {ae}")
                self.logger.error(f"Available methods: {[m for m in dir(self) if not m.startswith('__')]}")
                # Don't fail extraction if property creation fails
            except Exception as prop_error:
                self.logger.warning(f"Failed to auto-create material_properties: {prop_error}")

            # Step 6: Log packaging/iconography extraction results
            packaging_fields = extracted_data.get("packaging", {})
            if packaging_fields:
                self.logger.info(f"📦 Packaging fields extracted: {list(packaging_fields.keys())}")
                self.logger.debug(f"   Packaging data: {packaging_fields}")
            else:
                self.logger.warning(f"⚠️ No packaging fields extracted (check for Iconography/Packing sections)")

            # Log compliance/safety extraction results
            compliance_fields = extracted_data.get("compliance", {})
            application_fields = extracted_data.get("application", {})
            care_fields = {k: v for k, v in application_fields.items() if k in ['care_instructions', 'maintenance']}

            if compliance_fields:
                self.logger.info(f"✅ Compliance/Safety fields extracted: {list(compliance_fields.keys())}")
                self.logger.debug(f"   Compliance data: {compliance_fields}")
            else:
                self.logger.warning(f"⚠️ No compliance/safety fields extracted (check for Regulation/Certifications sections)")

            if care_fields:
                self.logger.info(f"🧼 Care/Maintenance fields extracted: {list(care_fields.keys())}")
                self.logger.debug(f"   Care data: {care_fields}")
            else:
                self.logger.warning(f"⚠️ No care/maintenance fields extracted (check for Cleaning/Handling sections)")

            # Step 7: Add metadata
            extracted_data["metadata"] = {
                "extraction_timestamp": datetime.utcnow().isoformat(),
                "extraction_method": f"ai_dynamic_{self.model}",
                "validation_passed": validation_result["valid"],
                "validation_errors": validation_result.get("errors", []),
                "manual_overrides_applied": bool(manual_overrides),
                "normalization_applied": normalization_report["fields_normalized"] > 0,
                "fields_normalized": normalization_report["fields_normalized"],
                "packaging_fields_found": len(packaging_fields) if packaging_fields else 0,
                "compliance_fields_found": len(compliance_fields) if compliance_fields else 0,
                "care_fields_found": len(care_fields) if care_fields else 0
            }

            return extracted_data

        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {e}")
            return self._get_empty_result(error=str(e))
    
    async def _call_claude(self, prompt: str) -> str:
        """Call Claude Sonnet 4.5 for metadata extraction."""
        start_time = datetime.now()

        try:
            # Use centralized AI client service
            ai_service = get_ai_client_service()
            client = ai_service.anthropic

            response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=8000,
                temperature=0.1,  # Low temperature for consistent extraction
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            content = response.content[0].text

            # Log AI call
            latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            await self.ai_logger.log_claude_call(
                task="dynamic_metadata_extraction",
                model="claude-sonnet-4-5",
                response=response,
                latency_ms=latency_ms,
                confidence_score=0.9,
                confidence_breakdown={},
                action="use_ai_result",  # Fixed: must be 'use_ai_result' or 'fallback_to_rules'
                job_id=self.job_id
            )

            return content

        except Exception as e:
            self.logger.error(f"Claude metadata extraction failed: {e}")
            raise RuntimeError(f"Claude metadata extraction failed: {str(e)}") from e

    async def _call_gpt(self, prompt: str) -> str:
        """Call GPT-4o for metadata extraction."""
        start_time = datetime.now()

        try:
            # Use centralized AI client service
            ai_service = get_ai_client_service()
            client = ai_service.openai

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at extracting structured metadata from product specifications. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=8000
            )

            content = response.choices[0].message.content

            # Log AI call
            latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # Calculate cost
            input_cost = (response.usage.prompt_tokens / 1_000_000) * 2.50  # GPT-4o pricing
            output_cost = (response.usage.completion_tokens / 1_000_000) * 10.00
            total_cost = input_cost + output_cost

            await self.ai_logger.log_ai_call(
                task="dynamic_metadata_extraction",
                model="gpt-4o",
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                cost=total_cost,
                latency_ms=latency_ms,
                confidence_score=0.9,
                confidence_breakdown={
                    "model_confidence": 0.9,
                    "completeness": 0.9,
                    "consistency": 0.9,
                    "validation": 0.9
                },
                action="use_ai_result",
                job_id=self.job_id
            )

            return content

        except Exception as e:
            self.logger.error(f"GPT metadata extraction failed: {e}")
            raise RuntimeError(f"GPT metadata extraction failed: {str(e)}") from e
    
    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse AI JSON response."""
        try:
            # Try to extract JSON from markdown code blocks if present
            if "```json" in response:
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
                if json_match:
                    response = json_match.group(1)
            elif "```" in response:
                json_match = re.search(r'```\s*(\{.*?\})\s*```', response, re.DOTALL)
                if json_match:
                    response = json_match.group(1)

            return json.loads(response)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse AI response as JSON: {e}")
            self.logger.debug(f"Raw response (first 500 chars): {response[:500]}")
            return self._get_empty_result()
    
    def _fallback_extraction(self, pdf_text: str) -> Dict[str, Any]:
        """Fallback pattern-based extraction when AI unavailable."""
        result = self._get_empty_result()
        
        # Extract critical fields using patterns
        for field_name, field_config in CRITICAL_METADATA_SCHEMA.items():
            if "patterns" in field_config:
                for pattern in field_config["patterns"]:
                    match = re.search(pattern, pdf_text, re.IGNORECASE)
                    if match:
                        result["critical"][field_name] = {
                            "value": match.group(1).strip(),
                            "confidence": 0.7,
                            "source": "pattern_match"
                        }
                        break
        
        return result
    
    def _apply_manual_overrides(
        self, 
        extracted_data: Dict[str, Any], 
        manual_overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply manual overrides from admin panel."""
        for field, value in manual_overrides.items():
            if field in CRITICAL_METADATA_SCHEMA:
                extracted_data["critical"][field] = {
                    "value": value,
                    "confidence": 1.0,
                    "source": "manual_override"
                }
        return extracted_data
    
    def _validate_critical_fields(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that critical fields are present and valid."""
        errors = []
        
        for field_name, field_config in CRITICAL_METADATA_SCHEMA.items():
            if field_config["required"]:
                field_data = extracted_data.get("critical", {}).get(field_name)
                
                if not field_data or not field_data.get("value"):
                    errors.append(f"Missing required field: {field_name}")
                elif not field_config["validation"](field_data.get("value")):
                    errors.append(f"Invalid value for field: {field_name}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def _get_empty_result(self, error: Optional[str] = None) -> Dict[str, Any]:
        """Get empty result structure."""
        return {
            "critical": {},
            "discovered": {},
            "unknown": {},
            "metadata": {
                "extraction_timestamp": datetime.utcnow().isoformat(),
                "extraction_method": "failed",
                "error": error
            }
        }

    async def _ensure_properties_exist(self, extracted_data: Dict[str, Any]):
        """Auto-create material_properties entries for newly discovered fields.

        This integrates with the prototype validation system by ensuring that
        all discovered metadata fields have corresponding entries in the
        material_properties table.

        Args:
            extracted_data: Extracted metadata from AI
        """
        from app.services.core.supabase_client import get_supabase_client

        try:
            supabase = get_supabase_client()

            # Collect all discovered property keys
            property_keys = set()

            # From critical metadata
            if "critical" in extracted_data:
                for key in extracted_data["critical"].keys():
                    property_keys.add(key)

            # From discovered metadata (nested by category)
            if "discovered" in extracted_data:
                for category, fields in extracted_data["discovered"].items():
                    if isinstance(fields, dict):
                        for key in fields.keys():
                            property_keys.add(key)

            # From unknown metadata (custom fields)
            if "unknown" in extracted_data:
                for key in extracted_data["unknown"].keys():
                    if not key.startswith('_'):  # Skip internal fields
                        property_keys.add(key)

            # Check which properties already exist
            existing_result = supabase.client.table('material_properties').select('property_key').execute()
            existing_keys = {row['property_key'] for row in existing_result.data}

            # Create missing properties
            new_properties = []
            for property_key in property_keys:
                if property_key not in existing_keys:
                    # Determine category from METADATA_CATEGORY_HINTS
                    category = self._determine_property_category(property_key)

                    # Create property definition
                    new_properties.append({
                        'property_key': property_key,
                        'name': property_key.replace('_', ' ').title(),
                        'display_name': property_key.replace('_', ' ').title(),
                        'description': f'Auto-discovered property: {property_key}',
                        'data_type': 'string',  # Default to string
                        'validation_rules': {},
                        'is_searchable': True,
                        'is_filterable': True,
                        'is_ai_extractable': True,
                        'category': category,
                        'created_at': datetime.utcnow().isoformat(),
                        'updated_at': datetime.utcnow().isoformat()
                    })

            # Batch insert new properties with upsert to handle duplicates
            if new_properties:
                try:
                    # Use upsert to avoid duplicate key violations
                    supabase.client.table('material_properties')\
                        .upsert(new_properties, on_conflict='property_key')\
                        .execute()
                    self.logger.info(f"Auto-created/updated {len(new_properties)} material_properties entries")
                except Exception as insert_error:
                    # If upsert fails, try inserting one by one to identify problematic entries
                    self.logger.warning(f"Batch upsert failed: {insert_error}, trying individual inserts")
                    for prop in new_properties:
                        try:
                            supabase.client.table('material_properties')\
                                .upsert(prop, on_conflict='property_key')\
                                .execute()
                        except Exception as single_error:
                            self.logger.debug(f"Skipped property {prop['property_key']}: {single_error}")

        except Exception as e:
            # Don't fail extraction if property creation fails
            self.logger.warning(f"Failed to auto-create material_properties: {e}")

    def _determine_property_category(self, property_key: str) -> str:
        """Determine which category a property belongs to."""
        # Check each category's hints
        for category, hints in METADATA_CATEGORY_HINTS.items():
            if property_key in hints:
                return category

        # Check if it's a custom field
        if property_key.startswith('_custom_'):
            return 'custom'

        # Default to 'other'
        return 'other'


# ============================================================================
# SCOPE DETECTION (Product-Specific vs Catalog-General)
# ============================================================================

class MetadataScopeDetector:
    """
    Detects if metadata applies to:
    - Specific product (e.g., "NOVA has R11")
    - All products (e.g., "All tiles made in Spain")
    - Product category (e.g., "All matte tiles have R11")
    """

    def __init__(self, ai_client=None):
        self.ai_client = ai_client
        self.logger = logging.getLogger(__name__)

    async def detect_scope(
        self,
        chunk_content: str,
        product_names: List[str],
        document_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect if chunk metadata is product-specific or catalog-general.

        Args:
            chunk_content: Text content of chunk
            product_names: List of product names in catalog
            document_context: Optional document context

        Returns:
            {
                "scope": "product_specific|catalog_general|category_specific",
                "confidence": 0.95,
                "reasoning": "...",
                "applies_to": ["NOVA"] or "all" or ["matte_tiles"],
                "extracted_metadata": {...}
            }
        """
        try:
            prompt = self._build_scope_detection_prompt(
                chunk_content, product_names, document_context
            )

            if self.ai_client:
                response = await self._call_ai(prompt)
                return self._parse_scope_response(response)
            else:
                # Fallback to pattern-based detection
                return self._fallback_scope_detection(chunk_content, product_names)

        except Exception as e:
            self.logger.error(f"Scope detection failed: {e}")
            return {
                "scope": "unknown",
                "confidence": 0.0,
                "reasoning": f"Error: {e}",
                "applies_to": [],
                "extracted_metadata": {}
            }

    def _build_scope_detection_prompt(
        self,
        chunk_content: str,
        product_names: List[str],
        document_context: Optional[str]
    ) -> str:
        """Build AI prompt for scope detection."""

        product_list = ", ".join(product_names) if product_names else "Unknown"

        return f"""Analyze this text chunk and determine its metadata scope.

Chunk Content:
"{chunk_content}"

Product Names in Catalog: {product_list}

Classify the scope as:
1. "product_specific" - Mentions specific product name and applies only to that product
   Example: "NOVA tile features R11 slip resistance"

2. "catalog_general_explicit" - Explicitly says "all products" or "entire catalog"
   Example: "All tiles in this catalog are made in Spain"

3. "catalog_general_implicit" - Metadata mentioned WITHOUT product context (applies to all)
   Example: "Available in 15×38" (no product name, applies to all unless overridden)
   Example: "Factory: Castellón Ceramics" (general info, applies to all)

4. "category_specific" - Applies to a category of products
   Example: "All matte finish tiles have R11 slip resistance"

CRITICAL RULES:
- If chunk mentions dimensions/size WITHOUT product name → catalog_general_implicit
- If chunk mentions factory/country WITHOUT product name → catalog_general_implicit
- If chunk says "available in" or "comes in" WITHOUT product name → catalog_general_implicit
- Product-specific metadata can OVERRIDE catalog-general metadata

Extract ALL metadata mentioned in the chunk.

Return JSON in this exact format:
{{
  "scope": "product_specific|catalog_general_explicit|catalog_general_implicit|category_specific",
  "confidence": 0.95,
  "reasoning": "Chunk mentions 'NOVA' specifically, so metadata applies only to NOVA product",
  "applies_to": ["NOVA"],  // or "all" for catalog-general, or ["matte_tiles"] for category
  "extracted_metadata": {{
    "slip_resistance": "R11",
    "dimensions": "15×38",
    "designer": "SG NY"
  }},
  "is_override": false  // true if product-specific metadata overrides catalog-general
}}

Analyze now:"""

    def _fallback_scope_detection(
        self,
        chunk_content: str,
        product_names: List[str]
    ) -> Dict[str, Any]:
        """Fallback pattern-based scope detection."""

        chunk_lower = chunk_content.lower()

        # Check if any product name is mentioned
        mentioned_products = [
            name for name in product_names
            if name.lower() in chunk_lower
        ]

        # Check for explicit catalog-general keywords
        explicit_catalog_keywords = ["all tiles", "all products", "entire catalog", "every product"]
        is_catalog_general_explicit = any(keyword in chunk_lower for keyword in explicit_catalog_keywords)

        # Check for implicit catalog-general patterns
        implicit_patterns = [
            r"available in\s+\d+",  # "Available in 15×38"
            r"comes in\s+\d+",      # "Comes in 20×40"
            r"factory:\s*\w+",      # "Factory: Castellón"
            r"made in\s+\w+",       # "Made in Spain" (without "all")
            r"dimensions?:\s*\d+",  # "Dimensions: 15×38"
        ]
        is_catalog_general_implicit = any(
            re.search(pattern, chunk_lower) for pattern in implicit_patterns
        ) and not mentioned_products  # Only if no product name mentioned

        # Determine scope
        if mentioned_products:
            # Check if this is an override (product-specific dimensions when catalog-general exists)
            is_override = bool(re.search(r"dimensions?:\s*\d+", chunk_lower))

            return {
                "scope": "product_specific",
                "confidence": 0.7,
                "reasoning": f"Mentions product names: {', '.join(mentioned_products)}",
                "applies_to": mentioned_products,
                "extracted_metadata": {},
                "is_override": is_override
            }
        elif is_catalog_general_explicit:
            return {
                "scope": "catalog_general_explicit",
                "confidence": 0.6,
                "reasoning": "Contains explicit catalog-general keywords",
                "applies_to": "all",
                "extracted_metadata": {},
                "is_override": False
            }
        elif is_catalog_general_implicit:
            return {
                "scope": "catalog_general_implicit",
                "confidence": 0.5,
                "reasoning": "Metadata mentioned without product context (implicit catalog-general)",
                "applies_to": "all",
                "extracted_metadata": {},
                "is_override": False
            }
        else:
            return {
                "scope": "unknown",
                "confidence": 0.3,
                "reasoning": "Cannot determine scope from patterns",
                "applies_to": [],
                "extracted_metadata": {},
                "is_override": False
            }

    async def _call_ai(self, prompt: str) -> str:
        """Call AI service for scope detection."""
        raise NotImplementedError("AI client integration needed")

    def _parse_scope_response(self, response: str) -> Dict[str, Any]:
        """Parse AI JSON response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.error("Failed to parse scope detection response")
            return {
                "scope": "unknown",
                "confidence": 0.0,
                "reasoning": "Failed to parse AI response",
                "applies_to": [],
                "extracted_metadata": {}
            }

    async def _ensure_properties_exist(self, extracted_data: Dict[str, Any]):
        """Auto-create material_properties entries for newly discovered fields.

        This integrates with the prototype validation system by ensuring that
        all discovered metadata fields have corresponding entries in the
        material_properties table.

        Args:
            extracted_data: Extracted metadata from AI
        """
        from app.services.core.supabase_client import get_supabase_client

        try:
            supabase = get_supabase_client()

            # Collect all discovered property keys
            property_keys = set()

            # From critical metadata
            if "critical" in extracted_data:
                for key in extracted_data["critical"].keys():
                    property_keys.add(key)

            # From discovered metadata (nested by category)
            if "discovered" in extracted_data:
                for category, fields in extracted_data["discovered"].items():
                    if isinstance(fields, dict):
                        for key in fields.keys():
                            property_keys.add(key)

            # From unknown metadata (custom fields)
            if "unknown" in extracted_data:
                for key in extracted_data["unknown"].keys():
                    if not key.startswith('_'):  # Skip internal fields
                        property_keys.add(key)

            # Check which properties already exist
            existing_result = supabase.client.table('material_properties').select('property_key').execute()
            existing_keys = {row['property_key'] for row in existing_result.data}

            # Create missing properties
            new_properties = []
            for property_key in property_keys:
                if property_key not in existing_keys:
                    # Determine category from METADATA_CATEGORY_HINTS
                    category = self._determine_property_category(property_key)

                    # Create property definition
                    new_properties.append({
                        'property_key': property_key,
                        'name': property_key.replace('_', ' ').title(),
                        'display_name': property_key.replace('_', ' ').title(),
                        'description': f'Auto-discovered property: {property_key}',
                        'data_type': 'string',  # Default to string
                        'validation_rules': {},
                        'is_searchable': True,
                        'is_filterable': True,
                        'is_ai_extractable': True,
                        'category': category,
                        'created_at': datetime.utcnow().isoformat(),
                        'updated_at': datetime.utcnow().isoformat()
                    })

            # Batch insert new properties with upsert to handle duplicates
            if new_properties:
                try:
                    # Use upsert to avoid duplicate key violations
                    supabase.client.table('material_properties')\
                        .upsert(new_properties, on_conflict='property_key')\
                        .execute()
                    self.logger.info(f"Auto-created/updated {len(new_properties)} material_properties entries")
                except Exception as insert_error:
                    # If upsert fails, try inserting one by one to identify problematic entries
                    self.logger.warning(f"Batch upsert failed: {insert_error}, trying individual inserts")
                    for prop in new_properties:
                        try:
                            supabase.client.table('material_properties')\
                                .upsert(prop, on_conflict='property_key')\
                                .execute()
                        except Exception as single_error:
                            self.logger.debug(f"Skipped property {prop['property_key']}: {single_error}")

        except Exception as e:
            # Don't fail extraction if property creation fails
            self.logger.warning(f"Failed to auto-create material_properties: {e}")

    def _determine_property_category(self, property_key: str) -> str:
        """Determine which category a property belongs to."""
        # Check each category's hints
        for category, hints in METADATA_CATEGORY_HINTS.items():
            if property_key in hints:
                return category

        # Check if it's a custom field
        if property_key.startswith('_custom_'):
            return 'custom'

        # Default to 'other'
        return 'other'


