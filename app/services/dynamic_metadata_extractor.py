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

logger = logging.getLogger(__name__)


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
Extract all other attributes you discover, organized by category:

- Material Properties: composition, type, blend, texture, finish, pattern, weight, density, durability
- Dimensions: length, width, height, thickness, diameter, size, area, volume
- Appearance: color, color_code, gloss_level, sheen, transparency, grain, visual_effects
- Performance: water_resistance, fire_rating, slip_resistance, wear_rating, strength ratings
- Application: recommended_use, installation_method, room_type, traffic_level, care_instructions
- Compliance: certifications, standards, eco_friendly, sustainability, safety_ratings
- Commercial: pricing, availability, supplier, sku, warranty
- Design: designer, studio, collection, series, aesthetic_style
- Technical: specifications, construction, manufacturing_process, grade, class, rating

IMPORTANT:
- If you find attributes NOT in the above list, STILL EXTRACT THEM!
- Group similar attributes together
- Use snake_case for attribute names (e.g., "slip_resistance" not "Slip Resistance")
- Include confidence scores (0.0-1.0) for each extraction
- If a value is uncertain, mark confidence < 0.7

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

PDF Content:
{pdf_text[:4000]}

Extract all metadata now:"""


# ============================================================================
# DYNAMIC METADATA EXTRACTOR CLASS
# ============================================================================

class DynamicMetadataExtractor:
    """
    Extracts metadata dynamically using AI, without hardcoded attribute checks.
    """
    
    def __init__(self, ai_client=None):
        """
        Initialize extractor.
        
        Args:
            ai_client: AI client (Claude, GPT, etc.) for extraction
        """
        self.ai_client = ai_client
        self.logger = logging.getLogger(__name__)
    
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
            # Step 1: AI extraction
            prompt = get_dynamic_extraction_prompt(pdf_text, category_hint)
            
            if self.ai_client:
                ai_response = await self._call_ai(prompt)
                extracted_data = self._parse_ai_response(ai_response)
            else:
                # Fallback to pattern-based extraction
                extracted_data = self._fallback_extraction(pdf_text)
            
            # Step 2: Apply manual overrides
            if manual_overrides:
                extracted_data = self._apply_manual_overrides(extracted_data, manual_overrides)
            
            # Step 3: Validate critical fields
            validation_result = self._validate_critical_fields(extracted_data)
            
            # Step 4: Add metadata
            extracted_data["metadata"] = {
                "extraction_timestamp": datetime.utcnow().isoformat(),
                "extraction_method": "ai_dynamic" if self.ai_client else "pattern_fallback",
                "validation_passed": validation_result["valid"],
                "validation_errors": validation_result.get("errors", []),
                "manual_overrides_applied": bool(manual_overrides)
            }
            
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {e}")
            return self._get_empty_result(error=str(e))
    
    async def _call_ai(self, prompt: str) -> str:
        """Call AI service for extraction."""
        # Implementation depends on your AI client (Claude, GPT, etc.)
        # This is a placeholder
        raise NotImplementedError("AI client integration needed")
    
    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse AI JSON response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.error("Failed to parse AI response as JSON")
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

