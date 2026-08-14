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


from app.services.core.ai_call_logger import AICallLogger
from app.services.core.ai_client_service import get_ai_client_service
from app.services.metadata.metadata_normalizer import normalize_metadata, get_normalization_report
from app.services.metadata.field_registry import field_registry
from app.services.core.supabase_client import get_supabase_client
from app.services.utilities.prompt_registry import get_cached, load_prompt, prefetch, render

logger = logging.getLogger(__name__)

# Get API keys from environment
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


# ============================================================================
# TIER 1: CRITICAL METADATA (Always Required)
# ============================================================================

CRITICAL_METADATA_SCHEMA = {
    "material_category": {
        "description": "Primary material category — covers tiles, wood, furniture, lighting, heating, sanitary, kitchen, paint, decor, and general materials",
        "extraction_method": "ai_with_keywords",
        "keywords": [
            "tile", "porcelain", "ceramic", "stone", "marble", "granite",
            "wood", "laminate", "parquet", "vinyl", "hardwood",
            "metal", "glass", "composite", "concrete", "quartz",
            "paint", "wallpaper", "plaster",
            "sofa", "chair", "table", "cabinet", "bed", "desk", "shelf",
            "radiator", "boiler", "fireplace", "towel_rail", "convector",
            "toilet", "basin", "bathtub", "shower", "bidet", "tap", "faucet",
            "kitchen", "hood", "sink", "worktop",
            "light", "lamp", "pendant", "chandelier", "spotlight",
            "rug", "curtain", "cushion", "vase", "mirror", "sculpture",
        ],
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
# The extraction prompt lives in the `prompts` table
# (prompt_type='extraction', stage='entity_creation', category='products') and is loaded by
# _load_prompt_from_database. A hardcoded builder used to live here as well: 7,558 characters
# that read like the authoritative prompt and were called by nothing, so anyone editing "the
# extraction prompt" in code had even odds of editing the dead one. Deleted in #347 phase 3P.



# ============================================================================
# DYNAMIC METADATA EXTRACTOR CLASS
# ============================================================================

class DynamicMetadataExtractor:
    """
    Extracts metadata dynamically using AI, without hardcoded attribute checks.
    """

    def __init__(self, model: str = "claude", job_id: Optional[str] = None, workspace_id: str = None):
        """
        Initialize extractor.

        Args:
            model: AI model family — supports "claude", "claude-vision", "claude-haiku-vision"
            job_id: Optional job ID for AI call logging
            workspace_id: Workspace ID for loading custom prompts from database
        """
        from app.config import get_settings
        self.model = model
        self.job_id = job_id
        self.workspace_id = workspace_id or get_settings().default_workspace_id
        self.logger = logging.getLogger(__name__)
        self.ai_logger = AICallLogger()
        self.supabase = get_supabase_client()

        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set - cannot use Claude")

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
    
    def _extract_relevant_sections(self, pdf_text: str, max_chars: int = 100000) -> str:
        """
        Smart extraction: Find and extract only relevant sections from PDF.

        This reduces token usage while preserving critical information.

        Strategy:
        1. Always include first 12000 chars (product name, description, basic specs)
        2. Search for section headers (Packaging, Compliance, Care, Technical, etc.)
           in multiple languages (EN, IT, FR, ES, DE, EL)
        3. Extract 6000 chars around each relevant section (3000 before + 3000 after)
        4. Always include last 8000 chars (often contains packaging/compliance)

        Args:
            pdf_text: Full PDF text
            max_chars: Maximum characters to return (default: 100000)

        Returns:
            Intelligently extracted text with relevant sections
        """
        if len(pdf_text) <= max_chars:
            return pdf_text  # No need to extract if already small

        # Section keywords to search for (case-insensitive)
        # Multilingual: English, Italian, French, Spanish, German, Greek
        section_keywords = [
            # Packaging (EN + multilingual)
            r'\b(packaging|packing|iconography|box|pallet|pieces per box|coverage|confezionamento|imballaggio|emballage|conditionnement|embalaje|empaque|verpackung|συσκευασία)\b',
            # Compliance & Safety (EN + multilingual)
            r'\b(regulation|compliance|certification|standard|safety|eco.?friendly|sustainability|voc|leed|iso|regolamento|certificazione|réglementation|certification|regulación|certificación|zertifizierung|vorschriften|πιστοποίηση)\b',
            # Care & Maintenance (EN + multilingual)
            r'\b(care|maintenance|cleaning|handling|installation|recommended use|manutenzione|pulizia|entretien|nettoyage|mantenimiento|limpieza|reinigung|pflege|wartung|καθαρισμός|συντήρηση)\b',
            # Technical specs (EN + multilingual)
            r'\b(technical|specification|properties|performance|dimensions|weight|thickness|specifiche tecniche|spécifications|especificaciones|technische daten|τεχνικά)\b',
        ]

        extracted_sections = []

        # 1. Always include beginning (product name, description, basic info)
        extracted_sections.append(("START", pdf_text[:12000]))

        # 1.5. Include index/TOC pages (usually pages 8-20) for dimension data
        # Index pages often contain product dimensions in table format
        import re
        index_start = 12000  # Approximate start of index pages
        index_end = min(35000, len(pdf_text))  # Include up to ~page 25
        if len(pdf_text) > index_end:
            # Look for dimension patterns in index area
            index_area = pdf_text[index_start:index_end]
            if re.search(r'\d+[.,]?\d*\s*[x×]\s*\d+[.,]?\d*\s*(cm|mm|inch)?', index_area, re.IGNORECASE):
                extracted_sections.append(("INDEX_DIMENSIONS", index_area))
                self.logger.debug("   📐 Found dimensions in index area - including")

        # 2. Search for relevant sections
        for keyword_pattern in section_keywords:
            for match in re.finditer(keyword_pattern, pdf_text, re.IGNORECASE):
                start = max(0, match.start() - 3000)  # 3000 chars before
                end = min(len(pdf_text), match.end() + 3000)  # 3000 chars after
                section_text = pdf_text[start:end]
                extracted_sections.append((f"SECTION@{match.start()}", section_text))

        # 3. Always include end (often has packaging/compliance tables)
        extracted_sections.append(("END", pdf_text[-8000:]))

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
        manual_overrides: Optional[Dict[str, Any]] = None,
        product_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract metadata dynamically from PDF text.

        Args:
            pdf_text: Text content from PDF
            category_hint: Optional material category hint
            manual_overrides: Manual values from admin (override AI extraction)
            product_name: Current product name — used for Option A (prompt scoping) and
                          Option B (normalization filter) to prevent cross-product SKU contamination

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
            smart_text = self._extract_relevant_sections(pdf_text, max_chars=100000)

            # Step 1: Load prompt AND field registry from the database — NO FALLBACK for either.
            # The field list, category tips, skip list and subtype vocabulary all come from
            # material_metadata_fields / material_categories (#347 phase 3.2). There is no
            # hardcoded copy left to fall back to; the hardcoded copy WAS the bug.
            await field_registry.ensure_loaded()
            db_prompt_template = await self._load_prompt_from_database(stage="entity_creation", category="products")

            if db_prompt_template:
                # Use database prompt - replace placeholders
                category_context = f"\nMaterial Category Hint: This appears to be a {category_hint} product." if category_hint else ""
                # [Option A] Inject product name scoping rule so the AI only extracts SKUs for THIS product
                if product_name:
                    product_name_context = render(await load_prompt("extraction", "product_name_scope", stage="entity_creation"),
                        product_name=product_name)
                else:
                    product_name_context = ""

                # ── Category-specific field guidance ─────────────────────────
                # Inject priority fields + extraction hints based on the upload
                # category so the AI knows what to hunt for (e.g. grout codes
                # for tiles, lumens for lighting) and what to skip.
                category_fields_context = ""
                if category_hint:
                    cat_key = category_hint.lower().strip()
                    # Priority fields, subtype vocabulary, category tips and the skip list all
                    # come from the DB registry now — one block, one source (#347 phase 3.2).
                    category_fields_context = "\n\n" + field_registry.category_prompt_block(cat_key)
                    category_fields_context += (
                        "\n\nIMPORTANT: Beyond the priority fields above, also extract ANY other "
                        "attributes you discover in the document. Capture everything — the priority "
                        "list tells you what to look for specifically, but never ignore useful data "
                        "just because it's not on the list. Place unexpected attributes in the "
                        "'unknown_attributes' section with a descriptive key name."
                    )
                    self.logger.info(
                        "📋 Injected DB field registry for '%s' (%d fields, %d skip-fields)",
                        cat_key,
                        len(field_registry.fields_for_category(cat_key)),
                        len(field_registry.skip_fields(cat_key)),
                    )

                prompt = (
                    db_prompt_template
                    .replace("{category_context}", category_context)
                    .replace("{category_fields}", category_fields_context)
                    .replace("{product_name_context}", product_name_context)
                    .replace("{pdf_text}", smart_text)
                )
                # If the DB prompt template doesn't have the {category_fields}
                # placeholder yet, append category guidance before the PDF text
                # so it still takes effect without requiring a prompt update.
                if "{category_fields}" not in db_prompt_template and category_fields_context:
                    # Insert before the PDF content (last section of the prompt)
                    prompt = prompt + "\n" + category_fields_context
                    self.logger.info("📋 Appended category fields (DB prompt missing {category_fields} placeholder)")

                self.logger.info(f"✅ Using DATABASE prompt for metadata extraction (product='{product_name or 'unknown'}')")
            else:
                error_msg = "CRITICAL: Metadata extraction prompt not found in database. Add prompt via /admin/ai-configs with prompt_type='extraction', stage='entity_creation', category='products'"
                self.logger.error(f"❌ {error_msg}")
                raise ValueError(error_msg)

            # Step 2: Always use Claude for metadata extraction
            ai_response = await self._call_claude(prompt)

            extracted_data = self._parse_ai_response(ai_response)

            # Step 2: Apply manual overrides
            if manual_overrides:
                extracted_data = self._apply_manual_overrides(extracted_data, manual_overrides)

            # Step 3: Validate critical fields
            validation_result = self._validate_critical_fields(extracted_data)

            # Step 4: Normalize metadata to standardized schema
            original_data = extracted_data.copy()
            normalized_data = {}

            # Normalize each category — pass product_name for Option B SKU filtering
            for category in ["critical", "discovered", "unknown"]:
                if category in extracted_data:
                    normalized_data[category] = normalize_metadata(
                        {category: extracted_data[category]},
                        product_name=product_name
                    ).get(category, {})

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
                self.logger.warning("⚠️ No packaging fields extracted (check for Iconography/Packing sections)")

            # Log compliance/safety extraction results
            compliance_fields = extracted_data.get("compliance", {})
            application_fields = extracted_data.get("application", {})
            care_fields = {k: v for k, v in application_fields.items() if k in ['care_instructions', 'maintenance']}

            if compliance_fields:
                self.logger.info(f"✅ Compliance/Safety fields extracted: {list(compliance_fields.keys())}")
                self.logger.debug(f"   Compliance data: {compliance_fields}")
            else:
                self.logger.warning("⚠️ No compliance/safety fields extracted (check for Regulation/Certifications sections)")

            if care_fields:
                self.logger.info(f"🧼 Care/Maintenance fields extracted: {list(care_fields.keys())}")
                self.logger.debug(f"   Care data: {care_fields}")
            else:
                self.logger.warning("⚠️ No care/maintenance fields extracted (check for Cleaning/Handling sections)")

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
    
    # P0-6 cost control: in-process content-hash cache. Same product text
    # → same metadata, no second Claude call. Capped at 64 entries per
    # extractor instance (typical job has 5-30 products).
    _CALL_CACHE: Dict[str, str] = {}
    _CALL_CACHE_MAX = 64

    async def _call_claude(self, prompt: str) -> str:
        """Call Claude for metadata extraction.

        P0-6 cost control:
          - Uses claude-haiku-4-5 by default (12× cheaper than Opus, accurate
            enough for structured field extraction). Override via env
            METADATA_EXTRACTOR_MODEL if you need to escalate.
          - SHA-256 cache by prompt: identical inputs return cached responses
            instead of re-billing.
        """
        import hashlib
        import os as _os

        cache_key = hashlib.sha256(prompt.encode('utf-8', errors='replace')).hexdigest()
        cached = DynamicMetadataExtractor._CALL_CACHE.get(cache_key)
        if cached is not None:
            self.logger.info("💰 Metadata extraction cache hit — skipping Claude call")
            return cached

        start_time = datetime.now()
        model = _os.environ.get("METADATA_EXTRACTOR_MODEL", "claude-haiku-4-5")

        try:
            # Use centralized AI client service
            ai_service = get_ai_client_service()
            client = ai_service.anthropic

            response = client.messages.create(
                model=model,
                max_tokens=16000,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            content = response.content[0].text

            # Cache (LRU-ish: drop oldest when full)
            cache = DynamicMetadataExtractor._CALL_CACHE
            if len(cache) >= DynamicMetadataExtractor._CALL_CACHE_MAX:
                cache.pop(next(iter(cache)))
            cache[cache_key] = content

            # Log AI call
            latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            await self.ai_logger.log_claude_call(
                task="dynamic_metadata_extraction",
                model=model,
                response=response,
                latency_ms=latency_ms,
                confidence_score=0.9,
                confidence_breakdown={},
                action="use_ai_result",  # must be 'use_ai_result' or 'fallback_to_rules'
                job_id=self.job_id
            )

            return content

        except Exception as e:
            self.logger.error(f"Claude metadata extraction failed: {e}")
            raise RuntimeError(f"Claude metadata extraction failed: {str(e)}") from e

    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse AI JSON response.

        On parse failure, logs a structured error with the truncated raw response
        so we can diagnose prompt/model drift from Sentry + journalctl without
        silently producing an empty metadata skeleton.
        """
        original_response = response
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
            # Log the truncated raw response so we can debug what the AI actually
            # returned. The empty-result fallback is tagged with the error so it
            # propagates into products.metadata._extraction_error and surfaces in
            # the admin UI instead of showing as "just missing data".
            truncated = (original_response or "")[:800]
            self.logger.error(
                "❌ Failed to parse AI metadata response as JSON | error=%s | "
                "response_len=%d | response[:800]=%r",
                e, len(original_response or ""), truncated
            )
            return self._get_empty_result(
                error=f"JSONDecodeError: {e} | raw[:200]={truncated[:200]!r}"
            )
    
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

    def __init__(self):
        self.ai_client = get_ai_client_service()
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
        # _build_scope_detection_prompt is sync and reads the cache.
        await prefetch(("classification", "chunk_scope", "chunking"))
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

        return render(get_cached("classification", "chunk_scope", stage="chunking"),
            chunk_content=chunk_content, product_list=product_list)

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
        from app.services.core.claude_helper import tracked_claude_call_async
        response = await tracked_claude_call_async(
            task="metadata_scope_detection",
            model="claude-haiku-4-5",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

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


