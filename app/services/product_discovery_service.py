"""
Product Discovery Service - Stage 0 Implementation

ARCHITECTURE:
1. **Products + Metadata** (ALWAYS extracted together - inseparable)
   - Products are discovered with ALL metadata in one pass
   - Metadata stored in product.metadata JSONB (dimensions, designer, factory, etc.)
   - This is the PRIMARY service that always runs

2. **Document Entities** (OPTIONAL - separate knowledge base)
   - Certificates, Logos, Specifications = Document entities
   - Stored in document_entities table with category system
   - Connected to products via product_document_relationships
   - Managed in "Docs" admin page
   - Can be extracted DURING or AFTER product processing

DISCOVERY PROCESS:
- Stage 0A: Discover products with metadata (ALWAYS)
- Stage 0B: Discover document entities (OPTIONAL - based on extract_categories)
- Both stages identify content location and classification
- Subsequent stages create semantic chunks for RAG search

EXTENSIBILITY:
This service is designed to support future extraction types:
- Marketing content extraction
- Bank statement extraction
- Custom document type extraction
"""

import logging
import asyncio
import base64
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import os

import anthropic
import openai
from PIL import Image
import io

from app.services.ai_call_logger import AICallLogger
from app.services.dynamic_metadata_extractor import DynamicMetadataExtractor

logger = logging.getLogger(__name__)

# Get API keys from environment
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


@dataclass
class ProductInfo:
    """
    Information about a discovered product.

    ARCHITECTURE: Products + Metadata are INSEPARABLE.
    All product metadata (designer, dimensions, factory, technical specs, etc.)
    is stored in the metadata JSONB field and saved to product.metadata in database.

    Technical specifications are ALSO extracted as semantic chunks for RAG search,
    but the primary source of truth is product.metadata.
    """
    name: str
    page_range: List[int]  # Pages where this product appears
    description: Optional[str] = None  # Product description

    # ALL product metadata stored here (inseparable from product)
    metadata: Dict[str, Any] = None
    """
    Metadata structure:
    {
        # Design information
        "designer": "SG NY",
        "studio": "SG NY",
        "category": "tiles",

        # Dimensions and variants
        "dimensions": ["15×38", "20×40"],
        "variants": [{"type": "color", "value": "beige"}],

        # Factory/Group identification (for agentic queries)
        "factory": "Castellón Factory",
        "factory_group": "Harmony Group",
        "manufacturer": "Harmony Materials",
        "country_of_origin": "Spain",

        # Technical specifications
        "slip_resistance": "R11",
        "fire_rating": "A1",
        "thickness": "8mm",
        "water_absorption": "Class 3",
        "finish": "matte",
        "material": "ceramic",

        # Discovery metadata
        "page_range": [12, 13, 14],
        "confidence": 0.95,
        "extraction_method": "ai_discovery"
    }
    """

    image_indices: List[int] = None  # Which images belong to this product
    confidence: float = 0.0


@dataclass
class CertificateInfo:
    """Information about a discovered certificate"""
    name: str
    page_range: List[int]
    certificate_type: Optional[str] = None  # ISO, CE, fire rating, etc.
    issuer: Optional[str] = None
    issue_date: Optional[str] = None
    expiry_date: Optional[str] = None
    standards: List[str] = None  # e.g., ["ISO 9001", "EN 14411"]
    confidence: float = 0.0


@dataclass
class LogoInfo:
    """Information about a discovered logo"""
    name: str
    page_range: List[int]
    logo_type: Optional[str] = None  # company, brand, certification, etc.
    description: Optional[str] = None
    confidence: float = 0.0


@dataclass
class SpecificationInfo:
    """Information about discovered specifications"""
    name: str
    page_range: List[int]
    spec_type: Optional[str] = None  # technical, installation, maintenance, etc.
    description: Optional[str] = None
    confidence: float = 0.0


@dataclass
class ProductCatalog:
    """
    Complete catalog discovered from PDF.

    ARCHITECTURE:
    - Products (ALWAYS extracted with metadata)
    - Document entities (OPTIONAL - certificates, logos, specifications)

    Products contain ALL metadata in ProductInfo.metadata field.
    Document entities are stored separately in document_entities table.
    """
    # Products (ALWAYS extracted with metadata)
    products: List[ProductInfo]

    # Document entities (OPTIONAL - based on extract_categories parameter)
    certificates: List[CertificateInfo] = None
    logos: List[LogoInfo] = None
    specifications: List[SpecificationInfo] = None

    # Metadata
    total_pages: int = 0
    total_images: int = 0
    content_classification: Dict[int, str] = None  # page_number -> "product" | "certificate" | "logo" | "specification" | "marketing" | "admin"

    # Processing info
    processing_time_ms: float = 0.0
    model_used: str = ""
    confidence_score: float = 0.0

    def __post_init__(self):
        """Initialize empty lists if None"""
        if self.certificates is None:
            self.certificates = []
        if self.logos is None:
            self.logos = []
        if self.specifications is None:
            self.specifications = []
        if self.content_classification is None:
            self.content_classification = {}


class ProductDiscoveryService:
    """
    Analyzes PDF to discover products BEFORE processing.
    Uses Claude Sonnet 4.5 or GPT-5 for intelligent product identification.
    """
    
    def __init__(self, model: str = "claude"):
        """
        Initialize service.
        
        Args:
            model: "claude" for Claude Sonnet 4.5 or "gpt" for GPT-5
        """
        self.logger = logger
        self.model = model
        self.ai_logger = AICallLogger()
        
        if model == "claude" and not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set - cannot use Claude")
        if model == "gpt" and not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set - cannot use GPT")
    
    async def discover_products(
        self,
        pdf_content: bytes,
        pdf_text: str,
        total_pages: int,
        categories: List[str] = None,
        agent_prompt: Optional[str] = None,
        workspace_id: str = "ffafc28b-1b8b-4b0d-b226-9f9a6154004e",
        enable_prompt_enhancement: bool = True,
        job_id: Optional[str] = None,
        pdf_path: Optional[str] = None
    ) -> ProductCatalog:
        """
        TWO-STAGE DISCOVERY ARCHITECTURE for handling large catalogs (1000+ pages).

        **Stage 0A: Index Scan (Quick Discovery)**
        - Analyzes first 50-100 pages (TOC/Index) to identify product names and page ranges
        - Uses minimal tokens (~50K characters)
        - Fast and cost-effective

        **Stage 0B: Focused Extraction (Deep Analysis)**
        - Extracts ONLY the specific pages for each discovered product
        - Performs detailed metadata extraction per product
        - No token limits - can handle catalogs of ANY size
        - Processes products in parallel for speed

        Args:
            pdf_content: Raw PDF bytes
            pdf_text: Extracted text from PDF (markdown format) - used for Stage 0A index scan
            total_pages: Total number of pages in PDF
            categories: Categories to discover (products, certificates, logos, specifications). Default: ["products"]
            agent_prompt: Optional natural language prompt from agent (e.g., "extract products", "search for NOVA")
            workspace_id: Workspace ID for custom prompts
            enable_prompt_enhancement: Whether to enhance prompts with admin templates
            job_id: Optional job ID for tracking
            pdf_path: Optional path to PDF file for page-range extraction (Stage 0B)

        Returns:
            ProductCatalog with all discovered content across categories
        """
        start_time = datetime.now()

        # Default to products only if not specified
        if categories is None:
            categories = ["products"]

        try:
            self.logger.info(f"🔍 Starting TWO-STAGE discovery for {total_pages} pages using {self.model.upper()}")
            self.logger.info(f"   Categories: {', '.join(categories)}")
            if agent_prompt:
                self.logger.info(f"   Agent Prompt: '{agent_prompt}'")
            if enable_prompt_enhancement:
                self.logger.info(f"   Prompt Enhancement: ENABLED")

            # ============================================================
            # STAGE 0A: INDEX SCAN - Quick discovery from TOC/Index
            # ============================================================
            self.logger.info(f"📋 STAGE 0A: Scanning index/TOC for product locations...")

            # Use first 50-100 pages for index scan (most catalogs have TOC in first pages)
            index_scan_pages = min(100, total_pages)
            index_text = self._extract_index_text(pdf_text, index_scan_pages, total_pages)

            self.logger.info(f"   Analyzing first {index_scan_pages} pages ({len(index_text)} chars)")

            # Build index scan prompt (lightweight, focused on finding product names + page ranges)
            index_prompt = await self._build_index_scan_prompt(
                index_text,
                total_pages,
                categories,
                agent_prompt,
                workspace_id,
                enable_prompt_enhancement
            )

            # Call AI model for index scan
            if self.model == "claude":
                index_result = await self._discover_with_claude(index_prompt, job_id)
            else:
                index_result = await self._discover_with_gpt(index_prompt, job_id)

            # Parse index scan results (product names + page ranges)
            catalog = self._parse_discovery_results(index_result, total_pages, categories)

            self.logger.info(f"✅ STAGE 0A complete: Found {len(catalog.products)} products")
            for product in catalog.products:
                self.logger.info(f"   📦 {product.name}: pages {product.page_range}")

            # ============================================================
            # STAGE 0B: FOCUSED EXTRACTION - Deep analysis per product
            # ============================================================
            if "products" in categories and catalog.products and pdf_path:
                self.logger.info(f"🔍 STAGE 0B: Extracting detailed metadata for each product...")
                catalog = await self._enrich_products_with_focused_extraction(
                    catalog,
                    pdf_path,
                    job_id
                )
            elif "products" in categories and catalog.products:
                # Fallback: Use full PDF text if pdf_path not provided
                self.logger.warning("⚠️ pdf_path not provided, using fallback metadata extraction")
                catalog = await self._enrich_products_with_metadata(catalog, pdf_text, job_id)

            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            catalog.processing_time_ms = processing_time
            catalog.model_used = self.model

            # Log comprehensive results
            self.logger.info(f"✅ TWO-STAGE Discovery complete in {processing_time:.0f}ms:")
            if "products" in categories:
                self.logger.info(f"   📦 Products: {len(catalog.products)}")
            if "certificates" in categories:
                self.logger.info(f"   📜 Certificates: {len(catalog.certificates)}")
            if "logos" in categories:
                self.logger.info(f"   🎨 Logos: {len(catalog.logos)}")
            if "specifications" in categories:
                self.logger.info(f"   📋 Specifications: {len(catalog.specifications)}")

            return catalog
            
        except Exception as e:
            self.logger.error(f"❌ Product discovery failed: {e}")
            raise
    
    async def _build_discovery_prompt(
        self,
        pdf_text: str,
        total_pages: int,
        categories: List[str],
        agent_prompt: Optional[str] = None,
        workspace_id: str = "ffafc28b-1b8b-4b0d-b226-9f9a6154004e",
        enable_prompt_enhancement: bool = True
    ) -> str:
        """
        Build comprehensive prompt for category-based discovery.

        Uses admin-configured prompts as templates and enhances with agent context.
        """

        # If prompt enhancement is enabled, use PromptEnhancementService
        if enable_prompt_enhancement and agent_prompt:
            try:
                from app.services.prompt_enhancement_service import PromptEnhancementService

                enhancement_service = PromptEnhancementService()

                # Enhance prompt for each category
                enhanced_prompts = []
                for category in categories:
                    enhanced = await enhancement_service.enhance_prompt(
                        agent_prompt=agent_prompt,
                        stage="discovery",
                        category=category,
                        workspace_id=workspace_id,
                        context={
                            "total_pages": total_pages,
                            "pdf_text_preview": pdf_text[:2000],  # First 2000 chars for context
                            "categories": categories
                        }
                    )
                    enhanced_prompts.append({
                        "category": category,
                        "enhanced": enhanced.enhanced_prompt,
                        "version": enhanced.prompt_version
                    })

                    self.logger.info(f"   ✅ Enhanced prompt for {category} (v{enhanced.prompt_version})")

                # Build combined prompt from enhanced templates
                return await self._build_enhanced_discovery_prompt(
                    pdf_text,
                    total_pages,
                    categories,
                    enhanced_prompts,
                    agent_prompt
                )

            except Exception as e:
                self.logger.warning(f"⚠️ Prompt enhancement failed, using default: {e}")
                # Fall through to default prompt building

        # Default prompt building (fallback or when enhancement disabled)
        return self._build_default_discovery_prompt(pdf_text, total_pages, categories, agent_prompt)

    def _build_default_discovery_prompt(
        self,
        pdf_text: str,
        total_pages: int,
        categories: List[str],
        agent_prompt: Optional[str] = None
    ) -> str:
        """Build default discovery prompt without enhancement"""

        # Build category-specific instructions
        category_instructions = []

        if "products" in categories:
            category_instructions.append("""
**PRODUCTS (with ALL metadata - inseparable):**
- Identify ONLY MAIN FEATURED PRODUCTS with dedicated presentations (e.g., "NOVA", "BEAT", "FOLD")
- EXCLUDE products that appear only in:
  * Index pages (table of contents, product lists, thumbnails)
  * Cross-references or "related products" sections
  * Small preview images or catalog grids
  * Footer/header references

**PRODUCT IDENTIFICATION CRITERIA (use ANY of these to identify a MAIN product):**

1. **Page Spread Method** (most common):
   - Dedicated page spread (typically 1-12 consecutive pages)
   - Large hero images showing the product prominently
   - Detailed product description and specifications
   - Designer/studio attribution (usually present)

2. **Metadata Presence Method** (for compact catalogs):
   - Product has comprehensive metadata including:
     * Product name prominently displayed (large font, title position)
     * Dimensions/sizes listed (e.g., "15×38", "20×40", "60x60cm")
     * Designer or studio name mentioned
     * Technical specifications (material, finish, thickness, etc.)
     * Factory/manufacturer information
   - Even if on a single page, if metadata is comprehensive, it's a MAIN product

3. **Visual Prominence Method** (for mixed layouts):
   - Product image is significantly larger than surrounding content (>30% of page)
   - Product name in prominent typography (larger than body text)
   - Dedicated section with clear visual separation from other content
   - Product has its own design space, not part of a grid/list

4. **Content Depth Method** (for text-heavy catalogs):
   - Detailed product description (>100 words)
   - Multiple paragraphs about the product
   - Technical specifications table or detailed specs
   - Application examples or use cases described

**CRITICAL RULES:**
- A product appearing in BOTH index AND dedicated section = count ONLY the dedicated section (not the index mention)
- A product with ONLY thumbnail in index = EXCLUDE (no dedicated pages)
- A product with small reference in footer/header = EXCLUDE
- A product with comprehensive metadata even on 1 page = INCLUDE
- A "dedicated section" means: product name as heading + ANY product details (dimensions, colors, designer, images, description)
- If a product has its own page(s) with product information, it MUST be included even if also mentioned in index
- When in doubt, INCLUDE the product - better to have false positives than miss real products

**Extract ALL available metadata for each MAIN product:**
- Basic info: name, description, category
- Design: designer, studio
- Dimensions: all size variants (e.g., ["15×38", "20×40"])
- Variants: color, finish, texture variants
- Factory/Group: factory name, factory group, manufacturer, country of origin
- Technical specs: slip resistance, fire rating, thickness, water absorption, finish, material
- Any other relevant metadata found in the PDF
- Image pages for each product (the full page range of the product spread)
- Products and metadata are INSEPARABLE - always extract together""")

        if "certificates" in categories:
            category_instructions.append("""
**CERTIFICATES:**
- Identify ALL certificates (ISO, CE, fire ratings, quality certifications)
- Extract: name, type, issuer, issue/expiry dates, standards (e.g., "ISO 9001", "EN 14411")
- Page range where certificate appears""")

        if "logos" in categories:
            category_instructions.append("""
**LOGOS:**
- Identify company logos, brand marks, certification logos
- Extract: name, type (company/brand/certification), description
- Page range where logo appears""")

        if "specifications" in categories:
            category_instructions.append("""
**SPECIFICATIONS:**
- Identify technical specs, installation guides, maintenance instructions
- Extract: name, type (technical/installation/maintenance), description
- Page range where specification appears""")

        # Build agent prompt context
        agent_context = ""
        if agent_prompt:
            agent_context = f"""
**AGENT REQUEST:**
The user requested: "{agent_prompt}"
Focus your analysis on fulfilling this specific request while still providing comprehensive results.
"""

        prompt = f"""You are analyzing a material/product catalog PDF with {total_pages} pages.

Your task is to identify and extract content across the following categories:
{chr(10).join(category_instructions)}

{agent_context}

**GENERAL INSTRUCTIONS:**
1. Be comprehensive - identify EVERY instance in each category
2. **CRITICAL: For page_range, you MUST identify the COMPLETE page range for each product:**
   - Include ALL consecutive pages where the product appears (not just 2 representative pages)
   - Example: If a product spans pages 22-27, return [22, 23, 24, 25, 26, 27], NOT just [22, 23]
   - Look for: product name continuity, related images, variant displays, technical specs across multiple pages
   - A product's page range ends when a new product begins or when content becomes unrelated
3. Classify each page as: "product", "certificate", "logo", "specification", "marketing", "admin", or "transitional"
4. Provide confidence scores (0.0-1.0) for each item

**OUTPUT FORMAT (JSON):**
```json
{{
  "products": [
    {{
      "name": "NOVA",
      "description": "Modern ceramic tile collection",
      "page_range": [12, 13, 14],
      "image_pages": [12, 13],
      "confidence": 0.95,
      "metadata": {{
        "designer": "SG NY",
        "studio": "SG NY",
        "category": "tiles",
        "dimensions": ["15×38", "20×40"],
        "variants": [
          {{"type": "color", "value": "beige"}},
          {{"type": "finish", "value": "matte"}}
        ],
        "factory": "Castellón Factory",
        "factory_group": "Harmony Group",
        "manufacturer": "Harmony Materials",
        "country_of_origin": "Spain",
        "slip_resistance": "R11",
        "fire_rating": "A1",
        "thickness": "8mm",
        "water_absorption": "Class 3",
        "finish": "matte",
        "material": "ceramic"
      }}
    }}
  ],
  "certificates": [
    {{
      "name": "ISO 9001:2015",
      "certificate_type": "quality_management",
      "issuer": "TÜV SÜD",
      "issue_date": "2023-01-15",
      "expiry_date": "2026-01-15",
      "standards": ["ISO 9001:2015"],
      "page_range": [45, 46],
      "confidence": 0.92
    }}
  ],
  "logos": [
    {{
      "name": "Company Logo",
      "logo_type": "company",
      "description": "Main company brand logo",
      "page_range": [1, 2],
      "confidence": 0.98
    }}
  ],
  "specifications": [
    {{
      "name": "Installation Guide",
      "spec_type": "installation",
      "description": "Step-by-step installation instructions",
      "page_range": [50, 52],
      "confidence": 0.90
    }}
  ],
  "page_classification": {{
    "1": "logo",
    "2": "marketing",
    "12": "product",
    "13": "product",
    "45": "certificate",
    "50": "specification"
  }},
  "total_products": 14,
  "total_certificates": 3,
  "total_logos": 5,
  "total_specifications": 2,
  "confidence_score": 0.92
}}
```

**PDF CONTENT:**
{pdf_text[:200000]}

Analyze the above content and return ONLY valid JSON with ALL content discovered across the requested categories."""

        return prompt

    async def _build_index_scan_prompt(
        self,
        index_text: str,
        total_pages: int,
        categories: List[str],
        agent_prompt: str,
        workspace_id: str,
        enable_prompt_enhancement: bool
    ) -> str:
        """
        Build lightweight prompt for Stage 0A index scanning.

        This prompt is optimized for FAST discovery of product names and page ranges
        from TOC/Index pages. It does NOT extract detailed metadata (that's Stage 0B).

        Args:
            index_text: Text from index/TOC pages
            total_pages: Total pages in PDF
            categories: Categories to discover
            agent_prompt: User's request
            workspace_id: Workspace ID
            enable_prompt_enhancement: Whether to use admin templates

        Returns:
            Optimized prompt for index scanning
        """

        prompt = f"""You are analyzing the TABLE OF CONTENTS / INDEX section of a product catalog PDF with {total_pages} total pages.

**YOUR TASK**: Quickly identify ALL products and their page locations from the index/TOC.

**CRITICAL INSTRUCTIONS**:
1. **Focus on the INDEX/TOC** - Look for product listings with page numbers
2. **Extract product names and page ranges** - This is the PRIMARY goal
3. **DO NOT extract detailed metadata yet** - That comes in Stage 2
4. **Be comprehensive** - Find ALL products mentioned in the index
5. **Page ranges**: If a product spans multiple pages (e.g., "NOVA ... 24-27"), include ALL pages [24,25,26,27]

**WHAT TO LOOK FOR**:
- Product names in uppercase or bold (e.g., "VALENOVA", "FOLD", "PIQUÉ")
- Page numbers next to product names (e.g., "NOVA ... 24", "FOLD ... 32-35")
- Section headers indicating product categories
- Designer/studio names associated with products

**OUTPUT FORMAT** (JSON only):
```json
{{
  "products": [
    {{
      "name": "VALENOVA",
      "page_range": [24, 25, 26, 27],
      "description": "Brief description if available in index",
      "confidence": 0.95,
      "metadata": {{
        "designer": "SG NY",
        "category": "tiles"
      }}
    }}
  ],
  "confidence_score": 0.92
}}
```

**INDEX/TOC CONTENT:**
{index_text}

Return ONLY valid JSON with ALL products found in the index."""

        return prompt

    async def _build_enhanced_discovery_prompt(
        self,
        pdf_text: str,
        total_pages: int,
        categories: List[str],
        enhanced_prompts: List[Dict[str, Any]],
        agent_prompt: str
    ) -> str:
        """
        Build discovery prompt using admin-enhanced templates.

        Combines admin-configured prompts with agent context and PDF content.
        """

        # Build category-specific sections from enhanced prompts
        category_sections = []
        for ep in enhanced_prompts:
            category_sections.append(f"""
**{ep['category'].upper()}:**
{ep['enhanced']}
(Using admin template v{ep['version']})
""")

        # Build comprehensive prompt
        prompt = f"""You are analyzing a material/product catalog PDF with {total_pages} pages.

**USER REQUEST:** "{agent_prompt}"

Your task is to identify and extract content across the following categories:
{chr(10).join(category_sections)}

**GENERAL INSTRUCTIONS:**
1. For PRODUCTS: Identify ONLY main featured products using ANY of these criteria:
   - EXCLUDE index pages, thumbnails, cross-references, and catalog grids (this is critical!)
   - INCLUDE if product has: dedicated page spread (1-12 pages) OR comprehensive metadata OR visual prominence OR content depth
   - Use multiple identification methods: page spread, metadata presence, visual prominence, or content depth
   - Even single-page products count if they have comprehensive metadata and visual prominence
2. **CRITICAL: For page_range, you MUST identify the COMPLETE page range for each product:**
   - Include ALL consecutive pages where the product appears (not just 2 representative pages)
   - Example: If a product spans pages 22-27, return [22, 23, 24, 25, 26, 27], NOT just [22, 23]
   - Look for: product name continuity, related images, variant displays, technical specs across multiple pages
   - A product's page range ends when a new product begins or when content becomes unrelated
3. For other categories: Be comprehensive - identify EVERY instance
4. Focus on fulfilling the user's request: "{agent_prompt}"
5. Classify each page as: "product", "certificate", "logo", "specification", "marketing", "admin", or "transitional"
6. Provide confidence scores (0.0-1.0) for each item

**SPECIAL HANDLING FOR USER REQUEST:**
- If user mentions specific product names (e.g., "NOVA"), prioritize finding those products
- If user requests "search", provide comprehensive results with high confidence scores
- If user requests "extract", focus on complete data extraction with all metadata

**OUTPUT FORMAT (JSON):**
```json
{{
  "products": [
    {{
      "name": "NOVA",
      "description": "Modern ceramic tile collection",
      "page_range": [12, 13, 14],
      "image_pages": [12, 13],
      "confidence": 0.95,
      "metadata": {{
        "designer": "SG NY",
        "studio": "SG NY",
        "category": "tiles",
        "dimensions": ["15×38", "20×40"],
        "variants": [
          {{"type": "color", "value": "beige"}},
          {{"type": "finish", "value": "matte"}}
        ],
        "factory": "Castellón Factory",
        "factory_group": "Harmony Group",
        "manufacturer": "Harmony Materials",
        "country_of_origin": "Spain",
        "slip_resistance": "R11",
        "fire_rating": "A1",
        "thickness": "8mm",
        "water_absorption": "Class 3",
        "finish": "matte",
        "material": "ceramic"
      }}
    }}
  ],
  "certificates": [
    {{
      "name": "ISO 9001:2015",
      "certificate_type": "quality_management",
      "issuer": "TÜV SÜD",
      "issue_date": "2023-01-15",
      "expiry_date": "2026-01-15",
      "standards": ["ISO 9001:2015"],
      "page_range": [45, 46],
      "confidence": 0.92
    }}
  ],
  "logos": [
    {{
      "name": "Company Logo",
      "logo_type": "company",
      "description": "Main company brand logo",
      "page_range": [1, 2],
      "confidence": 0.98
    }}
  ],
  "specifications": [
    {{
      "name": "Installation Guide",
      "spec_type": "installation",
      "description": "Step-by-step installation instructions",
      "page_range": [50, 52],
      "confidence": 0.90
    }}
  ],
  "page_classification": {{
    "1": "logo",
    "2": "marketing",
    "12": "product",
    "13": "product",
    "45": "certificate",
    "50": "specification"
  }},
  "total_products": 14,
  "total_certificates": 3,
  "total_logos": 5,
  "total_specifications": 2,
  "confidence_score": 0.92
}}
```

**PDF CONTENT:**
{pdf_text[:200000]}

Analyze the above content and return ONLY valid JSON with ALL content discovered across the requested categories."""

        return prompt

    def _repair_json(self, json_str: str) -> str:
        """Attempt to repair common JSON issues"""
        # Remove trailing commas before closing brackets/braces
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        # Fix missing commas between array elements
        json_str = re.sub(r'}\s*{', r'},{', json_str)
        # Fix missing commas between object properties
        json_str = re.sub(r'"\s*"', r'","', json_str)
        return json_str

    async def _discover_with_claude(
        self,
        prompt: str,
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Use Claude Sonnet 4.5 for product discovery"""
        start_time = datetime.now()
        
        try:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            
            response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=16000,  # Large response for comprehensive catalog
                temperature=0.1,  # Low temperature for consistent extraction
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            content = response.content[0].text.strip()
            
            # Parse JSON from response
            try:
                # Extract JSON from markdown code blocks if present
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    content = content[json_start:json_end].strip()
                elif "```" in content:
                    json_start = content.find("```") + 3
                    json_end = content.find("```", json_start)
                    content = content[json_start:json_end].strip()

                try:
                    result = json.loads(content)
                except json.JSONDecodeError as first_error:
                    self.logger.warning(f"First JSON parse failed, attempting repair: {first_error}")
                    try:
                        repaired = self._repair_json(content)
                        result = json.loads(repaired)
                        self.logger.info("Successfully repaired and parsed JSON")
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Failed to parse Claude response as JSON even after repair: {e}")
                        self.logger.debug(f"Raw response (first 1000 chars): {content[:1000]}")
                        raise RuntimeError(f"Claude returned invalid JSON: {e}")

                # DEBUG: Log how many products Claude found
                products_found = len(result.get("products", []))
                self.logger.info(f"🔍 Claude Sonnet 4.5 discovered {products_found} products")
                if products_found > 0:
                    product_names = [p.get("name", "Unknown") for p in result.get("products", [])]
                    self.logger.info(f"   Product names: {product_names}")

                # Log AI call
                latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                await self.ai_logger.log_claude_call(
                    task="product_discovery",
                    model="claude-sonnet-4-5",
                    response=response,
                    latency_ms=latency_ms,
                    confidence_score=result.get("confidence_score", 0.9),
                    confidence_breakdown={},
                    action="use_ai_result",  # Fixed: must be 'use_ai_result' or 'fallback_to_rules'
                    job_id=job_id
                )
                
                return result
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse Claude response as JSON: {e}")
                self.logger.debug(f"Raw response (first 500 chars): {content[:500]}")
                raise RuntimeError(f"Claude returned invalid JSON: {e}")
                
        except Exception as e:
            self.logger.error(f"Claude product discovery failed: {e}")
            raise RuntimeError(f"Claude product discovery failed: {str(e)}") from e
    
    async def _discover_with_gpt(
        self,
        prompt: str,
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Use GPT-5 for product discovery"""
        start_time = datetime.now()
        
        try:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            
            response = client.chat.completions.create(
                model="gpt-4o",  # Will be GPT-5 when available
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing product catalogs and extracting structured product information. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=8000,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content.strip()
            result = json.loads(content)

            # DEBUG: Log how many products GPT found
            products_found = len(result.get("products", []))
            self.logger.info(f"🔍 GPT-4o discovered {products_found} products")
            if products_found > 0:
                product_names = [p.get("name", "Unknown") for p in result.get("products", [])]
                self.logger.info(f"   Product names: {product_names}")

            # Log AI call
            latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            confidence_score = result.get("confidence_score", 0.9)

            # Calculate cost
            input_cost = (response.usage.prompt_tokens / 1_000_000) * 2.50  # GPT-4o pricing
            output_cost = (response.usage.completion_tokens / 1_000_000) * 10.00
            total_cost = input_cost + output_cost

            await self.ai_logger.log_ai_call(
                task="product_discovery",
                model="gpt-4o",
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                cost=total_cost,
                latency_ms=latency_ms,
                confidence_score=confidence_score,
                confidence_breakdown={"overall": confidence_score},
                action="use_ai_result",
                job_id=job_id
            )

            return result
            
        except Exception as e:
            self.logger.error(f"GPT product discovery failed: {e}")
            raise RuntimeError(f"GPT product discovery failed: {str(e)}") from e
    
    def _parse_discovery_results(
        self,
        result: Dict[str, Any],
        total_pages: int,
        categories: List[str]
    ) -> ProductCatalog:
        """Parse and validate discovery results across all categories"""

        # Parse products
        products = []
        for p in result.get("products", []):
            # Extract metadata (new architecture - products + metadata inseparable)
            metadata = p.get("metadata", {})

            # If metadata is not in the new format, build it from old fields for backward compatibility
            if not metadata:
                metadata = {
                    "designer": p.get("designer"),
                    "studio": p.get("studio"),
                    "dimensions": p.get("dimensions", []),
                    "variants": p.get("variants", []),
                    "category": p.get("category"),
                    "page_range": p.get("page_range", []),
                    "confidence": p.get("confidence", 0.8)
                }
                # Remove None values
                metadata = {k: v for k, v in metadata.items() if v is not None}

            product = ProductInfo(
                name=p.get("name", "Unknown"),
                page_range=p.get("page_range", []),
                description=p.get("description", ""),
                metadata=metadata,
                image_indices=p.get("image_pages", []),
                confidence=p.get("confidence", 0.8)
            )
            products.append(product)

        # Parse certificates
        certificates = []
        for c in result.get("certificates", []):
            certificate = CertificateInfo(
                name=c.get("name", "Unknown"),
                page_range=c.get("page_range", []),
                certificate_type=c.get("certificate_type"),
                issuer=c.get("issuer"),
                issue_date=c.get("issue_date"),
                expiry_date=c.get("expiry_date"),
                standards=c.get("standards", []),
                confidence=c.get("confidence", 0.8)
            )
            certificates.append(certificate)

        # Parse logos
        logos = []
        for l in result.get("logos", []):
            logo = LogoInfo(
                name=l.get("name", "Unknown"),
                page_range=l.get("page_range", []),
                logo_type=l.get("logo_type"),
                description=l.get("description"),
                confidence=l.get("confidence", 0.8)
            )
            logos.append(logo)

        # Parse specifications
        specifications = []
        for s in result.get("specifications", []):
            spec = SpecificationInfo(
                name=s.get("name", "Unknown"),
                page_range=s.get("page_range", []),
                spec_type=s.get("spec_type"),
                description=s.get("description"),
                confidence=s.get("confidence", 0.8)
            )
            specifications.append(spec)

        # Build page classification
        page_classification = {}
        for page_str, classification in result.get("page_classification", {}).items():
            page_classification[int(page_str)] = classification

        catalog = ProductCatalog(
            products=products,
            certificates=certificates,
            logos=logos,
            specifications=specifications,
            total_pages=total_pages,
            total_images=0,  # Will be updated later
            content_classification=page_classification,
            processing_time_ms=0,  # Will be set by caller
            model_used=self.model,
            confidence_score=result.get("confidence_score", 0.85)
        )

        return catalog

    async def _enrich_products_with_focused_extraction(
        self,
        catalog: ProductCatalog,
        pdf_path: str,
        job_id: Optional[str] = None
    ) -> ProductCatalog:
        """
        STAGE 0B: Extract detailed metadata for each product using focused page extraction.

        This is the core of the Two-Stage Discovery system:
        1. For each product, extract ONLY its specific pages from the PDF
        2. Send focused text to AI for detailed metadata extraction
        3. No token limits - can handle products with 50+ pages each

        Args:
            catalog: Product catalog from Stage 0A (with page ranges)
            pdf_path: Path to PDF file for page extraction
            job_id: Optional job ID for logging

        Returns:
            Catalog with fully enriched product metadata
        """
        try:
            from app.core.extractor import extract_pdf_to_markdown
            import pymupdf4llm

            # Initialize metadata extractor
            metadata_extractor = DynamicMetadataExtractor(model=self.model, job_id=job_id)

            enriched_products = []

            # ⚡ OPTIMIZATION: Extract all product pages in ONE pass instead of sequentially
            # Collect all unique pages needed across all products
            all_product_pages = set()
            product_page_mapping = {}  # Map product index to its pages

            for i, product in enumerate(catalog.products):
                page_indices = [p - 1 for p in product.page_range if p > 0]
                if page_indices:
                    all_product_pages.update(page_indices)
                    product_page_mapping[i] = page_indices

            # Extract ALL product pages in a single PyMuPDF4LLM call
            if all_product_pages:
                self.logger.info(f"   ⚡ OPTIMIZED: Extracting {len(all_product_pages)} unique pages in ONE pass for {len(catalog.products)} products")
                sorted_pages = sorted(all_product_pages)

                # Single extraction for all products (MUCH faster than sequential)
                all_pages_text = pymupdf4llm.to_markdown(pdf_path, pages=sorted_pages)

                # Parse the markdown to extract page-specific content
                # PyMuPDF4LLM returns markdown with page markers
                page_texts = self._split_markdown_by_pages(all_pages_text, sorted_pages)

                self.logger.info(f"      ✅ Extracted {len(all_pages_text)} characters from {len(sorted_pages)} pages")
            else:
                page_texts = {}
                self.logger.warning("   ⚠️ No valid pages found for any products")

            # Now process each product with its pre-extracted text
            for i, product in enumerate(catalog.products):
                try:
                    self.logger.info(f"   🔍 [{i+1}/{len(catalog.products)}] Processing {product.name} metadata...")

                    # Get this product's pages from the mapping
                    page_indices = product_page_mapping.get(i)

                    if not page_indices:
                        self.logger.warning(f"   ⚠️ No valid pages for {product.name}, skipping")
                        enriched_products.append(product)
                        continue

                    # Combine text from this product's pages
                    product_text = "\n\n".join(
                        page_texts.get(page_idx, "")
                        for page_idx in page_indices
                    )

                    self.logger.info(f"      Using {len(product_text)} characters from {len(page_indices)} pages")

                    # Get category hint from existing metadata
                    category_hint = product.metadata.get("category") or product.metadata.get("material")

                    # Extract comprehensive metadata from focused text
                    extracted = await metadata_extractor.extract_metadata(
                        pdf_text=product_text,
                        category_hint=category_hint
                    )

                    # Merge extracted metadata with existing metadata
                    # Priority: existing metadata > extracted critical > extracted discovered
                    enriched_metadata = {
                        **extracted.get("discovered", {}),  # Lowest priority
                        **extracted.get("critical", {}),    # Medium priority
                        **product.metadata,                 # Highest priority (from discovery)
                        "_extraction_metadata": extracted.get("metadata", {})
                    }

                    # Flatten nested values (extract "value" from {"value": "...", "confidence": ...})
                    flattened_metadata = {}
                    for key, value in enriched_metadata.items():
                        if isinstance(value, dict) and "value" in value:
                            flattened_metadata[key] = value["value"]
                        else:
                            flattened_metadata[key] = value

                    # Update product with enriched metadata
                    product.metadata = flattened_metadata
                    enriched_products.append(product)

                    self.logger.info(f"      ✅ Extracted {len(flattened_metadata)} metadata fields")

                except Exception as e:
                    self.logger.error(f"Failed to enrich metadata for {product.name}: {e}")
                    # Keep original product if enrichment fails
                    enriched_products.append(product)

            # Update catalog with enriched products
            catalog.products = enriched_products

            return catalog

        except Exception as e:
            self.logger.error(f"Focused extraction failed: {e}")
            # Return original catalog if enrichment fails
            return catalog

    async def _enrich_products_with_metadata(
        self,
        catalog: ProductCatalog,
        pdf_text: str,
        job_id: Optional[str] = None
    ) -> ProductCatalog:
        """
        FALLBACK: Enrich products with metadata using full PDF text.

        This is used when pdf_path is not available for focused extraction.
        Less efficient than focused extraction but still works.

        Args:
            catalog: Product catalog from discovery
            pdf_text: Full PDF text content
            job_id: Optional job ID for logging

        Returns:
            Catalog with enriched product metadata
        """
        try:
            # Initialize metadata extractor with same model as discovery
            metadata_extractor = DynamicMetadataExtractor(model=self.model, job_id=job_id)

            enriched_products = []

            for product in catalog.products:
                try:
                    # Extract product-specific text from page range (limited)
                    product_text = self._extract_product_text(pdf_text, product.page_range)

                    # Get category hint from existing metadata
                    category_hint = product.metadata.get("category") or product.metadata.get("material")

                    # Extract comprehensive metadata
                    self.logger.info(f"   🔍 Extracting metadata for: {product.name}")
                    extracted = await metadata_extractor.extract_metadata(
                        pdf_text=product_text,
                        category_hint=category_hint
                    )

                    # Merge extracted metadata with existing metadata
                    # Priority: existing metadata > extracted critical > extracted discovered
                    enriched_metadata = {
                        **extracted.get("discovered", {}),  # Lowest priority
                        **extracted.get("critical", {}),    # Medium priority
                        **product.metadata,                 # Highest priority (from discovery)
                        "_extraction_metadata": extracted.get("metadata", {})
                    }

                    # Flatten nested values (extract "value" from {"value": "...", "confidence": ...})
                    flattened_metadata = {}
                    for key, value in enriched_metadata.items():
                        if isinstance(value, dict) and "value" in value:
                            flattened_metadata[key] = value["value"]
                        else:
                            flattened_metadata[key] = value

                    # Update product with enriched metadata
                    product.metadata = flattened_metadata
                    enriched_products.append(product)

                    self.logger.info(f"      ✅ Extracted {len(flattened_metadata)} metadata fields")

                except Exception as e:
                    self.logger.error(f"Failed to enrich metadata for {product.name}: {e}")
                    # Keep original product if enrichment fails
                    enriched_products.append(product)

            # Update catalog with enriched products
            catalog.products = enriched_products

            return catalog

        except Exception as e:
            self.logger.error(f"Metadata enrichment failed: {e}")
            # Return original catalog if enrichment fails
            return catalog

    def _split_markdown_by_pages(self, markdown_text: str, page_indices: list) -> dict:
        """
        Split PyMuPDF4LLM markdown output into individual pages.

        PyMuPDF4LLM returns markdown with page markers like:
        -----
        # Page 1
        content...
        -----
        # Page 2
        content...

        Args:
            markdown_text: Full markdown text from PyMuPDF4LLM
            page_indices: List of 0-indexed page numbers that were extracted

        Returns:
            Dictionary mapping page_index -> page_text
        """
        import re

        page_texts = {}

        # Split by page markers (PyMuPDF4LLM uses "-----" as separator)
        # Pattern: -----\n# Page N\n or similar
        pages = re.split(r'-{3,}\s*(?:#\s*Page\s*\d+)?', markdown_text)

        # Map extracted pages to their indices
        for i, page_text in enumerate(pages):
            if i < len(page_indices) and page_text.strip():
                page_idx = page_indices[i]
                page_texts[page_idx] = page_text.strip()

        return page_texts

    def _extract_index_text(self, pdf_text: str, index_pages: int, total_pages: int) -> str:
        """
        Extract text from index/TOC pages for Stage 0A discovery.

        Uses smart extraction to get the most relevant content:
        1. First 30 pages (usually contains TOC/Index)
        2. Last 10 pages (sometimes contains product index)
        3. Limits to reasonable size for AI processing

        Args:
            pdf_text: Full PDF text
            index_pages: Number of pages to scan for index
            total_pages: Total pages in PDF

        Returns:
            Extracted index text (limited to ~100K chars)
        """
        # Strategy: Extract first N pages which usually contain TOC/Index
        # Most catalogs have comprehensive index in first 20-50 pages

        # Calculate character limit based on index pages
        # Assume ~1500 chars per page average
        char_limit = index_pages * 1500

        # Cap at 100K characters to avoid token limits in Stage 0A
        char_limit = min(char_limit, 100000)

        index_text = pdf_text[:char_limit]

        self.logger.info(f"   Extracted {len(index_text)} characters from first {index_pages} pages")

        return index_text

    def _extract_product_text(self, pdf_text: str, page_range: List[int]) -> str:
        """
        Extract text for specific product pages.

        Args:
            pdf_text: Full PDF text
            page_range: List of page numbers for this product

        Returns:
            Text content for the product pages
        """
        # For now, return full PDF text
        # TODO: Implement page-specific text extraction if needed
        # This would require storing page boundaries during PDF extraction
        return pdf_text[:10000]  # Limit to first 10k chars to avoid token limits

