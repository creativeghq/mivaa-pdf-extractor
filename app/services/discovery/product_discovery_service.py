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

from app.services.core.ai_call_logger import AICallLogger
from app.services.metadata.dynamic_metadata_extractor import DynamicMetadataExtractor
from app.services.core.ai_client_service import get_ai_client_service
from app.utils.page_converter import PageConverter, PageNumber  # ✅ NEW: Centralized page management
from app.config import get_settings


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

        # Dimensions (all unique sizes across variants)
        "dimensions": ["15×38", "20×40", "7×14.8"],

        # Product variants (SKU codes with colors, shapes, patterns)
        # Each variant is a complete product configuration
        "variants": [
            {
                "sku": "37885",
                "name": "FOLD WHITE/15X38",
                "color": "WHITE",
                "shape": "FOLD",
                "pattern": null,
                "size": "15×38",
                "pattern_count": null,
                "mapei_code": "100"
            },
            {
                "sku": "39656",
                "name": "VALENOVA WHITE LT/11,8X11,8",
                "color": "WHITE LT",
                "shape": null,
                "pattern": null,
                "size": "11.8×11.8",
                "pattern_count": 12,
                "mapei_code": "100"
            },
            {
                "sku": "40123",
                "name": "CHEVRON OAK/20X120",
                "color": "OAK",
                "shape": null,
                "pattern": "CHEVRON",
                "size": "20×120",
                "pattern_count": null,
                "mapei_code": null
            }
        ],

        # Available colors (when listed without individual SKUs)
        "available_colors": ["clay", "sand", "white", "taupe"],

        # Packaging details (CRITICAL for quote management)
        "packaging": {
            "pieces_per_box": 12,
            "boxes_per_pallet": 48,
            "weight_per_box_kg": 18.5,
            "coverage_per_box_m2": 1.14,
            "coverage_per_box_sqft": 12.27
        },

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
    page_types: Dict[int, str] = None  # Page type classification: {page_num: "TEXT"|"IMAGE"|"MIXED"|"EMPTY"}
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

    # Catalog-level factory info (inherited by products without factory info)
    catalog_factory: Optional[str] = None  # e.g., "HARMONY"
    catalog_factory_group: Optional[str] = None  # e.g., "Peronda Group"
    catalog_manufacturer: Optional[str] = None  # e.g., "Harmony Materials"
    pages_per_sheet: int = 1  # 1 for standard, 2 for 2-page spreads

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

    NEW: Supports vision-guided extraction for precise image cropping.
    """

    def __init__(self, model: str = "claude"):
        """
        Initialize service.

        Args:
            model: AI model to use - supports "claude", "gpt"
        """
        self.logger = logger
        self.model = model
        self.ai_logger = AICallLogger()
        self.settings = get_settings()

        # Check API keys based on model family (claude/gpt)
        if "claude" in model.lower() and not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set - cannot use Claude")
        if "gpt" in model.lower() and not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set - cannot use GPT")
    
    async def discover_products_from_text(
        self,
        markdown_text: str,
        source_type: str = "web_scraping",
        categories: List[str] = None,
        agent_prompt: Optional[str] = None,
        workspace_id: str = "ffafc28b-1b8b-4b0d-b226-9f9a6154004e",
        enable_prompt_enhancement: bool = True,
        job_id: Optional[str] = None
    ) -> ProductCatalog:
        """
        Discover products from markdown text (web scraping, XML, or any text source).

        This is the UNIFIED product discovery method that works with ANY markdown text source:
        - Web scraping (Firecrawl markdown)
        - XML imports (converted to markdown)
        - Manual text input
        - Future text sources

        **ARCHITECTURE:**
        - Reuses existing text-based discovery pipeline
        - Reuses existing metadata extraction
        - NO changes to PDF pipeline
        - NO changes to XML pipeline

        Args:
            markdown_text: Markdown-formatted text content to analyze
            source_type: Type of source ("web_scraping", "xml_import", "manual")
            categories: Categories to discover (products, certificates, logos, specifications). Default: ["products"]
            agent_prompt: Optional natural language prompt from agent
            workspace_id: Workspace ID for custom prompts
            enable_prompt_enhancement: Whether to enhance prompts with admin templates
            job_id: Optional job ID for tracking

        Returns:
            ProductCatalog with all discovered content

        Example:
            ```python
            # Web scraping use case
            service = ProductDiscoveryService(model="claude")
            catalog = await service.discover_products_from_text(
                markdown_text=firecrawl_markdown,
                source_type="web_scraping",
                categories=["products"]
            )
            ```
        """
        start_time = datetime.now()

        # Default to products only if not specified
        if categories is None:
            categories = ["products"]

        try:
            self.logger.info(f"🔍 Starting TEXT-BASED discovery from {source_type.upper()}")
            self.logger.info(f"   Text length: {len(markdown_text):,} characters")
            self.logger.info(f"   Categories: {', '.join(categories)}")
            if agent_prompt:
                self.logger.info(f"   Agent Prompt: '{agent_prompt}'")

            # Estimate total pages from text length (rough estimate: 2000 chars per page)
            estimated_pages = max(1, len(markdown_text) // 2000)

            # Use iterative batch discovery (same as PDF text-based discovery)
            catalog = await self._iterative_batch_discovery(
                pdf_text=markdown_text,
                total_pages=estimated_pages,
                categories=categories,
                agent_prompt=agent_prompt,
                workspace_id=workspace_id,
                enable_prompt_enhancement=enable_prompt_enhancement,
                job_id=job_id
            )

            self.logger.info(f"✅ Discovery complete: Found {len(catalog.products)} products")
            for product in catalog.products:
                self.logger.info(f"   📦 {product.name}")

            # Enrich products with metadata using full text
            if "products" in categories and catalog.products:
                self.logger.info(f"🔍 Extracting detailed metadata for each product...")
                catalog = await self._enrich_products_with_metadata(
                    catalog,
                    markdown_text,
                    job_id
                )

            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            catalog.processing_time_ms = processing_time
            catalog.model_used = self.model

            # Log comprehensive results
            self.logger.info(f"✅ Text-based discovery complete in {processing_time:.0f}ms:")
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
            self.logger.error(f"❌ Text-based product discovery failed: {e}")
            raise

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
        pdf_path: Optional[str] = None,
        tracker: Optional[Any] = None
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
            self.logger.info(f"🔍 Starting TEXT-BASED discovery for {total_pages} pages using {self.model.upper()}")
            self.logger.info(f"   Categories: {', '.join(categories)}")
            if agent_prompt:
                self.logger.info(f"   Agent Prompt: '{agent_prompt}'")
            if enable_prompt_enhancement:
                self.logger.info(f"   Prompt Enhancement: ENABLED")

            # ============================================================
            # TEXT-BASED DISCOVERY
            # ============================================================
            self.logger.info(f"📋 TEXT MODE: Iterative batch discovery with early stopping...")

            # Extract text from PDF if not provided
            if pdf_text is None:
                if pdf_path is None:
                    raise ValueError("Either pdf_text or pdf_path must be provided for text-based discovery")

                self.logger.info(f"   Extracting full PDF text for iterative discovery...")
                import pymupdf4llm

                # Extract ALL pages (SLOW: 10+ minutes for 71 pages)
                pdf_text = pymupdf4llm.to_markdown(pdf_path)
                self.logger.info(f"   Extracted {len(pdf_text)} characters from {total_pages} pages")

            # Iterative batch discovery
            catalog = await self._iterative_batch_discovery(
                pdf_text,
                total_pages,
                categories,
                agent_prompt,
                workspace_id,
                enable_prompt_enhancement,
                job_id
                )

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
                    pdf_text=pdf_text,
                    job_id=job_id,
                    tracker=tracker
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
                from app.services.utilities.prompt_enhancement_service import PromptEnhancementService

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

**FIRST: EXTRACT DOCUMENT-LEVEL INFORMATION**
Look at the cover page, intro pages, and headers/footers to identify:
1. **catalog_factory**: The main factory/brand name for this catalog (e.g., "HARMONY", "Porcelanosa")
2. **catalog_factory_group**: The parent company or group (e.g., "Peronda Group", "Porcelanosa Group")
3. **catalog_manufacturer**: The manufacturer if different from factory

This information typically appears on:
- Cover page (large logo/brand name)
- Footer/header of pages
- "About Us" or intro sections
- Copyright notices

**THEN: Extract content across the following categories:**
{chr(10).join(category_instructions)}

{agent_context}

**GENERAL INSTRUCTIONS:**
1. Be comprehensive - identify EVERY product name in the catalog
2. **DO NOT include page numbers** - we will detect pages automatically using vision/text search
3. Focus on extracting product names and metadata accurately
4. Provide confidence scores (0.0-1.0) for each item

**OUTPUT FORMAT (JSON):**
```json
{{
  "catalog_factory": "HARMONY",
  "catalog_factory_group": "Peronda Group",
  "catalog_manufacturer": "Peronda Group",
  "products": [
    {{
      "name": "NOVA",
      "description": "Modern ceramic tile collection",
      "confidence": 0.95,
      "metadata": {{
        "designer": "SG NY",
        "studio": "SG NY",
        "material_category": "Ceramic Tile",
        "dimensions": ["15×38", "20×40"],

        // ✅ CRITICAL: Product variants with SKU codes, colors, shapes, patterns
        // Each variant is a COMPLETE product configuration with its own SKU
        "variants": [
          {{
            "sku": "37885",
            "name": "FOLD WHITE/15X38",
            "color": "WHITE",
            "shape": "FOLD",
            "pattern": null,
            "size": "15×38",
            "pattern_count": null,
            "mapei_code": "100"
          }},
          {{
            "sku": "37889",
            "name": "FOLD CLAY/15X38",
            "color": "CLAY",
            "shape": "FOLD",
            "pattern": null,
            "size": "15×38",
            "pattern_count": null,
            "mapei_code": "145"
          }},
          {{
            "sku": "38343",
            "name": "TRI. FOLD WHITE/7X14,8",
            "color": "WHITE",
            "shape": "TRI. FOLD",
            "pattern": null,
            "size": "7×14.8",
            "pattern_count": null,
            "mapei_code": "100"
          }},
          {{
            "sku": "39656",
            "name": "VALENOVA WHITE LT/11,8X11,8",
            "color": "WHITE LT",
            "shape": null,
            "pattern": null,
            "size": "11.8×11.8",
            "pattern_count": 12,
            "mapei_code": "100"
          }},
          {{
            "sku": "40123",
            "name": "CHEVRON OAK/20X120",
            "color": "OAK",
            "shape": null,
            "pattern": "CHEVRON",
            "size": "20×120",
            "pattern_count": null,
            "mapei_code": null
          }}
        ],

        // Available colors (when listed without individual SKUs)
        "available_colors": ["clay", "sand", "white", "taupe"],

        "factory_name": "HARMONY",
        "factory_group_name": "Peronda Group",
        "country_of_origin": "Spain",
        "slip_resistance": "R11",
        "fire_rating": "A1",
        "thickness": "8mm",
        "water_absorption": "Class 3",
        "finish": "matte",
        "material": "ceramic",

        // Packaging details (CRITICAL for quote management)
        "packaging": {{
          "pieces_per_box": 12,
          "boxes_per_pallet": 48,
          "weight_per_box_kg": 18.5,
          "coverage_per_box_m2": 1.14,
          "coverage_per_box_sqft": 12.27
        }}
      }}
    }}
  ],
  "certificates": [...],
  "logos": [...],
  "specifications": [...],
  "page_classification": {{...}},
  "total_products": 14,
  "confidence_score": 0.92
}}
```

**CRITICAL: Product Variant Extraction Rules**

Products often have multiple SKU codes representing different variants (colors, shapes, patterns, sizes).
You MUST identify the MAIN PRODUCT NAME and extract ALL variants as separate entries.

**Example 1 - Color & Shape Variants:**
```
37885 FOLD WHITE/15X38
37889 FOLD CLAY/15X38
37888 FOLD GREEN/15X38
38343 TRI. FOLD WHITE/7X14,8
38341 TRI. FOLD CLAY/7X14,8
```

**Extraction:**
- Main Product: "FOLD"
- Variants:
  * SKU 37885: color="WHITE", shape="FOLD", size="15×38"
  * SKU 37889: color="CLAY", shape="FOLD", size="15×38"
  * SKU 37888: color="GREEN", shape="FOLD", size="15×38"
  * SKU 38343: color="WHITE", shape="TRI. FOLD", size="7×14.8"
  * SKU 38341: color="CLAY", shape="TRI. FOLD", size="7×14.8"

**Example 2 - Pattern Variants:**
```
39656 VALENOVA WHITE LT/11,8X11,8
12 patterns · * 100 Mapei
39659 VALENOVA CLAY LT/11,8X11,8
12 patterns · ** 40 Kerakoll
```

**Extraction:**
- Main Product: "VALENOVA"
- Variants:
  * SKU 39656: color="WHITE LT", size="11.8×11.8", pattern_count=12, mapei_code="100"
  * SKU 39659: color="CLAY LT", size="11.8×11.8", pattern_count=12, mapei_code="40"

**Variant Extraction Rules:**
1. **Identify the base product name** (e.g., "FOLD", "VALENOVA", "NOVA")
2. **Extract ALL SKU codes** - these are typically 5-digit numbers at the start of each line
3. **Parse variant attributes from the full name:**
   - Color: Extract color name (WHITE, CLAY, GREEN, SAND, TAUPE, etc.)
   - Shape/Pattern: Extract shape OR pattern modifier (TRI., RECT., HEX., CHEVRON, HERRINGBONE, etc.)
   - Size: Extract dimensions (15×38, 11.8×11.8, etc.)
4. **Extract pattern count** if mentioned (e.g., "12 patterns")
5. **Extract reference codes** (Mapei codes, Kerakoll codes, etc.)
6. **Extract available colors from lists** - Look for color lists like "clay · sand · white · taupe" and create variants for each
7. **DO NOT create separate products for each color** - they are variants of the SAME product
8. **DO NOT confuse product names with colors** - "VALENOVA CLAY" is product "VALENOVA" with color "CLAY", NOT a product called "VALENOVA CLAY"

**Color List Extraction:**
When you see color lists like:
```
VALENOVA by SG NY
White Body Tile
11,8x11,8 cm
clay · sand · white · taupe
```

Extract as:
- Product: "VALENOVA"
- Available colors: ["clay", "sand", "white", "taupe"]
- If SKUs are listed separately for each color, create full variants
- If only color list is shown, store as "available_colors" in metadata

**Packaging Details Extraction (CRITICAL for Quote Management):**

Extract ALL packaging information found in the PDF:
- **pieces_per_box**: Number of pieces/tiles per box
- **boxes_per_pallet**: Number of boxes per pallet
- **weight_per_box_kg**: Weight per box in kilograms
- **weight_per_box_lb**: Weight per box in pounds (if provided)
- **coverage_per_box_m2**: Coverage area per box in square meters
- **coverage_per_box_sqft**: Coverage area per box in square feet

Common patterns to look for:
- "12 pcs/box", "pieces per box: 12"
- "48 boxes/pallet", "boxes per pallet: 48"
- "18.5 kg/box", "weight: 18.5 kg"
- "1.14 m²/box", "coverage: 1.14 m²"
- "12.27 sqft/box", "coverage: 12.27 sqft"

**IMPORTANT:**
- ALWAYS extract catalog_factory from cover/intro pages - this is the brand that makes ALL products
- Products inherit factory_name from catalog_factory if not specified individually
- Use consistent field names: factory_name (not factory), factory_group_name (not factory_group), material_category (not category)
- Each variant MUST have: sku, name (full variant name), color, size
- Optional variant fields: shape, pattern, pattern_count, mapei_code, kerakoll_code
- ALWAYS extract packaging details when available - critical for quote calculations

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

        prompt = f"""You are analyzing a product catalog PDF with {total_pages} pages (numbered 1 to {total_pages} in this file).

**YOUR TASK**: Identify ALL products that actually exist in THIS PDF by analyzing the content.

**⚠️ CRITICAL - PDF EXCERPT DETECTION**:
This PDF may be an EXCERPT from a larger catalog. The index/TOC might reference page numbers from the ORIGINAL catalog that don't exist in this file.

**INSTRUCTIONS**:
1. **Look for any product index/listing section** - May have various names like "INDEX", "COLLECTIONS", "PRODUCTS", etc.
2. **Scan the content** to identify all products by looking for:
   - Product names in uppercase or bold
   - Page numbers associated with products
   - Designer/studio attributions
3. **Validate page numbers**: This PDF has pages 1-{total_pages}. Any page number > {total_pages} does NOT exist in this file
4. **For each product found**:
   - If index says "Product X ... pages 50-55" but this PDF only has {total_pages} pages
   - Check if Product X actually appears in the PDF content on pages 1-{total_pages}
   - If YES: Include it with the ACTUAL pages where it appears in THIS PDF
   - If NO: SKIP this product entirely (it's not in this excerpt)
5. **Page ranges**: Include ALL consecutive pages where the product appears in THIS PDF
6. **Be comprehensive**: Find ALL products that actually exist in pages 1-{total_pages}

**WHAT TO LOOK FOR**:
- Product names in uppercase or bold (e.g., "VALENOVA", "FOLD", "PIQUÉ")
- Page numbers next to product names (e.g., "— **24**", "FOLD ... 32-35")
- Designer names (e.g., "by SG NY", "by ESTUDI{{H}}AC", "by DSIGNIO")
- Section headers indicating product categories

**⚠️ IMPORTANT**: Only include products that you can accurately map to page numbers within the range 1-{total_pages}. If a product is mentioned but its page is outside this range or cannot be determined, SKIP it.

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

**FIRST: EXTRACT DOCUMENT-LEVEL INFORMATION**
Look at the cover page, intro pages, and headers/footers to identify:
1. **catalog_factory**: The main factory/brand name for this catalog (e.g., "HARMONY", "Porcelanosa")
2. **catalog_factory_group**: The parent company or group (e.g., "Peronda Group", "Porcelanosa Group")
3. **catalog_manufacturer**: The manufacturer if different from factory

This information typically appears on:
- Cover page (large logo/brand name)
- Footer/header of pages
- "About Us" or intro sections
- Copyright notices

**THEN: Extract content across the following categories:**
{chr(10).join(category_sections)}

**⚠️ CRITICAL - PDF EXCERPT HANDLING**:
This PDF has {total_pages} pages (numbered 1 to {total_pages}). It may be an EXCERPT from a larger catalog.

**VALIDATION RULES**:
1. **Scan ACTUAL CONTENT** - Don't just copy page numbers from the index/TOC
2. **Page number validation**: ALL page_range values MUST be between 1 and {total_pages}
3. **If a product's index entry references pages > {total_pages}**:
   - Search for that product name in the ACTUAL PDF content (pages 1-{total_pages})
   - If found: Use the ACTUAL pages where it appears in THIS PDF
   - If NOT found: SKIP this product (it's not in this excerpt)
4. **Example**: Index says "LOG ... pages 74-79" but PDF has only {total_pages} pages
   - Search pages 1-{total_pages} for "LOG" product
   - If found on pages 45-48: Use [45,46,47,48]
   - If not found: SKIP LOG entirely

**PRODUCT IDENTIFICATION**:
1. Identify ONLY main featured products using ANY of these criteria:
   - EXCLUDE index pages, thumbnails, cross-references, and catalog grids (this is critical!)
   - INCLUDE if product has: dedicated page spread (1-12 pages) OR comprehensive metadata OR visual prominence OR content depth
   - Use multiple identification methods: page spread, metadata presence, visual prominence, or content depth
   - Even single-page products count if they have comprehensive metadata and visual prominence
2. **For page_range**: Include ALL consecutive pages where the product appears in THIS PDF (pages 1-{total_pages})
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
  "catalog_factory": "HARMONY",
  "catalog_factory_group": "Peronda Group",
  "catalog_manufacturer": "Peronda Group",
  "products": [
    {{
      "name": "NOVA",
      "description": "Modern ceramic tile collection",
      "confidence": 0.95,
      "metadata": {{
        "designer": "SG NY",
        "studio": "SG NY",
        "material_category": "Ceramic Tile",
        "dimensions": ["15×38", "20×40"],
        "variants": [
          {{"type": "color", "value": "beige"}},
          {{"type": "finish", "value": "matte"}}
        ],
        "factory_name": "HARMONY",
        "factory_group_name": "Peronda Group",
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
  "certificates": [...],
  "logos": [...],
  "specifications": [...],
  "page_classification": {{...}},
  "total_products": 14,
  "confidence_score": 0.92
}}
```

**IMPORTANT:**
- ALWAYS extract catalog_factory from cover/intro pages - this is the brand that makes ALL products
- Products inherit factory_name from catalog_factory if not specified individually
- Use consistent field names: factory_name (not factory), factory_group_name (not factory_group), material_category (not category)

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
            # Use centralized AI client service
            ai_service = get_ai_client_service()
            client = ai_service.anthropic

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
            # Use centralized AI client service
            ai_service = get_ai_client_service()
            client = ai_service.openai

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
        products_missing_pages = []
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

            # Parse page_types (convert string keys to int)
            page_types_raw = p.get("page_types", {})
            page_types = {}
            if page_types_raw:
                for page_str, page_type in page_types_raw.items():
                    page_types[int(page_str)] = page_type

            # Page range is now optional - will be detected later using vision/text search
            page_range = p.get("page_range", [])
            product = ProductInfo(
                name=p.get("name", "Unknown"),
                page_range=page_range,  # Can be empty - will be filled by vision detection
                description=p.get("description", ""),
                metadata=metadata,
                # Image indices will be detected later if not provided
                image_indices=p.get("image_pages", []),
                page_types=page_types if page_types else None,
                confidence=p.get("confidence", 0.8)
            )
            products.append(product)

            # Track products with missing page ranges for fallback processing
            if not page_range or len(page_range) == 0:
                products_missing_pages.append(product.name)

            # Log if image_pages was missing (helps debug Claude Vision behavior)
            if not p.get("image_pages"):
                self.logger.warning(f"   ⚠️ Product '{product.name}' missing image_pages - using page_range as fallback")

        # Log products that need fallback page detection
        if products_missing_pages:
            self.logger.warning(
                f"   ⚠️ {len(products_missing_pages)} products missing page ranges - will need fallback detection: "
                f"{', '.join(products_missing_pages[:5])}"
                f"{' and ' + str(len(products_missing_pages) - 5) + ' more' if len(products_missing_pages) > 5 else ''}"
            )

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

        # Extract catalog-level factory info (from cover/intro pages)
        catalog_factory = result.get("catalog_factory")
        catalog_factory_group = result.get("catalog_factory_group")
        catalog_manufacturer = result.get("catalog_manufacturer")

        # Extract pages_per_sheet (layout info)
        pages_per_sheet = result.get("pages_per_sheet", 1)  # Default to 1 if not provided

        # Log if catalog-level factory info was found
        if catalog_factory:
            self.logger.info(f"   🏭 Catalog factory: {catalog_factory}")
        if catalog_factory_group:
            self.logger.info(f"   🏢 Catalog factory group: {catalog_factory_group}")

        catalog = ProductCatalog(
            products=products,
            certificates=certificates,
            logos=logos,
            specifications=specifications,
            catalog_factory=catalog_factory,
            catalog_factory_group=catalog_factory_group,
            catalog_manufacturer=catalog_manufacturer,
            total_pages=total_pages,
            pages_per_sheet=pages_per_sheet, # ✅ PERSIST LAYOUT
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
        pdf_text: Optional[str] = None,
        job_id: Optional[str] = None,
        tracker: Optional[Any] = None
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

            # Store tracker for heartbeat updates
            self.tracker = tracker

            enriched_products = []

            # ⚡ OPTIMIZATION: Extract all product pages in ONE pass instead of sequentially
            # Collect all unique pages needed across all products
            all_product_pages = set()
            product_page_mapping = {}  # Map product index to its pages

            # 🔍 VALIDATION: Get PDF page count and detect layout using PageConverter
            from app.utils.page_converter import PageConverter

            converter = PageConverter.from_pdf_path(pdf_path)
            pdf_page_count = converter.total_pdf_pages
            pages_per_sheet = converter.pages_per_sheet

            if pages_per_sheet == 2:
                self.logger.info(f"   ✅ DOMINANT LAYOUT: 2-page spreads")
                self.logger.info(f"      → Catalog pages 1-{converter.total_catalog_pages} mapped to PDF pages 1-{pdf_page_count}")
            else:
                self.logger.info(f"   ✅ DOMINANT LAYOUT: Standard")

            self.logger.info(f"   📄 PDF has {pdf_page_count} pages ({converter.total_catalog_pages} catalog pages)")

            # Track products that need vision-based page detection
            products_needing_vision = []

            for i, product in enumerate(catalog.products):
                # Convert catalog pages to PDF pages and validate
                page_indices = []
                invalid_pages = []

                # ✅ CHECK: If product has empty page_range, mark for vision-based detection
                if not product.page_range or len(product.page_range) == 0:
                    self.logger.warning(
                        f"   ⚠️ Product '{product.name}' has EMPTY page_range - "
                        f"will use vision-based detection to find pages"
                    )
                    products_needing_vision.append((i, product))
                    continue  # Skip normal page mapping for now

                for catalog_page in product.page_range:
                    if catalog_page > 0:
                        try:
                            # ✅ USE PAGE CONVERTER: Type-safe conversion
                            page = converter.from_catalog_page(catalog_page)
                            page_indices.append(page.array_index)
                        except ValueError as e:
                            # Page is out of bounds
                            invalid_pages.append(catalog_page)

                if invalid_pages:
                    # Log each invalid page individually for clarity
                    for invalid_page in invalid_pages:
                        if pages_per_sheet == 2:
                            self.logger.warning(
                                f"   ⚠️ Skipping hallucinated page {invalid_page} "
                                f"(PDF has only {pdf_page_count * 2} catalog pages = {pdf_page_count} PDF pages)"
                            )
                        else:
                            self.logger.warning(
                                f"   ⚠️ Skipping hallucinated page {invalid_page} "
                                f"(PDF has only {pdf_page_count} pages)"
                            )

                    # Summary log
                    if pages_per_sheet == 2:
                        self.logger.warning(
                            f"   ⚠️ Product '{product.name}' has {len(invalid_pages)} invalid catalog pages "
                            f"(PDF has {pdf_page_count} pages = {pdf_page_count * 2} catalog pages) - skipping these pages"
                        )
                    else:
                        self.logger.warning(
                            f"   ⚠️ Product '{product.name}' has {len(invalid_pages)} invalid pages "
                            f"(PDF has {pdf_page_count} pages) - skipping these pages"
                        )

                    # ✅ SANITIZATION: Remove invalid pages from the product's page_range
                    # This prevents downstream services from trying to process non-existent pages
                    product.page_range = [p for p in product.page_range if p not in invalid_pages]

                if page_indices:
                    all_product_pages.update(page_indices)
                    product_page_mapping[i] = page_indices

            # ============================================================
            # TEXT-BASED PAGE DETECTION FOR PRODUCTS WITH EMPTY PAGE_RANGE
            # ============================================================
            if products_needing_vision:
                self.logger.info(
                    f"🔍 TEXT-BASED PAGE DETECTION: {len(products_needing_vision)} products need page detection"
                )

                # Ensure we have pdf_text
                if not pdf_text:
                    self.logger.info("   Extracting text for page detection...")
                    import pymupdf4llm
                    pdf_text = pymupdf4llm.to_markdown(pdf_path)

                for i, product in products_needing_vision:
                    try:
                        self.logger.info(f"   🔍 Detecting pages for: {product.name}")

                        detected_pages = await self._detect_product_pages_with_text(
                            pdf_text=pdf_text,
                            product_name=product.name,
                            total_pages=pdf_page_count
                        )

                        if detected_pages:
                            self.logger.info(
                                f"   ✅ Found {len(detected_pages)} pages for '{product.name}': {detected_pages}"
                            )
                        else:
                            self.logger.error(
                                f"   ❌ Could not detect pages for '{product.name}' - "
                                f"product will be SKIPPED to prevent hallucinated data"
                            )

                        # Update product and mapping
                        if detected_pages:
                            product.page_range = detected_pages
                            page_indices = [p - 1 for p in detected_pages]  # Convert to 0-based
                            all_product_pages.update(page_indices)
                            product_page_mapping[i] = page_indices

                    except Exception as e:
                        self.logger.error(
                            f"   ❌ Detection failed for '{product.name}': {e}"
                        )
                        continue

            # ============================================================
            # INTELLIGENT PAGE EXTRACTION BASED ON PAGE TYPES
            # ============================================================
            page_texts = {}

            if all_product_pages:
                sorted_pages = sorted(all_product_pages)

                # Separate pages by type for optimal processing
                text_pages = []
                image_pages = []
                mixed_pages = []

                # Collect page types from all products
                for product in catalog.products:
                    if product.page_types:
                        for page_num, page_type in product.page_types.items():
                            page_idx = page_num - 1  # Convert to 0-based
                            if page_idx in sorted_pages:
                                if page_type == "TEXT":
                                    text_pages.append(page_idx)
                                elif page_type == "IMAGE":
                                    image_pages.append(page_idx)
                                elif page_type == "MIXED":
                                    mixed_pages.append(page_idx)

                # Remove duplicates
                text_pages = sorted(set(text_pages))
                image_pages = sorted(set(image_pages))
                mixed_pages = sorted(set(mixed_pages))

                self.logger.info(f"   📊 Page type distribution: {len(text_pages)} TEXT, {len(image_pages)} IMAGE, {len(mixed_pages)} MIXED")

                # ✅ MEMORY OPTIMIZATION: Store page type classifications only (not extracted text)
                # We'll extract text per-product to avoid keeping all 52 pages in memory
                page_types = {}  # page_idx -> "TEXT" | "IMAGE" | "MIXED" | "UNCLASSIFIED"

                for page_idx in text_pages:
                    page_types[page_idx] = "TEXT"
                for page_idx in image_pages:
                    page_types[page_idx] = "IMAGE"
                for page_idx in mixed_pages:
                    page_types[page_idx] = "MIXED"

                # Mark unclassified pages
                unclassified_pages = [p for p in sorted_pages if p not in text_pages and p not in image_pages and p not in mixed_pages]
                for page_idx in unclassified_pages:
                    page_types[page_idx] = "UNCLASSIFIED"

                if unclassified_pages:
                    self.logger.warning(f"   ⚠️ {len(unclassified_pages)} pages have no type classification")

                self.logger.info(f"   ✅ Page type classification complete: {len(page_types)} pages classified")
                self.logger.info(f"   💾 MEMORY OPTIMIZATION: Text will be extracted per-product (not all at once)")
            else:
                self.logger.warning("   ⚠️ No valid pages found for any products")
                page_types = {}

            # Now process each product - extract text per-product to minimize memory
            for i, product in enumerate(catalog.products):
                try:
                    # ✅ UPDATE PROGRESS: Show real-time progress during metadata extraction
                    # Stage 0B progress: 0-10% (Stage 0A was discovery, this is metadata enrichment)
                    progress_pct = int((i / len(catalog.products)) * 10)  # 0-10% range
                    if self.tracker:
                        await self.tracker.update_heartbeat()
                        # Update progress with current stage info
                        self.tracker.manual_progress_override = progress_pct
                        await self.tracker._sync_to_database(stage='product_discovery')

                    self.logger.info(f"   🔍 [{i+1}/{len(catalog.products)}] Processing {product.name} metadata... ({progress_pct}%)")

                    # Get this product's pages from the mapping
                    page_indices = product_page_mapping.get(i)

                    if not page_indices:
                        if pages_per_sheet == 2:
                            self.logger.warning(f"   ⚠️ Product '{product.name}' has no valid pages in this PDF - REMOVING from catalog")
                            self.logger.warning(f"      Catalog pages: {product.page_range} (PDF has {pdf_page_count} pages = {pdf_page_count * 2} catalog pages)")
                        else:
                            self.logger.warning(f"   ⚠️ Product '{product.name}' has no valid pages in this PDF - REMOVING from catalog")
                            self.logger.warning(f"      Original page_range: {product.page_range} (PDF has {pdf_page_count} pages)")
                        # DO NOT add to enriched_products - this product doesn't exist in this PDF
                        continue

                    # ✅ MEMORY OPTIMIZATION: Extract text ONLY for this product's pages
                    # This keeps memory low (~500MB per product instead of 3-4GB for all pages)
                    product_page_texts = {}

                    # Separate pages by type for this product
                    product_text_pages = [p for p in page_indices if page_types.get(p) == "TEXT"]
                    product_image_pages = [p for p in page_indices if page_types.get(p) == "IMAGE"]
                    product_mixed_pages = [p for p in page_indices if page_types.get(p) == "MIXED"]
                    product_unclassified_pages = [p for p in page_indices if page_types.get(p) == "UNCLASSIFIED"]

                    # Extract TEXT pages for this product
                    if product_text_pages:
                        try:
                            text_markdown = pymupdf4llm.to_markdown(pdf_path, pages=product_text_pages)
                            text_page_texts = self._split_markdown_by_pages(text_markdown, product_text_pages)
                            product_page_texts.update(text_page_texts)
                        except Exception as e:
                            self.logger.warning(f"      ⚠️ PyMuPDF4LLM failed for TEXT pages: {e}")
                            for page_idx in product_text_pages:
                                product_page_texts[page_idx] = ""

                    # IMAGE pages - use empty text (vision data already in metadata)
                    for page_idx in product_image_pages:
                        product_page_texts[page_idx] = ""

                    # Extract MIXED pages for this product
                    if product_mixed_pages:
                        try:
                            mixed_markdown = pymupdf4llm.to_markdown(pdf_path, pages=product_mixed_pages)
                            mixed_page_texts = self._split_markdown_by_pages(mixed_markdown, product_mixed_pages)
                            product_page_texts.update(mixed_page_texts)
                        except Exception as e:
                            self.logger.warning(f"      ⚠️ PyMuPDF4LLM failed for MIXED pages: {e}")
                            for page_idx in product_mixed_pages:
                                product_page_texts[page_idx] = ""

                    # Extract UNCLASSIFIED pages for this product
                    if product_unclassified_pages:
                        try:
                            unclass_markdown = pymupdf4llm.to_markdown(pdf_path, pages=product_unclassified_pages)
                            unclass_page_texts = self._split_markdown_by_pages(unclass_markdown, product_unclassified_pages)
                            product_page_texts.update(unclass_page_texts)
                        except Exception as e:
                            self.logger.warning(f"      ⚠️ PyMuPDF4LLM failed for unclassified pages: {e}")
                            for page_idx in product_unclassified_pages:
                                product_page_texts[page_idx] = ""

                    # Combine text from this product's pages
                    product_text = "\n\n".join(
                        product_page_texts.get(page_idx, "")
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

                    # ✅ MEMORY CLEANUP: Delete extracted text immediately after processing
                    # This frees ~500MB-1GB per product, preventing memory accumulation
                    del product_text
                    del product_page_texts
                    del extracted
                    del enriched_metadata
                    del flattened_metadata

                    # Force garbage collection to free memory immediately
                    import gc
                    gc.collect()

                except Exception as e:
                    self.logger.error(f"Failed to enrich metadata for {product.name}: {e}")
                    # Keep original product if enrichment fails
                    enriched_products.append(product)

                    # Cleanup even on error
                    import gc
                    gc.collect()

            # ✅ UPDATE PROGRESS: Mark Stage 0B complete (10% progress)
            if self.tracker:
                self.tracker.manual_progress_override = 10
                await self.tracker._sync_to_database(stage='product_discovery')
                self.logger.info(f"✅ Stage 0B complete: Metadata extraction finished (10%)")

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

                    # NEW: Validate metadata against prototypes
                    from app.services.metadata.metadata_prototype_validator import get_metadata_validator

                    try:
                        validator = get_metadata_validator(job_id=job_id)
                        validation_result = await validator.validate_metadata(
                            extracted_metadata=extracted,
                            confidence_threshold=0.80
                        )

                        validated_metadata = validation_result["validated_metadata"]
                        validation_info = validation_result["validation_info"]

                        self.logger.info(f"      ✅ Validated {len(validation_info)} metadata fields")
                    except Exception as e:
                        self.logger.warning(f"Metadata validation failed, using unvalidated: {e}")
                        # Fallback: flatten without validation
                        validated_metadata = {}
                        for category, fields in extracted.get("discovered", {}).items():
                            if isinstance(fields, dict):
                                validated_metadata.update(fields)
                        validated_metadata.update(extracted.get("critical", {}))
                        validation_info = {}

                    # Merge validated metadata with existing metadata
                    # Priority: existing metadata > validated metadata > extraction metadata
                    enriched_metadata = {
                        **validated_metadata,               # Validated extracted metadata
                        **product.metadata,                 # Highest priority (from discovery)
                        "_extraction_metadata": extracted.get("metadata", {}),
                        "_validation": validation_info      # Track validation details
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

                    validated_count = sum(1 for v in validation_info.values() if v.get("prototype_matched"))
                    self.logger.info(f"      ✅ Extracted {len(flattened_metadata)} fields ({validated_count} validated)")

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

    async def _iterative_batch_discovery(
        self,
        pdf_text: str,
        total_pages: int,
        categories: List[str],
        agent_prompt: str,
        workspace_id: str,
        enable_prompt_enhancement: bool,
        job_id: Optional[str] = None
    ) -> ProductCatalog:
        """
        Iterative batch discovery with early stopping.

        Sends text in 100K char batches until no new products are found.
        This is much smarter than arbitrary limits:
        - Small PDFs: Stops after 1-2 batches
        - Large PDFs: Continues until all products found
        - Saves money by not processing unnecessary text

        Args:
            pdf_text: Full PDF text
            total_pages: Total pages in PDF
            categories: Categories to extract
            agent_prompt: Custom prompt
            workspace_id: Workspace ID
            enable_prompt_enhancement: Whether to enhance prompts
            job_id: Job ID for logging

        Returns:
            ProductCatalog with all discovered products
        """
        BATCH_SIZE = 100000  # 100K chars per batch
        all_products = []
        all_certificates = []
        all_logos = []
        all_specifications = []
        seen_product_names = set()

        batch_num = 0
        offset = 0

        while offset < len(pdf_text):
            batch_num += 1
            batch_text = pdf_text[offset:offset + BATCH_SIZE]

            self.logger.info(f"   📦 Batch {batch_num}: Processing chars {offset:,} to {offset + len(batch_text):,}")

            # Build prompt for this batch
            index_prompt = await self._build_index_scan_prompt(
                batch_text,
                total_pages,
                categories,
                agent_prompt,
                workspace_id,
                enable_prompt_enhancement
            )

            # Call AI model based on model family
            if "claude" in self.model.lower():
                batch_result = await self._discover_with_claude(index_prompt, job_id)
            else:
                batch_result = await self._discover_with_gpt(index_prompt, job_id)

            # Parse results
            batch_catalog = self._parse_discovery_results(batch_result, total_pages, categories)

            # Count new products found in this batch
            new_products_count = 0
            for product in batch_catalog.products:
                if product.name not in seen_product_names:
                    all_products.append(product)
                    seen_product_names.add(product.name)
                    new_products_count += 1

            # Add other categories
            all_certificates.extend(batch_catalog.certificates)
            all_logos.extend(batch_catalog.logos)
            all_specifications.extend(batch_catalog.specifications)

            self.logger.info(f"      ✅ Found {new_products_count} NEW products (total: {len(all_products)})")

            # Early stopping: If no new products found, stop
            if new_products_count == 0:
                self.logger.info(f"   🛑 EARLY STOP: No new products in batch {batch_num}, stopping discovery")
                break

            # Move to next batch
            offset += BATCH_SIZE

            # Safety limit: Max 10 batches (1M chars)
            if batch_num >= 10:
                self.logger.warning(f"   ⚠️ Reached max batch limit (10 batches = 1M chars), stopping")
                break

        # Create final catalog
        catalog = ProductCatalog(
            products=all_products,
            certificates=all_certificates,
            logos=all_logos,
            specifications=all_specifications,
            confidence_score=0.9  # High confidence from iterative discovery
        )

        return catalog

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
        # ✅ FIX: Increased from 10000 to 100000 chars to include end-of-document sections
        # (packaging, compliance, care/maintenance info is typically at the end)
        return pdf_text[:100000]


    async def _detect_product_pages_with_text(
        self,
        pdf_text: str,
        product_name: str,
        total_pages: int
    ) -> List[int]:
        """
        Use text-based analysis to detect which pages contain a specific product.

        This is a fallback mechanism for when both AI discovery and vision-based
        detection fail to assign page ranges.

        Args:
            pdf_text: Full PDF markdown text with page markers
            product_name: Name of product to search for
            total_pages: Total number of pages in PDF

        Returns:
            List of page numbers (1-based) where product was found
        """
        if not pdf_text or not product_name:
            return []

        import re
        detected_pages = set()
        clean_name = product_name.lower().strip()

        # 1. Try exact match first
        # 2. Try partial match if no exact match found
        
        # Split text into pages using PyMuPDF4LLM markers
        # Pattern: -----\n# Page N\n (matches _split_markdown_by_pages)
        markers = list(re.finditer(r'-{3,}\s*(?:#\s*Page\s*(\d+))?', pdf_text))
        
        pages_content = []
        if not markers:
            # Fallback: Treat whole text as Page 1 if no markers
            pages_content.append((1, pdf_text))
        else:
            # Add text before first marker to Page 1
            first_text = pdf_text[:markers[0].start()].strip()
            if first_text:
                pages_content.append((1, first_text))
            
            for i in range(len(markers)):
                start = markers[i].end()
                end = markers[i+1].start() if i+1 < len(markers) else len(pdf_text)
                
                page_num_str = markers[i].group(1)
                # If no page number in marker, use sequence
                page_num = int(page_num_str) if page_num_str else (i + 1)
                
                content = pdf_text[start:end].strip()
                if content:
                    pages_content.append((page_num, content))

        # Search for product name in each page
        for page_num, content in pages_content:
            if page_num > total_pages:
                continue
                
            content_lower = content.lower()
            
            # Use word boundaries for better accuracy
            if re.search(r'\b' + re.escape(clean_name) + r'\b', content_lower):
                detected_pages.add(page_num)
            # Fallback to simple containment if no word boundary match
            elif clean_name in content_lower:
                detected_pages.add(page_num)

        return sorted(list(detected_pages))

