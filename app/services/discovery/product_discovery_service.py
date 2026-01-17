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
- Stage 0A: Claude discovers PRODUCT NAMES + metadata (NO page numbers!)
- Stage 0B: DETERMINISTIC page detection using:
  * Text search (PyMuPDF) to find pages containing product names
  * YOLO layout validation to confirm product-relevant content
- This eliminates catalog vs PDF page number confusion
- Subsequent stages create semantic chunks for RAG search

KEY DESIGN DECISION:
Claude does NOT return page_range. Instead, we detect pages deterministically:
1. Add explicit page markers (--- # Page N ---) during PDF text extraction
2. Search for product names in the text
3. Use YOLO to validate pages have product content (TITLE, IMAGE, TEXT regions)

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
from app.services.utilities.prompt_templates import get_prompt_template_from_db
# PageConverter removed - using simple PDF page numbers instead
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
    # pages_per_sheet removed - we only use PDF pages now

    # Metadata
    total_pages: int = 0  # Total PHYSICAL pages (accounting for spreads)
    total_pdf_pages: int = 0  # Actual PDF page count
    total_images: int = 0
    content_classification: Dict[int, str] = None  # page_number -> "product" | "certificate" | "logo" | "specification" | "marketing" | "admin"

    # Spread layout info (for catalogs with 2-page spreads)
    has_spread_layout: bool = False
    # Mapping: physical_page -> (pdf_page_idx, 'left'|'right'|'single'|'full')
    physical_to_pdf_map: Dict[int, tuple] = None

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
        if self.physical_to_pdf_map is None:
            self.physical_to_pdf_map = {}


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
            # Store tracker reference for progress updates
            self.tracker = tracker

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

                self.logger.info(f"📄 Extracting PDF text page-by-page with progress tracking...")
                self.logger.info(f"   Total pages: {total_pages}")
                import pymupdf4llm
                import fitz

                # Extract page by page with progress logging
                doc = fitz.open(pdf_path)
                pdf_text_parts = []
                pages_extracted = 0
                pages_failed = 0

                # Log progress every N pages
                log_interval = max(1, total_pages // 10)  # Log ~10 times during extraction

                for page_num in range(len(doc)):
                    try:
                        page_text = pymupdf4llm.to_markdown(pdf_path, pages=[page_num])
                        # ✅ FIX: Add explicit page markers that Claude can use
                        # PyMuPDF4LLM does NOT include page numbers - only "-----" separators
                        # Without these markers, Claude uses printed catalog page numbers from INDEX
                        # which are DIFFERENT from actual PDF page numbers
                        page_marker = f"\n\n--- # Page {page_num + 1} ---\n\n"
                        pdf_text_parts.append(page_marker + page_text)
                        pages_extracted += 1

                        # Log progress at intervals
                        if (page_num + 1) % log_interval == 0 or (page_num + 1) == total_pages:
                            progress_pct = int(((page_num + 1) / total_pages) * 100)
                            self.logger.info(f"   📊 Progress: {page_num + 1}/{total_pages} pages ({progress_pct}%) - {len(''.join(pdf_text_parts)):,} chars extracted")
                            # Update tracker with extraction progress (0-3% of Stage 0)
                            if self.tracker:
                                extraction_progress = int((progress_pct / 100) * 3)  # 0-3%
                                self.tracker.manual_progress_override = extraction_progress
                                self.tracker.current_step = f"Extracting text: {page_num + 1}/{total_pages} pages"
                                await self.tracker.update_heartbeat()

                    except Exception as page_error:
                        pages_failed += 1
                        self.logger.warning(f"   ⚠️ Skipping page {page_num + 1} due to error: {page_error}")
                        continue

                doc.close()
                pdf_text = "\n\n".join(pdf_text_parts)

                self.logger.info(f"✅ PDF text extraction complete:")
                self.logger.info(f"   📄 Pages extracted: {pages_extracted}/{total_pages}")
                if pages_failed > 0:
                    self.logger.warning(f"   ⚠️ Pages failed: {pages_failed}")
                self.logger.info(f"   📝 Total characters: {len(pdf_text):,}")

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
            # Update progress: Stage 0A complete (discovery scan) = 5%
            if self.tracker:
                self.tracker.manual_progress_override = 5
                self.tracker.current_step = f"Discovery complete: {len(catalog.products)} products found"
                await self.tracker.update_heartbeat()
                await self.tracker._sync_to_database(stage="product_discovery")
            for product in catalog.products:
                self.logger.info(f"   📦 {product.name}")

            # ============================================================
            # STAGE 0B: DETERMINISTIC PAGE DETECTION + FOCUSED EXTRACTION
            # Claude returns ONLY product names - we detect pages using text search + YOLO
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

        Fetches prompt template from database and fills in variables.
        All prompts are managed via /admin/ai-configs.
        """
        try:
            # Fetch discovery prompt from database
            prompt_template = await get_prompt_template_from_db(
                workspace_id=workspace_id,
                stage="discovery",
                category="products"
            )

            self.logger.info(f"   ✅ Loaded discovery prompt from database")

            # Replace template variables
            prompt = prompt_template.replace("{total_pages}", str(total_pages))
            prompt = prompt.replace("{categories}", ", ".join(categories))
            prompt = prompt.replace("{pdf_text}", pdf_text[:200000])

            if agent_prompt:
                prompt = prompt.replace("{agent_prompt}", agent_prompt)
            else:
                prompt = prompt.replace("{agent_prompt}", "Extract all products from this catalog")

            return prompt

        except Exception as e:
            self.logger.error(f"Failed to load discovery prompt from database: {e}")
            raise ValueError(
                f"Discovery prompt not found in database for workspace={workspace_id}, "
                f"stage=discovery, category=products. Please add it via /admin/ai-configs."
            )


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

        Fetches prompt template from database and fills in variables.
        All prompts are managed via /admin/ai-configs.

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
        try:
            # Fetch index scan prompt from database
            prompt_template = await get_prompt_template_from_db(
                workspace_id=workspace_id,
                stage="discovery",
                category="index_scan"
            )

            self.logger.info(f"   ✅ Loaded index_scan prompt from database")

            # Replace template variables
            prompt = prompt_template.replace("{total_pages}", str(total_pages))
            prompt = prompt.replace("{index_text}", index_text)
            prompt = prompt.replace("{categories}", ", ".join(categories))

            if agent_prompt:
                prompt = prompt.replace("{agent_prompt}", agent_prompt)
            else:
                prompt = prompt.replace("{agent_prompt}", "Extract all products from this catalog")

            return prompt

        except Exception as e:
            self.logger.error(f"Failed to load index_scan prompt from database: {e}")
            raise ValueError(
                f"Index scan prompt not found in database for workspace={workspace_id}, "
                f"stage=discovery, category=index_scan. Please add it via /admin/ai-configs."
            )


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
                temperature=0,  # Zero temperature for maximum consistency
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

        # Parse products (Claude returns product names + metadata, NO page_range - that's detected separately)
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

            # page_range is intentionally not returned by Claude - we detect pages using text search + YOLO
            # This is by design - Claude only returns product names, we handle page detection deterministically

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
        STAGE 0B: Deterministic page detection + detailed metadata extraction.

        This is the core of the Two-Stage Discovery system:
        1. DETERMINISTIC PAGE DETECTION: Use text search + YOLO to find pages for each product
           (Claude returns ONLY product names, NOT page numbers)
        2. For each product, extract ONLY its detected pages from the PDF
        3. Send focused text to AI for detailed metadata extraction
        4. No token limits - can handle products with 50+ pages each

        Args:
            catalog: Product catalog from Stage 0A (product names only, NO page_range)
            pdf_path: Path to PDF file for page extraction
            pdf_text: Optional pre-extracted PDF text with page markers
            job_id: Optional job ID for logging
            tracker: Optional progress tracker

        Returns:
            Catalog with detected page_range and fully enriched product metadata
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

            # ============================================================
            # SPREAD LAYOUT DETECTION - Handle catalogs with 2-page spreads
            # ============================================================
            from app.utils.pdf_to_images import (
                analyze_pdf_layout,
                extract_text_with_physical_pages,
                PDFLayoutAnalysis
            )

            # Analyze PDF layout to detect spread pages
            pdf_layout = analyze_pdf_layout(pdf_path)
            pdf_page_count = pdf_layout.total_physical_pages  # Use PHYSICAL page count

            self.logger.info(f"   📄 PDF has {pdf_layout.total_pdf_pages} PDF pages -> {pdf_page_count} physical pages")
            if pdf_layout.has_spread_layout:
                self.logger.info(f"   📐 Spread layout detected - using physical page numbers for product mapping")

            # ============================================================
            # DETERMINISTIC PAGE DETECTION FOR ALL PRODUCTS (OPTIMIZED)
            # Claude returns ONLY product names - we detect pages using text search + YOLO
            # ============================================================
            self.logger.info(
                f"🔍 DETERMINISTIC PAGE DETECTION: Detecting pages for {len(catalog.products)} products using text search + YOLO"
            )

            # Ensure we have pdf_text with PHYSICAL page markers (handles spread layouts)
            if not pdf_text:
                self.logger.info("   Extracting text for page detection (with spread layout handling)...")
                pdf_text, _ = extract_text_with_physical_pages(pdf_path)

            # ⚡ OPTIMIZATION: Parse PDF text into pages ONCE (not per-product)
            pages_content = self._parse_pdf_text_into_pages(pdf_text, pdf_page_count)
            self.logger.info(f"   📄 Parsed {len(pages_content)} physical pages from text (one-time operation)")

            # ⚡ OPTIMIZATION: Initialize YOLO detector ONCE (not per-product)
            yolo_detector = None
            yolo_enabled = False
            try:
                from app.config import get_settings
                settings = get_settings()
                yolo_enabled = settings.yolo_enabled
                if yolo_enabled:
                    from app.services.pdf.yolo_layout_detector import YoloLayoutDetector
                    yolo_config = settings.get_yolo_config()
                    yolo_detector = YoloLayoutDetector(config=yolo_config)
                    self.logger.info("   🎯 YOLO detector initialized (reused for all products)")
            except Exception as e:
                self.logger.warning(f"   ⚠️ YOLO initialization failed: {e}, using text-only detection")

            # Detect pages for ALL products using pre-parsed pages
            for i, product in enumerate(catalog.products):
                try:
                    self.logger.info(f"   🔍 Detecting pages for: {product.name}")

                    # Step 1: Text-based page detection (uses pre-parsed pages)
                    detected_pages = self._detect_product_pages_optimized(
                        pages_content=pages_content,
                        product_name=product.name,
                        total_pages=pdf_page_count
                    )

                    # Step 2: YOLO validation (if enabled, uses shared detector)
                    if detected_pages and yolo_enabled and yolo_detector:
                        validated_pages = await self._validate_pages_with_yolo_optimized(
                            pdf_path=pdf_path,
                            detected_pages=detected_pages,
                            product_name=product.name,
                            detector=yolo_detector,
                            pdf_layout=pdf_layout  # Pass layout for spread handling
                        )
                        # Use validated pages if YOLO returned any, otherwise keep text-detected pages
                        if validated_pages:
                            detected_pages = validated_pages

                    if detected_pages:
                        self.logger.info(
                            f"   ✅ Found {len(detected_pages)} pages for '{product.name}': {detected_pages}"
                        )
                        product.page_range = detected_pages
                        page_indices = [p - 1 for p in detected_pages]  # Convert to 0-based
                        all_product_pages.update(page_indices)
                        product_page_mapping[i] = page_indices
                    else:
                        self.logger.error(
                            f"   ❌ Could not detect pages for '{product.name}' - "
                            f"product will be SKIPPED to prevent hallucinated data"
                        )

                except Exception as e:
                    self.logger.error(
                        f"   ❌ Detection failed for '{product.name}': {e}"
                    )
                    continue

            # ============================================================
            # EXCLUDE TOC/INDEX PAGES (pages appearing in most products)
            # ============================================================
            # TOC/Index pages list all product names, so they match every product.
            # We detect them by finding pages that appear in 70%+ of all products.
            if len(catalog.products) >= 3:  # Only meaningful with 3+ products
                page_occurrence_count: Dict[int, int] = {}
                products_with_pages = 0

                for product in catalog.products:
                    if product.page_range:
                        products_with_pages += 1
                        for page_num in product.page_range:
                            page_occurrence_count[page_num] = page_occurrence_count.get(page_num, 0) + 1

                if products_with_pages >= 3:
                    # Find pages appearing in 70%+ of products - likely TOC/index pages
                    toc_threshold = int(products_with_pages * 0.7)
                    toc_pages = {
                        page_num for page_num, count in page_occurrence_count.items()
                        if count >= toc_threshold
                    }

                    if toc_pages:
                        self.logger.warning(
                            f"   🚫 Detected {len(toc_pages)} likely TOC/Index pages (appear in 70%+ of products): {sorted(toc_pages)}"
                        )

                        # Remove TOC pages from each product's page_range
                        for i, product in enumerate(catalog.products):
                            if product.page_range:
                                original_pages = product.page_range.copy()
                                product.page_range = [p for p in product.page_range if p not in toc_pages]

                                removed_pages = set(original_pages) - set(product.page_range)
                                if removed_pages:
                                    self.logger.info(
                                        f"   ✂️  Removed TOC pages {sorted(removed_pages)} from '{product.name}'"
                                    )

                                # Update mapping
                                if i in product_page_mapping:
                                    product_page_mapping[i] = [p - 1 for p in product.page_range]

                        # Update all_product_pages to exclude TOC pages
                        toc_page_indices = {p - 1 for p in toc_pages}
                        all_product_pages -= toc_page_indices

                        self.logger.info(f"   ✅ TOC page exclusion complete. Remaining pages: {len(all_product_pages)}")

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

            # Store spread layout info in catalog for downstream use
            catalog.total_pages = pdf_layout.total_physical_pages
            catalog.total_pdf_pages = pdf_layout.total_pdf_pages
            catalog.has_spread_layout = pdf_layout.has_spread_layout
            catalog.physical_to_pdf_map = pdf_layout.physical_to_pdf_map

            self.logger.info(f"   📐 Catalog layout info: {catalog.total_pdf_pages} PDF pages -> {catalog.total_pages} physical pages")

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

        # Split by page markers (our explicit markers: --- # Page N ---)
        # Also handles old format: -----\n# Page N\n for backward compatibility
        pages = re.split(r'-{3,}\s*#?\s*Page\s*\d+\s*-*', markdown_text, flags=re.IGNORECASE)

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
        MIN_CHARS_BEFORE_EARLY_STOP = 200000  # Process at least 200K chars before allowing early stop
        CONSECUTIVE_EMPTY_BATCHES_TO_STOP = 2  # Require 2 consecutive empty batches

        all_products = []
        all_certificates = []
        all_logos = []
        all_specifications = []
        seen_product_names = set()

        batch_num = 0
        offset = 0
        consecutive_empty_batches = 0  # Track consecutive batches with no new products

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

            # Track consecutive empty batches for early stopping
            if new_products_count == 0:
                consecutive_empty_batches += 1
            else:
                consecutive_empty_batches = 0  # Reset counter when we find new products

            # Move to next batch
            offset += BATCH_SIZE
            chars_processed = offset

            # Early stopping conditions:
            # 1. Must have processed minimum chars (200K) before considering early stop
            # 2. Must have 2 consecutive batches with no new products
            if chars_processed >= MIN_CHARS_BEFORE_EARLY_STOP and consecutive_empty_batches >= CONSECUTIVE_EMPTY_BATCHES_TO_STOP:
                self.logger.info(f"   🛑 EARLY STOP: {consecutive_empty_batches} consecutive empty batches after {chars_processed:,} chars, stopping discovery")
                break

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


    # Pre-compiled regex pattern for page markers (compiled once, reused)
    _PAGE_MARKER_PATTERN = re.compile(r'-{3,}\s*#?\s*Page\s*(\d+)\s*-*', re.IGNORECASE)

    def _parse_pdf_text_into_pages(
        self,
        pdf_text: str,
        total_pages: int
    ) -> Dict[int, str]:
        """
        ⚡ OPTIMIZED: Parse PDF text into pages ONCE.

        This is called once before processing all products, avoiding
        repeated regex parsing for each product.

        Args:
            pdf_text: Full PDF markdown text with page markers
            total_pages: Total number of pages in PDF

        Returns:
            Dictionary mapping page_num (1-based) -> page_content (lowercased for search)
        """
        pages_content = {}

        if not pdf_text:
            return pages_content

        # Use pre-compiled pattern
        markers = list(self._PAGE_MARKER_PATTERN.finditer(pdf_text))

        if not markers:
            # Fallback: Treat whole text as Page 1 if no markers
            pages_content[1] = pdf_text.lower()
        else:
            # Add text before first marker to Page 1
            first_text = pdf_text[:markers[0].start()].strip()
            if first_text:
                pages_content[1] = first_text.lower()

            for i in range(len(markers)):
                start = markers[i].end()
                end = markers[i + 1].start() if i + 1 < len(markers) else len(pdf_text)

                page_num_str = markers[i].group(1)
                page_num = int(page_num_str) if page_num_str else (i + 1)

                if page_num <= total_pages:
                    content = pdf_text[start:end].strip()
                    if content:
                        # Store lowercased content for faster search
                        pages_content[page_num] = content.lower()

        return pages_content

    def _detect_product_pages_optimized(
        self,
        pages_content: Dict[int, str],
        product_name: str,
        total_pages: int
    ) -> List[int]:
        """
        ⚡ OPTIMIZED: Detect product pages using pre-parsed page content.

        This method uses pre-parsed pages (from _parse_pdf_text_into_pages)
        instead of re-parsing the entire PDF text for each product.

        Args:
            pages_content: Pre-parsed dictionary of page_num -> lowercased_content
            product_name: Name of product to search for
            total_pages: Total number of pages in PDF

        Returns:
            List of page numbers (1-based) where product was found
        """
        if not pages_content or not product_name:
            return []

        detected_pages = set()
        clean_name = product_name.lower().strip()

        # Pre-compile word boundary pattern for this product
        word_boundary_pattern = re.compile(r'\b' + re.escape(clean_name) + r'\b')

        # Search in pre-parsed pages (already lowercased)
        for page_num, content_lower in pages_content.items():
            if page_num > total_pages:
                continue

            # Use word boundaries ONLY - no substring fallback
            # (Fixes false matches like "internacional" matching "ONA")
            if word_boundary_pattern.search(content_lower):
                detected_pages.add(page_num)
            # NOTE: Removed substring fallback - it caused false positives
            # Products like ONA matched "internacional", "tradicional", "emocional" etc.

        return sorted(list(detected_pages))

    async def _detect_product_pages_with_text(
        self,
        pdf_text: str,
        product_name: str,
        total_pages: int
    ) -> List[int]:
        """
        DEPRECATED: Use _detect_product_pages_optimized instead.

        Kept for backward compatibility. Parses PDF text each time (inefficient).
        """
        # Parse pages (this is inefficient - called for each product)
        pages_content = self._parse_pdf_text_into_pages(pdf_text, total_pages)
        return self._detect_product_pages_optimized(pages_content, product_name, total_pages)

    async def _validate_pages_with_yolo_optimized(
        self,
        pdf_path: str,
        detected_pages: List[int],
        product_name: str,
        detector: Any,
        pdf_layout: Optional[Any] = None
    ) -> List[int]:
        """
        ⚡ OPTIMIZED: Validate pages using a pre-initialized YOLO detector.

        Args:
            pdf_path: Path to PDF file
            detected_pages: List of PHYSICAL page numbers (1-based) from text search
            product_name: Product name (for logging)
            detector: Pre-initialized YoloLayoutDetector (shared across products)
            pdf_layout: Optional PDFLayoutAnalysis for spread layout handling

        Returns:
            List of validated page numbers (1-based) where YOLO confirms product content
        """
        validated_pages = []

        for page_num in detected_pages:
            # Handle spread layout: convert physical page to PDF page + position
            if pdf_layout and pdf_layout.has_spread_layout and page_num in pdf_layout.physical_to_pdf_map:
                pdf_page_idx, position = pdf_layout.physical_to_pdf_map[page_num]
                # For spreads, we run YOLO on the full PDF page
                # The position ('left', 'right', 'full') tells us which half to focus on
                page_idx = pdf_page_idx
            else:
                page_idx = page_num - 1  # Convert to 0-based for YOLO (non-spread case)

            try:
                # Run YOLO layout detection on the PDF page
                result = await detector.detect_layout_regions(pdf_path, page_idx)

                if result and result.regions:
                    # Check if page has product-relevant content
                    has_title = result.title_regions > 0
                    has_image = result.image_regions > 0
                    has_text = result.text_regions > 0

                    # A product page should have at least:
                    # - TITLE + IMAGE (typical product spread), OR
                    # - IMAGE + TEXT (product with description), OR
                    # - Just IMAGE (image-only page in product spread)
                    is_product_page = (has_title and has_image) or (has_image and has_text) or has_image

                    if is_product_page:
                        validated_pages.append(page_num)
                        self.logger.debug(
                            f"      ✅ Page {page_num} validated: TITLE={has_title}, IMAGE={has_image}, TEXT={has_text}"
                        )
                    else:
                        # Still include pages with text but log as uncertain
                        if has_text:
                            validated_pages.append(page_num)
                            self.logger.debug(
                                f"      ⚠️ Page {page_num} has text only, including but uncertain"
                            )
                        else:
                            self.logger.debug(
                                f"      ❌ Page {page_num} excluded: no relevant content found"
                            )
                else:
                    # YOLO returned no regions - keep the page from text search
                    validated_pages.append(page_num)
                    self.logger.debug(f"      ⚠️ Page {page_num}: YOLO returned no regions, keeping from text search")

            except Exception as e:
                # If YOLO fails for a page, keep it from text search
                validated_pages.append(page_num)
                self.logger.warning(f"      ⚠️ YOLO failed for page {page_num}: {e}, keeping from text search")

        # Log summary
        if len(validated_pages) != len(detected_pages):
            removed = set(detected_pages) - set(validated_pages)
            self.logger.info(
                f"   🎯 YOLO validation: {len(detected_pages)} → {len(validated_pages)} pages "
                f"(removed: {sorted(removed)})"
            )

        return sorted(validated_pages)

    async def _validate_pages_with_yolo(
        self,
        pdf_path: str,
        detected_pages: List[int],
        product_name: str
    ) -> List[int]:
        """
        DEPRECATED: Use _validate_pages_with_yolo_optimized with pre-initialized detector.

        Kept for backward compatibility. Initializes YOLO detector each time (inefficient).
        """
        try:
            from app.config import get_settings
            settings = get_settings()

            # Skip YOLO validation if disabled
            if not settings.yolo_enabled:
                self.logger.debug(f"   YOLO disabled, skipping validation for '{product_name}'")
                return detected_pages

            from app.services.pdf.yolo_layout_detector import YoloLayoutDetector

            # Initialize YOLO detector (inefficient - called for each product)
            yolo_config = settings.get_yolo_config()
            detector = YoloLayoutDetector(config=yolo_config)

            return await self._validate_pages_with_yolo_optimized(
                pdf_path, detected_pages, product_name, detector
            )

        except Exception as e:
            self.logger.warning(f"   ⚠️ YOLO validation failed for '{product_name}': {e}, using text-detected pages")
            return detected_pages

