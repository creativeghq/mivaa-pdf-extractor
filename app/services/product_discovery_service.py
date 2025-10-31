"""
Product Discovery Service - Stage 0 Implementation

This service analyzes the entire PDF BEFORE processing to identify:
1. All products (names, page ranges, variants)
2. Product metadata (dimensions, designers, categories)
3. Metafield categories (R11 ratings, fire ratings, sizes, etc.)
4. Image-to-product mapping
5. Content classification (product vs marketing vs admin)

This enables focused extraction - only processing relevant content.
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

logger = logging.getLogger(__name__)

# Get API keys from environment
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


@dataclass
class ProductInfo:
    """Information about a discovered product"""
    name: str
    page_range: List[int]  # Pages where this product appears
    description: Optional[str] = None  # Product description
    designer: Optional[str] = None
    studio: Optional[str] = None
    dimensions: List[str] = None  # e.g., ["15×38", "20×40"]
    variants: List[Dict[str, Any]] = None  # Color/finish variants
    category: Optional[str] = None
    metafields: Dict[str, Any] = None  # R11, fire ratings, etc.
    image_indices: List[int] = None  # Which images belong to this product
    confidence: float = 0.0


@dataclass
class ProductCatalog:
    """Complete product catalog discovered from PDF"""
    products: List[ProductInfo]
    total_pages: int
    total_images: int
    metafield_categories: Dict[str, List[str]]  # e.g., {"slip_resistance": ["R9", "R10", "R11"]}
    content_classification: Dict[int, str]  # page_number -> "product" | "marketing" | "admin"
    processing_time_ms: float
    model_used: str
    confidence_score: float


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
        job_id: Optional[str] = None
    ) -> ProductCatalog:
        """
        Analyze PDF to discover all products and their metadata.
        
        Args:
            pdf_content: Raw PDF bytes
            pdf_text: Extracted text from PDF (markdown format)
            total_pages: Total number of pages in PDF
            job_id: Optional job ID for tracking
            
        Returns:
            ProductCatalog with all discovered products
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"🔍 Starting product discovery for {total_pages} pages using {self.model.upper()}")
            
            # Build comprehensive prompt for product discovery
            prompt = self._build_discovery_prompt(pdf_text, total_pages)
            
            # Call AI model
            if self.model == "claude":
                result = await self._discover_with_claude(prompt, job_id)
            else:
                result = await self._discover_with_gpt(prompt, job_id)
            
            # Parse and validate results
            catalog = self._parse_discovery_results(result, total_pages)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            catalog.processing_time_ms = processing_time
            catalog.model_used = self.model
            
            self.logger.info(f"✅ Product discovery complete: {len(catalog.products)} products found in {processing_time:.0f}ms")
            
            return catalog
            
        except Exception as e:
            self.logger.error(f"❌ Product discovery failed: {e}")
            raise
    
    def _build_discovery_prompt(self, pdf_text: str, total_pages: int) -> str:
        """Build comprehensive prompt for product discovery"""
        
        prompt = f"""You are analyzing a material/product catalog PDF with {total_pages} pages.

Your task is to identify ALL products and extract their complete metadata.

**IMPORTANT INSTRUCTIONS:**
1. Identify EVERY distinct product (not just the first few)
2. For each product, extract:
   - Product name (e.g., "NOVA", "BEAT", "FOLD")
   - Designer/Studio (e.g., "SG NY", "ESTUDI{{H}}AC")
   - Page range where product appears (e.g., [12, 13, 14])
   - All available dimensions/sizes (e.g., ["15×38", "20×40", "8×45"])
   - Variants (colors, finishes, patterns)
   - Category (e.g., "tiles", "flooring", "wall_covering")
   - Metafields:
     * Slip resistance (R9, R10, R11, R12, R13)
     * Fire rating (A1, A2, B, C, etc.)
     * Water absorption class
     * Thickness
     * Weight
     * Any other technical specifications

3. Classify each page as:
   - "product" - Contains product information
   - "marketing" - Marketing/branding content
   - "admin" - Index, TOC, legal, contact info
   - "transitional" - Dividers, section breaks

4. Identify which images belong to which products (by page number)

**OUTPUT FORMAT (JSON):**
```json
{{
  "products": [
    {{
      "name": "NOVA",
      "designer": "SG NY",
      "studio": "SG NY",
      "page_range": [12, 13, 14],
      "dimensions": ["15×38", "20×40"],
      "variants": [
        {{"type": "color", "value": "beige"}},
        {{"type": "finish", "value": "matte"}}
      ],
      "category": "tiles",
      "metafields": {{
        "slip_resistance": "R11",
        "fire_rating": "A1",
        "thickness": "8mm",
        "water_absorption": "Class 3"
      }},
      "image_pages": [12, 13],
      "confidence": 0.95
    }}
  ],
  "metafield_categories": {{
    "slip_resistance": ["R9", "R10", "R11"],
    "fire_rating": ["A1", "A2"],
    "thickness": ["6mm", "8mm", "10mm"]
  }},
  "page_classification": {{
    "1": "admin",
    "2": "marketing",
    "12": "product",
    "13": "product"
  }},
  "total_products": 14,
  "confidence_score": 0.92
}}
```

**PDF CONTENT:**
{pdf_text[:50000]}  

Analyze the above content and return ONLY valid JSON with ALL products discovered."""

        return prompt
    
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
                max_tokens=8000,  # Large response for comprehensive catalog
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
                
                result = json.loads(content)
                
                # Log AI call
                latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                await self.ai_logger.log_claude_call(
                    task="product_discovery",
                    model="claude-sonnet-4-5",
                    response=response,
                    latency_ms=latency_ms,
                    confidence_score=result.get("confidence_score", 0.9),
                    confidence_breakdown={},
                    action="product_discovery",
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
            
            # Log AI call
            latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            await self.ai_logger.log_openai_call(
                task="product_discovery",
                model="gpt-4o",
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                latency_ms=latency_ms,
                confidence_score=result.get("confidence_score", 0.9),
                job_id=job_id
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"GPT product discovery failed: {e}")
            raise RuntimeError(f"GPT product discovery failed: {str(e)}") from e
    
    def _parse_discovery_results(
        self,
        result: Dict[str, Any],
        total_pages: int
    ) -> ProductCatalog:
        """Parse and validate discovery results"""
        
        products = []
        for p in result.get("products", []):
            product = ProductInfo(
                name=p.get("name", "Unknown"),
                page_range=p.get("page_range", []),
                description=p.get("description", ""),
                designer=p.get("designer"),
                studio=p.get("studio"),
                dimensions=p.get("dimensions", []),
                variants=p.get("variants", []),
                category=p.get("category"),
                metafields=p.get("metafields", {}),
                image_indices=p.get("image_pages", []),
                confidence=p.get("confidence", 0.8)
            )
            products.append(product)
        
        # Build metafield categories
        metafield_categories = result.get("metafield_categories", {})
        
        # Build page classification
        page_classification = {}
        for page_str, classification in result.get("page_classification", {}).items():
            page_classification[int(page_str)] = classification
        
        catalog = ProductCatalog(
            products=products,
            total_pages=total_pages,
            total_images=0,  # Will be updated later
            metafield_categories=metafield_categories,
            content_classification=page_classification,
            processing_time_ms=0,  # Will be set by caller
            model_used=self.model,
            confidence_score=result.get("confidence_score", 0.85)
        )
        
        return catalog

