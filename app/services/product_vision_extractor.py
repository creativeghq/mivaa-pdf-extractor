"""
Product Vision Extractor - Enhanced PDF Product Detection

This service uses Llama 4 Scout Vision to extract product information from PDF images.
Llama 4 Scout excels at:
- OCR (# 1 open source model)
- Table extraction (superior to Llama 3.2 90B)
- Diagram understanding (69.4% MMMU benchmark)
- Material property detection

This enhances the existing PDF processing pipeline by adding vision-based product detection.
"""

import logging
import asyncio
import base64
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class ProductVisionResult:
    """Result of vision-based product extraction"""
    product_name: str
    product_code: Optional[str]
    dimensions: Optional[Dict[str, Any]]  # e.g., {"width": "15cm", "height": "38cm"}
    colors: List[str]
    materials: List[str]
    finish: Optional[str]  # matte, glossy, satin, etc.
    pattern: Optional[str]  # solid, striped, geometric, etc.
    designer: Optional[str]
    collection: Optional[str]
    page_number: int
    confidence: float
    raw_analysis: Dict[str, Any]


class ProductVisionExtractor:
    """
    Extract product information from PDF images using Llama 4 Scout Vision.
    
    This service integrates with the existing RealImageAnalysisService to provide
    enhanced product detection during PDF processing.
    """
    
    def __init__(self):
        self.logger = logger
        # Import here to avoid circular dependencies
        from .real_image_analysis_service import RealImageAnalysisService
        self.vision_service = RealImageAnalysisService()
    
    async def extract_products_from_images(
        self,
        extracted_images: List[Dict[str, Any]],
        document_context: Optional[Dict[str, Any]] = None
    ) -> List[ProductVisionResult]:
        """
        Extract product information from a list of extracted PDF images.
        
        Args:
            extracted_images: List of image dictionaries with 'path', 'page_number', etc.
            document_context: Optional context about the document (catalog name, brand, etc.)
            
        Returns:
            List of ProductVisionResult objects
        """
        products = []
        
        for image_info in extracted_images:
            try:
                image_path = image_info.get('path')
                if not image_path:
                    continue
                
                # Read image
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                
                # Analyze with Llama 4 Scout Vision
                product_info = await self._analyze_product_image(
                    image_base64=image_base64,
                    page_number=image_info.get('page_number', 1),
                    context=document_context
                )
                
                if product_info:
                    products.append(product_info)
                    self.logger.info(f"✅ Extracted product: {product_info.product_name} (page {product_info.page_number})")
                
            except Exception as e:
                self.logger.error(f"Failed to extract product from image: {e}")
                continue
        
        return products
    
    async def _analyze_product_image(
        self,
        image_base64: str,
        page_number: int,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[ProductVisionResult]:
        """
        Analyze a single image for product information using Llama 4 Scout Vision.
        
        Uses a specialized prompt optimized for product catalog extraction.
        """
        try:
            import httpx
            import os
            
            TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
            if not TOGETHER_API_KEY:
                self.logger.warning("TOGETHER_API_KEY not set - skipping vision analysis")
                return None
            
            # Build context-aware prompt
            context_str = ""
            if context:
                catalog_name = context.get('catalog_name', '')
                brand = context.get('brand', '')
                if catalog_name:
                    context_str += f"\nCatalog: {catalog_name}"
                if brand:
                    context_str += f"\nBrand: {brand}"
            
            prompt = f"""Analyze this product catalog image and extract ALL product information visible. This is a material/design catalog page.{context_str}

Extract and return in JSON format:
{{
  "products": [
    {{
      "product_name": "<exact product name from image>",
      "product_code": "<product code/SKU if visible>",
      "dimensions": {{"width": "<value>", "height": "<value>", "depth": "<value>"}},
      "colors": ["<color1>", "<color2>"],
      "materials": ["<material1>", "<material2>"],
      "finish": "<matte/glossy/satin/textured/etc>",
      "pattern": "<solid/striped/geometric/etc>",
      "designer": "<designer name if visible>",
      "collection": "<collection name if visible>",
      "description": "<brief description>",
      "confidence": <0.0-1.0>
    }}
  ],
  "page_type": "<product_page/moodboard/collection_overview/technical_specs>",
  "layout_type": "<single_product/multi_product/grid/comparison_table>",
  "text_extracted": ["<any text visible in image>"]
}}

IMPORTANT:
- Extract ALL products visible on the page (there may be multiple)
- Pay special attention to tables, diagrams, and technical specifications
- Extract dimensions even if they're in tables or diagrams
- Look for product codes, SKUs, or reference numbers
- Identify designer/studio names (e.g., "ESTUDI{{H}}AC", "SG NY")
- Note finish types (matte, glossy, polished, textured, etc.)
- Respond ONLY with valid JSON, no additional text."""

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "https://api.together.xyz/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {TOGETHER_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_base64}"
                                        }
                                    }
                                ]
                            }
                        ],
                        "max_tokens": 2048,  # Increased for multiple products
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "stop": ["```"]
                    }
                )
                
                if response.status_code != 200:
                    self.logger.error(f"Llama API error {response.status_code}: {response.text}")
                    return None
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Parse JSON response
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                analysis = json.loads(content)
                
                # Extract first product (or combine if multiple)
                products_data = analysis.get("products", [])
                if not products_data:
                    return None
                
                # For now, return the first product
                # TODO: Handle multiple products per page
                product_data = products_data[0]
                
                return ProductVisionResult(
                    product_name=product_data.get("product_name", "Unknown"),
                    product_code=product_data.get("product_code"),
                    dimensions=product_data.get("dimensions"),
                    colors=product_data.get("colors", []),
                    materials=product_data.get("materials", []),
                    finish=product_data.get("finish"),
                    pattern=product_data.get("pattern"),
                    designer=product_data.get("designer"),
                    collection=product_data.get("collection"),
                    page_number=page_number,
                    confidence=product_data.get("confidence", 0.0),
                    raw_analysis=analysis
                )
                
        except Exception as e:
            self.logger.error(f"Product vision analysis failed: {e}")
            return None
    
    async def enrich_existing_products(
        self,
        products: List[Dict[str, Any]],
        document_id: str
    ) -> List[Dict[str, Any]]:
        """
        Enrich existing products with vision-based analysis.
        
        This can be called after initial product extraction to add
        visual analysis data to products that were detected through
        text-based methods.
        """
        enriched_products = []
        
        for product in products:
            try:
                # Get associated images for this product
                # This would query the database for images linked to this product
                # For now, we'll skip this and just return the original product
                enriched_products.append(product)
                
            except Exception as e:
                self.logger.error(f"Failed to enrich product: {e}")
                enriched_products.append(product)
        
        return enriched_products

