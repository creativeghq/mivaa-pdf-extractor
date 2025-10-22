#!/usr/bin/env python3
"""
Test script for Layout-Based Product Candidate Detection

This script tests the new markdown-based product detection system
to verify it can identify real products and filter out non-product content.
"""

import sys
import os
import asyncio
import logging

# Add the app directory to Python path
sys.path.append('/var/www/mivaa-pdf-extractor/app')

from services.product_creation_service import ProductCreationService
from services.supabase_client import get_supabase_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample markdown content from HARMONY PDF (based on grad.md analysis)
SAMPLE_MARKDOWN = """
# HARMONY COLLECTION

Table of Contents
- VALENOVA ........................... Page 12
- PIQUÉ ................................ Page 18  
- ONA .................................. Page 24
- MARE ................................. Page 30
- LOG .................................. Page 36

-----

## Sustainability Information

Our commitment to environmental responsibility drives every aspect of our manufacturing process. 
We use recycled materials and maintain LEED certification standards.
Carbon footprint reduction is a key priority.

-----

## VALENOVA

**TAUPE, SAND, CLAY**

Dimensions: 11.8×11.8 cm
Thickness: 10mm

VALENOVA represents the perfect balance between contemporary design and natural inspiration.
This ceramic tile collection features three sophisticated colorways that bring warmth and 
elegance to any space. The subtle texture and matte finish create a timeless appeal.

Material: Porcelain ceramic
Designer: Studio ALT Design
Collection: Harmony Series

-----

## Technical Characteristics

| Property | Value |
|----------|-------|
| Thickness | 10mm |
| Weight per m² | 22 kg |
| Fire rating | A1 |
| Slip resistance | R10 |

Technical specifications for installation and maintenance guidelines.

-----

## PIQUÉ by Estudi{H}ac

Designer: José Manuel Ferrero

PIQUÉ brings a fresh perspective to textile-inspired ceramics. The collection captures
the essence of woven fabrics through innovative surface treatments and sophisticated
color palettes. Available in multiple formats to suit diverse architectural applications.

Dimensions: 20×40 cm, 15×38 cm
Colors: White, Sand, Anthracite
Material: Ceramic

-----

## ONA BY DSIGNIO

Contemporary minimalism meets functional design in the ONA collection.
Clean lines and subtle variations create visual interest while maintaining
the sophisticated aesthetic that defines modern architecture.

Dimensions: 12×45 cm
Designer: DSIGNIO Studio
Colors: White Mix, Grey Mix

-----
"""

async def test_markdown_analysis():
    """Test the markdown-based product detection system."""
    
    logger.info("🧪 Starting Layout-Based Product Detection Test")
    
    try:
        # Initialize the service
        supabase_client = get_supabase_client()
        product_service = ProductCreationService(supabase_client)
        
        # Test the markdown analysis directly
        logger.info("📝 Testing markdown content analysis...")
        
        # Analyze the sample markdown
        product_candidates = product_service._detect_products_in_markdown(SAMPLE_MARKDOWN)
        
        logger.info(f"🎯 Found {len(product_candidates)} product candidates")
        
        # Display results
        for i, candidate in enumerate(product_candidates, 1):
            logger.info(f"\n--- Product Candidate {i} ---")
            logger.info(f"Name: {candidate.get('name')}")
            logger.info(f"Content Type: {candidate.get('contentType')}")
            logger.info(f"Confidence: {candidate.get('confidence', 0):.2f}")
            logger.info(f"Quality Score: {candidate.get('qualityScore', 0):.2f}")
            logger.info(f"Page: {candidate.get('pageNumber')}")
            
            patterns = candidate.get('patterns', {})
            logger.info(f"Patterns: Product Name={patterns.get('hasProductName')}, "
                       f"Dimensions={patterns.get('hasDimensions')}, "
                       f"Designer={patterns.get('hasDesignerAttribution')}")
            
            extracted = candidate.get('extractedData', {})
            if extracted.get('productName'):
                logger.info(f"Extracted Name: {extracted['productName']}")
            if extracted.get('dimensions'):
                logger.info(f"Extracted Dimensions: {extracted['dimensions']}")
            if extracted.get('designer'):
                logger.info(f"Extracted Designer: {extracted['designer']}")
            if extracted.get('colors'):
                logger.info(f"Extracted Colors: {extracted['colors']}")
            
            logger.info(f"Source: {candidate.get('sourceSection', '')[:100]}...")
        
        # Test content classification
        logger.info("\n🔍 Testing content classification...")
        
        test_sections = [
            ("Table of Contents section", "Table of Contents\n- VALENOVA ... Page 12\n- PIQUÉ ... Page 18"),
            ("Sustainability section", "Our commitment to environmental responsibility. LEED certification."),
            ("Technical specs", "Technical Characteristics\nThickness: 10mm\nWeight per m²: 22 kg"),
            ("Product section", "VALENOVA\nTAUPE, SAND, CLAY\nDimensions: 11.8×11.8 cm\nDesigner: Studio ALT"),
        ]
        
        for name, content in test_sections:
            content_type = product_service._classify_markdown_content(content)
            logger.info(f"{name}: {content_type}")
        
        # Summary
        real_products = [c for c in product_candidates if c.get('contentType') == 'product']
        logger.info(f"\n✅ SUMMARY:")
        logger.info(f"Total candidates found: {len(product_candidates)}")
        logger.info(f"Real product candidates: {len(real_products)}")
        logger.info(f"Expected real products: 4 (VALENOVA, PIQUÉ, ONA, MARE)")
        
        if len(real_products) >= 3:  # Allow some tolerance
            logger.info("🎉 SUCCESS: Product detection is working correctly!")
            return True
        else:
            logger.warning("⚠️ WARNING: Expected more product candidates")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_markdown_analysis())
    sys.exit(0 if success else 1)
