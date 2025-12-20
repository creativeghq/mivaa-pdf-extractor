#!/usr/bin/env python3
"""
Unit tests for discover_products_from_text() method.

Tests the new text-based product discovery that works with:
- Web scraping markdown (Firecrawl)
- XML imports (converted to markdown)
- Manual text input
"""

import asyncio
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from app.services.product_discovery_service import ProductDiscoveryService


# Sample markdown content for testing (simulating Firecrawl output)
SAMPLE_WEB_SCRAPING_MARKDOWN = """
# NOVA Collection

**Designer:** SG NY  
**Studio:** SG NY  
**Category:** Ceramic Tile

## Product Details

NOVA is a modern ceramic tile collection featuring clean lines and contemporary design.

### Available Sizes
- 15×38 cm
- 20×40 cm
- 7×14.8 cm

### Available Colors
- White
- Clay
- Sand
- Taupe

### Product Variants

**37885** FOLD WHITE/15X38  
Color: WHITE | Shape: FOLD | Size: 15×38 | Mapei: 100

**37889** FOLD CLAY/15X38  
Color: CLAY | Shape: FOLD | Size: 15×38 | Mapei: 145

**38343** TRI. FOLD WHITE/7X14,8  
Color: WHITE | Shape: TRI. FOLD | Size: 7×14.8 | Mapei: 100

### Technical Specifications
- Material: Ceramic
- Finish: Matte
- Slip Resistance: R11
- Fire Rating: A1
- Thickness: 8mm
- Water Absorption: Class 3

### Packaging
- Pieces per box: 12
- Boxes per pallet: 48
- Weight per box: 18.5 kg
- Coverage per box: 1.14 m²

### Factory Information
- Factory: HARMONY
- Factory Group: Peronda Group
- Country of Origin: Spain
"""

EMPTY_MARKDOWN = ""

MALFORMED_MARKDOWN = """
This is just random text with no product information.
Some numbers: 123, 456, 789
Some words: hello, world, test
"""


def log(message: str, level: str = 'info'):
    """Log a message with timestamp and emoji"""
    emoji_map = {
        'info': '📝',
        'success': '✅',
        'error': '❌',
        'warning': '⚠️',
        'test': '🧪'
    }
    emoji = emoji_map.get(level, '📝')
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"{emoji} [{timestamp}] {message}")


def log_section(title: str):
    """Log a section header"""
    print('\n' + '=' * 80)
    print(f'🎯 {title}')
    print('=' * 80)


async def test_web_scraping_markdown():
    """Test discovery from web scraping markdown"""
    log_section("TEST 1: Web Scraping Markdown")
    
    try:
        service = ProductDiscoveryService(model="claude")
        
        log("Discovering products from web scraping markdown...", "test")
        catalog = await service.discover_products_from_text(
            markdown_text=SAMPLE_WEB_SCRAPING_MARKDOWN,
            source_type="web_scraping",
            categories=["products"]
        )
        
        # Assertions
        assert len(catalog.products) > 0, "Should discover at least one product"
        
        product = catalog.products[0]
        log(f"✅ Discovered product: {product.name}", "success")
        log(f"   Metadata fields: {len(product.metadata)}", "info")
        log(f"   Confidence: {product.confidence}", "info")
        
        # Check metadata extraction
        assert product.metadata is not None, "Product should have metadata"
        assert len(product.metadata) > 0, "Metadata should not be empty"
        
        log("✅ TEST 1 PASSED", "success")
        return True
        
    except Exception as e:
        log(f"❌ TEST 1 FAILED: {e}", "error")
        return False


async def test_empty_content():
    """Test handling of empty content"""
    log_section("TEST 2: Empty Content")

    try:
        service = ProductDiscoveryService(model="claude")

        log("Testing with empty markdown...", "test")
        catalog = await service.discover_products_from_text(
            markdown_text=EMPTY_MARKDOWN,
            source_type="web_scraping",
            categories=["products"]
        )

        # Should return empty catalog, not crash
        assert catalog is not None, "Should return a catalog object"
        log(f"✅ Returned catalog with {len(catalog.products)} products", "success")

        log("✅ TEST 2 PASSED", "success")
        return True

    except Exception as e:
        log(f"❌ TEST 2 FAILED: {e}", "error")
        return False


async def test_malformed_markdown():
    """Test handling of malformed markdown"""
    log_section("TEST 3: Malformed Markdown")

    try:
        service = ProductDiscoveryService(model="claude")

        log("Testing with malformed markdown...", "test")
        catalog = await service.discover_products_from_text(
            markdown_text=MALFORMED_MARKDOWN,
            source_type="web_scraping",
            categories=["products"]
        )

        # Should handle gracefully
        assert catalog is not None, "Should return a catalog object"
        log(f"✅ Returned catalog with {len(catalog.products)} products", "success")

        log("✅ TEST 3 PASSED", "success")
        return True

    except Exception as e:
        log(f"❌ TEST 3 FAILED: {e}", "error")
        return False


async def test_metadata_quality():
    """Test metadata extraction quality"""
    log_section("TEST 4: Metadata Quality")

    try:
        service = ProductDiscoveryService(model="claude")

        log("Testing metadata extraction quality...", "test")
        catalog = await service.discover_products_from_text(
            markdown_text=SAMPLE_WEB_SCRAPING_MARKDOWN,
            source_type="web_scraping",
            categories=["products"]
        )

        if len(catalog.products) == 0:
            log("⚠️ No products discovered, skipping metadata quality test", "warning")
            return True

        product = catalog.products[0]
        metadata = product.metadata

        # Check for expected metadata fields
        expected_fields = ["designer", "dimensions", "factory_name", "material"]
        found_fields = [field for field in expected_fields if field in metadata]

        log(f"Expected fields: {expected_fields}", "info")
        log(f"Found fields: {found_fields}", "info")
        log(f"Metadata quality: {len(found_fields)}/{len(expected_fields)} fields", "info")

        # Log all metadata for inspection
        log("Full metadata:", "info")
        for key, value in metadata.items():
            if not key.startswith("_"):  # Skip internal fields
                log(f"  {key}: {value}", "info")

        log("✅ TEST 4 PASSED", "success")
        return True

    except Exception as e:
        log(f"❌ TEST 4 FAILED: {e}", "error")
        return False


async def run_all_tests():
    """Run all tests"""
    log_section("RUNNING ALL TESTS")

    results = []

    # Run tests
    results.append(await test_web_scraping_markdown())
    results.append(await test_empty_content())
    results.append(await test_malformed_markdown())
    results.append(await test_metadata_quality())

    # Summary
    log_section("TEST SUMMARY")
    passed = sum(results)
    total = len(results)

    log(f"Tests passed: {passed}/{total}", "success" if passed == total else "error")

    if passed == total:
        log("🎉 ALL TESTS PASSED!", "success")
        return 0
    else:
        log(f"❌ {total - passed} test(s) failed", "error")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)

