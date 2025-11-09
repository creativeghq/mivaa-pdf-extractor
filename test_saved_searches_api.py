#!/usr/bin/env python3
"""
Test script for Saved Searches API endpoints.
Tests the API without requiring the full service to be running.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.search_deduplication_service import SearchDeduplicationService
from app.config import get_settings

async def test_deduplication_service():
    """Test the deduplication service."""
    print("=" * 80)
    print("Testing Search Deduplication Service")
    print("=" * 80)
    
    # Initialize settings
    settings = get_settings()
    print(f"\n✓ Settings loaded")
    print(f"  - Supabase URL: {settings.supabase_url}")
    print(f"  - Anthropic API Key: {'✓ Set' if settings.ANTHROPIC_API_KEY else '✗ Missing'}")
    print(f"  - OpenAI API Key: {'✓ Set' if settings.OPENAI_API_KEY else '✗ Missing'}")
    
    # Initialize service
    try:
        service = SearchDeduplicationService()
        print(f"\n✓ Deduplication service initialized")
    except Exception as e:
        print(f"\n✗ Failed to initialize service: {e}")
        return False
    
    # Test 1: Analyze a search query
    print("\n" + "-" * 80)
    print("Test 1: Analyze Search Query")
    print("-" * 80)
    
    test_query = "I need a cement tile for my kitchen floor"
    print(f"Query: '{test_query}'")
    
    try:
        analysis = await service.analyze_search_query(test_query)
        print(f"\n✓ Analysis complete:")
        print(f"  - Core Material: {analysis.core_material}")
        print(f"  - Application Context: {analysis.application_context}")
        print(f"  - Intent Category: {analysis.intent_category}")
        print(f"  - Normalized Query: {analysis.normalized_query}")
        print(f"  - Attributes: {analysis.attributes}")
        print(f"  - Semantic Fingerprint: {len(analysis.semantic_fingerprint)} dimensions")
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Find or merge search
    print("\n" + "-" * 80)
    print("Test 2: Find or Merge Search")
    print("-" * 80)
    
    test_user_id = "test-user-123"
    test_filters = {"priceRange": [0, 100]}
    test_material_filters = {"materialTypes": ["cement_tile"]}
    
    print(f"User ID: {test_user_id}")
    print(f"Query: '{test_query}'")
    print(f"Filters: {test_filters}")
    print(f"Material Filters: {test_material_filters}")
    
    try:
        existing_id, should_merge, suggestion = await service.find_or_merge_search(
            user_id=test_user_id,
            query=test_query,
            filters=test_filters,
            material_filters=test_material_filters
        )
        
        print(f"\n✓ Deduplication check complete:")
        print(f"  - Existing Search ID: {existing_id or 'None'}")
        print(f"  - Should Merge: {should_merge}")
        if suggestion:
            print(f"  - Merge Suggestion:")
            print(f"    - Similarity Score: {suggestion.get('similarity_score', 'N/A')}")
            print(f"    - Reason: {suggestion.get('reason', 'N/A')}")
    except Exception as e:
        print(f"\n✗ Deduplication check failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
    return True

if __name__ == "__main__":
    result = asyncio.run(test_deduplication_service())
    sys.exit(0 if result else 1)

