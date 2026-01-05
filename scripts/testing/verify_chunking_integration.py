#!/usr/bin/env python3
"""
Verify Chunking Service Integration

Tests that the integrated chunking services work correctly:
1. MetadataFirstChunkingService - Excludes metadata pages
2. ChunkContextEnrichmentService - Adds product context
3. ChunkTypeClassificationService - Classifies chunk types
4. UnifiedChunkingService - Uses single strategy selection method
"""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.chunking.metadata_first_chunking_service import MetadataFirstChunkingService
from app.services.chunking.chunk_context_enrichment_service import ChunkContextEnrichmentService
from app.services.chunking.chunk_type_classification_service import ChunkTypeClassificationService
from app.services.chunking.unified_chunking_service import UnifiedChunkingService, ChunkingConfig, ChunkingStrategy


async def test_metadata_first_service():
    """Test MetadataFirstChunkingService"""
    print("\n" + "="*80)
    print("TEST 1: MetadataFirstChunkingService")
    print("="*80)
    
    service = MetadataFirstChunkingService(enabled=True)
    
    # Mock product with page range
    products = [{
        'id': 'product_1',
        'name': 'Test Product',
        'page_range': [1, 2, 3]
    }]
    
    excluded_pages = await service.get_pages_to_exclude(
        products=products,
        document_id='test_doc'
    )
    
    print(f"✅ Excluded pages: {excluded_pages}")
    assert excluded_pages == {1, 2, 3}, "Should exclude pages 1, 2, 3"
    print("✅ MetadataFirstChunkingService works correctly!")


async def test_chunk_context_enrichment():
    """Test ChunkContextEnrichmentService"""
    print("\n" + "="*80)
    print("TEST 2: ChunkContextEnrichmentService")
    print("="*80)
    
    service = ChunkContextEnrichmentService(enabled=True)
    
    # Mock chunks
    chunks = [
        {
            'content': 'Test chunk 1',
            'metadata': {'page_number': 1}
        },
        {
            'content': 'Test chunk 2',
            'metadata': {'page_number': 2}
        }
    ]
    
    # Mock product
    products = [{
        'id': 'product_1',
        'name': 'Test Product',
        'page_range': [1, 2, 3]
    }]
    
    enriched_chunks = await service.enrich_chunks(
        chunks=chunks,
        products=products,
        document_id='test_doc'
    )
    
    print(f"✅ Enriched {len(enriched_chunks)} chunks")
    assert enriched_chunks[0]['metadata']['product_id'] == 'product_1'
    assert enriched_chunks[0]['metadata']['product_name'] == 'Test Product'
    print("✅ ChunkContextEnrichmentService works correctly!")


async def test_chunk_type_classification():
    """Test ChunkTypeClassificationService"""
    print("\n" + "="*80)
    print("TEST 3: ChunkTypeClassificationService")
    print("="*80)
    
    service = ChunkTypeClassificationService()
    
    # Test product description
    product_text = "AURORA LAMP\n20×40 cm\nWood and metal construction"
    classification = await service.classify_chunk(product_text)
    
    print(f"✅ Classified as: {classification.chunk_type.value}")
    print(f"   Confidence: {classification.confidence}")
    print(f"   Reasoning: {classification.reasoning}")
    assert classification.chunk_type.value in ['product_description', 'technical_specs']
    print("✅ ChunkTypeClassificationService works correctly!")


async def test_unified_chunking_service():
    """Test UnifiedChunkingService strategy selection"""
    print("\n" + "="*80)
    print("TEST 4: UnifiedChunkingService")
    print("="*80)
    
    config = ChunkingConfig(
        strategy=ChunkingStrategy.HYBRID,
        max_chunk_size=500,
        overlap_size=50
    )
    service = UnifiedChunkingService(config)
    
    # Test chunk_pages (preferred method)
    pages = [
        {'metadata': {'page': 0}, 'text': 'This is a test page with some content. ' * 20},
        {'metadata': {'page': 1}, 'text': 'This is another test page with more content. ' * 20}
    ]
    
    chunks = await service.chunk_pages(
        pages=pages,
        document_id='test_doc',
        metadata={'test': True}
    )
    
    print(f"✅ Created {len(chunks)} chunks using chunk_pages()")
    assert len(chunks) > 0, "Should create at least one chunk"
    assert all(hasattr(chunk, 'metadata') for chunk in chunks), "All chunks should have metadata"
    print("✅ UnifiedChunkingService works correctly!")


async def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("CHUNKING SERVICE INTEGRATION VERIFICATION")
    print("="*80)
    
    try:
        await test_metadata_first_service()
        await test_chunk_context_enrichment()
        await test_chunk_type_classification()
        await test_unified_chunking_service()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nIntegration Summary:")
        print("  ✅ MetadataFirstChunkingService - Working")
        print("  ✅ ChunkContextEnrichmentService - Working")
        print("  ✅ ChunkTypeClassificationService - Working")
        print("  ✅ UnifiedChunkingService - Working")
        print("\nThe chunking service integration is complete and functional!")
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ TEST FAILED!")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())

