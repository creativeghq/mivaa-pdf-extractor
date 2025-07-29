#!/usr/bin/env python3
"""
Test script to validate embedding model migration from text-embedding-ada-002 to text-embedding-3-small.
This script tests both models and compares their outputs to ensure the migration is successful.
"""

import os
import sys
import logging
from typing import Dict, List, Any
import numpy as np

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from services.llamaindex_service import LlamaIndexService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_embedding_model(model_name: str) -> Dict[str, Any]:
    """Test embedding generation with a specific model."""
    logger.info(f"Testing embedding model: {model_name}")
    
    # Test configuration
    config = {
        'embedding_model': model_name,
        'llm_model': 'gpt-3.5-turbo',
        'chunk_size': 512,
        'chunk_overlap': 50
    }
    
    try:
        # Initialize service
        service = LlamaIndexService(config)
        
        if not service.available:
            return {
                'model': model_name,
                'success': False,
                'error': 'LlamaIndex service not available',
                'dimensions': None,
                'embedding_sample': None
            }
        
        # Test text for embedding
        test_text = "This is a test document for embedding generation. It contains multiple sentences to test the embedding quality and dimensions."
        
        # Get embedding directly from the embedding model
        embedding = service.embeddings.get_text_embedding(test_text)
        
        return {
            'model': model_name,
            'success': True,
            'error': None,
            'dimensions': len(embedding),
            'embedding_sample': embedding[:5],  # First 5 dimensions for comparison
            'embedding_norm': np.linalg.norm(embedding)
        }
        
    except Exception as e:
        logger.error(f"Error testing {model_name}: {str(e)}")
        return {
            'model': model_name,
            'success': False,
            'error': str(e),
            'dimensions': None,
            'embedding_sample': None
        }

def compare_embeddings(old_result: Dict, new_result: Dict) -> Dict[str, Any]:
    """Compare embeddings from old and new models."""
    comparison = {
        'dimension_change': None,
        'dimension_reduction_ratio': None,
        'both_successful': False,
        'compatibility_check': 'UNKNOWN'
    }
    
    if old_result['success'] and new_result['success']:
        comparison['both_successful'] = True
        comparison['dimension_change'] = new_result['dimensions'] - old_result['dimensions']
        comparison['dimension_reduction_ratio'] = new_result['dimensions'] / old_result['dimensions']
        
        # Check if we achieved the expected dimension reduction (1536 -> 768)
        if old_result['dimensions'] == 1536 and new_result['dimensions'] == 768:
            comparison['compatibility_check'] = 'SUCCESS'
        elif new_result['dimensions'] == 768:
            comparison['compatibility_check'] = 'PARTIAL_SUCCESS'
        else:
            comparison['compatibility_check'] = 'UNEXPECTED_DIMENSIONS'
    
    return comparison

def main():
    """Main test function."""
    logger.info("Starting embedding model migration test...")
    
    # Check if OpenAI API key is available
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY environment variable not set. Cannot test OpenAI embeddings.")
        return
    
    # Test both models
    old_model_result = test_embedding_model('text-embedding-ada-002')
    new_model_result = test_embedding_model('text-embedding-3-small')
    
    # Print results
    print("\n" + "="*60)
    print("EMBEDDING MODEL MIGRATION TEST RESULTS")
    print("="*60)
    
    print(f"\n📊 OLD MODEL (text-embedding-ada-002):")
    print(f"   Success: {old_model_result['success']}")
    if old_model_result['success']:
        print(f"   Dimensions: {old_model_result['dimensions']}")
        print(f"   Sample: {old_model_result['embedding_sample']}")
        print(f"   Norm: {old_model_result.get('embedding_norm', 'N/A'):.4f}")
    else:
        print(f"   Error: {old_model_result['error']}")
    
    print(f"\n🚀 NEW MODEL (text-embedding-3-small):")
    print(f"   Success: {new_model_result['success']}")
    if new_model_result['success']:
        print(f"   Dimensions: {new_model_result['dimensions']}")
        print(f"   Sample: {new_model_result['embedding_sample']}")
        print(f"   Norm: {new_model_result.get('embedding_norm', 'N/A'):.4f}")
    else:
        print(f"   Error: {new_model_result['error']}")
    
    # Comparison
    comparison = compare_embeddings(old_model_result, new_model_result)
    print(f"\n🔄 COMPARISON:")
    print(f"   Both models successful: {comparison['both_successful']}")
    if comparison['both_successful']:
        print(f"   Dimension change: {comparison['dimension_change']}")
        print(f"   Reduction ratio: {comparison['dimension_reduction_ratio']:.2f}")
        print(f"   Compatibility: {comparison['compatibility_check']}")
    
    # Final assessment
    print(f"\n✅ MIGRATION ASSESSMENT:")
    if new_model_result['success'] and new_model_result['dimensions'] == 768:
        print("   ✅ SUCCESS: New model generates 768-dimension embeddings as expected")
        print("   ✅ Platform standardization achieved")
        if comparison['compatibility_check'] == 'SUCCESS':
            print("   ✅ Dimension reduction from 1536→768 confirmed")
    elif new_model_result['success']:
        print(f"   ⚠️  WARNING: New model generates {new_model_result['dimensions']} dimensions (expected 768)")
    else:
        print("   ❌ FAILURE: New model failed to generate embeddings")
        print(f"   Error: {new_model_result['error']}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()