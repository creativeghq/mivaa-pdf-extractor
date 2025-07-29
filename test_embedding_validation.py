#!/usr/bin/env python3
"""
Test script to validate embedding model standardization to text-embedding-3-small.
This script tests the new model configuration and validates 768-dimension output.
"""

import os
import sys
import logging
from typing import Dict, Any

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from services.llamaindex_service import LlamaIndexService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_new_embedding_model() -> Dict[str, Any]:
    """Test the new embedding model configuration."""
    logger.info("Testing new embedding model: text-embedding-3-small")
    
    # Test with default configuration (should use text-embedding-3-small)
    config = {
        'llm_model': 'gpt-3.5-turbo',
        'chunk_size': 512,
        'chunk_overlap': 50
    }
    
    try:
        # Initialize service
        service = LlamaIndexService(config)
        
        if not service.available:
            return {
                'success': False,
                'error': 'LlamaIndex service not available',
                'model_used': None,
                'dimensions': None,
                'embedding_sample': None
            }
        
        # Test text for embedding
        test_text = "This is a test document for embedding generation. It contains multiple sentences to test the embedding quality and dimensions."
        
        # Get embedding directly from the embedding model
        embedding = service.embeddings.get_text_embedding(test_text)
        
        return {
            'success': True,
            'error': None,
            'model_used': service.embedding_model,
            'dimensions': len(embedding),
            'embedding_sample': embedding[:5],  # First 5 dimensions for comparison
            'embedding_norm': sum(x*x for x in embedding)**0.5  # Manual norm calculation
        }
        
    except Exception as e:
        logger.error(f"Error testing embedding model: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'model_used': None,
            'dimensions': None,
            'embedding_sample': None
        }

def main():
    """Main test function."""
    logger.info("Starting embedding model standardization validation...")
    
    # Check if OpenAI API key is available
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY environment variable not set. Cannot test OpenAI embeddings.")
        return
    
    # Test the new model
    result = test_new_embedding_model()
    
    # Print results
    print("\n" + "="*60)
    print("EMBEDDING MODEL STANDARDIZATION VALIDATION")
    print("="*60)
    
    print(f"\n🚀 NEW MODEL CONFIGURATION TEST:")
    print(f"   Success: {result['success']}")
    if result['success']:
        print(f"   Model Used: {result['model_used']}")
        print(f"   Dimensions: {result['dimensions']}")
        print(f"   Sample (first 5): {result['embedding_sample']}")
        print(f"   Vector Norm: {result.get('embedding_norm', 'N/A'):.4f}")
    else:
        print(f"   Error: {result['error']}")
    
    # Final assessment
    print(f"\n✅ STANDARDIZATION ASSESSMENT:")
    if result['success']:
        if result['model_used'] == 'text-embedding-3-small' and result['dimensions'] == 768:
            print("   ✅ SUCCESS: Service correctly uses text-embedding-3-small")
            print("   ✅ SUCCESS: Generates expected 768-dimension embeddings")
            print("   ✅ SUCCESS: Platform standardization achieved")
        elif result['dimensions'] == 768:
            print("   ✅ SUCCESS: Generates 768-dimension embeddings")
            print(f"   ⚠️  INFO: Model used: {result['model_used']}")
        else:
            print(f"   ⚠️  WARNING: Unexpected dimensions: {result['dimensions']} (expected 768)")
            print(f"   ⚠️  WARNING: Model used: {result['model_used']}")
    else:
        print("   ❌ FAILURE: Could not validate embedding generation")
        print(f"   Error: {result['error']}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()