#!/usr/bin/env python3
"""
Fix Missing Image-Product Relationships

This script creates relationships for existing documents that have images and products
but no relationships between them.
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.supabase_client import get_supabase_client
from app.services.entity_linking_service import EntityLinkingService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def fix_document_relationships(document_id: str):
    """Fix relationships for a single document."""
    try:
        supabase = get_supabase_client()
        linking_service = EntityLinkingService()
        
        logger.info(f"🔧 Fixing relationships for document {document_id}")
        
        # Get products for this document
        products_response = supabase.client.table('products')\
            .select('id, name, metadata')\
            .eq('source_document_id', document_id)\
            .execute()
        
        if not products_response.data:
            logger.warning(f"⚠️ No products found for document {document_id}")
            return 0
        
        # Build product_name_to_id mapping
        product_name_to_id = {p['name']: p['id'] for p in products_response.data}
        
        # Link images to products (will use page proximity fallback)
        linked_count = await linking_service.link_images_to_products(
            document_id=document_id,
            image_to_product_mapping={},  # Empty - will use page proximity
            product_name_to_id=product_name_to_id
        )
        
        # Link images to chunks
        chunk_links = await linking_service.link_images_to_chunks(
            document_id=document_id
        )
        
        logger.info(f"✅ Fixed document {document_id}:")
        logger.info(f"   - Image-Product links: {linked_count}")
        logger.info(f"   - Image-Chunk links: {chunk_links}")
        
        return linked_count
        
    except Exception as e:
        logger.error(f"❌ Failed to fix document {document_id}: {e}")
        return 0


async def fix_all_documents():
    """Fix relationships for all documents with missing relationships."""
    try:
        supabase = get_supabase_client()
        
        # Find documents with images but no product-image relationships
        logger.info("🔍 Finding documents with missing relationships...")
        
        # Get all documents with images
        images_response = supabase.client.table('document_images')\
            .select('document_id')\
            .execute()
        
        document_ids = list(set([img['document_id'] for img in images_response.data]))
        logger.info(f"📄 Found {len(document_ids)} documents with images")
        
        # Check which ones have no relationships
        relationships_response = supabase.client.table('product_image_relationships')\
            .select('id')\
            .execute()
        
        total_relationships = len(relationships_response.data)
        logger.info(f"🔗 Current total relationships: {total_relationships}")
        
        # Fix each document
        total_fixed = 0
        for doc_id in document_ids:
            fixed = await fix_document_relationships(doc_id)
            total_fixed += fixed
        
        logger.info(f"\n✅ COMPLETE: Created {total_fixed} new relationships across {len(document_ids)} documents")
        
    except Exception as e:
        logger.error(f"❌ Failed to fix documents: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Fix specific document
        document_id = sys.argv[1]
        asyncio.run(fix_document_relationships(document_id))
    else:
        # Fix all documents
        asyncio.run(fix_all_documents())

