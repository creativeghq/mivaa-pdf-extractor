#!/usr/bin/env python3
"""
Fix Missing Image-Product Relationships

This script creates relationships for existing documents that have images and products
but no relationships between them.
"""

import asyncio
import sys
import os
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up minimal environment
os.environ.setdefault('ENVIRONMENT', 'production')

from supabase import create_client
from datetime import datetime
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Supabase client
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY')

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


async def link_images_to_products_simple(document_id: str, product_name_to_id: dict):
    """Simplified version of entity linking using page proximity."""
    try:
        logger.info(f"🔗 Linking images to products for document {document_id}")

        # Get all images for this document
        images_response = supabase.table('document_images')\
            .select('id, page_number, metadata')\
            .eq('document_id', document_id)\
            .execute()

        if not images_response.data:
            logger.warning(f"⚠️ No images found for document {document_id}")
            return 0

        # Get all products with page ranges
        products_response = supabase.table('products')\
            .select('id, name, metadata')\
            .eq('source_document_id', document_id)\
            .execute()

        product_page_ranges = {}
        for product in products_response.data:
            metadata = product.get('metadata', {})
            page_range = metadata.get('page_range', [])
            product_page_ranges[product['id']] = page_range

        linked_count = 0
        relationships = []

        for image in images_response.data:
            image_id = image['id']
            page_number = image['page_number']

            # Find product whose page_range contains this image's page
            product_id = None
            product_name = None

            for pid, page_range in product_page_ranges.items():
                if page_range and len(page_range) >= 2:
                    start_page, end_page = page_range[0], page_range[1]
                    if start_page <= page_number <= end_page:
                        product_id = pid
                        # Find product name
                        for pname, pid_check in product_name_to_id.items():
                            if pid_check == pid:
                                product_name = pname
                                break
                        break

            if product_id:
                # Calculate simple relevance score (0.8 for same page)
                relevance_score = 0.8

                # Create relationship entry
                relationships.append({
                    'id': str(uuid.uuid4()),
                    'image_id': image_id,
                    'product_id': product_id,
                    'relevance_score': relevance_score,
                    'relationship_type': 'depicts',
                    'created_at': datetime.utcnow().isoformat()
                })

                linked_count += 1
                logger.debug(f"✅ Linked image {image_id} to product {product_name} (page {page_number})")

        # Batch insert all relationships
        if relationships:
            supabase.table('product_image_relationships')\
                .insert(relationships)\
                .execute()
            logger.info(f"✅ Created {len(relationships)} product-image relationships")

        logger.info(f"✅ Linked {linked_count}/{len(images_response.data)} images to products")
        return linked_count

    except Exception as e:
        logger.error(f"❌ Failed to link images: {e}")
        return 0


async def fix_document_relationships(document_id: str):
    """Fix relationships for a single document."""
    try:
        logger.info(f"🔧 Fixing relationships for document {document_id}")

        # Get products for this document
        products_response = supabase.table('products')\
            .select('id, name, metadata')\
            .eq('source_document_id', document_id)\
            .execute()

        if not products_response.data:
            logger.warning(f"⚠️ No products found for document {document_id}")
            return 0

        # Build product_name_to_id mapping
        product_name_to_id = {p['name']: p['id'] for p in products_response.data}

        # Link images to products using simplified version
        linked_count = await link_images_to_products_simple(
            document_id=document_id,
            product_name_to_id=product_name_to_id
        )

        logger.info(f"✅ Fixed document {document_id}: {linked_count} image-product links")
        return linked_count

    except Exception as e:
        logger.error(f"❌ Failed to fix document {document_id}: {e}")
        return 0


async def fix_all_documents():
    """Fix relationships for all documents with missing relationships."""
    try:
        # Find documents with images but no product-image relationships
        logger.info("🔍 Finding documents with missing relationships...")

        # Get all documents with images
        images_response = supabase.table('document_images')\
            .select('document_id')\
            .execute()

        document_ids = list(set([img['document_id'] for img in images_response.data]))
        logger.info(f"📄 Found {len(document_ids)} documents with images")

        # Check which ones have no relationships
        relationships_response = supabase.table('product_image_relationships')\
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

