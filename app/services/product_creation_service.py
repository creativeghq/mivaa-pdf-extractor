"""
Product Creation Service

This service automatically creates products from processed PDF chunks.
It analyzes chunks to identify product-like content and creates product records
with proper metadata, embeddings, and relationships.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ProductCreationService:
    """Service for creating products from PDF chunks"""

    def __init__(self, supabase_client):
        """
        Initialize product creation service.
        
        Args:
            supabase_client: Supabase client instance for database operations
        """
        self.supabase = supabase_client
        self.logger = logging.getLogger(__name__)

    async def create_products_from_chunks(
        self,
        document_id: str,
        workspace_id: str = "ffafc28b-1b8b-4b0d-b226-9f9a6154004e",
        max_products: Optional[int] = None,
        min_chunk_length: int = 100
    ) -> Dict[str, Any]:
        """
        Create products from document chunks.
        
        Args:
            document_id: UUID of the processed document
            workspace_id: UUID of the workspace
            max_products: Maximum number of products to create (None = unlimited)
            min_chunk_length: Minimum chunk content length to consider for products
            
        Returns:
            Dictionary with creation statistics
        """
        try:
            self.logger.info(f"🏭 Starting product creation for document {document_id}")
            
            # Fetch all chunks for this document
            chunks_response = self.supabase.client.table('document_chunks').select('*').eq('document_id', document_id).order('chunk_index').execute()
            
            if not chunks_response.data:
                self.logger.warning(f"No chunks found for document {document_id}")
                return {
                    "success": True,
                    "products_created": 0,
                    "chunks_processed": 0,
                    "message": "No chunks found to create products from"
                }
            
            chunks = chunks_response.data
            self.logger.info(f"📦 Found {len(chunks)} chunks to process")
            
            # Filter chunks by minimum length
            eligible_chunks = [
                chunk for chunk in chunks 
                if len(chunk.get('content', '')) >= min_chunk_length
            ]
            
            self.logger.info(f"✅ {len(eligible_chunks)} chunks meet minimum length requirement")
            
            # Limit number of products if specified
            if max_products:
                eligible_chunks = eligible_chunks[:max_products]
                self.logger.info(f"📊 Limited to {max_products} products")
            
            products_created = 0
            products_failed = 0
            
            for i, chunk in enumerate(eligible_chunks):
                try:
                    # Create product from chunk
                    product_data = self._create_product_from_chunk(
                        chunk=chunk,
                        document_id=document_id,
                        workspace_id=workspace_id,
                        index=i
                    )
                    
                    # Insert product into database
                    product_response = self.supabase.client.table('products').insert(product_data).execute()
                    
                    if product_response.data:
                        products_created += 1
                        product_id = product_response.data[0]['id']
                        self.logger.info(f"✅ Created product {i+1}/{len(eligible_chunks)}: {product_id}")
                    else:
                        products_failed += 1
                        self.logger.warning(f"⚠️ Failed to create product from chunk {chunk['id']}")
                        
                except Exception as e:
                    products_failed += 1
                    self.logger.error(f"❌ Error creating product from chunk {chunk.get('id')}: {e}")
                    continue
            
            result = {
                "success": True,
                "products_created": products_created,
                "products_failed": products_failed,
                "chunks_processed": len(eligible_chunks),
                "total_chunks": len(chunks),
                "message": f"Created {products_created} products from {len(eligible_chunks)} chunks"
            }
            
            self.logger.info(f"🎉 Product creation complete: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Product creation failed: {e}", exc_info=True)
            return {
                "success": False,
                "products_created": 0,
                "error": str(e),
                "message": f"Product creation failed: {str(e)}"
            }

    def _create_product_from_chunk(
        self,
        chunk: Dict[str, Any],
        document_id: str,
        workspace_id: str,
        index: int
    ) -> Dict[str, Any]:
        """
        Create a product record from a chunk.
        
        Args:
            chunk: Chunk data from database
            document_id: UUID of source document
            workspace_id: UUID of workspace
            index: Index of chunk in processing order
            
        Returns:
            Product data dictionary ready for insertion
        """
        content = chunk.get('content', '')
        chunk_id = chunk.get('id')
        chunk_index = chunk.get('chunk_index', index)
        page_number = chunk.get('page_number')
        
        # Generate product name from content
        # Take first line or first 50 characters as name
        lines = content.split('\n')
        first_line = lines[0].strip() if lines else content[:50]
        product_name = f"Product from Chunk {chunk_index} - {int(datetime.now().timestamp() * 1000)}"
        
        # Use first 200 characters as description
        description = content[:200].strip()
        if len(content) > 200:
            description += "..."
        
        # Use full content as long description (up to 1000 chars)
        long_description = content[:1000].strip()
        if len(content) > 1000:
            long_description += "..."
        
        # Build product data
        product_data = {
            "name": product_name,
            "description": description,
            "long_description": long_description,
            "source_document_id": document_id,
            "source_chunks": [chunk_id],  # Array of chunk IDs
            "properties": {
                "source_chunk_id": chunk_id,
                "document_id": document_id,
                "chunk_index": chunk_index,
                "page_number": page_number,
                "content_length": len(content),
                "auto_generated": True,
                "generation_timestamp": datetime.utcnow().isoformat()
            },
            "metadata": {
                "extracted_from": "pdf_chunk",
                "chunk_metadata": chunk.get('metadata', {}),
                "extraction_date": datetime.utcnow().isoformat(),
                "auto_created": True,
                "workspace_id": workspace_id
            },
            "status": "draft",
            "created_from_type": "pdf_processing",
            # Note: embedding will be generated separately if needed
        }
        
        return product_data

    async def get_product_creation_stats(self, document_id: str) -> Dict[str, Any]:
        """
        Get statistics about products created from a document.
        
        Args:
            document_id: UUID of the document
            
        Returns:
            Statistics dictionary
        """
        try:
            # Count products created from this document
            products_response = self.supabase.client.table('products').select('id, name, status, created_at').eq('source_document_id', document_id).execute()
            
            products = products_response.data or []
            
            return {
                "success": True,
                "document_id": document_id,
                "total_products": len(products),
                "products_by_status": self._count_by_status(products),
                "products": products
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get product stats: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _count_by_status(self, products: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count products by status"""
        counts = {}
        for product in products:
            status = product.get('status', 'unknown')
            counts[status] = counts.get(status, 0) + 1
        return counts

