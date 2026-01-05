"""
Stage 2: Text Chunking

This module handles text chunking for individual products in the product-centric pipeline.
"""

import logging
from typing import Dict, Any, Set


async def process_product_chunking(
    file_content: bytes,
    document_id: str,
    workspace_id: str,
    job_id: str,
    product: Any,
    product_pages: Set[int],
    catalog: Any,
    pdf_result: Any,
    config: Dict[str, Any],
    supabase: Any,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Create text chunks for a single product (product-centric pipeline).

    This is the single-product version used in the product-centric pipeline.
    It creates chunks ONLY for pages belonging to one product.

    Args:
        file_content: PDF file bytes
        document_id: Document identifier
        workspace_id: Workspace identifier
        job_id: Job identifier
        product: Single product object
        product_pages: Set of page numbers for this product
        catalog: Full catalog (for context)
        pdf_result: PDF extraction result
        config: Processing configuration (chunk_size, chunk_overlap, etc.)
        supabase: Supabase client
        logger: Logger instance

    Returns:
        Dictionary with chunks_created count
    """
    from app.services.search.rag_service import RAGService

    logger.info(f"📝 Creating chunks for product: {product.name}")
    logger.info(f"   Pages: {sorted(product_pages)}")

    # Initialize RAG service
    rag_service = RAGService(config={
        'chunk_size': config.get('chunk_size', 1000),
        'chunk_overlap': config.get('chunk_overlap', 200)
    })

    # Filter PDF content to only include product pages
    # Extract text from product pages only
    product_text = ""
    if pdf_result and pdf_result.markdown_content:
        # TODO: Implement page-based text filtering
        # For now, use all text (will be filtered by page metadata)
        product_text = pdf_result.markdown_content

    # Create chunks with product-specific metadata
    chunk_result = await rag_service.index_pdf_content(
        pdf_content=file_content,
        document_id=document_id,
        metadata={
            'filename': f"{product.name}_chunks",
            'title': product.name,
            'page_count': len(product_pages),
            'product_pages': sorted(product_pages),
            'product_name': product.name,
            'product_id': f"product_{product.name.replace(' ', '_')}",
            'chunk_size': config.get('chunk_size', 1000),
            'chunk_overlap': config.get('chunk_overlap', 200),
            'workspace_id': workspace_id,
            'job_id': job_id
        },
        catalog=catalog,
        pre_extracted_text=product_text if product_text else None
    )

    chunks_created = chunk_result.get('chunks_created', 0)
    logger.info(f"   ✅ Created {chunks_created} chunks for {product.name}")

    return {
        'chunks_created': chunks_created,
        'chunk_ids': chunk_result.get('chunk_ids', [])
    }

