"""
Stage 2: Text Chunking

This module handles text chunking for individual products in the product-centric pipeline.
Includes metadata-first chunking, context enrichment, and type classification.
"""

import logging
from typing import Dict, Any, Set, List, Optional


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
    logger: logging.Logger,
    product_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create text chunks for a single product (product-centric pipeline).

    This is the single-product version used in the product-centric pipeline.
    It creates chunks ONLY for pages belonging to one product.

    Pipeline:
    1. Metadata-First: Exclude product metadata pages from chunking
    2. Create Chunks: Generate text chunks from remaining pages
    3. Enrich Context: Add product metadata to chunks
    4. Classify Types: Classify chunk types and extract structured metadata

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
        Dictionary with chunks_created count and processing stats
    """
    from app.services.search.rag_service import RAGService
    from app.services.chunking.metadata_first_chunking_service import MetadataFirstChunkingService
    from app.services.chunking.chunk_context_enrichment_service import ChunkContextEnrichmentService
    from app.services.chunking.chunk_type_classification_service import ChunkTypeClassificationService

    logger.info(f"📝 Creating chunks for product: {product.name}")
    logger.info(f"   Initial pages: {sorted(product_pages)}")

    # ========================================================================
    # STEP 1: Metadata-First - Exclude product metadata pages
    # ========================================================================
    metadata_first_service = MetadataFirstChunkingService(
        enabled=config.get('enable_metadata_first', True)
    )

    excluded_pages = await metadata_first_service.get_pages_to_exclude(
        products=[product],
        document_id=document_id
    )

    # Filter pages to chunk (exclude metadata pages to prevent duplication)
    chunkable_pages = product_pages - excluded_pages
    logger.info(f"   Chunkable pages: {sorted(chunkable_pages)} ({len(chunkable_pages)} pages)")
    if excluded_pages:
        logger.info(f"   Excluded pages: {sorted(excluded_pages)} ({len(excluded_pages)} metadata pages)")

    # ========================================================================
    # EARLY EXIT: Skip chunking if no chunkable pages
    # ========================================================================
    if not chunkable_pages:
        logger.info(f"   ⏭️ Skipping chunking - all pages are metadata pages")
        return {
            'chunks_created': 0,
            'chunk_ids': [],
            'pages_chunked': 0,
            'pages_excluded': len(excluded_pages),
            'chunks': []
        }

    # ========================================================================
    # STEP 1.5: Load Layout Regions (for layout-aware chunking)
    # ========================================================================
    layout_regions_by_page = {}
    if product_id and config.get('enable_layout_aware_chunking', True):
        logger.info(f"   📐 Loading layout regions for layout-aware chunking...")
        layout_regions_by_page = await get_layout_regions(
            product_id=product_id,
            supabase=supabase,
            logger=logger
        )

    # ========================================================================
    # STEP 2: Create Chunks (existing logic)
    # ========================================================================
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
    # NOTE: We use chunkable_pages (metadata pages excluded)
    chunk_result = await rag_service.index_pdf_content(
        pdf_content=file_content,
        document_id=document_id,
        metadata={
            'filename': f"{product.name}_chunks",
            'title': product.name,
            'page_count': len(chunkable_pages),  # ← Changed: use chunkable_pages
            'product_pages': sorted(chunkable_pages),  # ← Changed: use chunkable_pages
            'chunk_size': config.get('chunk_size', 1000),
            'chunk_overlap': config.get('chunk_overlap', 200),
            'workspace_id': workspace_id,
            'job_id': job_id
        },
        catalog=catalog,
        pre_extracted_text=product_text if product_text else None,
        layout_regions_by_page=layout_regions_by_page  # Pass layout regions for layout-aware chunking
    )

    # Get chunks from result
    chunks = chunk_result.get('chunks', [])
    logger.info(f"   Created {len(chunks)} raw chunks")

    # ========================================================================
    # STEP 3: Enrich Chunks with Product Context
    # ========================================================================
    enrichment_service = ChunkContextEnrichmentService(
        enabled=config.get('enable_chunk_enrichment', True)
    )

    # Convert product to dict format expected by enrichment service
    product_dict = {
        'id': getattr(product, 'id', f"product_{product.name.replace(' ', '_')}"),
        'name': product.name,
        'page_range': getattr(product, 'page_range', list(product_pages))
    }

    enriched_chunks = await enrichment_service.enrich_chunks(
        chunks=chunks,
        products=[product_dict],
        document_id=document_id
    )
    logger.info(f"   Enriched {len(enriched_chunks)} chunks with product context")

    # ========================================================================
    # STEP 4: Classify Chunk Types
    # ========================================================================
    classification_service = ChunkTypeClassificationService()

    if config.get('enable_chunk_classification', True):
        logger.info(f"   Classifying {len(enriched_chunks)} chunks...")

        for chunk in enriched_chunks:
            try:
                classification = await classification_service.classify_chunk(chunk.get('content', ''))

                # Add classification metadata
                if 'metadata' not in chunk:
                    chunk['metadata'] = {}

                chunk['metadata']['chunk_type'] = classification.chunk_type.value
                chunk['metadata']['chunk_confidence'] = classification.confidence
                chunk['metadata']['classification_reasoning'] = classification.reasoning

                # Merge extracted metadata
                if classification.metadata:
                    chunk['metadata'].update(classification.metadata)
            except Exception as e:
                logger.warning(f"   ⚠️ Failed to classify chunk: {e}")
                # Continue with unclassified chunk

        logger.info(f"   ✅ Classified {len(enriched_chunks)} chunks")
    else:
        logger.info(f"   ⏭️ Chunk classification disabled")

    # ========================================================================
    # STEP 5: Return Results
    # ========================================================================
    chunks_created = len(enriched_chunks)
    logger.info(f"   ✅ Created {chunks_created} enriched chunks for {product.name}")

    return {
        'chunks_created': chunks_created,
        'chunk_ids': chunk_result.get('chunk_ids', []),
        'pages_chunked': len(chunkable_pages),
        'pages_excluded': len(excluded_pages),
        'chunks': enriched_chunks  # Return enriched chunks for further processing
    }


async def get_layout_regions(
    product_id: str,
    supabase: Any,
    logger: logging.Logger
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Fetch layout regions from database for a product.

    Args:
        product_id: Database product ID
        supabase: Supabase client
        logger: Logger instance

    Returns:
        Dict mapping page_number -> list of regions sorted by reading_order
        Example: {1: [region1, region2], 2: [region3, region4]}
    """
    try:
        # Fetch all layout regions for this product
        result = supabase.client.table('product_layout_regions')\
            .select('*')\
            .eq('product_id', product_id)\
            .order('page_number')\
            .order('reading_order')\
            .execute()

        if not result.data:
            logger.info(f"   No layout regions found for product {product_id}")
            return {}

        # Group regions by page number
        regions_by_page = {}
        for region in result.data:
            page_num = region['page_number']
            if page_num not in regions_by_page:
                regions_by_page[page_num] = []
            regions_by_page[page_num].append(region)

        logger.info(f"   Loaded {len(result.data)} layout regions across {len(regions_by_page)} pages")
        return regions_by_page

    except Exception as e:
        logger.error(f"   ❌ Failed to fetch layout regions: {e}")
        return {}

