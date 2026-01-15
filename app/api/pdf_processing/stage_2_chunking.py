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

    # ✅ FIX: Extract text from product pages when pdf_result is None
    # In product-centric pipeline, pdf_result is None, so we need to extract text on-demand
    product_text = ""
    if pdf_result and pdf_result.markdown_content:
        # Legacy path: use pre-extracted text
        product_text = pdf_result.markdown_content
    else:
        # Product-centric path: extract text from specific product pages
        logger.info(f"   📄 Extracting text from {len(product_pages)} product pages...")
        try:
            import pymupdf4llm
            import tempfile
            import os

            # Write PDF bytes to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file_content)
                tmp_pdf_path = tmp_file.name

            try:
                # Convert product_pages (0-based array indices) to 1-based catalog pages
                from app.utils.page_converter import PageConverter
                pages_per_sheet = getattr(catalog, 'pages_per_sheet', 1)
                converter = PageConverter(pages_per_sheet=pages_per_sheet)

                catalog_pages = []
                for array_index in product_pages:
                    try:
                        page = converter.from_array_index(array_index)
                        catalog_pages.append(page.catalog_page)
                    except ValueError:
                        continue

                # Extract text from specific pages using PyMuPDF4LLM
                if catalog_pages:
                    logger.info(f"   📄 Extracting text from catalog pages: {sorted(catalog_pages)}")
                    markdown_result = pymupdf4llm.to_markdown(
                        tmp_pdf_path,
                        pages=sorted(catalog_pages),
                        page_chunks=False  # Get full text, not page chunks
                    )
                    product_text = str(markdown_result) if markdown_result else ""
                    logger.info(f"   ✅ Extracted {len(product_text)} characters from {len(catalog_pages)} pages")
                else:
                    logger.warning(f"   ⚠️ No valid catalog pages found for product")
            finally:
                # Clean up temp file
                if os.path.exists(tmp_pdf_path):
                    os.unlink(tmp_pdf_path)
        except Exception as e:
            logger.error(f"   ❌ Failed to extract product text: {e}")
            product_text = ""

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
            'job_id': job_id,
            'product_id': product_id,  # ✅ FIX: Add product_id to metadata for chunk-product association
            'product_name': product.name  # ✅ FIX: Add product_name for easier querying
        },
        catalog=catalog,
        pre_extracted_text=product_text if product_text else None,
        layout_regions_by_page=layout_regions_by_page  # Pass layout regions for layout-aware chunking
    )

    # Get chunks count from result
    # NOTE: Chunks are already created and stored in DB by index_pdf_content
    # We don't need to re-process them here
    chunks_created = chunk_result.get('chunks_created', 0)
    logger.info(f"   ✅ Created {chunks_created} chunks for {product.name} (already stored in DB)")

    # TODO: If we need chunk enrichment and classification, we should:
    # 1. Fetch chunks from DB using chunk_ids
    # 2. Enrich and classify them
    # 3. Update them in DB
    # For now, we skip this since chunks are already stored with basic metadata

    return {
        'chunks_created': chunks_created,
        'chunk_ids': chunk_result.get('chunk_ids', []),
        'embeddings_generated': chunk_result.get('embeddings_generated', 0),  # ✅ FIX: Return embeddings count for tracker update
        'pages_chunked': len(chunkable_pages),
        'pages_excluded': len(excluded_pages)
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

