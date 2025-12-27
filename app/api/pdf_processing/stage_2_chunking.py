"""
Stage 2: Text Chunking

This module handles text chunking and embedding generation for PDF content.
"""

import logging
from typing import Dict, Any, Set
from app.schemas.jobs import ProcessingStage
from app.services.checkpoint_recovery_service import ProcessingStage as CheckpointStage
from app.utils.timeout_guard import with_timeout, ProgressiveTimeoutStrategy

logger = logging.getLogger(__name__)


async def process_stage_2_chunking(
    file_content: bytes,
    document_id: str,
    workspace_id: str,
    job_id: str,
    filename: str,
    title: str,
    description: str,
    document_tags: list,
    catalog: Any,
    pdf_result: Any,
    product_pages: Set[int],
    focused_extraction: bool,
    discovery_model: str,
    chunk_size: int,
    chunk_overlap: int,
    tracker: Any,
    checkpoint_recovery_service: Any,
    supabase: Any,
    logger: Any
) -> Dict[str, Any]:
    """
    Stage 2: Text Chunking

    Creates document record and processes text chunks for vector database.
    
    Args:
        file_content: PDF file bytes
        document_id: Unique document identifier
        workspace_id: Workspace identifier
        job_id: Job identifier for tracking
        filename: Original filename
        title: Document title
        description: Document description
        document_tags: Document tags
        catalog: Product catalog from Stage 0
        pdf_result: PDF extraction result
        product_pages: Set of page numbers to process
        focused_extraction: Whether focused extraction is enabled
        discovery_model: AI model used for discovery
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        tracker: Job progress tracker
        checkpoint_recovery_service: Checkpoint service
        supabase: Supabase client
        logger: Logger instance
        
    Returns:
        Dictionary containing:
        - chunk_result: Chunking result with chunk IDs and count
    """
    from app.services.rag_service import RAGService

    logger.info("📝 [STAGE 2] Chunking - Starting with RAG service...")
    await tracker.update_stage(ProcessingStage.SAVING_TO_DATABASE, stage_name="chunking")

    # Initialize RAG service with chunking configuration
    rag_service = RAGService(config={
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap
    })
    
    # Create document in database
    doc_metadata = {
        'title': title or filename,
        'description': description,
        'page_count': pdf_result.page_count,
        'tags': document_tags,
        'products_discovered': len(catalog.products),
        'product_names': [p.name for p in catalog.products],
        'focused_extraction': focused_extraction,
        'discovery_model': discovery_model
    }

    # Add catalog-level factory info if discovered
    if catalog.catalog_factory:
        doc_metadata['catalog_factory'] = catalog.catalog_factory
    if catalog.catalog_factory_group:
        doc_metadata['catalog_factory_group'] = catalog.catalog_factory_group
    if catalog.catalog_manufacturer:
        doc_metadata['catalog_manufacturer'] = catalog.catalog_manufacturer

    doc_data = {
        'id': document_id,
        'workspace_id': workspace_id,
        'filename': filename,
        'content_type': 'application/pdf',
        'file_size': len(file_content),
        'file_path': f'pdf-documents/{document_id}/{filename}',
        'processing_status': 'processing',
        'metadata': doc_metadata
    }
    supabase.client.table('documents').upsert(doc_data).execute()
    logger.info(f"✅ Created documents table record for {document_id}")
    if catalog.catalog_factory:
        logger.info(f"   🏭 Catalog factory: {catalog.catalog_factory}")

    # ============================================================
    # 🚀 MEMORY OPTIMIZATION: Retrieve pre-extracted text from Stage 1
    # ============================================================
    # Instead of re-extracting text from PDF bytes (which causes memory bloat),
    # retrieve the already-extracted text from processed_documents table
    pre_extracted_text = None
    try:
        result = supabase.client.table('processed_documents')\
            .select('content')\
            .eq('id', document_id)\
            .execute()

        if result.data and len(result.data) > 0:
            pre_extracted_text = result.data[0].get('content')
            if pre_extracted_text:
                logger.info(f"✅ Retrieved pre-extracted text ({len(pre_extracted_text)} chars) from Stage 1")
            else:
                logger.warning("⚠️ No pre-extracted text found, will extract from PDF")
        else:
            logger.warning("⚠️ No processed_documents record found, will extract from PDF")
    except Exception as e:
        logger.warning(f"⚠️ Failed to retrieve pre-extracted text: {e}, will extract from PDF")

    # Process chunks using RAG service (with progressive timeout)
    if pre_extracted_text:
        logger.info(f"📝 Calling index_pdf_content with pre-extracted text, product_pages={sorted(product_pages)}")
    else:
        logger.info(f"📝 Calling index_pdf_content with {len(file_content)} bytes, product_pages={sorted(product_pages)}")
    logger.info(f"📝 RAG service available: {rag_service.available}")

    # 🚀 PROGRESSIVE TIMEOUT: Calculate timeout based on page count
    chunking_timeout = ProgressiveTimeoutStrategy.calculate_chunking_timeout(
        page_count=pdf_result.page_count,
        chunk_size=chunk_size
    )
    logger.info(f"📊 Chunking: {pdf_result.page_count} pages, chunk_size={chunk_size} → timeout: {chunking_timeout:.0f}s")

    chunk_result = await with_timeout(
        rag_service.index_pdf_content(
            pdf_content=file_content if not pre_extracted_text else None,  # ✅ Only pass PDF if no pre-extracted text
            document_id=document_id,
            metadata={
                'filename': filename,
                'title': title,
                'page_count': pdf_result.page_count,
                'product_pages': sorted(product_pages),
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'workspace_id': workspace_id,
                'job_id': job_id  # ✅ NEW: Pass job_id for source tracking
            },
            catalog=catalog,  # ✅ NEW: Pass catalog for category tagging
            pre_extracted_text=pre_extracted_text  # ✅ NEW: Pass pre-extracted text to skip PDF extraction
        ),
        timeout_seconds=chunking_timeout,
        operation_name="Chunking operation"
    )
    
    logger.info(f"📝 index_pdf_content returned: {chunk_result}")

    tracker.chunks_created = chunk_result.get('chunks_created', 0)
    # NOTE: Don't pass chunks_created to update_database_stats because it increments!
    # We already set tracker.chunks_created directly above.
    await tracker.update_database_stats(
        kb_entries=tracker.chunks_created,
        sync_to_db=True
    )

    # Stage 2 progress: 30% → 50% (fixed when complete)
    await tracker.update_stage(
        ProcessingStage.SAVING_TO_DATABASE,
        stage_name="chunking",
        progress_percentage=50
    )

    # Sync progress to database with stage name
    await tracker._sync_to_database(stage="chunking")

    logger.info(f"✅ [STAGE 2] Chunking Complete: {tracker.chunks_created} chunks created")
    logger.info(f"📊 Progress updated: 50% (Stage 2 complete - {tracker.chunks_created} chunks + embeddings)")
    
    # Create CHUNKS_CREATED checkpoint
    await checkpoint_recovery_service.create_checkpoint(
        job_id=job_id,
        stage=CheckpointStage.CHUNKS_CREATED,
        data={
            "document_id": document_id,
            "chunks_created": tracker.chunks_created,
            "chunk_ids": chunk_result.get('chunk_ids', [])
        },
        metadata={
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "product_pages": sorted(product_pages)
        }
    )
    logger.info(f"✅ Created CHUNKS_CREATED checkpoint for job {job_id}")
    
    # Force garbage collection after chunking to free memory
    import gc
    gc.collect()
    logger.info("💾 Memory freed after Stage 2 (Chunking)")
    
    return {
        "chunk_result": chunk_result
    }

