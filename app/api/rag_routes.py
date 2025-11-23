"""
RAG (Retrieval-Augmented Generation) API Routes - MODULAR VERSION

This module provides the main router that includes all modular sub-routers
and defines shared background processing functions.

File Structure:
- routes/data.py - Data retrieval endpoints (chunks, images, products, embeddings, relevancies)
- routes/system.py - System endpoints (health, stats)
- routes/jobs.py - Job management endpoints
- routes/search.py - Search endpoints (query, chat, search)
- routes/documents.py - Document management endpoints
- routes/models.py - Shared Pydantic models
- routes/shared.py - Shared utilities and dependencies
"""

import logging
import os
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.services.llamaindex_service import LlamaIndexService
from app.services.job_recovery_service import JobRecoveryService
from app.services.checkpoint_recovery_service import checkpoint_recovery_service, ProcessingStage
from app.services.supabase_client import get_supabase_client
from app.utils.logging import PDFProcessingLogger
from app.utils.progress_tracker import ProgressTracker
from app.services.lazy_loader import get_component_manager
from app.services.resource_manager import get_resource_manager

# Import modular PDF processing stages
from app.api.pdf_processing import (
    process_stage_0_discovery,
    process_stage_1_focused_extraction,
    process_stage_2_chunking,
    process_stage_3_images,
    process_stage_4_products,
    process_stage_5_quality
)

# Import modular route routers
from app.api.routes import (
    data_router,
    system_router,
    jobs_router,
    search_router,
    documents_router
)

# Import shared utilities
from app.api.routes.shared import job_storage, run_async_in_background

logger = logging.getLogger(__name__)

# Initialize main router
router = APIRouter(prefix="/api/rag", tags=["RAG"])

# Job recovery service (initialized on startup)
job_recovery_service: Optional[JobRecoveryService] = None


# ============================================================================
# Startup/Shutdown Functions
# ============================================================================
async def initialize_job_recovery():
    """
    Initialize job recovery service and mark any interrupted jobs.
    This should be called on application startup.
    """
    global job_recovery_service

    try:
        logger.info("🔄 Initializing job recovery service...")

        supabase_client = get_supabase_client()
        job_recovery_service = JobRecoveryService(supabase_client)

        # Mark all processing jobs as interrupted (they were interrupted by restart)
        interrupted_count = await job_recovery_service.mark_all_processing_as_interrupted(
            reason="Service restart detected"
        )

        if interrupted_count > 0:
            logger.warning(f"🛑 Marked {interrupted_count} jobs as interrupted due to service restart")

        # Get statistics
        stats = await job_recovery_service.get_job_statistics()
        logger.info(f"📊 Job statistics: {stats}")

        logger.info("✅ Job recovery service initialized successfully")

    except Exception as e:
        logger.error(f"❌ Failed to initialize job recovery service: {e}", exc_info=True)
        job_recovery_service = None


# ============================================================================
# Background Processing Functions
# ============================================================================
async def process_document_background(
    job_id: str,
    document_id: str,
    file_content: bytes,
    filename: str,
    title: Optional[str],
    description: Optional[str],
    document_tags: List[str],
    chunk_size: int,
    chunk_overlap: int,
    llamaindex_service: Optional[LlamaIndexService] = None
):
    """
    Background task for standard document processing (legacy).

    NOTE: This is the legacy processing function. New uploads should use
    process_document_with_discovery() which includes product discovery.

    This function is kept for backward compatibility but delegates to
    the modular pipeline with default settings.
    """
    logger.info(f"⚠️  [LEGACY] process_document_background called - delegating to modular pipeline")

    # Delegate to modular pipeline with default discovery settings
    await process_document_with_discovery(
        job_id=job_id,
        document_id=document_id,
        file_content=file_content,
        filename=filename,
        title=title,
        description=description,
        document_tags=document_tags,
        discovery_model="claude-sonnet-4.5",
        focused_extraction=False,  # Process all pages
        extract_categories=["all"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        workspace_id="ffafc28b-1b8b-4b0d-b226-9f9a6154004e",
        agent_prompt=None,
        enable_prompt_enhancement=True
    )


async def process_document_with_discovery(
    job_id: str,
    document_id: str,
    file_content: bytes,
    filename: str,
    title: Optional[str],
    description: Optional[str],
    document_tags: List[str],
    discovery_model: str,
    focused_extraction: bool,
    extract_categories: List[str],
    chunk_size: int,
    chunk_overlap: int,
    workspace_id: str = "ffafc28b-1b8b-4b0d-b226-9f9a6154004e",
    agent_prompt: Optional[str] = None,
    enable_prompt_enhancement: bool = True
):
    """
    Background task for document processing with product discovery - MODULAR VERSION.

    This function orchestrates the 6-stage PDF processing pipeline using
    modular stage functions for better maintainability and operational management.

    Stages:
    0. Product Discovery - Identify products using Claude/GPT-5
    1. Focused Extraction - Filter to product pages only
    2. Chunking - Create text chunks with LlamaIndex
    3. Image Processing - Extract images, generate CLIP embeddings, analyze with Llama Vision
    4. Product Creation - Create product entities and relationships
    5. Quality Enhancement - Async Claude validation for low-quality images
    """
    start_time = datetime.utcnow()
    logger.info("=" * 80)
    logger.info(f"🚀 [MODULAR PIPELINE] Starting PDF processing with discovery")
    logger.info(f"📋 Job ID: {job_id}")
    logger.info(f"📄 Document ID: {document_id}")
    logger.info(f"📝 Filename: {filename}")
    logger.info(f"📦 File size: {len(file_content)} bytes")
    logger.info(f"🤖 Discovery model: {discovery_model}")
    logger.info(f"🎯 Focused extraction: {focused_extraction}")
    logger.info(f"📂 Categories: {extract_categories}")
    logger.info("=" * 80)

    # Initialize services
    supabase = get_supabase_client()
    pdf_logger = PDFProcessingLogger(job_id, document_id)
    tracker = ProgressTracker(job_id, document_id, supabase)
    component_manager = get_component_manager()
    resource_manager = get_resource_manager()
    loaded_components = []

    try:
        # ====================================================================
        # STAGE 0: Product Discovery
        # ====================================================================
        logger.info("🔍 [STAGE 0] Starting Product Discovery...")
        stage_0_result = await process_stage_0_discovery(
            file_content=file_content,
            document_id=document_id,
            workspace_id=workspace_id,
            job_id=job_id,
            filename=filename,
            title=title or filename,
            description=description or "",
            extract_categories=extract_categories,
            discovery_model=discovery_model,
            agent_prompt=agent_prompt or "",
            enable_prompt_enhancement=enable_prompt_enhancement,
            tracker=tracker,
            checkpoint_recovery_service=checkpoint_recovery_service,
            logger=pdf_logger
        )
        catalog = stage_0_result['catalog']
        pdf_result = stage_0_result['pdf_result']
        temp_pdf_path = stage_0_result['temp_pdf_path']
        logger.info(f"✅ [STAGE 0] Product Discovery complete - found {len(catalog.products)} products")

        # ====================================================================
        # STAGE 1: Focused Extraction
        # ====================================================================
        logger.info("📄 [STAGE 1] Starting Focused Extraction...")
        stage_1_result = await process_stage_1_focused_extraction(
            catalog=catalog,
            pdf_result=pdf_result,
            focused_extraction=focused_extraction,
            tracker=tracker,
            logger=pdf_logger
        )
        product_pages = stage_1_result['product_pages']
        logger.info(f"✅ [STAGE 1] Focused Extraction complete - {len(product_pages)} product pages identified")

        # ====================================================================
        # STAGE 2: Chunking
        # ====================================================================
        logger.info("✂️  [STAGE 2] Starting Chunking...")
        stage_2_result = await process_stage_2_chunking(
            file_content=file_content,
            document_id=document_id,
            workspace_id=workspace_id,
            job_id=job_id,
            filename=filename,
            title=title or filename,
            description=description or "",
            document_tags=document_tags,
            catalog=catalog,
            pdf_result=pdf_result,
            product_pages=product_pages,
            focused_extraction=focused_extraction,
            discovery_model=discovery_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tracker=tracker,
            checkpoint_recovery_service=checkpoint_recovery_service,
            supabase=supabase,
            logger=pdf_logger
        )
        chunk_result = stage_2_result['chunk_result']
        logger.info(f"✅ [STAGE 2] Chunking complete - {chunk_result.get('chunks_created', 0)} chunks created")

        # ====================================================================
        # STAGE 3: Image Processing
        # ====================================================================
        logger.info("🖼️  [STAGE 3] Starting Image Processing...")
        stage_3_result = await process_stage_3_images(
            file_content=file_content,
            document_id=document_id,
            workspace_id=workspace_id,
            job_id=job_id,
            page_count=pdf_result.page_count,
            product_pages=product_pages,
            focused_extraction=focused_extraction,
            extract_categories=extract_categories,
            component_manager=component_manager,
            loaded_components=loaded_components,
            tracker=tracker,
            checkpoint_recovery_service=checkpoint_recovery_service,
            resource_manager=resource_manager,
            pdf_result_with_images=None
        )
        logger.info(f"✅ [STAGE 3] Image Processing complete")

        # ====================================================================
        # STAGE 4: Product Creation
        # ====================================================================
        logger.info("🏭 [STAGE 4] Starting Product Creation...")
        stage_4_result = await process_stage_4_products(
            catalog=catalog,
            document_id=document_id,
            workspace_id=workspace_id,
            job_id=job_id,
            extract_categories=extract_categories,
            tracker=tracker,
            checkpoint_recovery_service=checkpoint_recovery_service,
            logger=pdf_logger
        )
        logger.info(f"✅ [STAGE 4] Product Creation complete")

        # ====================================================================
        # STAGE 5: Quality Enhancement
        # ====================================================================
        logger.info("⭐ [STAGE 5] Starting Quality Enhancement...")
        stage_5_result = await process_stage_5_quality(
            job_id=job_id,
            document_id=document_id,
            workspace_id=workspace_id,
            tracker=tracker,
            logger=pdf_logger
        )
        logger.info(f"✅ [STAGE 5] Quality Enhancement complete")

        # ====================================================================
        # COMPLETION
        # ====================================================================
        await tracker.complete_job()

        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()

        logger.info("=" * 80)
        logger.info(f"✅ [MODULAR PIPELINE] PDF processing completed successfully!")
        logger.info(f"⏱️  Total processing time: {processing_time:.2f} seconds")
        logger.info(f"📊 Products: {len(catalog.products)}")
        logger.info(f"📄 Chunks: {chunk_result.get('chunks_created', 0)}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ [MODULAR PIPELINE] Processing failed: {e}", exc_info=True)
        await tracker.fail_job(error=e)
        raise
    finally:
        # Cleanup
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
                logger.info(f"🗑️  Cleaned up temporary PDF: {temp_pdf_path}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to cleanup temp PDF: {e}")


# ============================================================================
# Include Modular Routers
# ============================================================================
router.include_router(data_router)
router.include_router(system_router)
router.include_router(jobs_router)
router.include_router(search_router)
router.include_router(documents_router)

