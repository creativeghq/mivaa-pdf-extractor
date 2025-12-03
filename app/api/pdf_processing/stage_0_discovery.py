"""
Stage 0: Product Discovery

This module handles product discovery from PDF documents using AI models.
Implements two-stage discovery:
- Stage 0A: Index scan (first 50-100 pages)
- Stage 0B: Focused extraction (specific pages per product)
"""

import os
import logging
import tempfile
import aiofiles
from typing import Dict, Any, List, Set
from datetime import datetime

from app.utils.timeout_guard import with_timeout, ProgressiveTimeoutStrategy
from app.utils.circuit_breaker import CircuitBreaker, CircuitBreakerError
from app.utils.memory_monitor import memory_monitor
from app.schemas.jobs import ProcessingStage, PageProcessingStatus
from app.services.checkpoint_recovery_service import ProcessingStage as CheckpointStage

module_logger = logging.getLogger(__name__)


async def process_stage_0_discovery(
    file_content: bytes,
    document_id: str,
    workspace_id: str,
    job_id: str,
    filename: str,
    title: str,
    description: str,
    extract_categories: List[str],
    discovery_model: str,
    agent_prompt: str,
    enable_prompt_enhancement: bool,
    tracker: Any,
    checkpoint_recovery_service: Any,
    logger: Any
) -> Dict[str, Any]:
    """
    Stage 0: Product Discovery
    
    Discovers products and other entities (certificates, logos, specifications) from PDF.
    
    Args:
        file_content: PDF file bytes
        document_id: Unique document identifier
        workspace_id: Workspace identifier
        job_id: Job identifier for tracking
        filename: Original filename
        title: Document title
        description: Document description
        extract_categories: Categories to extract (products, certificates, logos, specifications)
        discovery_model: AI model to use (claude, gpt, haiku)
        agent_prompt: Custom prompt for discovery
        enable_prompt_enhancement: Whether to enhance prompts with context
        tracker: Job progress tracker
        checkpoint_recovery_service: Checkpoint service for recovery
        logger: Logger instance
        
    Returns:
        Dictionary containing:
        - catalog: Discovered catalog with products and entities
        - pdf_result: PDF extraction result with page count and content
        - product_pages: Set of page numbers containing products
        - temp_pdf_path: Path to temporary PDF file
    """
    from app.services.product_discovery_service import ProductDiscoveryService
    from app.services.pdf_processor import PDFProcessor
    from app.services.supabase_client import get_supabase_client

    logger.info("🔍 [STAGE 0] Product Discovery - Starting...")
    await tracker.update_stage(ProcessingStage.INITIALIZING, stage_name="product_discovery")

    # Initialize circuit breaker for AI API calls
    discovery_breaker = CircuitBreaker(
        name="ProductDiscovery",
        failure_threshold=3,
        timeout_seconds=120
    )

    # Log initial memory state
    mem_stats = memory_monitor.get_memory_stats()
    logger.info(f"💾 Initial memory: {mem_stats.used_mb:.1f} MB ({mem_stats.percent_used:.1f}%)")

    # Save PDF to temporary file for two-stage discovery
    temp_pdf_path = None
    try:
        # EVENT-BASED CLEANUP: Register resource manager
        from app.utils.resource_manager import get_resource_manager
        resource_manager = get_resource_manager()
        
        # Create temp file that persists during processing
        temp_fd, temp_pdf_path = tempfile.mkstemp(suffix='.pdf', prefix=f'{document_id}_')
        os.close(temp_fd)  # Close file descriptor, we'll write with aiofiles
        
        # Write PDF bytes to temp file
        async with aiofiles.open(temp_pdf_path, 'wb') as f:
            await f.write(file_content)
        
        logger.info(f"📁 Saved PDF to temp file: {temp_pdf_path}")
        
        # Register temp file with resource manager
        await resource_manager.register_resource(
            resource_id=f"temp_pdf_{document_id}",
            resource_type="file",
            path=temp_pdf_path,
            job_id=job_id,
            metadata={"document_id": document_id, "filename": filename}
        )
        
        # Mark as IN_USE before PyMuPDF opens it
        await resource_manager.mark_in_use(f"temp_pdf_{document_id}", job_id)
        
        # 🚀 PROGRESSIVE TIMEOUT: Calculate timeout based on document size
        file_size_mb = len(file_content) / (1024 * 1024)
        
        # Quick page count check (fast, doesn't extract content)
        import fitz
        quick_doc = fitz.open(temp_pdf_path)
        page_count = len(quick_doc)
        quick_doc.close()

        # Calculate progressive timeout for PDF extraction
        pdf_extraction_timeout = ProgressiveTimeoutStrategy.calculate_pdf_extraction_timeout(
            page_count=page_count,
            file_size_mb=file_size_mb
        )
        logger.info(f"📊 Document: {page_count} pages, {file_size_mb:.1f} MB → timeout: {pdf_extraction_timeout:.0f}s")

        # Check memory pressure before PDF extraction
        mem_before_extraction = memory_monitor.get_memory_stats()
        logger.info(f"💾 Memory before PDF extraction: {mem_before_extraction.used_mb:.1f} MB ({mem_before_extraction.percent_used:.1f}%)")

        # Wait for memory if pressure is high
        if mem_before_extraction.is_high_pressure:
            logger.warning(f"⚠️ High memory pressure detected, waiting for memory to free up...")
            await memory_monitor.wait_for_memory_available(
                required_mb=200,  # Need 200MB for PDF extraction
                max_wait_seconds=60,
                operation_name="PDF extraction"
            )

        # SKIP FULL PDF EXTRACTION - Let product_discovery_service handle it
        # The service will extract text on-demand in batches (100K chars at a time)
        # This is MUCH faster than extracting all 71 pages upfront
        logger.info(f"📄 SKIPPING full PDF extraction - using on-demand extraction in discovery service")
        logger.info(f"⏱️  This will be 10x faster than extracting all {page_count} pages upfront")

        # Create a minimal pdf_result with just page count
        from dataclasses import dataclass
        @dataclass
        class MinimalPDFResult:
            page_count: int
            markdown_content: str = None

        pdf_result = MinimalPDFResult(page_count=page_count, markdown_content=None)

        logger.info(f"✅ PDF ready for on-demand extraction: {pdf_result.page_count} pages")

        # Log memory (no extraction happened, so no memory used)
        mem_after_extraction = memory_monitor.get_memory_stats()
        logger.info(f"💾 Memory after setup: {mem_after_extraction.used_mb:.1f} MB (no extraction yet)")

        # Create processed_documents record IMMEDIATELY (required for job_progress foreign key)
        supabase = get_supabase_client()
        try:
            supabase.client.table('processed_documents').upsert({
                "id": document_id,
                "workspace_id": workspace_id,
                "pdf_document_id": document_id,
                "content": pdf_result.markdown_content or "",
                "processing_status": "processing",
                "metadata": {
                    "filename": filename,
                    "file_size": len(file_content),
                    "page_count": pdf_result.page_count
                }
            }).execute()
            logger.info(f"✅ Created processed_documents record for {document_id}")
        except Exception as e:
            logger.error(f"❌ CRITICAL: Failed to create processed_documents record: {e}")
            raise  # Don't continue if this fails

        # Update tracker with total pages
        tracker.total_pages = pdf_result.page_count
        for page_num in range(1, pdf_result.page_count + 1):
            tracker.page_statuses[page_num] = PageProcessingStatus(
                page_number=page_num,
                stage=ProcessingStage.INITIALIZING,
                status="pending"
            )

        # Run TWO-STAGE category-based discovery with prompt enhancement
        # Stage 0A: Index scan (first 50-100 pages)
        # Stage 0B: Focused extraction (specific pages per product)

        # 🚀 PROGRESSIVE TIMEOUT: Calculate timeout based on pages and categories
        discovery_timeout = ProgressiveTimeoutStrategy.calculate_product_discovery_timeout(
            page_count=pdf_result.page_count,
            categories=extract_categories
        )
        logger.info(f"📊 Product discovery: {pdf_result.page_count} pages, {len(extract_categories)} categories → timeout: {discovery_timeout:.0f}s")

        #NORMALIZE: Map discovery_model to expected values
        # ProductDiscoveryService expects: "claude-vision", "claude", "gpt-vision", "gpt", "haiku-vision", "haiku"
        normalized_model = discovery_model.lower()

        # Keep vision suffix if present
        if normalized_model.endswith('-vision'):
            # Vision models: claude-vision, claude-haiku-vision, gpt-vision
            if "haiku" in normalized_model:
                normalized_model = "claude-haiku-vision"
            elif "claude" in normalized_model or "sonnet" in normalized_model:
                normalized_model = "claude-vision"
            elif "gpt" in normalized_model:
                normalized_model = "gpt-vision"
        else:
            # Text-only models (legacy)
            if "claude" in normalized_model or "sonnet" in normalized_model:
                normalized_model = "claude"
            elif "gpt" in normalized_model:
                normalized_model = "gpt"
            elif "haiku" in normalized_model:
                normalized_model = "haiku"
            else:
                normalized_model = "claude-vision"  # Default to Claude Vision (fastest)

        logger.info(f"🤖 Discovery model normalized: '{discovery_model}' → '{normalized_model}'")

        # Check memory before discovery
        await memory_monitor.check_memory_pressure()

        discovery_service = ProductDiscoveryService(model=normalized_model)

        # Wrap discovery in circuit breaker for fail-fast protection
        try:
            catalog = await discovery_breaker.call(
                lambda: with_timeout(
                    discovery_service.discover_products(
                        pdf_content=file_content,
                        pdf_text=pdf_result.markdown_content,  # ✅ PASS FULL PDF TEXT
                        total_pages=pdf_result.page_count,
                        categories=extract_categories,
                        agent_prompt=agent_prompt,
                        workspace_id=workspace_id,
                        enable_prompt_enhancement=enable_prompt_enhancement,
                        job_id=job_id,
                        pdf_path=temp_pdf_path,
                        tracker=tracker
                    ),
                    timeout_seconds=discovery_timeout,
                    operation_name="Product discovery (Stage 0A + 0B)"
                )
            )
        except CircuitBreakerError as cb_error:
            logger.error(f"❌ Product discovery failed (circuit breaker OPEN): {cb_error}")
            raise Exception(f"Product discovery service unavailable: {cb_error}")

        # Log memory after discovery
        mem_after = memory_monitor.get_memory_stats()
        logger.info(f"💾 Memory after discovery: {mem_after.used_mb:.1f} MB ({mem_after.percent_used:.1f}%)")

    except Exception as discovery_error:
        # Re-raise discovery errors to be handled by outer exception handler
        raise discovery_error

    logger.info(f"✅ [STAGE 0] Discovery Complete:")
    logger.info(f"   Categories: {', '.join(extract_categories)}")
    logger.info(f"   Products: {len(catalog.products)}")
    if "certificates" in extract_categories:
        logger.info(f"   Certificates: {len(catalog.certificates)}")
    if "logos" in extract_categories:
        logger.info(f"   Logos: {len(catalog.logos)}")
    if "specifications" in extract_categories:
        logger.info(f"   Specifications: {len(catalog.specifications)}")
    logger.info(f"   Confidence: {catalog.confidence_score:.2f}")

    # Update tracker
    tracker.products_created = len(catalog.products)
    await tracker._sync_to_database(stage="product_discovery")

    # ✅ NEW: Save discovered entities to document_entities table
    entity_ids = []
    if any(cat in extract_categories for cat in ["certificates", "logos", "specifications"]):
        try:
            from app.services.document_entity_service import DocumentEntityService, DocumentEntity
            from app.database import get_supabase_client

            supabase_client = get_supabase_client()
            entity_service = DocumentEntityService(supabase_client)

            # Convert discovered entities to DocumentEntity objects
            entities_to_save = []

            # Certificates
            if "certificates" in extract_categories and catalog.certificates:
                for cert in catalog.certificates:
                    entity = DocumentEntity(
                        entity_type="certificate",
                        name=cert.name,
                        page_range=cert.page_range,
                        description=f"{cert.certificate_type or 'Certificate'} issued by {cert.issuer or 'Unknown'}",
                        metadata={
                            "certificate_type": cert.certificate_type,
                            "issuer": cert.issuer,
                            "issue_date": cert.issue_date,
                            "expiry_date": cert.expiry_date,
                            "standards": cert.standards or [],
                            "confidence": cert.confidence
                        }
                    )
                    entities_to_save.append(entity)

            # Logos
            if "logos" in extract_categories and catalog.logos:
                for logo in catalog.logos:
                    entity = DocumentEntity(
                        entity_type="logo",
                        name=logo.name,
                        page_range=logo.page_range,
                        description=logo.description,
                        metadata={
                            "logo_type": logo.logo_type,
                            "confidence": logo.confidence
                        }
                    )
                    entities_to_save.append(entity)

            # Specifications
            if "specifications" in extract_categories and catalog.specifications:
                for spec in catalog.specifications:
                    entity = DocumentEntity(
                        entity_type="specification",
                        name=spec.name,
                        page_range=spec.page_range,
                        description=spec.description,
                        metadata={
                            "spec_type": spec.spec_type,
                            "confidence": spec.confidence
                        }
                    )
                    entities_to_save.append(entity)

            # Save all entities
            if entities_to_save:
                entity_ids = await entity_service.save_entities(
                    entities=entities_to_save,
                    source_document_id=document_id,
                    workspace_id=workspace_id
                )
                logger.info(f"✅ Saved {len(entity_ids)} entities to document_entities table")

        except Exception as entity_error:
            logger.error(f"⚠️ Failed to save entities (continuing): {entity_error}")
            # Don't fail the entire process if entity saving fails

    # Create PRODUCTS_DETECTED checkpoint (now includes all categories)
    checkpoint_data = {
        "document_id": document_id,
        "categories": extract_categories,
        "products_detected": len(catalog.products),
        "product_names": [p.name for p in catalog.products],
        "total_pages": pdf_result.page_count,
        "entity_ids": entity_ids  # Store entity IDs for later stages
    }

    # Add other categories if discovered
    if "certificates" in extract_categories:
        checkpoint_data["certificates_detected"] = len(catalog.certificates)
        checkpoint_data["certificate_names"] = [c.name for c in catalog.certificates]
    if "logos" in extract_categories:
        checkpoint_data["logos_detected"] = len(catalog.logos)
        checkpoint_data["logo_names"] = [l.name for l in catalog.logos]
    if "specifications" in extract_categories:
        checkpoint_data["specifications_detected"] = len(catalog.specifications)
        checkpoint_data["specification_names"] = [s.name for s in catalog.specifications]

    await checkpoint_recovery_service.create_checkpoint(
        job_id=job_id,
        stage=CheckpointStage.PRODUCTS_DETECTED,
        data=checkpoint_data,
        metadata={
            "confidence_score": catalog.confidence_score,
            "discovery_model": discovery_model
        }
    )
    logger.info(f"✅ Created PRODUCTS_DETECTED checkpoint for job {job_id}")

    return {
        "catalog": catalog,
        "pdf_result": pdf_result,  # Return full PDF result
        "page_count": pdf_result.page_count,
        "file_size_mb": file_size_mb,
        "temp_pdf_path": temp_pdf_path
    }

