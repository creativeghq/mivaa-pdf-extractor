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
from typing import Dict, Any, List, Set, Optional
from datetime import datetime
import sentry_sdk

from app.utils.timeout_guard import with_timeout, ProgressiveTimeoutStrategy
from app.utils.circuit_breaker import CircuitBreaker, CircuitBreakerError
from app.utils.memory_monitor import memory_monitor
from app.schemas.jobs import ProcessingStage, PageProcessingStatus
from app.services.tracking.checkpoint_recovery_service import ProcessingStage as CheckpointStage

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
    logger: Any,
    temp_pdf_path: Optional[str] = None,  # ✅ NEW: Optional existing temp path
    test_single_product: bool = False  # 🧪 TEST MODE: Process only first product
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
        - page_count: Total number of PDF pages
        - all_physical_pages: Set of physical page numbers (1-based) containing products
        - temp_pdf_path: Path to temporary PDF file (returned existing or new)
    """
    from app.services.discovery.product_discovery_service import ProductDiscoveryService
    from app.services.pdf.pdf_processor import PDFProcessor
    from app.services.core.supabase_client import get_supabase_client

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

    # Save PDF to temporary file for two-stage discovery (if not provided)
    pdf_resource_registered = False
    try:
        from app.utils.resource_manager import get_resource_manager
        resource_manager = get_resource_manager()

        if temp_pdf_path and os.path.exists(temp_pdf_path):
            logger.info(f"♻️ [STAGE 0] Reusing existing temp PDF: {temp_pdf_path}")
            # Ensure it's registered (though it should be by rag_routes)
            await resource_manager.register_resource(
                resource_id=f"temp_pdf_{document_id}",
                resource_type="file",
                path=temp_pdf_path,
                job_id=job_id,
                metadata={"document_id": document_id, "filename": filename}
            )
            pdf_resource_registered = True
        else:
            logger.info(f"🔧 [STAGE 0] Step 1/7: Creating temporary PDF file...")
            # Create temp file that persists during processing
            temp_fd, temp_pdf_path = tempfile.mkstemp(suffix='.pdf', prefix=f'{document_id}_')
            os.close(temp_fd)
            logger.info(f"✅ [STAGE 0] Temp file created: {temp_pdf_path}")

            logger.info(f"🔧 [STAGE 0] Step 2/7: Writing PDF bytes to temp file ({len(file_content)} bytes)...")
            async with aiofiles.open(temp_pdf_path, 'wb') as f:
                await f.write(file_content)
            
            # Register temp file with resource manager
            await resource_manager.register_resource(
                resource_id=f"temp_pdf_{document_id}",
                resource_type="file",
                path=temp_pdf_path,
                job_id=job_id,
                metadata={"document_id": document_id, "filename": filename}
            )
            pdf_resource_registered = True
            logger.info(f"✅ [STAGE 0] PDF registered with ResourceManager")

        # Mark as IN_USE
        await resource_manager.mark_in_use(f"temp_pdf_{document_id}", job_id)
        logger.info(f"✅ [STAGE 0] Resource marked as IN_USE")

        logger.info(f"🔧 [STAGE 0] Step 6/7: Analyzing PDF structure...")
        # 🚀 PROGRESSIVE TIMEOUT: Calculate timeout based on document size
        file_size_mb = len(file_content) / (1024 * 1024)
        logger.info(f"📊 File size: {file_size_mb:.1f} MB")

        # Quick page count check (fast, doesn't extract content)
        logger.info(f"🔧 Opening PDF with PyMuPDF to count pages...")
        import fitz
        quick_doc = fitz.open(temp_pdf_path)
        page_count = len(quick_doc)
        quick_doc.close()
        logger.info(f"✅ PDF opened successfully: {page_count} pages")

        # Calculate progressive timeout for PDF extraction
        pdf_extraction_timeout = ProgressiveTimeoutStrategy.calculate_pdf_extraction_timeout(
            page_count=page_count,
            file_size_mb=file_size_mb
        )
        logger.info(f"📊 Document: {page_count} pages, {file_size_mb:.1f} MB → timeout: {pdf_extraction_timeout:.0f}s")

        # Check memory pressure before PDF extraction
        logger.info(f"🔧 Checking memory pressure...")
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
            logger.info(f"✅ Memory pressure resolved")

        # Full PDF text extraction is skipped. product_discovery_service extracts
        # text on-demand in batches (100K chars at a time), which is ~10× faster
        # than reading every page up front.
        logger.info(f"📄 Skipping full PDF extraction ({page_count} pages) — discovery service will extract on-demand")

        mem_after_extraction = memory_monitor.get_memory_stats()
        logger.info(f"💾 Memory after setup: {mem_after_extraction.used_mb:.1f} MB (no extraction yet)")

        supabase = get_supabase_client()
        try:
            supabase.client.table('processed_documents').upsert({
                "id": document_id,
                "workspace_id": workspace_id,
                "pdf_document_id": document_id,
                "content": "",
                "processing_status": "processing",
                "metadata": {
                    "filename": filename,
                    "file_size": len(file_content),
                    "page_count": page_count,
                },
            }).execute()
            logger.info(f"✅ Created processed_documents record for {document_id}")
        except Exception as e:
            logger.error(f"❌ CRITICAL: Failed to create processed_documents record: {e}")
            raise

        tracker.total_pages = page_count
        for page_num in range(1, page_count + 1):
            tracker.page_statuses[page_num] = PageProcessingStatus(
                page_number=page_num,
                stage=ProcessingStage.INITIALIZING,
                status="pending",
            )
        logger.info(f"✅ Tracker updated with {page_count} pages")

        logger.info(f"🔧 [STAGE 0] Step 7/7: Preparing product discovery...")
        # Run TWO-STAGE category-based discovery with prompt enhancement
        # Stage 0A: Index scan (first 50-100 pages)
        # Stage 0B: Focused extraction (specific pages per product)

        discovery_timeout = ProgressiveTimeoutStrategy.calculate_product_discovery_timeout(
            page_count=page_count,
            categories=extract_categories,
        )
        logger.info(f"📊 Product discovery: {page_count} pages, {len(extract_categories)} categories → timeout: {discovery_timeout:.0f}s")

        logger.info(f"🔧 Normalizing discovery model '{discovery_model}'...")
        #NORMALIZE: Map discovery_model to expected values
        # ProductDiscoveryService expects: "claude-vision", "claude", "gpt-vision", "gpt", "haiku-vision", "haiku"
        normalized_model = discovery_model.lower()

        # Keep vision suffix if present
        if normalized_model.endswith('-vision'):
            # Vision models: claude-vision, claude-haiku-vision, gpt-vision
            if "haiku" in normalized_model:
                normalized_model = "claude-haiku-vision"
            elif "claude" in normalized_model:
                normalized_model = "claude-vision"
            elif "gpt" in normalized_model:
                normalized_model = "gpt-vision"
        else:
            # Text-only models (legacy)
            if "claude" in normalized_model:
                normalized_model = "claude"
            elif "gpt" in normalized_model:
                normalized_model = "gpt"
            elif "haiku" in normalized_model:
                normalized_model = "haiku"
            else:
                normalized_model = "claude-vision"  # Default to Claude Vision (fastest)

        logger.info(f"🤖 Discovery model normalized: '{discovery_model}' → '{normalized_model}'")

        logger.info(f"🔧 Checking memory pressure before discovery...")
        # Check memory before discovery
        await memory_monitor.check_memory_pressure()
        logger.info(f"✅ Memory check passed")

        logger.info(f"🔧 Initializing ProductDiscoveryService with model '{normalized_model}'...")
        discovery_service = ProductDiscoveryService(model=normalized_model)
        logger.info(f"✅ ProductDiscoveryService initialized")

        logger.info(f"🚀 [STAGE 0] STARTING PRODUCT DISCOVERY (this may take several minutes)...")
        logger.info(f"   📄 Pages: {page_count}")
        logger.info(f"   📦 Categories: {', '.join(extract_categories)}")
        logger.info(f"   🤖 Model: {normalized_model}")
        logger.info(f"   ⏱️  Timeout: {discovery_timeout:.0f}s")

        try:
            catalog = await discovery_breaker.call(
                lambda: with_timeout(
                    discovery_service.discover_products(
                        pdf_content=file_content,
                        pdf_text=None,
                        total_pages=page_count,
                        categories=extract_categories,
                        agent_prompt=agent_prompt,
                        workspace_id=workspace_id,
                        enable_prompt_enhancement=enable_prompt_enhancement,
                        job_id=job_id,
                        pdf_path=temp_pdf_path,
                        tracker=tracker,
                    ),
                    timeout_seconds=discovery_timeout,
                    operation_name="Product discovery (Stage 0A + 0B)",
                )
            )
            logger.info(f"✅ [STAGE 0] DISCOVERY COMPLETED SUCCESSFULLY")
        except CircuitBreakerError as cb_error:
            logger.error(f"❌ Product discovery failed (circuit breaker OPEN): {cb_error}")
            sentry_sdk.capture_exception(cb_error)
            raise Exception(f"Product discovery service unavailable: {cb_error}")

        # Log memory after discovery
        mem_after = memory_monitor.get_memory_stats()
        logger.info(f"💾 Memory after discovery: {mem_after.used_mb:.1f} MB ({mem_after.percent_used:.1f}%)")

    except Exception as discovery_error:
        # Capture exception in Sentry
        sentry_sdk.capture_exception(discovery_error)
        # Re-raise discovery errors to be handled by outer exception handler
        raise discovery_error

    # ============================================================
    # MONITORING: Track Stage 0 metrics
    # ============================================================
    products_discovered = len(catalog.products)
    certificates_discovered = len(catalog.certificates) if "certificates" in extract_categories else 0
    logos_discovered = len(catalog.logos) if "logos" in extract_categories else 0
    specifications_discovered = len(catalog.specifications) if "specifications" in extract_categories else 0
    total_entities = products_discovered + certificates_discovered + logos_discovered + specifications_discovered

    # Calculate processing time
    discovery_time_ms = catalog.processing_time_ms

    logger.info(f"✅ [STAGE 0] Discovery Complete:")
    logger.info(f"   Categories: {', '.join(extract_categories)}")
    logger.info(f"   Products: {products_discovered}")
    if "certificates" in extract_categories:
        logger.info(f"   Certificates: {certificates_discovered}")
    if "logos" in extract_categories:
        logger.info(f"   Logos: {logos_discovered}")
    if "specifications" in extract_categories:
        logger.info(f"   Specifications: {specifications_discovered}")
    logger.info(f"   Total Entities: {total_entities}")
    logger.info(f"   Confidence: {catalog.confidence_score:.2f}")
    logger.info(f"   Processing Time: {discovery_time_ms:.0f}ms")
    logger.info(f"   Model Used: {catalog.model_used}")

    # Update tracker with comprehensive metadata
    tracker.products_created = products_discovered

    # Stage 0 progress: 0% → 10% (fixed when complete)
    await tracker.update_stage(
        ProcessingStage.INITIALIZING,
        stage_name="product_discovery",
        progress_percentage=10
    )

    await tracker._sync_to_database(stage="product_discovery")

    logger.info(f"📊 Progress updated: 10% (Stage 0 complete - {products_discovered} products discovered)")

    # ✅ NEW: Save discovered entities to document_entities table
    entity_ids = []
    if any(cat in extract_categories for cat in ["certificates", "logos", "specifications"]):
        try:
            from app.services.discovery.document_entity_service import DocumentEntityService, DocumentEntity
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
            sentry_sdk.capture_exception(entity_error)
            # Don't fail the entire process if entity saving fails

    # Create PRODUCTS_DETECTED checkpoint (now includes all categories)
    checkpoint_data = {
        "document_id": document_id,
        "categories": extract_categories,
        "products_detected": len(catalog.products),
        "product_names": [p.name for p in catalog.products],
        "total_pages": page_count,
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
            "discovery_model": discovery_model,
            # NEW: Add comprehensive metrics to checkpoint
            "products_discovered": products_discovered,
            "certificates_discovered": certificates_discovered,
            "logos_discovered": logos_discovered,
            "specifications_discovered": specifications_discovered,
            "total_entities": total_entities,
            "processing_time_ms": discovery_time_ms,
            "model_used": catalog.model_used
        }
    )
    logger.info(f"✅ Created PRODUCTS_DETECTED checkpoint for job {job_id}")

    # ✅ CREATE PRODUCTS IN DATABASE IMMEDIATELY AFTER DISCOVERY
    # This creates the actual product records so all subsequent stages just update them
    if catalog.products:
        from app.services.tracking.product_progress_tracker import ProductProgressTracker
        from app.schemas.product_progress import ProductStatus
        from app.api.pdf_processing.stage_4_products import create_single_product

        # 🧪 TEST MODE: Only create first product in DB if test_single_product=True
        if test_single_product:
            logger.warning("=" * 80)
            logger.warning("🧪 TEST MODE ENABLED: Creating ONLY the first product in database")
            logger.warning("   This is for testing/debugging purposes only")
            logger.warning("   Set test_single_product=False to create all products")
            logger.warning("=" * 80)
            products_to_create = catalog.products[:1]  # Only first product
        else:
            products_to_create = catalog.products  # All products

        product_tracker = ProductProgressTracker(job_id=job_id)
        logger.info(f"🏭 Creating {len(products_to_create)} products in database (discovered: {len(catalog.products)})...")

        product_db_ids = []  # Store created product IDs

        for i, product in enumerate(products_to_create, start=1):
            try:
                # Generate product_id (same format as in product_processor.py)
                product_id = f"product_{i}_{product.name.replace(' ', '_')}"

                # 1. Initialize product_progress tracking
                await product_tracker.initialize_product(
                    product_id=product_id,
                    product_name=product.name,
                    product_index=i,
                    metadata={
                        "page_range": product.page_range,
                        "confidence": product.confidence,
                        "description": product.description
                    }
                )

                # 2. Create product in database immediately
                logger.info(f"   🏭 [{i}/{len(catalog.products)}] Creating product in DB: {product.name}")
                product_creation_result = await create_single_product(
                    product=product,
                    document_id=document_id,
                    workspace_id=workspace_id,
                    job_id=job_id,
                    catalog=catalog,
                    supabase=supabase,
                    logger=logger
                )

                product_db_id = product_creation_result.get('product_id')
                product_db_ids.append(product_db_id)

                # 3. Update product_progress with DB ID
                await product_tracker.update_product_metadata(
                    product_id=product_id,
                    metadata={"product_db_id": product_db_id}
                )

                logger.info(f"   ✅ [{i}/{len(catalog.products)}] Created product: {product.name} (DB ID: {product_db_id})")

            except Exception as e:
                logger.error(f"   ❌ Failed to create product {product.name}: {e}")
                sentry_sdk.capture_exception(e)
                # Continue with other products even if one fails

        logger.info(f"✅ Created {len(product_db_ids)} products in database")

        # Store product DB IDs in checkpoint for later stages
        checkpoint_data["product_db_ids"] = product_db_ids

    # ============================================================
    # RETURN DATA: Include comprehensive metrics
    # ============================================================
    return {
        "catalog": catalog,
        "page_count": page_count,
        "file_size_mb": file_size_mb,
        "temp_pdf_path": temp_pdf_path,
        # NEW: Return Stage 0 metrics
        "products_discovered": products_discovered,
        "certificates_discovered": certificates_discovered,
        "logos_discovered": logos_discovered,
        "specifications_discovered": specifications_discovered,
        "total_entities": total_entities,
        "discovery_time_ms": discovery_time_ms,
        "discovery_model": catalog.model_used,
        "confidence_score": catalog.confidence_score
    }


