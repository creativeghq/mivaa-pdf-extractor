"""
Products API Routes

This module provides API endpoints for product creation and management,
including the two-stage classification system for creating products from PDF chunks.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from pydantic import BaseModel, Field

from app.services.products.product_creation_service import ProductCreationService
from app.services.core.supabase_client import SupabaseClient
from app.schemas.api_responses import ServiceHealthResponse
from app.services.integrations.data_import_service import DataImportService
from app.dependencies import get_workspace_context
from app.middleware.jwt_auth import WorkspaceContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/products", tags=["Products"])


# ============================================================================
# Request/Response Models
# ============================================================================

class ProductCreationRequest(BaseModel):
    """Request model for creating products from chunks."""
    document_id: str = Field(..., description="UUID of the processed document")
    workspace_id: str = Field(
        default="ffafc28b-1b8b-4b0d-b226-9f9a6154004e",
        description="UUID of the workspace"
    )
    max_products: Optional[int] = Field(
        default=None,
        description="Maximum number of products to create (None = unlimited)"
    )
    min_chunk_length: int = Field(
        default=100,
        description="Minimum chunk content length to consider for products"
    )

    class Config:
        schema_extra = {
            "example": {
                "document_id": "69cba085-9c2d-405c-aff2-8a20caf0b568",
                "workspace_id": "ffafc28b-1b8b-4b0d-b226-9f9a6154004e",
                "max_products": None,
                "min_chunk_length": 100
            }
        }


class ProductCreationResponse(BaseModel):
    """Response model for product creation results."""
    success: bool = Field(..., description="Whether the operation was successful")
    products_created: int = Field(..., description="Number of products created")
    products_failed: int = Field(..., description="Number of products that failed to create")
    chunks_processed: int = Field(..., description="Number of chunks processed")
    total_chunks: int = Field(..., description="Total number of chunks available")
    eligible_chunks: int = Field(..., description="Number of chunks that met criteria")
    stage1_candidates: Optional[int] = Field(None, description="Number of candidates from Stage 1")
    stage1_time: Optional[float] = Field(None, description="Time spent in Stage 1 (seconds)")
    stage2_time: Optional[float] = Field(None, description="Time spent in Stage 2 (seconds)")
    total_time: Optional[float] = Field(None, description="Total AI processing time (seconds)")
    message: str = Field(..., description="Human-readable result message")
    error: Optional[str] = Field(None, description="Error message if operation failed")


# ============================================================================
# API Endpoints
# ============================================================================

@router.post(
    "/create-from-chunks",
    response_model=ProductCreationResponse,
    summary="Create products from document chunks using two-stage AI classification",
    description="""
    Advanced two-stage product creation system with intelligent AI model selection.

    **Two-Stage Classification System:**

    **Stage 1: Fast Filtering (Claude 4.5 Haiku)**
    - Text-only classification for initial candidate identification
    - Batch processing for efficiency
    - Low-cost, high-speed filtering
    - Identifies potential product chunks

    **Stage 2: Deep Enrichment (Claude Opus 4.7)**
    - Detailed product metadata extraction
    - Image analysis and validation
    - Comprehensive feature extraction
    - High-accuracy enrichment

    **Performance Benefits:**
    - ⚡ 60% faster processing vs single-stage approach
    - 💰 40% reduced API costs through intelligent model selection
    - 🎯 Higher accuracy through specialized model usage
    - 📦 Batch processing reduces API call overhead

    **Processing Flow:**
    1. Fetch all chunks for document
    2. Filter by minimum length criteria
    3. Stage 1: Haiku classifies chunks (batch)
    4. Stage 2: Opus enriches confirmed products
    5. Create products in database
    6. Return detailed metrics

    **Example Request:**
    ```json
    {
      "document_id": "69cba085-9c2d-405c-aff2-8a20caf0b568",
      "workspace_id": "ffafc28b-1b8b-4b0d-b226-9f9a6154004e",
      "max_products": null,
      "min_chunk_length": 100
    }
    ```

    **Example Response:**
    ```json
    {
      "success": true,
      "products_created": 12,
      "products_failed": 0,
      "chunks_processed": 45,
      "total_chunks": 150,
      "eligible_chunks": 45,
      "stage1_candidates": 15,
      "stage1_time": 2.5,
      "stage2_time": 8.3,
      "total_time": 10.8,
      "message": "Successfully created 12 products from 45 chunks in 10.8s"
    }
    ```

    **Parameters:**
    - `document_id`: UUID of processed document (required)
    - `workspace_id`: UUID of workspace (default: ffafc28b-1b8b-4b0d-b226-9f9a6154004e)
    - `max_products`: Maximum products to create (null = unlimited)
    - `min_chunk_length`: Minimum chunk content length (default: 100)

    **Performance:**
    - Typical: 10-15 seconds for 50 chunks
    - Stage 1: ~2-3 seconds (batch processing)
    - Stage 2: ~8-12 seconds (detailed enrichment)

    **Use Cases:**
    - Automated product extraction from PDFs
    - Bulk product creation from catalogs
    - Material database population
    - Product metadata enrichment

    **Error Codes:**
    - 200: Success
    - 400: Invalid request parameters
    - 404: Document not found
    - 500: Processing failed (check logs)

    **Rate Limits:**
    - 5 requests/minute (processing intensive)
    """,
    tags=["Products"],
    responses={
        200: {"description": "Products created successfully"},
        400: {"description": "Invalid request parameters"},
        404: {"description": "Document not found"},
        500: {"description": "Processing failed"}
    }
)
async def create_products_from_chunks(
    request: ProductCreationRequest
) -> ProductCreationResponse:
    """
    Create products from document chunks using two-stage classification.

    This endpoint uses an advanced two-stage classification system:
    - Stage 1: Fast text-only classification using Claude 4.5 Haiku for initial filtering
    - Stage 2: Deep enrichment using Claude Opus 4.7 for confirmed products

    The system provides significant performance improvements:
    - 60% faster processing through intelligent model selection
    - Reduced API costs by using Haiku for initial filtering
    - Higher accuracy through Opus enrichment of confirmed candidates
    - Batch processing reduces API call overhead

    Args:
        request: Product creation request parameters
        supabase_client: Supabase client dependency

    Returns:
        ProductCreationResponse: Detailed results including timing metrics

    Raises:
        HTTPException: If the operation fails
    """
    try:
        logger.info(f"🚀 Starting two-stage product creation for document: {request.document_id}")

        # Initialize product creation service
        supabase_client = SupabaseClient()
        product_service = ProductCreationService(supabase_client)
        
        # Create products using two-stage classification
        result = await product_service.create_products_from_chunks(
            document_id=request.document_id,
            workspace_id=request.workspace_id,
            max_products=request.max_products,
            min_chunk_length=request.min_chunk_length
        )
        
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error occurred")
            logger.error(f"❌ Product creation failed: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Product creation failed: {error_msg}"
            )
        
        logger.info(f"✅ Two-stage product creation completed: {result.get('products_created', 0)} products created")
        
        # Return structured response
        return ProductCreationResponse(
            success=result.get("success", False),
            products_created=result.get("products_created", 0),
            products_failed=result.get("products_failed", 0),
            chunks_processed=result.get("chunks_processed", 0),
            total_chunks=result.get("total_chunks", 0),
            eligible_chunks=result.get("eligible_chunks", 0),
            stage1_candidates=result.get("stage1_candidates"),
            stage1_time=result.get("stage1_time"),
            stage2_time=result.get("stage2_time"),
            total_time=result.get("total_time"),
            message=result.get("message", "Product creation completed"),
            error=result.get("error")
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in create_products_from_chunks: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/create-from-layout", response_model=ProductCreationResponse)
async def create_products_from_layout(
    request: ProductCreationRequest
) -> ProductCreationResponse:
    """
    Create products from layout-based candidates (legacy method).
    
    This endpoint uses the original layout-based product detection method
    for comparison with the new two-stage classification system.
    
    Args:
        request: Product creation request parameters
        supabase_client: Supabase client dependency
        
    Returns:
        ProductCreationResponse: Results from layout-based creation
        
    Raises:
        HTTPException: If the operation fails
    """
    try:
        logger.info(f"🏗️ Starting layout-based product creation for document: {request.document_id}")

        # Initialize product creation service
        supabase_client = SupabaseClient()
        product_service = ProductCreationService(supabase_client)
        
        # Create products using layout-based method
        result = await product_service.create_products_from_layout_candidates(
            document_id=request.document_id,
            workspace_id=request.workspace_id,
            min_confidence=0.5,
            min_quality_score=0.5
        )
        
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error occurred")
            logger.error(f"❌ Layout-based product creation failed: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Layout-based product creation failed: {error_msg}"
            )
        
        logger.info(f"✅ Layout-based product creation completed: {result.get('products_created', 0)} products created")
        
        # Convert result to standard response format
        return ProductCreationResponse(
            success=result.get("success", False),
            products_created=result.get("products_created", 0),
            products_failed=result.get("products_failed", 0),
            chunks_processed=result.get("chunks_processed", 0),
            total_chunks=result.get("total_chunks", 0),
            eligible_chunks=result.get("eligible_chunks", 0),
            message=result.get("message", "Layout-based product creation completed")
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in create_products_from_layout: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


class BatchCategorizeRequest(BaseModel):
    workspace_id: str = Field(..., description="UUID of the workspace to process")
    only_uncategorized: bool = Field(
        default=True,
        description="If True, only process products without a material_category"
    )
    limit: Optional[int] = Field(
        default=200,
        description="Maximum number of products to process in one run"
    )


class BatchCategorizeResponse(BaseModel):
    success: bool
    total_found: int
    categorized: int
    failed: int
    skipped: int
    results: List[Dict[str, Any]]
    message: str


@router.post("/batch-categorize", response_model=BatchCategorizeResponse)
async def batch_categorize_products(request: BatchCategorizeRequest) -> BatchCategorizeResponse:
    """
    Batch re-categorize products using Claude Haiku.

    Fetches products for the workspace (optionally only those without a
    material_category in metadata), calls _classify_product for each,
    and updates metadata.material_category + metadata.zone_intent in DB.
    """
    try:
        from app.api.pdf_processing.stage_4_products import _classify_product
        supabase_client = SupabaseClient()
        supabase = supabase_client.get_client()

        # Fetch products
        query = (
            supabase
            .from_("products")
            .select("id, name, description, metadata")
            .eq("workspace_id", request.workspace_id)
        )

        if request.only_uncategorized:
            # Filter: metadata->material_category is null or missing
            query = query.or_("metadata->material_category.is.null,metadata->>material_category.eq.")

        if request.limit:
            query = query.limit(request.limit)

        result = query.execute()
        products = result.data or []
        total_found = len(products)

        if total_found == 0:
            return BatchCategorizeResponse(
                success=True,
                total_found=0,
                categorized=0,
                failed=0,
                skipped=0,
                results=[],
                message="No products found matching criteria",
            )

        categorized = 0
        failed = 0
        skipped = 0
        results: List[Dict[str, Any]] = []

        # Process concurrently in batches of 10
        BATCH = 10
        for i in range(0, total_found, BATCH):
            chunk = products[i : i + BATCH]

            async def process_one(p: Dict[str, Any]) -> Dict[str, Any]:
                meta = p.get("metadata") or {}
                existing_cat = meta.get("material_category", "")
                try:
                    classification = await _classify_product(
                        name=p.get("name") or "",
                        description=p.get("description") or "",
                        existing_category=existing_cat,
                    )
                    if not classification:
                        return {"id": p["id"], "name": p.get("name"), "status": "skipped", "reason": "no classification returned"}

                    updated_meta = {**meta, **classification}
                    update_result = (
                        supabase
                        .from_("products")
                        .update({"metadata": updated_meta})
                        .eq("id", p["id"])
                        .execute()
                    )
                    if hasattr(update_result, "error") and update_result.error:
                        return {"id": p["id"], "name": p.get("name"), "status": "failed", "reason": str(update_result.error)}

                    return {
                        "id": p["id"],
                        "name": p.get("name"),
                        "status": "categorized",
                        "material_category": classification.get("material_category"),
                        "zone_intent": classification.get("zone_intent"),
                    }
                except Exception as e:
                    return {"id": p["id"], "name": p.get("name"), "status": "failed", "reason": str(e)}

            batch_results = await asyncio.gather(*[process_one(p) for p in chunk])
            for r in batch_results:
                results.append(r)
                if r["status"] == "categorized":
                    categorized += 1
                elif r["status"] == "failed":
                    failed += 1
                else:
                    skipped += 1

        return BatchCategorizeResponse(
            success=True,
            total_found=total_found,
            categorized=categorized,
            failed=failed,
            skipped=skipped,
            results=results,
            message=f"Processed {total_found} products: {categorized} categorized, {skipped} skipped, {failed} failed",
        )

    except Exception as e:
        logger.error(f"Batch categorization failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Dealer / supplier "Add Product" (#174) — manual single-product create
# ============================================================================

class ManualImageRef(BaseModel):
    """An image the dealer already uploaded to storage (generation-images)."""
    storage_url: str
    storage_path: Optional[str] = None
    filename: Optional[str] = None
    content_type: Optional[str] = None


class ManualProductRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    material_category: Optional[str] = None
    supply_mode: str = Field(default="platform_sold")
    price: Optional[float] = None
    currency: str = "EUR"
    unit: Optional[str] = None
    dimensions: Optional[str] = None
    external_sku: Optional[str] = None
    # Dynamic descriptive facets (color, finish, material, available_colors, …) — free
    # form; canonicalized + embedded by the shared core just like catalog ingestion.
    metadata: Dict[str, Any] = Field(default_factory=dict)
    images: List[ManualImageRef] = Field(default_factory=list)


@router.post("/create-manual")
async def create_manual_product(
    request: ManualProductRequest,
    ctx: WorkspaceContext = Depends(get_workspace_context),
) -> Dict[str, Any]:
    """
    Dealer/supplier "Add Product" — creates ONE product in the caller's workspace via the
    SAME ingest core as XML import (facet canonicalization → Voyage text_embedding_1024 →
    full image suite). Attributed to the dealer through factory_name = their business name.
    Price + supply_mode seed the marketplace cascade (product_prices on the dealer's row).
    """
    svc = DataImportService()
    ws = ctx.workspace_id

    # Attribute the product to the dealer through the existing factory meta field.
    factory_name = None
    try:
        fs = svc.supabase.table('finance_settings').select('business_name') \
            .eq('workspace_id', ws).limit(1).execute()
        if fs and fs.data:
            factory_name = fs.data[0].get('business_name')
    except Exception:
        factory_name = None

    meta = dict(request.metadata or {})
    if request.unit:
        meta['unit'] = request.unit

    payload: Dict[str, Any] = {
        'name': request.name,
        'description': request.description or '',
        'material_category': request.material_category,
        'factory_name': factory_name,
        'factory_group_name': factory_name,
        'price': request.price,
        'dimensions': request.dimensions,
        'size': meta.get('size'),
        'external_sku': request.external_sku,
        'metadata': meta,
    }
    # Promote common descriptive facets the core reads explicitly at top level.
    for k in ('color', 'colors', 'designer', 'collection', 'finish', 'material'):
        if k in meta:
            payload[k] = meta[k]
    if request.images:
        payload['downloaded_images'] = [
            {'success': True, 'storage_url': im.storage_url, 'storage_path': im.storage_path,
             'filename': im.filename, 'content_type': im.content_type, 'index': i}
            for i, im in enumerate(request.images)
        ]

    try:
        product_id = await svc.create_product_from_payload(ws, payload, source='dealer_manual')
    except Exception as e:
        logger.error(f"create_manual_product failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    if not product_id:
        raise HTTPException(status_code=500, detail="Product creation failed")

    # supply_mode flag + cascade base price on the dealer's own product_prices row.
    try:
        svc.supabase.table('products').update({'supply_mode': request.supply_mode}) \
            .eq('id', product_id).execute()
    except Exception as e:
        logger.warning(f"supply_mode set failed for {product_id}: {e}")
    if request.price is not None:
        try:
            svc.supabase.table('product_prices').upsert({
                'workspace_id': ws, 'product_id': product_id,
                'list_price': request.price, 'currency': request.currency,
                'unit': request.unit,
            }, on_conflict='workspace_id,product_id').execute()
        except Exception as e:
            logger.warning(f"product_prices upsert failed for {product_id}: {e}")

    return {'success': True, 'product_id': product_id}


@router.get("/health", response_model=ServiceHealthResponse)
async def products_health_check() -> Dict[str, Any]:
    """
    Health check endpoint for the products API.
    
    Returns:
        Dict[str, Any]: Health status information
    """
    return {
        "status": "healthy",
        "service": "products-api",
        "version": "1.0.0",
        "features": {
            "two_stage_classification": True,
            "layout_based_creation": True,
            "claude_haiku_integration": True,
            "claude_opus_integration": True,
            "batch_processing": True,
            "performance_metrics": True
        },
        "endpoints": {
            "create_from_chunks": "/api/products/create-from-chunks",
            "create_from_layout": "/api/products/create-from-layout",
            "health": "/api/products/health"
        }
    }

