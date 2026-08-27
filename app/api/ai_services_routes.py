"""
AI Services API Routes

Endpoints for Phase 1-4 AI services:
- Document classification
- Boundary detection
- Product validation
- Consensus validation
- Escalation metrics

Auth (audit #23 M10-2, and #22 M9-1): every model-calling route below declares its
own ``Depends(get_workspace_context)``. The prefix is not in
``JWTAuthMiddleware.exclude_paths``, so a token was already required — but invariant 5
requires the route to gate itself rather than rely on the middleware alone, and these
routes had no dependency of any kind. ``/health`` stays open: the ``health-check`` edge
function polls it and it reads nothing.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

from app.dependencies import current_user_id, get_current_user, get_workspace_context
from app.schemas.auth import WorkspaceContext
from app.services.integrations.credit_router import debit_credits, refund_credits
from app.utils.paid_door import metered_door

from app.schemas.api_responses import (
    DataResponse, ClassificationResponse, BatchClassificationResponse,
    BoundaryDetectionResponse, ProductGroupingResponse, ValidationResponse,
    ServiceHealthResponse,
)
from app.services.ai_validation.document_classifier import DocumentClassifier
from app.services.ai_validation.boundary_detector import BoundaryDetector
from app.services.products.product_validator import ProductValidator
from app.services.ai_validation.consensus_validator import ConsensusValidator
# REMOVED: EnhancedPDFProcessor was imported but never used - consolidated into process_document_with_discovery

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ai-services", tags=["AI Services"])

# ============================================================================
# INPUT BOUNDS  (audit #23 M10-2 / M10-3)
# ============================================================================
# Every request model below reaches a provider once per array element, so the
# array length IS the bill. `classify_batch` is the sharpest: it `asyncio.gather`s
# one Claude call per item with no semaphore, so an N-element array is N
# concurrent calls. These caps are the amplification bound, not a taste
# preference — without them one authenticated POST buys arbitrary spend on the
# platform's Anthropic account.
#
# The per-item character caps matter for the same reason: length drives input
# tokens, so an array that is short but whose items are megabytes is the same
# attack with a different shape.
MAX_BATCH_CONTENTS = 50        # → 50 parallel Claude calls, the hard ceiling per request
MAX_CONTENT_CHARS = 100_000    # ≈ 25k tokens, well past any real page or section
MAX_CHUNKS = 200               # → one embedding call per chunk, sequential
MAX_IMAGES = 100


# ============================================================================
# PRICE  (audit #23 M10-4)
# ============================================================================
# Capping the arrays bounded the amplification; it did not make anyone pay. None of
# these doors debited or rate-limited before its provider call, so the cost landed in
# `ai_usage_logs` looking perfectly healthy while nobody was ever charged — the same
# shape audit #18 M5-3 found on thirteen monitoring doors.
#
# The price scales with the work because the work scales with the request: one Claude
# call per item on the batch door, one embedding per chunk on the others. A flat price
# would make the 50-item call the cheap way to buy 50 Claude calls.
CREDIT_PER_CLASSIFY = 1        # one Haiku call
CREDIT_PER_CHUNK_BATCH = 20    # chunks per credit — embeddings are ~an order cheaper
CREDIT_CONSENSUS = 3           # two voters, one of them Opus (M10-3)


def _chunk_cost(n_chunks: int) -> int:
    """At least one credit, then one per CREDIT_PER_CHUNK_BATCH chunks.

    Never zero: a debit of zero is refused by the credit router, and a door that asks
    for nothing is a door that charges nothing.
    """
    return max(1, -(-n_chunks // CREDIT_PER_CHUNK_BATCH))

# Initialize services
document_classifier = DocumentClassifier()
boundary_detector = BoundaryDetector()
product_validator = ProductValidator()
consensus_validator = ConsensusValidator()


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ClassifyRequest(BaseModel):
    """Request for document classification."""
    content: str = Field(..., max_length=MAX_CONTENT_CHARS, description="Text content to classify")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional context (page number, images, etc.)")
    job_id: Optional[str] = Field(None, description="Optional job ID for tracking")


class ClassifyBatchRequest(BaseModel):
    """Request for batch document classification."""
    contents: List[str] = Field(
        ..., min_length=1, max_length=MAX_BATCH_CONTENTS,
        description=f"List of text contents to classify (max {MAX_BATCH_CONTENTS}; one Claude call each, issued in parallel)",
    )
    contexts: Optional[List[Dict[str, Any]]] = Field(
        None, max_length=MAX_BATCH_CONTENTS, description="Optional contexts for each content",
    )
    job_id: Optional[str] = Field(None, description="Optional job ID for tracking")


class DetectBoundariesRequest(BaseModel):
    """Request for boundary detection."""
    chunks: List[Dict[str, Any]] = Field(
        ..., min_length=1, max_length=MAX_CHUNKS,
        description=f"List of document chunks (max {MAX_CHUNKS}; one embedding call each)",
    )
    job_id: Optional[str] = Field(None, description="Optional job ID for tracking")


class ValidateProductRequest(BaseModel):
    """Request for product validation."""
    product_data: Dict[str, Any] = Field(..., description="Product metadata")
    chunks: List[Dict[str, Any]] = Field(
        ..., min_length=1, max_length=MAX_CHUNKS,
        description=f"Associated chunks (max {MAX_CHUNKS}; one embedding call each)",
    )
    images: Optional[List[Dict[str, Any]]] = Field(
        None, max_length=MAX_IMAGES, description="Associated images",
    )


class ConsensusValidateRequest(BaseModel):
    """Request for consensus validation."""
    content: str = Field(..., max_length=MAX_CONTENT_CHARS, description="Content to validate")
    extraction_type: str = Field(..., max_length=200, description="Type of extraction (e.g., 'product_name', 'material_type')")
    job_id: Optional[str] = Field(None, description="Optional job ID for tracking")


# ============================================================================
# DOCUMENT CLASSIFICATION ENDPOINTS
# ============================================================================

@router.post(
    "/classify-document",
    response_model=ClassificationResponse,
    summary="Classify document content into semantic categories",
    description="""
    AI-powered content classification using Claude 4.5 Haiku for fast, accurate categorization.

    **Classification Categories:**

    **1. Product** (`product`)
    - Product information and specifications
    - Features and benefits
    - Technical specifications
    - Product variants and options
    - Pricing and availability

    **2. Supporting Information** (`supporting`)
    - Technical details and guides
    - Certifications and compliance
    - Installation instructions
    - Maintenance guides
    - Warranty information

    **3. Administrative** (`administrative`)
    - Company information
    - Legal notices and disclaimers
    - Contact information
    - Terms and conditions
    - Privacy policies

    **4. Transitional** (`transitional`)
    - Table of contents
    - Headers and footers
    - Page numbers
    - Navigation elements
    - Section dividers

    **Example Request:**
    ```json
    {
      "content": "NOVA Oak Flooring - Premium engineered wood flooring with 3mm oak veneer...",
      "context": {
        "page_number": 5,
        "has_images": true,
        "section": "Products"
      },
      "job_id": "job_123"
    }
    ```

    **Example Response:**
    ```json
    {
      "success": true,
      "classification": {
        "category": "product",
        "confidence": 0.95,
        "reasoning": "Content contains product name, specifications, and features",
        "metadata": {
          "product_indicators": ["specifications", "features", "pricing"],
          "quality_score": 0.92
        }
      }
    }
    ```

    **Use Cases:**
    - PDF content categorization
    - Automated document organization
    - Product extraction pipelines
    - Content filtering

    **Performance:**
    - Typical: 300-500ms per classification
    - Batch processing: Use `/classify-batch` for better performance

    **Accuracy:**
    - Average confidence: 0.87
    - Fallback rate: <5%

    **Rate Limits:**
    - 60 requests/minute

    **Error Codes:**
    - 200: Success
    - 400: Invalid content or parameters
    - 500: Classification failed
    """,
    tags=["AI Services"],
    responses={
        200: {"description": "Classification successful"},
        400: {"description": "Invalid request"},
        500: {"description": "Classification failed"}
    }
)
async def classify_document(
    request: ClassifyRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    workspace: WorkspaceContext = Depends(get_workspace_context),
):
    """
    Classify document content into categories.

    Categories:
    - product: Product information, specifications, features
    - supporting: Technical details, certifications, guides
    - administrative: Company info, legal, contact
    - transitional: TOC, headers, footers, navigation
    """
    try:
        async with metered_door(
            user_id=current_user_id(user),
            workspace_id=workspace.workspace_id,
            cost=CREDIT_PER_CLASSIFY,
            operation_type="ai_services.classify_document",
            debit=debit_credits, refund=refund_credits,
        ) as paid:
            result = await document_classifier.classify_content(
                content=request.content,
                context=request.context,
                job_id=request.job_id
            )
            if not result:
                paid.refund("the classifier returned nothing")

        return {
            "success": True,
            "classification": result,
            "credits_debited": paid.charged,
        }
        
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classify-batch", response_model=BatchClassificationResponse)
async def classify_batch(
    request: ClassifyBatchRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    workspace: WorkspaceContext = Depends(get_workspace_context),
):
    """
    Classify multiple document contents in parallel.
    
    Efficient for processing multiple sections at once.
    """
    try:
        # Priced per item, because it IS per item: one Claude call each, in parallel.
        async with metered_door(
            user_id=current_user_id(user),
            workspace_id=workspace.workspace_id,
            cost=CREDIT_PER_CLASSIFY * len(request.contents),
            operation_type="ai_services.classify_batch",
            debit=debit_credits, refund=refund_credits,
        ) as paid:
            results = await document_classifier.classify_batch(
                contents=request.contents,
                contexts=request.contexts,
                job_id=request.job_id
            )
            if not results:
                paid.refund("the batch classifier returned nothing")

        return {
            "success": True,
            "classifications": results,
            "total": len(results),
            "product_count": sum(1 for r in results if r.get("is_product")),
            "credits_debited": paid.charged,
        }
        
    except Exception as e:
        logger.error(f"Batch classification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# BOUNDARY DETECTION ENDPOINTS
# ============================================================================

@router.post("/detect-boundaries", response_model=BoundaryDetectionResponse)
async def detect_boundaries(
    request: DetectBoundariesRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    workspace: WorkspaceContext = Depends(get_workspace_context),
):
    """
    Detect product boundaries in document chunks.
    
    Uses semantic similarity, structural markers, and page breaks
    to identify where one product ends and another begins.
    """
    try:
        async with metered_door(
            user_id=current_user_id(user),
            workspace_id=workspace.workspace_id,
            cost=_chunk_cost(len(request.chunks)),
            operation_type="ai_services.detect_boundaries",
            debit=debit_credits, refund=refund_credits,
        ) as paid:
            boundaries = await boundary_detector.detect_boundaries(
                chunks=request.chunks,
                job_id=request.job_id
            )

        return {
            "success": True,
            "boundaries": boundaries,
            "boundary_count": len(boundaries),
            "credits_debited": paid.charged,
        }
        
    except Exception as e:
        logger.error(f"Boundary detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/group-by-product", response_model=ProductGroupingResponse)
async def group_by_product(
    request: DetectBoundariesRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    workspace: WorkspaceContext = Depends(get_workspace_context),
):
    """
    Detect boundaries and group chunks into products.
    
    Returns product groups ready for extraction.
    """
    try:
        async with metered_door(
            user_id=current_user_id(user),
            workspace_id=workspace.workspace_id,
            cost=_chunk_cost(len(request.chunks)),
            operation_type="ai_services.group_by_product",
            debit=debit_credits, refund=refund_credits,
        ) as paid:
            # Detect boundaries
            boundaries = await boundary_detector.detect_boundaries(
                chunks=request.chunks,
                job_id=request.job_id
            )

            # Group chunks
            product_groups = await boundary_detector.group_chunks_by_product(
                chunks=request.chunks,
                boundaries=boundaries
            )
            if not product_groups:
                paid.refund("grouping produced no products")

        return {
            "success": True,
            "boundaries": boundaries,
            "product_groups": product_groups,
            "group_count": len(product_groups),
            "credits_debited": paid.charged,
        }
        
    except Exception as e:
        logger.error(f"Product grouping failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# PRODUCT VALIDATION ENDPOINTS
# ============================================================================

@router.post("/validate-product", response_model=ValidationResponse)
async def validate_product(
    request: ValidateProductRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    workspace: WorkspaceContext = Depends(get_workspace_context),
):
    """
    Validate product extraction quality.
    
    Checks:
    - Minimum content requirements
    - Substantive content (not just headers/footers)
    - Distinguishing features
    - Associated assets (images, specs)
    - Semantic coherence
    """
    try:
        async with metered_door(
            user_id=current_user_id(user),
            workspace_id=workspace.workspace_id,
            cost=_chunk_cost(len(request.chunks)),
            operation_type="ai_services.validate_product",
            debit=debit_credits, refund=refund_credits,
        ) as paid:
            validation = await product_validator.validate_product(
                product_data=request.product_data,
                chunks=request.chunks,
                images=request.images
            )
            if not validation:
                paid.refund("validation returned nothing")

        return {
            "success": True,
            "validation": validation,
            "credits_debited": paid.charged,
        }
        
    except Exception as e:
        logger.error(f"Product validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# CONSENSUS VALIDATION ENDPOINTS
# ============================================================================

@router.post("/consensus-validate", response_model=ValidationResponse)
async def consensus_validate(
    request: ConsensusValidateRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    workspace: WorkspaceContext = Depends(get_workspace_context),
):
    """
    Validate extraction using multi-model consensus.
    
    Runs 2-3 models in parallel and uses weighted voting
    to determine the most accurate result.
    
    Critical for:
    - Product name extraction
    - Material classification
    - Safety information
    - Compliance data
    - Technical specifications
    - Pricing data
    """
    try:
        # Two voters per request, one of them Opus — priced accordingly (M10-3).
        async with metered_door(
            user_id=current_user_id(user),
            workspace_id=workspace.workspace_id,
            cost=CREDIT_CONSENSUS,
            operation_type="ai_services.consensus_validate",
            debit=debit_credits, refund=refund_credits,
        ) as paid:
            result = await consensus_validator.validate_critical_extraction(
                content=request.content,
                extraction_type=request.extraction_type,
                job_id=request.job_id
            )
            if not result:
                paid.refund("consensus produced no verdict")

        return {
            "success": True,
            "consensus": result,
            "credits_debited": paid.charged,
        }
        
    except Exception as e:
        logger.error(f"Consensus validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/consensus/is-critical/{task_type}", response_model=DataResponse)
async def check_if_critical(
    task_type: str,
    workspace: WorkspaceContext = Depends(get_workspace_context),
):
    """
    Check if a task type requires consensus validation.
    
    Returns True for critical tasks like product_name_extraction,
    material_classification, safety_information, etc.
    """
    is_critical = ConsensusValidator.is_critical_task(task_type)
    
    return {
        "task_type": task_type,
        "is_critical": is_critical,
        "critical_tasks": list(ConsensusValidator.CRITICAL_TASKS)
    }


# ============================================================================
# ESCALATION METRICS ENDPOINTS
# ============================================================================

# GET /escalation/stats and the EscalationEngine behind it were deleted in audit
# #12. Neither execute_with_escalation nor execute_with_fallback had a single
# caller anywhere in the repo, so the only reachable part of a 321-line engine was
# this endpoint — reporting counters that could not be anything but zero, forever.
# That is the platform's own silent-zero shape: a metric that looks healthy
# because nothing is producing it. The confidence thresholds it used
# (app/config/confidence_thresholds.py) stay; unified_chunking_service uses them.


# ============================================================================
# ENHANCED PDF PROCESSING ENDPOINT
# ============================================================================

@router.post("/process-pdf-enhanced", response_model=DataResponse)
async def process_pdf_enhanced(
    document_id: str,
    workspace: WorkspaceContext = Depends(get_workspace_context),
    job_id: Optional[str] = None,
    background_tasks: BackgroundTasks = None
):
    """
    Process PDF with all enhanced AI services.
    
    This endpoint is for testing the full integration.
    In production, use the main PDF processing endpoint.
    """
    try:
        # This would need the PDF bytes - for now return info
        return {
            "success": True,
            "message": "Enhanced PDF processing endpoint ready",
            "services": {
                "document_classifier": "ready",
                "boundary_detector": "ready",
                "product_validator": "ready",
                "consensus_validator": "ready",
            }
        }
        
    except Exception as e:
        logger.error(f"Enhanced PDF processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/health", response_model=ServiceHealthResponse)
async def health_check():
    """Health check for AI services."""
    return {
        "status": "healthy",
        "services": {
            "document_classifier": "initialized",
            "boundary_detector": "initialized",
            "product_validator": "initialized",
            "consensus_validator": "initialized",
            "enhanced_processor": "initialized"
        }
    }


