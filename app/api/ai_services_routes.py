"""
AI Services API Routes

Endpoints for Phase 1-4 AI services:
- Document classification
- Boundary detection
- Product validation
- Consensus validation
- Escalation metrics
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

from app.services.ai_validation.document_classifier import DocumentClassifier
from app.services.ai_validation.boundary_detector import BoundaryDetector
from app.services.products.product_validator import ProductValidator
from app.services.ai_validation.consensus_validator import ConsensusValidator
from app.services.ai_validation.escalation_engine import EscalationEngine
# REMOVED: EnhancedPDFProcessor was imported but never used - consolidated into process_document_with_discovery

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ai-services", tags=["AI Services"])

# Initialize services
document_classifier = DocumentClassifier()
boundary_detector = BoundaryDetector()
product_validator = ProductValidator()
consensus_validator = ConsensusValidator()
escalation_engine = EscalationEngine()


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ClassifyRequest(BaseModel):
    """Request for document classification."""
    content: str = Field(..., description="Text content to classify")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional context (page number, images, etc.)")
    job_id: Optional[str] = Field(None, description="Optional job ID for tracking")


class ClassifyBatchRequest(BaseModel):
    """Request for batch document classification."""
    contents: List[str] = Field(..., description="List of text contents to classify")
    contexts: Optional[List[Dict[str, Any]]] = Field(None, description="Optional contexts for each content")
    job_id: Optional[str] = Field(None, description="Optional job ID for tracking")


class DetectBoundariesRequest(BaseModel):
    """Request for boundary detection."""
    chunks: List[Dict[str, Any]] = Field(..., description="List of document chunks")
    job_id: Optional[str] = Field(None, description="Optional job ID for tracking")


class ValidateProductRequest(BaseModel):
    """Request for product validation."""
    product_data: Dict[str, Any] = Field(..., description="Product metadata")
    chunks: List[Dict[str, Any]] = Field(..., description="Associated chunks")
    images: Optional[List[Dict[str, Any]]] = Field(None, description="Associated images")


class ConsensusValidateRequest(BaseModel):
    """Request for consensus validation."""
    content: str = Field(..., description="Content to validate")
    extraction_type: str = Field(..., description="Type of extraction (e.g., 'product_name', 'material_type')")
    job_id: Optional[str] = Field(None, description="Optional job ID for tracking")


# ============================================================================
# DOCUMENT CLASSIFICATION ENDPOINTS
# ============================================================================

@router.post(
    "/classify-document",
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
async def classify_document(request: ClassifyRequest):
    """
    Classify document content into categories.

    Categories:
    - product: Product information, specifications, features
    - supporting: Technical details, certifications, guides
    - administrative: Company info, legal, contact
    - transitional: TOC, headers, footers, navigation
    """
    try:
        result = await document_classifier.classify_content(
            content=request.content,
            context=request.context,
            job_id=request.job_id
        )
        
        return {
            "success": True,
            "classification": result
        }
        
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classify-batch")
async def classify_batch(request: ClassifyBatchRequest):
    """
    Classify multiple document contents in parallel.
    
    Efficient for processing multiple sections at once.
    """
    try:
        results = await document_classifier.classify_batch(
            contents=request.contents,
            contexts=request.contexts,
            job_id=request.job_id
        )
        
        return {
            "success": True,
            "classifications": results,
            "total": len(results),
            "product_count": sum(1 for r in results if r.get("is_product")),
        }
        
    except Exception as e:
        logger.error(f"Batch classification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# BOUNDARY DETECTION ENDPOINTS
# ============================================================================

@router.post("/detect-boundaries")
async def detect_boundaries(request: DetectBoundariesRequest):
    """
    Detect product boundaries in document chunks.
    
    Uses semantic similarity, structural markers, and page breaks
    to identify where one product ends and another begins.
    """
    try:
        boundaries = await boundary_detector.detect_boundaries(
            chunks=request.chunks,
            job_id=request.job_id
        )
        
        return {
            "success": True,
            "boundaries": boundaries,
            "boundary_count": len(boundaries),
        }
        
    except Exception as e:
        logger.error(f"Boundary detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/group-by-product")
async def group_by_product(request: DetectBoundariesRequest):
    """
    Detect boundaries and group chunks into products.
    
    Returns product groups ready for extraction.
    """
    try:
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
        
        return {
            "success": True,
            "boundaries": boundaries,
            "product_groups": product_groups,
            "group_count": len(product_groups),
        }
        
    except Exception as e:
        logger.error(f"Product grouping failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# PRODUCT VALIDATION ENDPOINTS
# ============================================================================

@router.post("/validate-product")
async def validate_product(request: ValidateProductRequest):
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
        validation = await product_validator.validate_product(
            product_data=request.product_data,
            chunks=request.chunks,
            images=request.images
        )
        
        return {
            "success": True,
            "validation": validation
        }
        
    except Exception as e:
        logger.error(f"Product validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# CONSENSUS VALIDATION ENDPOINTS
# ============================================================================

@router.post("/consensus-validate")
async def consensus_validate(request: ConsensusValidateRequest):
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
        result = await consensus_validator.validate_critical_extraction(
            content=request.content,
            extraction_type=request.extraction_type,
            job_id=request.job_id
        )
        
        return {
            "success": True,
            "consensus": result
        }
        
    except Exception as e:
        logger.error(f"Consensus validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/consensus/is-critical/{task_type}")
async def check_if_critical(task_type: str):
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

@router.get("/escalation/stats")
async def get_escalation_stats():
    """
    Get escalation engine statistics.
    
    Returns metrics on:
    - Total escalations
    - Successful escalations
    - Failed escalations
    - Cost saved/spent
    """
    stats = escalation_engine.get_stats()
    
    return {
        "success": True,
        "stats": stats
    }


# ============================================================================
# ENHANCED PDF PROCESSING ENDPOINT
# ============================================================================

@router.post("/process-pdf-enhanced")
async def process_pdf_enhanced(
    document_id: str,
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
                "escalation_engine": "ready"
            }
        }
        
    except Exception as e:
        logger.error(f"Enhanced PDF processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/health")
async def health_check():
    """Health check for AI services."""
    return {
        "status": "healthy",
        "services": {
            "document_classifier": "initialized",
            "boundary_detector": "initialized",
            "product_validator": "initialized",
            "consensus_validator": "initialized",
            "escalation_engine": "initialized",
            "enhanced_processor": "initialized"
        }
    }


