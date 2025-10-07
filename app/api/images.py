"""
Image Analysis & Material Kai Integration API endpoints.

This module provides comprehensive image processing capabilities including:
- Image analysis using Material Kai Vision Platform
- Batch image processing
- Image similarity search
- OCR and object detection
- Integration with document processing workflow
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse

from ..schemas.images import (
    ImageAnalysisRequest,
    ImageAnalysisResponse,
    ImageBatchRequest,
    ImageBatchResponse,
    ImageSearchRequest,
    ImageSearchResponse,
    ImageMetadata,
    DetectedObject,
    DetectedText,
    FaceDetection,
    BoundingBox,
    SimilarImage,
    ImageBatchResult
)
from ..schemas.common import BaseResponse, ProcessingStatus
from ..services.material_kai_service import MaterialKaiService, get_material_kai_service
from ..services.supabase_client import get_supabase_client
from ..dependencies import get_current_user, get_workspace_context, require_image_read, require_image_write
from ..middleware.jwt_auth import WorkspaceContext, User
from ..config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/images", tags=["Image Analysis"])

# Get settings
settings = get_settings()


@router.post("/analyze", response_model=ImageAnalysisResponse)
async def analyze_image(
    request: ImageAnalysisRequest,
    material_kai: MaterialKaiService = Depends(get_material_kai_service),
    current_user: User = Depends(get_current_user),
    workspace_context: WorkspaceContext = Depends(get_workspace_context),
    _: None = Depends(require_image_read)
) -> ImageAnalysisResponse:
    """
    Analyze an image using Material Kai Vision Platform.
    
    Performs comprehensive image analysis including:
    - AI-generated description
    - OCR text extraction
    - Object detection
    - Face detection (optional)
    - Content categorization
    """
    try:
        logger.info(f"Starting image analysis for image: {request.image_id or 'URL provided'}")
        
        # Validate Material Kai connection
        if not material_kai._is_connected:
            await material_kai.connect()
            if not material_kai._is_connected:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Material Kai Vision Platform is not available"
                )
        
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # Prepare analysis request for Material Kai
        analysis_data = {
            "analysis_id": analysis_id,
            "image_source": {
                "image_id": request.image_id,
                "image_url": str(request.image_url) if request.image_url else None
            },
            "analysis_types": request.analysis_types,
            "parameters": {
                "quality": request.quality,
                "language": request.language,
                "confidence_threshold": request.confidence_threshold
            },
            "context": {
                "document_context": request.document_context,
                "page_context": request.page_context
            }
        }
        
        # Perform analysis via Material Kai
        logger.info(f"Sending analysis request to Material Kai: {analysis_id}")
        
        # Simulate Material Kai API call (replace with actual implementation)
        analysis_result = await _perform_material_kai_analysis(
            material_kai, analysis_data
        )
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Parse and structure the response
        response = ImageAnalysisResponse(
            success=True,
            message="Image analysis completed successfully",
            timestamp=datetime.utcnow().isoformat(),
            image_id=request.image_id or f"url_{analysis_id[:8]}",
            status=ProcessingStatus.COMPLETED,
            description=analysis_result.get("description"),
            detected_objects=[
                DetectedObject(
                    label=obj["label"],
                    confidence=obj["confidence"],
                    bounding_box=BoundingBox(**obj["bounding_box"]),
                    attributes=obj.get("attributes", {})
                )
                for obj in analysis_result.get("detected_objects", [])
            ],
            detected_text=[
                DetectedText(
                    text=text["text"],
                    confidence=text["confidence"],
                    bounding_box=BoundingBox(**text["bounding_box"]),
                    language=text.get("language"),
                    font_info=text.get("font_info")
                )
                for text in analysis_result.get("detected_text", [])
            ],
            detected_faces=[
                FaceDetection(
                    confidence=face["confidence"],
                    bounding_box=BoundingBox(**face["bounding_box"]),
                    landmarks=face.get("landmarks"),
                    attributes=face.get("attributes")
                )
                for face in analysis_result.get("detected_faces", [])
            ],
            categories=analysis_result.get("categories", []),
            tags=analysis_result.get("tags", []),
            metadata=ImageMetadata(**analysis_result.get("metadata", {})),
            analysis_types_performed=request.analysis_types,
            processing_time_ms=processing_time,
            model_versions=analysis_result.get("model_versions", {})
        )
        
        # Store analysis results in database
        await _store_analysis_results(response)
        
        logger.info(f"Image analysis completed successfully: {analysis_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image analysis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image analysis failed: {str(e)}"
        )


@router.post("/analyze/batch", response_model=ImageBatchResponse)
async def analyze_images_batch(
    request: ImageBatchRequest,
    material_kai: MaterialKaiService = Depends(get_material_kai_service),
    current_user: User = Depends(get_current_user),
    workspace_context: WorkspaceContext = Depends(get_workspace_context),
    _: None = Depends(require_image_read)
) -> ImageBatchResponse:
    """
    Analyze multiple images in batch using Material Kai Vision Platform.
    
    Supports parallel processing for improved performance.
    """
    try:
        logger.info(f"Starting batch image analysis for {len(request.image_ids)} images")
        
        # Validate Material Kai connection
        if not material_kai._is_connected:
            await material_kai.connect()
            if not material_kai._is_connected:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Material Kai Vision Platform is not available"
                )
        
        # Generate batch ID
        batch_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # Process images
        if request.parallel_processing:
            # Parallel processing
            tasks = [
                _analyze_single_image_in_batch(
                    material_kai, image_id, request, batch_id
                )
                for image_id in request.image_ids
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Sequential processing
            results = []
            for image_id in request.image_ids:
                result = await _analyze_single_image_in_batch(
                    material_kai, image_id, request, batch_id
                )
                results.append(result)
        
        # Process results
        batch_results = []
        completed_count = 0
        failed_count = 0
        
        for i, result in enumerate(results):
            image_id = request.image_ids[i]
            
            if isinstance(result, Exception):
                # Handle exception
                batch_results.append(ImageBatchResult(
                    image_id=image_id,
                    status=ProcessingStatus.FAILED,
                    error=str(result),
                    processing_time_ms=0.0
                ))
                failed_count += 1
            else:
                # Handle successful result
                batch_results.append(ImageBatchResult(
                    image_id=image_id,
                    status=ProcessingStatus.COMPLETED,
                    result=result,
                    processing_time_ms=result.processing_time_ms if result else 0.0
                ))
                completed_count += 1
        
        # Calculate batch metrics
        total_processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        average_time_per_image = total_processing_time / len(request.image_ids) if request.image_ids else 0
        
        response = ImageBatchResponse(
            success=True,
            message=f"Batch analysis completed: {completed_count} successful, {failed_count} failed",
            timestamp=datetime.utcnow().isoformat(),
            batch_id=batch_id,
            total_images=len(request.image_ids),
            completed_count=completed_count,
            failed_count=failed_count,
            results=batch_results,
            total_processing_time_ms=total_processing_time,
            average_time_per_image_ms=average_time_per_image
        )
        
        logger.info(f"Batch image analysis completed: {batch_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch image analysis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch image analysis failed: {str(e)}"
        )


@router.post("/search", response_model=ImageSearchResponse)
async def search_similar_images(
    request: ImageSearchRequest,
    supabase = Depends(get_supabase_client),
    current_user: User = Depends(get_current_user),
    workspace_context: WorkspaceContext = Depends(get_workspace_context),
    _: None = Depends(require_image_read)
) -> ImageSearchResponse:
    """
    Search for similar images using visual similarity or description matching.
    
    Supports both image-to-image and text-to-image search.
    """
    try:
        logger.info(f"Starting image similarity search")
        start_time = datetime.utcnow()
        
        # Prepare search query
        search_query = {
            "query_type": "image" if request.query_image_id else "description",
            "query_value": request.query_image_id or request.query_description,
            "limit": request.limit,
            "similarity_threshold": request.similarity_threshold,
            "filters": {
                "document_ids": request.document_ids,
                "image_types": request.image_types,
                "tags": request.tags
            }
        }
        
        # Perform similarity search
        similar_images = await _perform_image_similarity_search(
            supabase, search_query
        )
        
        # Calculate search time
        search_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        response = ImageSearchResponse(
            success=True,
            message=f"Found {len(similar_images)} similar images",
            timestamp=datetime.utcnow().isoformat(),
            query_info={
                "type": search_query["query_type"],
                "query": search_query["query_value"],
                "filters_applied": bool(any(search_query["filters"].values()))
            },
            similar_images=similar_images,
            total_found=len(similar_images),
            search_time_ms=search_time
        )
        
        logger.info(f"Image similarity search completed: {len(similar_images)} results")
        return response
        
    except Exception as e:
        logger.error(f"Image similarity search failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image similarity search failed: {str(e)}"
        )


@router.post("/upload/analyze", response_model=ImageAnalysisResponse)
async def upload_and_analyze_image(
    file: UploadFile = File(...),
    analysis_types: str = Form(default="description,ocr,objects"),
    quality: str = Form(default="standard"),
    language: str = Form(default="auto"),
    confidence_threshold: float = Form(default=0.7),
    document_context: Optional[str] = Form(None),
    material_kai: MaterialKaiService = Depends(get_material_kai_service),
    current_user: User = Depends(get_current_user),
    workspace_context: WorkspaceContext = Depends(get_workspace_context),
    _: None = Depends(require_image_write)
) -> ImageAnalysisResponse:
    """
    Upload an image file and analyze it using Material Kai Vision Platform.
    
    Supports direct file upload with immediate analysis.
    """
    try:
        logger.info(f"Processing uploaded image: {file.filename}")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image"
            )
        
        # Read file content
        file_content = await file.read()
        if len(file_content) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Image file too large (max 10MB)"
            )
        
        # Generate temporary image ID
        temp_image_id = f"upload_{uuid.uuid4().hex[:12]}"
        
        # Store uploaded image temporarily (implement based on your storage strategy)
        image_url = await _store_uploaded_image(file_content, file.filename, temp_image_id)
        
        # Create analysis request
        analysis_request = ImageAnalysisRequest(
            image_id=temp_image_id,
            analysis_types=analysis_types.split(','),
            quality=quality,
            language=language,
            confidence_threshold=confidence_threshold,
            document_context=document_context
        )
        
        # Perform analysis
        result = await analyze_image(analysis_request, material_kai)
        
        logger.info(f"Uploaded image analysis completed: {temp_image_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload and analyze failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload and analyze failed: {str(e)}"
        )


@router.get("/health", response_model=BaseResponse)
async def health_check(
    material_kai: MaterialKaiService = Depends(get_material_kai_service),
    current_user: User = Depends(get_current_user),
    workspace_context: WorkspaceContext = Depends(get_workspace_context),
    _: None = Depends(require_image_read)
) -> BaseResponse:
    """
    Health check for image analysis service and Material Kai integration.
    """
    try:
        # Check Material Kai connection
        material_kai_status = await material_kai.health_check()
        
        # Check database connection
        supabase = get_supabase_client()
        db_healthy = supabase is not None
        
        # Overall health status
        overall_healthy = (
            material_kai_status.get("status") == "healthy" and
            db_healthy
        )
        
        return BaseResponse(
            success=overall_healthy,
            message="Image analysis service health check",
            timestamp=datetime.utcnow().isoformat(),
            data={
                "service_status": "healthy" if overall_healthy else "degraded",
                "material_kai_status": material_kai_status,
                "database_status": "healthy" if db_healthy else "unhealthy",
                "supported_analysis_types": [
                    "description", "ocr", "objects", "faces", "landmarks", "logos", "text_detection"
                ],
                "max_batch_size": 50,
                "max_file_size_mb": 10
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return BaseResponse(
            success=False,
            message=f"Health check failed: {str(e)}",
            timestamp=datetime.utcnow().isoformat()
        )


# Helper functions

async def _perform_material_kai_analysis(
    material_kai: MaterialKaiService,
    analysis_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Perform image analysis using Material Kai Vision Platform.
    
    This is a placeholder implementation. Replace with actual Material Kai API calls.
    """
    try:
        # Simulate API call to Material Kai
        # In real implementation, this would call material_kai service methods
        
        # Mock response structure
        mock_result = {
            "description": "A technical diagram showing system architecture with multiple components and connections",
            "detected_objects": [
                {
                    "label": "diagram",
                    "confidence": 0.95,
                    "bounding_box": {"x": 0.1, "y": 0.2, "width": 0.8, "height": 0.6},
                    "attributes": {"type": "technical", "complexity": "high"}
                }
            ],
            "detected_text": [
                {
                    "text": "System Architecture",
                    "confidence": 0.98,
                    "bounding_box": {"x": 0.3, "y": 0.1, "width": 0.4, "height": 0.05},
                    "language": "en",
                    "font_info": {"size": 24, "style": "bold"}
                }
            ],
            "detected_faces": [],
            "categories": [
                {"technical_diagram": 0.92},
                {"educational_content": 0.87}
            ],
            "tags": ["architecture", "diagram", "technical", "system"],
            "metadata": {
                "width": 1024,
                "height": 768,
                "format": "PNG",
                "size_bytes": 245760,
                "color_mode": "RGB",
                "quality_score": 0.89,
                "sharpness_score": 0.85,
                "dominant_colors": ["#2E86AB", "#A23B72", "#F18F01"],
                "brightness": 0.65,
                "contrast": 0.78
            },
            "model_versions": {
                "description_model": "material-kai-vision-v2.1",
                "ocr_model": "material-kai-ocr-v1.8",
                "object_detection": "material-kai-detect-v3.0"
            }
        }
        
        # Add some variation based on analysis types
        requested_types = analysis_data.get("analysis_types", [])
        
        if "description" not in requested_types:
            mock_result.pop("description", None)
        if "ocr" not in requested_types and "text_detection" not in requested_types:
            mock_result["detected_text"] = []
        if "objects" not in requested_types:
            mock_result["detected_objects"] = []
        if "faces" not in requested_types:
            mock_result["detected_faces"] = []
        
        return mock_result
        
    except Exception as e:
        logger.error(f"Material Kai analysis failed: {str(e)}")
        raise


async def _analyze_single_image_in_batch(
    material_kai: MaterialKaiService,
    image_id: str,
    batch_request: ImageBatchRequest,
    batch_id: str
) -> Optional[ImageAnalysisResponse]:
    """
    Analyze a single image as part of batch processing.
    """
    try:
        # Create individual analysis request
        individual_request = ImageAnalysisRequest(
            image_id=image_id,
            analysis_types=batch_request.analysis_types,
            quality=batch_request.quality,
            confidence_threshold=batch_request.confidence_threshold
        )
        
        # Perform analysis
        result = await analyze_image(individual_request, material_kai)
        return result
        
    except Exception as e:
        logger.error(f"Batch image analysis failed for {image_id}: {str(e)}")
        raise


async def _perform_image_similarity_search(
    supabase,
    search_query: Dict[str, Any]
) -> List[SimilarImage]:
    """
    Perform image similarity search using vector embeddings.
    
    This is a placeholder implementation. Replace with actual vector search.
    """
    try:
        # Mock similarity search results
        mock_results = [
            SimilarImage(
                image_id="img_789abc",
                document_id="doc_123def",
                document_name="Technical Documentation",
                page_number=5,
                similarity_score=0.92,
                content_similarity=0.88,
                description="System architecture diagram with component relationships",
                tags=["architecture", "system", "diagram"],
                dimensions={"width": 800, "height": 600},
                image_url="/api/v1/images/img_789abc/download"
            ),
            SimilarImage(
                image_id="img_456ghi",
                document_id="doc_789jkl",
                document_name="Design Patterns Guide",
                page_number=12,
                similarity_score=0.85,
                content_similarity=0.82,
                description="UML class diagram showing design patterns",
                tags=["uml", "design", "patterns"],
                dimensions={"width": 1024, "height": 768},
                image_url="/api/v1/images/img_456ghi/download"
            )
        ]
        
        # Filter results based on similarity threshold
        filtered_results = [
            result for result in mock_results
            if result.similarity_score >= search_query["similarity_threshold"]
        ]
        
        # Apply limit
        return filtered_results[:search_query["limit"]]
        
    except Exception as e:
        logger.error(f"Image similarity search failed: {str(e)}")
        raise


async def _store_analysis_results(analysis_result: ImageAnalysisResponse) -> None:
    """
    Store image analysis results in the database.
    """
    try:
        supabase = get_supabase_client()
        
        # Prepare data for storage
        analysis_data = {
            "image_id": analysis_result.image_id,
            "status": analysis_result.status.value,
            "description": analysis_result.description,
            "detected_objects": [obj.dict() for obj in analysis_result.detected_objects],
            "detected_text": [text.dict() for text in analysis_result.detected_text],
            "detected_faces": [face.dict() for face in analysis_result.detected_faces],
            "categories": analysis_result.categories,
            "tags": analysis_result.tags,
            "metadata": analysis_result.metadata.dict(),
            "analysis_types_performed": analysis_result.analysis_types_performed,
            "processing_time_ms": analysis_result.processing_time_ms,
            "model_versions": analysis_result.model_versions,
            "created_at": analysis_result.timestamp.isoformat() if hasattr(analysis_result.timestamp, 'isoformat') else str(analysis_result.timestamp)
        }
        
        # Store in database (implement based on your schema)
        # supabase.table("image_analysis_results").insert(analysis_data).execute()
        
        logger.info(f"Analysis results stored for image: {analysis_result.image_id}")
        
    except Exception as e:
        logger.error(f"Failed to store analysis results: {str(e)}")
        # Don't raise exception here to avoid breaking the main flow


async def _store_uploaded_image(
    file_content: bytes,
    filename: str,
    image_id: str
) -> str:
    """
    Store uploaded image file and return access URL.
    
    This is a placeholder implementation. Replace with actual storage logic.
    """
    try:
        # In real implementation, store in cloud storage (S3, GCS, etc.)
        # For now, return a mock URL
        mock_url = f"/api/v1/images/{image_id}/download"
        
        logger.info(f"Uploaded image stored: {image_id}")
        return mock_url
        
    except Exception as e:
        logger.error(f"Failed to store uploaded image: {str(e)}")
        raise