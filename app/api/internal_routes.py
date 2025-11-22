"""
Internal API Routes - Modular endpoints for PDF processing pipeline stages.

These endpoints are called internally by the main orchestrator to execute
individual stages of the PDF processing pipeline. Each endpoint is focused
on a single responsibility and can be tested/debugged independently.

Endpoints:
- POST /api/internal/classify-images/{job_id} - Classify images as material/non-material
- POST /api/internal/upload-images/{job_id} - Upload material images to storage
- POST /api/internal/save-images-db/{job_id} - Save images to DB and generate CLIP embeddings
- POST /api/internal/detect-products/{job_id} - Product discovery
- POST /api/internal/generate-chunks/{job_id} - Text chunking
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

from app.services.image_processing_service import ImageProcessingService
from app.services.chunking_service import ChunkingService
from app.services.relevancy_service import RelevancyService
from app.services.supabase_client import get_supabase_client
from app.services.job_tracker import JobTracker
from app.models.ai_config import AIModelConfig, DEFAULT_AI_CONFIG

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/internal",
    tags=["Internal Pipeline Stages"],
    responses={
        500: {"description": "Internal server error"},
        404: {"description": "Job not found"}
    }
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _extract_product_text(full_text: str, page_range: List[int]) -> str:
    """
    Extract text for a specific product based on page range.

    Args:
        full_text: Full PDF text
        page_range: List of page numbers for this product

    Returns:
        Product-specific text
    """
    if not page_range:
        return full_text

    # Split text by page markers (if available)
    # For now, return full text - can be enhanced with page-specific extraction
    return full_text


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ClassifyImagesRequest(BaseModel):
    """Request model for image classification."""
    job_id: str
    extracted_images: List[Dict[str, Any]]
    ai_config: Optional[AIModelConfig] = None  # Uses DEFAULT_AI_CONFIG if not provided


class ClassifyImagesResponse(BaseModel):
    """Response model for image classification."""
    success: bool
    material_images: List[Dict[str, Any]]
    non_material_images: List[Dict[str, Any]]
    total_classified: int
    material_count: int
    non_material_count: int


class UploadImagesRequest(BaseModel):
    """Request model for image upload."""
    job_id: str
    material_images: List[Dict[str, Any]]
    document_id: str


class UploadImagesResponse(BaseModel):
    """Response model for image upload."""
    success: bool
    uploaded_images: List[Dict[str, Any]]
    uploaded_count: int
    failed_count: int


class SaveImagesRequest(BaseModel):
    """Request model for saving images to DB and generating visual embeddings."""
    job_id: str
    material_images: List[Dict[str, Any]]
    document_id: str
    workspace_id: str
    ai_config: Optional[AIModelConfig] = None  # Uses DEFAULT_AI_CONFIG if not provided


class SaveImagesResponse(BaseModel):
    """Response model for saving images."""
    success: bool
    images_saved: int
    clip_embeddings_generated: int


class CreateChunksRequest(BaseModel):
    """Request model for creating chunks."""
    job_id: str
    document_id: str
    workspace_id: str
    extracted_text: str
    product_ids: Optional[List[str]] = None
    chunk_size: int = 512
    chunk_overlap: int = 50
    ai_config: Optional[AIModelConfig] = None  # Uses DEFAULT_AI_CONFIG if not provided


class CreateChunksResponse(BaseModel):
    """Response model for creating chunks."""
    success: bool
    chunks_created: int
    embeddings_generated: int
    relationships_created: int


class CreateRelationshipsRequest(BaseModel):
    """Request model for creating relationships."""
    job_id: str
    document_id: str
    product_ids: List[str]
    similarity_threshold: float = 0.5


class ExtractMetadataRequest(BaseModel):
    """Request model for metadata extraction."""
    job_id: str
    document_id: str
    product_ids: List[str]
    pdf_text: str
    ai_config: Optional[AIModelConfig] = None  # Uses DEFAULT_AI_CONFIG if not provided


class ExtractMetadataResponse(BaseModel):
    """Response model for metadata extraction."""
    success: bool
    products_enriched: int
    metadata_fields_extracted: int
    extraction_method: str
    model_used: str


class CreateRelationshipsResponse(BaseModel):
    """Response model for creating relationships."""
    success: bool
    chunk_image_relationships: int
    product_image_relationships: int


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/classify-images/{job_id}", response_model=ClassifyImagesResponse)
async def classify_images(
    job_id: str,
    request: ClassifyImagesRequest
):
    """
    Classify images as material or non-material using Llama Vision + Claude validation.
    
    This endpoint:
    1. Uses Llama Vision for fast initial classification
    2. Validates uncertain cases (confidence < threshold) with Claude Sonnet
    3. Returns separated lists of material and non-material images
    
    Args:
        job_id: Job ID for tracking
        request: Classification request with extracted images
        
    Returns:
        ClassifyImagesResponse with material and non-material images
    """
    try:
        # Use provided AI config or default
        ai_config = request.ai_config or DEFAULT_AI_CONFIG

        logger.info(f"🤖 [Job {job_id}] Starting image classification for {len(request.extracted_images)} images")
        logger.info(f"   AI Config: Primary={ai_config.classification_primary_model}, Validation={ai_config.classification_validation_model}, Threshold={ai_config.classification_confidence_threshold}")

        # Initialize tracker
        tracker = JobTracker(job_id)
        await tracker.update_stage("IMAGE_CLASSIFICATION", 0, sync_to_db=True)

        # Initialize service
        image_service = ImageProcessingService()

        # Classify images with AI config
        material_images, non_material_images = await image_service.classify_images(
            extracted_images=request.extracted_images,
            confidence_threshold=ai_config.classification_confidence_threshold,
            primary_model=ai_config.classification_primary_model,
            validation_model=ai_config.classification_validation_model
        )
        
        # Update tracker
        await tracker.update_stage(
            "IMAGE_CLASSIFICATION",
            100,
            metadata={'material_count': len(material_images), 'non_material_count': len(non_material_images)},
            sync_to_db=True
        )
        
        logger.info(f"✅ [Job {job_id}] Classification complete: {len(material_images)} material, {len(non_material_images)} non-material")
        
        return ClassifyImagesResponse(
            success=True,
            material_images=material_images,
            non_material_images=non_material_images,
            total_classified=len(material_images) + len(non_material_images),
            material_count=len(material_images),
            non_material_count=len(non_material_images)
        )
    
    except Exception as e:
        logger.error(f"❌ [Job {job_id}] Image classification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Image classification failed: {str(e)}")


@router.post("/upload-images/{job_id}", response_model=UploadImagesResponse)
async def upload_images(
    job_id: str,
    request: UploadImagesRequest
):
    """
    Upload material images to Supabase Storage.

    This endpoint:
    1. Uploads material images to Supabase Storage bucket
    2. Returns uploaded images with storage URLs

    Args:
        job_id: Job ID for tracking
        request: Upload request with material images

    Returns:
        UploadImagesResponse with uploaded images
    """
    try:
        logger.info(f"📤 [Job {job_id}] Starting upload for {len(request.material_images)} material images")

        # Initialize tracker
        tracker = JobTracker(job_id)
        await tracker.update_stage("IMAGE_UPLOAD", 0, sync_to_db=True)

        # Initialize service
        image_service = ImageProcessingService()

        # Upload images
        uploaded_images = await image_service.upload_images_to_storage(
            material_images=request.material_images,
            document_id=request.document_id
        )

        failed_count = len(request.material_images) - len(uploaded_images)

        # Update tracker
        await tracker.update_stage(
            "IMAGE_UPLOAD",
            100,
            metadata={'uploaded_count': len(uploaded_images), 'failed_count': failed_count},
            sync_to_db=True
        )

        logger.info(f"✅ [Job {job_id}] Upload complete: {len(uploaded_images)} uploaded, {failed_count} failed")

        return UploadImagesResponse(
            success=True,
            uploaded_images=uploaded_images,
            uploaded_count=len(uploaded_images),
            failed_count=failed_count
        )

    except Exception as e:
        logger.error(f"❌ [Job {job_id}] Image upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Image upload failed: {str(e)}")


@router.post("/save-images-db/{job_id}", response_model=SaveImagesResponse)
async def save_images_to_db(
    job_id: str,
    request: SaveImagesRequest
):
    """
    Save images to database and generate CLIP embeddings.

    This endpoint:
    1. Saves images to document_images table
    2. Generates CLIP embeddings (visual + specialized)
    3. Saves embeddings to database table AND VECS collection

    Args:
        job_id: Job ID for tracking
        request: Save request with material images

    Returns:
        SaveImagesResponse with counts
    """
    try:
        logger.info(f"💾 [Job {job_id}] Starting DB save and CLIP generation for {len(request.material_images)} images")

        # Initialize tracker
        tracker = JobTracker(job_id)
        await tracker.update_stage("IMAGE_SAVE_AND_CLIP", 0, sync_to_db=True)

        # Initialize service
        image_service = ImageProcessingService()

        # Save images and generate CLIP embeddings
        result = await image_service.save_images_and_generate_clips(
            material_images=request.material_images,
            document_id=request.document_id,
            workspace_id=request.workspace_id
        )

        # Update tracker
        await tracker.update_stage(
            "IMAGE_SAVE_AND_CLIP",
            100,
            metadata={
                'images_saved': result['images_saved'],
                'clip_embeddings_generated': result['clip_embeddings_generated']
            },
            sync_to_db=True
        )

        logger.info(f"✅ [Job {job_id}] DB save complete: {result['images_saved']} saved, {result['clip_embeddings_generated']} CLIP embeddings")

        return SaveImagesResponse(
            success=True,
            images_saved=result['images_saved'],
            clip_embeddings_generated=result['clip_embeddings_generated']
        )

    except Exception as e:
        logger.error(f"❌ [Job {job_id}] Image save and CLIP generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Image save and CLIP generation failed: {str(e)}")


@router.post("/create-chunks/{job_id}", response_model=CreateChunksResponse)
async def create_chunks(
    job_id: str,
    request: CreateChunksRequest
):
    """
    Create semantic chunks and generate text embeddings.

    This endpoint:
    1. Creates semantic chunks from extracted text
    2. Generates text embeddings for each chunk
    3. Creates chunk-to-product relationships

    Args:
        job_id: Job ID for tracking
        request: Chunking request with extracted text

    Returns:
        CreateChunksResponse with counts
    """
    try:
        logger.info(f"📝 [Job {job_id}] Starting chunking for document {request.document_id}")

        # Initialize tracker
        tracker = JobTracker(job_id)
        await tracker.update_stage("CHUNKING", 0, sync_to_db=True)

        # Initialize service
        chunking_service = ChunkingService()

        # Create chunks and embeddings
        result = await chunking_service.create_chunks_and_embeddings(
            document_id=request.document_id,
            workspace_id=request.workspace_id,
            extracted_text=request.extracted_text,
            product_ids=request.product_ids,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )

        # Update tracker
        await tracker.update_stage(
            "CHUNKING",
            100,
            metadata={
                'chunks_created': result['chunks_created'],
                'embeddings_generated': result['embeddings_generated'],
                'relationships_created': result['relationships_created']
            },
            sync_to_db=True
        )

        logger.info(f"✅ [Job {job_id}] Chunking complete: {result['chunks_created']} chunks, {result['embeddings_generated']} embeddings")

        return CreateChunksResponse(
            success=True,
            chunks_created=result['chunks_created'],
            embeddings_generated=result['embeddings_generated'],
            relationships_created=result['relationships_created']
        )

    except Exception as e:
        logger.error(f"❌ [Job {job_id}] Chunking failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chunking failed: {str(e)}")


@router.post("/create-relationships/{job_id}", response_model=CreateRelationshipsResponse)
async def create_relationships(
    job_id: str,
    request: CreateRelationshipsRequest
):
    """
    Create all relationships between chunks, images, and products.

    This endpoint:
    1. Creates chunk-to-image relationships based on embedding similarity
    2. Creates product-to-image relationships based on page ranges

    Args:
        job_id: Job ID for tracking
        request: Relationships request

    Returns:
        CreateRelationshipsResponse with counts
    """
    try:
        logger.info(f"🔗 [Job {job_id}] Starting relationship creation for document {request.document_id}")

        # Initialize tracker
        tracker = JobTracker(job_id)
        await tracker.update_stage("RELATIONSHIPS", 0, sync_to_db=True)

        # Initialize service
        relevancy_service = RelevancyService()

        # Create all relationships
        result = await relevancy_service.create_all_relationships(
            document_id=request.document_id,
            product_ids=request.product_ids,
            similarity_threshold=request.similarity_threshold
        )

        # Update tracker
        await tracker.update_stage(
            "RELATIONSHIPS",
            100,
            metadata={
                'chunk_image_relationships': result['chunk_image_relationships'],
                'product_image_relationships': result['product_image_relationships']
            },
            sync_to_db=True
        )

        logger.info(f"✅ [Job {job_id}] Relationships complete: {result['chunk_image_relationships']} chunk-image, {result['product_image_relationships']} product-image")

        return CreateRelationshipsResponse(
            success=True,
            chunk_image_relationships=result['chunk_image_relationships'],
            product_image_relationships=result['product_image_relationships']
        )

    except Exception as e:
        logger.error(f"❌ [Job {job_id}] Relationship creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-metadata/{job_id}", response_model=ExtractMetadataResponse)
async def extract_metadata(
    job_id: str,
    request: ExtractMetadataRequest
):
    """
    Extract comprehensive metadata from PDF text for products.

    This endpoint:
    1. Extracts product-specific text from PDF
    2. Uses AI (Claude or GPT) to extract structured metadata
    3. Enriches product records with extracted metadata
    4. Tracks which model was used and extraction method

    Args:
        job_id: Job ID for tracking
        request: Metadata extraction request with product IDs and PDF text

    Returns:
        ExtractMetadataResponse with extraction results
    """
    try:
        # Use provided AI config or default
        ai_config = request.ai_config or DEFAULT_AI_CONFIG

        logger.info(f"📋 [Job {job_id}] Starting metadata extraction for {len(request.product_ids)} products")
        logger.info(f"   AI Config: Model={ai_config.metadata_extraction_model}, Temperature={ai_config.metadata_temperature}, MaxTokens={ai_config.metadata_max_tokens}")

        # Initialize tracker
        tracker = JobTracker(job_id)
        await tracker.update_stage("METADATA_EXTRACTION", 0, sync_to_db=True)

        # Initialize metadata extractor
        from app.services.dynamic_metadata_extractor import DynamicMetadataExtractor
        metadata_extractor = DynamicMetadataExtractor(
            model=ai_config.metadata_extraction_model,
            job_id=job_id
        )

        # Get products from database
        supabase = get_supabase_client()
        products_response = supabase.table('products').select('*').in_('id', request.product_ids).execute()
        products = products_response.data

        total_metadata_fields = 0
        enriched_count = 0

        # Extract metadata for each product
        for i, product in enumerate(products):
            try:
                # Extract product-specific text from page range
                product_text = _extract_product_text(request.pdf_text, product.get('page_range', []))

                # Get category hint from existing metadata
                category_hint = product.get('metadata', {}).get('category') or product.get('metadata', {}).get('material')

                # Extract comprehensive metadata
                logger.info(f"   🔍 Extracting metadata for: {product.get('name')}")
                extracted = await metadata_extractor.extract_metadata(
                    pdf_text=product_text,
                    category_hint=category_hint
                )

                # Merge extracted metadata with existing metadata
                existing_metadata = product.get('metadata', {})
                enriched_metadata = {
                    **extracted.get("discovered", {}),  # Lowest priority
                    **extracted.get("critical", {}),    # Medium priority
                    **existing_metadata,                 # Highest priority (from discovery)
                    "_extraction_metadata": extracted.get("metadata", {})
                }

                # Update product in database
                supabase.table('products').update({
                    'metadata': enriched_metadata
                }).eq('id', product['id']).execute()

                enriched_count += 1
                total_metadata_fields += len(extracted.get("critical", {})) + len(extracted.get("discovered", {}))

                # Update progress
                progress = int((i + 1) / len(products) * 100)
                await tracker.update_stage("METADATA_EXTRACTION", progress, sync_to_db=True)

            except Exception as e:
                logger.error(f"   ❌ Failed to extract metadata for product {product.get('id')}: {e}")
                continue

        # Update tracker
        await tracker.update_stage(
            "METADATA_EXTRACTION",
            100,
            metadata={
                'products_enriched': enriched_count,
                'metadata_fields_extracted': total_metadata_fields,
                'model_used': ai_config.metadata_extraction_model
            },
            sync_to_db=True
        )

        logger.info(f"✅ [Job {job_id}] Metadata extraction complete: {enriched_count} products enriched, {total_metadata_fields} fields extracted")

        return ExtractMetadataResponse(
            success=True,
            products_enriched=enriched_count,
            metadata_fields_extracted=total_metadata_fields,
            extraction_method=f"ai_dynamic_{ai_config.metadata_extraction_model}",
            model_used=ai_config.metadata_extraction_model
        )

    except Exception as e:
        logger.error(f"❌ [Job {job_id}] Metadata extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Relationship creation failed: {str(e)}")

