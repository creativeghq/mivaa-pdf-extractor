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
- POST /api/internal/create-chunks/{job_id} - Text chunking with duplicate prevention
- POST /api/internal/create-relationships/{job_id} - Create chunk-image and product-image relationships
- POST /api/internal/extract-metadata/{job_id} - Extract product metadata using AI
- POST /api/internal/generate-product-embeddings - Generate embeddings for products without them
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

from app.services.images.image_processing_service import ImageProcessingService
from app.services.chunking.unified_chunking_service import UnifiedChunkingService, ChunkingConfig, ChunkingStrategy
from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
from app.services.search.relevancy_service import RelevancyService
from app.services.core.supabase_client import get_supabase_client, SupabaseClient
from app.services.tracking.progress_tracker import ProgressTracker
from app.services.core.async_queue_service import AsyncQueueService
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
    skipped: Optional[bool] = False
    existing_chunks: Optional[int] = 0
    existing_embeddings: Optional[int] = 0


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
    Classify images as material or non-material using Qwen Vision + Claude validation.

    This endpoint:
    1. Uses vision model for fast initial classification
    2. Validates uncertain cases (confidence < threshold) with Claude Sonnet
    3. Returns separated lists of material and non-material images

    AI Configuration (Optional):
    - classification_primary_model: Primary classification model (default: Qwen3-VL-8B)
    - classification_validation_model: Validation model (default: Claude Sonnet 4.5)
    - classification_confidence_threshold: Threshold for validation (default: 0.7)
    - classification_temperature: Temperature setting (default: 0.1)
    - classification_max_tokens: Max tokens for responses (default: 512)

    Example with custom AI config:
    ```json
    {
      "job_id": "abc123",
      "extracted_images": [...],
      "ai_config": {
        "classification_primary_model": "Qwen/Qwen3-VL-8B-Instruct",
        "classification_validation_model": "claude-sonnet-4-20250514",
        "classification_confidence_threshold": 0.8
      }
    }
    ```

    Args:
        job_id: Job ID for tracking
        request: Classification request with extracted images and optional AI config

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
    Save images to database and generate visual embeddings (SigLIP/CLIP).

    This endpoint:
    1. Saves images to document_images table
    2. Generates visual embeddings using SigLIP (primary) or CLIP (fallback)
    3. Creates 5 specialized embeddings per image (visual, color, texture, style, material)
    4. Saves embeddings to database table AND VECS collection

    AI Configuration (Optional):
    - visual_embedding_primary: Primary visual model (default: SigLIP ViT-SO400M)
    - visual_embedding_fallback: Fallback visual model (default: CLIP ViT-B/32)

    Example with custom AI config:
    ```json
    {
      "job_id": "abc123",
      "material_images": [...],
      "document_id": "doc123",
      "workspace_id": "ws123",
      "ai_config": {
        "visual_embedding_primary": "google/siglip-so400m-patch14-384",
        "visual_embedding_fallback": "openai/clip-vit-base-patch32"
      }
    }
    ```

    Args:
        job_id: Job ID for tracking
        request: Save request with material images and optional AI config

    Returns:
        SaveImagesResponse with counts and model used
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
    request: CreateChunksRequest,
    supabase_client: SupabaseClient = Depends(get_supabase_client)
):
    """
    Create semantic chunks and generate text embeddings.

    This endpoint:
    1. Creates semantic chunks from extracted text
    2. Saves chunks to database
    3. Generates text embeddings for each chunk
    4. Creates chunk-to-product relationships
    5. **Prevents duplicates** - skips if chunks already exist

    **Use Cases:**
    - Regenerate chunks after text extraction updates
    - Create chunks for documents that failed chunking
    - Manual chunk generation for testing

    Args:
        job_id: Job ID for tracking
        request: Chunking request with extracted text
        supabase_client: Supabase client for database operations

    Returns:
        CreateChunksResponse with counts
    """
    try:
        logger.info(f"📝 [Job {job_id}] Starting chunking for document {request.document_id}")

        # Initialize tracker
        tracker = JobTracker(job_id)
        await tracker.update_stage("CHUNKING", 0, sync_to_db=True)

        # ✅ Check if chunks already exist for this document (duplicate prevention)
        existing_chunks = supabase_client.client.table('document_chunks')\
            .select('id')\
            .eq('document_id', request.document_id)\
            .limit(1)\
            .execute()

        if existing_chunks.data and len(existing_chunks.data) > 0:
            logger.info(f"   ⏭️ [Job {job_id}] Skipping - document already has chunks")

            # Count existing embeddings
            existing_embeddings = supabase_client.client.table('document_chunks')\
                .select('id')\
                .eq('document_id', request.document_id)\
                .not_('text_embedding', 'is', None)\
                .execute()

            return CreateChunksResponse(
                success=True,
                chunks_created=0,
                embeddings_generated=0,
                relationships_created=0,
                skipped=True,
                existing_chunks=len(existing_chunks.data),
                existing_embeddings=len(existing_embeddings.data) if existing_embeddings.data else 0
            )

        # Initialize chunking service with hybrid strategy
        chunking_config = ChunkingConfig(
            strategy=ChunkingStrategy.HYBRID,
            max_chunk_size=request.chunk_size,
            overlap_size=request.chunk_overlap
        )
        chunking_service = UnifiedChunkingService(chunking_config)

        # Create chunks using chunk_pages() (preferred method for page metadata)
        # Convert text to pages format for consistency
        pages = [{'metadata': {'page': 0}, 'text': request.extracted_text}]
        chunks = await chunking_service.chunk_pages(
            pages=pages,
            document_id=request.document_id,
            metadata={
                'workspace_id': request.workspace_id,
                'product_ids': request.product_ids
            }
        )

        logger.info(f"   Created {len(chunks)} chunks")

        # Initialize embedding service
        embedding_service = RealEmbeddingsService()

        chunks_created = 0
        embeddings_generated = 0
        relationships_created = 0

        # Save chunks to database and generate embeddings
        for chunk in chunks:
            try:
                # Save chunk to database
                chunk_record = {
                    'document_id': request.document_id,
                    'workspace_id': request.workspace_id,
                    'content': chunk.content,
                    'chunk_index': chunk.chunk_index,
                    'metadata': chunk.metadata,
                    'quality_score': chunk.quality_score
                }

                result = supabase_client.client.table('document_chunks').insert(chunk_record).execute()

                if result.data and len(result.data) > 0:
                    chunks_created += 1
                    chunk_id = result.data[0]['id']

                    # Generate text embedding
                    embedding = await embedding_service.generate_text_embedding(chunk.content)

                    if embedding:
                        # Update chunk with embedding
                        supabase_client.client.table('document_chunks')\
                            .update({'text_embedding': embedding})\
                            .eq('id', chunk_id)\
                            .execute()
                        embeddings_generated += 1

                    # Create chunk-to-product relationships if product_ids provided
                    if request.product_ids:
                        for product_id in request.product_ids:
                            relationship = {
                                'chunk_id': chunk_id,
                                'product_id': product_id,
                                'workspace_id': request.workspace_id
                            }
                            supabase_client.client.table('chunk_product_relationships').insert(relationship).execute()
                            relationships_created += 1

            except Exception as e:
                logger.error(f"   Error processing chunk {chunk.chunk_index}: {e}")
                continue

        # Update tracker
        await tracker.update_stage(
            "CHUNKING",
            100,
            metadata={
                'chunks_created': chunks_created,
                'embeddings_generated': embeddings_generated,
                'relationships_created': relationships_created,
                'skipped': False
            },
            sync_to_db=True
        )

        logger.info(f"✅ [Job {job_id}] Chunking complete: {chunks_created} chunks, {embeddings_generated} embeddings")

        return CreateChunksResponse(
            success=True,
            chunks_created=chunks_created,
            embeddings_generated=embeddings_generated,
            relationships_created=relationships_created,
            skipped=False,
            existing_chunks=0,
            existing_embeddings=0
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
        from app.services.metadata.dynamic_metadata_extractor import DynamicMetadataExtractor
        metadata_extractor = DynamicMetadataExtractor(
            model=ai_config.metadata_extraction_model,
            job_id=job_id
        )

        # Get products from database
        supabase = get_supabase_client()
        products_response = supabase.client.table('products').select('*').in_('id', request.product_ids).execute()
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
                supabase.client.table('products').update({
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


# ============================================================================
# PRODUCT EMBEDDING GENERATION
# ============================================================================

class GenerateProductEmbeddingsRequest(BaseModel):
    """Request model for generating product embeddings."""
    document_id: Optional[str] = None
    workspace_id: str
    product_ids: Optional[List[str]] = None  # If None, generate for all products in document/workspace


class GenerateProductEmbeddingsResponse(BaseModel):
    """Response model for product embedding generation."""
    success: bool
    message: str
    products_processed: int
    chunks_created: int
    embeddings_queued: int
    errors: List[str] = []


@router.post("/generate-product-embeddings", response_model=GenerateProductEmbeddingsResponse)
async def generate_product_embeddings(
    request: GenerateProductEmbeddingsRequest,
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    """
    Generate embeddings for products that don't have them yet.

    This endpoint:
    1. Finds products without embeddings (no associated chunks)
    2. Creates chunks from product name + description
    3. Queues embedding generation jobs

    **Use Cases:**
    - Fix missing embeddings from old PDF processing
    - Regenerate embeddings after product updates
    - Bulk embedding generation for imported products

    Args:
        request: Request with workspace_id, optional document_id and product_ids

    Returns:
        GenerateProductEmbeddingsResponse with counts and errors
    """
    try:
        logger.info(f"🎨 Starting product embedding generation for workspace: {request.workspace_id}")

        # Build query to find products
        query = supabase.client.table('products').select('id, name, description, metadata, source_document_id')

        # Apply filters
        query = query.eq('workspace_id', request.workspace_id)

        if request.document_id:
            query = query.eq('source_document_id', request.document_id)

        if request.product_ids:
            query = query.in_('id', request.product_ids)

        products_response = query.execute()

        if not products_response.data:
            return GenerateProductEmbeddingsResponse(
                success=True,
                message="No products found matching criteria",
                products_processed=0,
                chunks_created=0,
                embeddings_queued=0
            )

        products = products_response.data
        logger.info(f"   Found {len(products)} products to process")

        # Initialize services
        async_queue = AsyncQueueService()

        products_processed = 0
        chunks_created = 0
        embeddings_queued = 0
        errors = []

        for product in products:
            try:
                product_id = product['id']
                product_name = product.get('name', '')
                description = product.get('description', '')
                document_id = product.get('source_document_id')

                # Skip if no document_id (required for chunks)
                if not document_id:
                    error_msg = f"Skipping product {product_name} - no source_document_id"
                    logger.warning(f"   ⚠️ {error_msg}")
                    errors.append(error_msg)
                    continue

                if not description:
                    error_msg = f"Skipping product {product_name} - no description"
                    logger.warning(f"   ⚠️ {error_msg}")
                    errors.append(error_msg)
                    continue

                # Check if product already has chunks with embeddings
                # First check if chunks exist for this product
                existing_chunks = supabase.client.table('document_chunks').select('id, text_embedding').eq(
                    'metadata->>product_id', product_id
                ).execute()

                if existing_chunks.data and len(existing_chunks.data) > 0:
                    # Check if any of the chunks have embeddings
                    has_embeddings = any(chunk.get('text_embedding') is not None for chunk in existing_chunks.data)
                    if has_embeddings:
                        logger.info(f"   ⏭️ Skipping product {product_name} - already has embeddings")
                        continue

                # Create chunk for product description
                chunk_text = f"{product_name}. {description}"
                metadata = product.get('metadata', {})
                page_number = metadata.get('page_range', [1])[0] if metadata.get('page_range') else 1

                chunk_record = {
                    'document_id': document_id,
                    'workspace_id': request.workspace_id,
                    'content': chunk_text, 
                    'chunk_index': 0,
                    'metadata': {
                        'product_id': product_id,
                        'product_name': product_name,
                        'source': 'product_description',
                        'generated_by': 'admin_embedding_generation',
                        'page_number': page_number
                    }
                }

                chunk_response = supabase.client.table('document_chunks').insert(chunk_record).execute()

                if chunk_response.data:
                    chunk_id = chunk_response.data[0]['id']
                    chunks_created += 1

                    # Queue for embedding generation
                    await async_queue.queue_ai_analysis_jobs(
                        document_id=document_id,
                        chunks=[{'id': chunk_id}],
                        analysis_type='embedding_generation',
                        priority=0
                    )

                    embeddings_queued += 1
                    products_processed += 1
                    logger.info(f"   ✅ Queued embedding for product: {product_name}")

            except Exception as e:
                error_msg = f"Failed to process product {product.get('name', product.get('id'))}: {str(e)}"
                logger.error(f"   ❌ {error_msg}")
                errors.append(error_msg)

        message = f"Generated embeddings for {products_processed} products"
        if errors:
            message += f" ({len(errors)} errors)"

        logger.info(f"✅ Product embedding generation complete: {products_processed} processed, {chunks_created} chunks created, {embeddings_queued} embeddings queued")

        return GenerateProductEmbeddingsResponse(
            success=True,
            message=message,
            products_processed=products_processed,
            chunks_created=chunks_created,
            embeddings_queued=embeddings_queued,
            errors=errors
        )

    except Exception as e:
        logger.error(f"❌ Product embedding generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate product embeddings: {str(e)}")


# ============================================================================
# IMAGE EMBEDDING REGENERATION
# ============================================================================

class RegenerateImageEmbeddingsRequest(BaseModel):
    """Request model for regenerating image embeddings."""
    document_id: Optional[str] = None
    workspace_id: str
    image_ids: Optional[List[str]] = None  # If None, regenerate for all images in document/workspace
    force_regenerate: bool = False  # If True, regenerate even if embeddings exist
    job_id: Optional[str] = None  # Optional job ID for progress tracking


class RegenerateImageEmbeddingsResponse(BaseModel):
    """Response model for image embedding regeneration."""
    success: bool
    message: str
    images_processed: int
    embeddings_generated: int
    skipped: int
    errors: List[str] = []


@router.post("/regenerate-image-embeddings", response_model=RegenerateImageEmbeddingsResponse)
async def regenerate_image_embeddings(
    request: RegenerateImageEmbeddingsRequest,
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    """
    Regenerate visual embeddings for existing images in the database.

    This endpoint:
    1. Fetches existing images from document_images table
    2. Downloads images from Supabase Storage
    3. Generates 5 CLIP embeddings per image (visual, color, texture, style, material)
    4. Saves embeddings to VECS collections

    **Use Cases:**
    - Fix missing embeddings from old PDF processing
    - Regenerate embeddings after model upgrades
    - Bulk embedding generation for imported images

    **Example Request:**
    ```json
    {
      "workspace_id": "ffafc28b-1b8b-4b0d-b226-9f9a6154004e",
      "document_id": "doc-123",  // Optional: limit to specific document
      "image_ids": ["img-1", "img-2"],  // Optional: specific images
      "force_regenerate": false  // Optional: regenerate even if embeddings exist
    }
    ```

    Args:
        request: Request with workspace_id, optional document_id and image_ids

    Returns:
        RegenerateImageEmbeddingsResponse with counts and errors
    """
    try:
        from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
        from app.services.embeddings.vecs_service import get_vecs_service
        import base64
        import aiohttp

        logger.info(f"🎨 Starting image embedding regeneration for workspace: {request.workspace_id}")

        # Build query to find images
        query = supabase.client.table('document_images').select('id, image_url, document_id, page_number, workspace_id')

        # Apply filters
        query = query.eq('workspace_id', request.workspace_id)

        if request.document_id:
            query = query.eq('document_id', request.document_id)

        if request.image_ids:
            query = query.in_('id', request.image_ids)

        images_response = query.execute()

        if not images_response.data:
            return RegenerateImageEmbeddingsResponse(
                success=True,
                message="No images found matching criteria",
                images_processed=0,
                embeddings_generated=0,
                skipped=0
            )

        images = images_response.data
        total_images = len(images)
        logger.info(f"   Found {total_images} images to process")

        # Initialize services
        embeddings_service = RealEmbeddingsService()
        vecs_service = get_vecs_service()

        images_processed = 0
        embeddings_generated = 0
        skipped = 0
        errors = []

        # ✅ Update job status to processing (if job_id provided)
        if request.job_id:
            supabase.client.table('background_jobs').update({
                'status': 'processing',
                'progress': 0,
                'started_at': datetime.utcnow().isoformat(),
                'last_heartbeat': datetime.utcnow().isoformat(),
                'metadata': {
                    'total_images': total_images,
                    'images_processed': 0,
                    'embeddings_generated': 0,
                    'skipped': 0
                }
            }).eq('id', request.job_id).execute()

        for image in images:
            try:
                image_id = image['id']
                image_url = image.get('image_url')

                if not image_url:
                    error_msg = f"Skipping image {image_id} - no image_url"
                    logger.warning(f"   ⚠️ {error_msg}")
                    errors.append(error_msg)
                    skipped += 1
                    continue

                # Check if embeddings already exist (unless force_regenerate)
                if not request.force_regenerate:
                    try:
                        collection = vecs_service.get_or_create_collection("image_siglip_embeddings", dimension=1152)
                        existing = collection.fetch(ids=[image_id])
                        if existing and len(existing) > 0:
                            logger.info(f"   ⏭️ Skipping image {image_id} - embeddings already exist")
                            skipped += 1
                            continue
                    except Exception as e:
                        logger.debug(f"   No existing embeddings for {image_id}: {e}")

                # Download image from Supabase Storage
                logger.info(f"   📥 Downloading image {image_id}...")
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_url) as response:
                        if response.status != 200:
                            error_msg = f"Failed to download image {image_id}: HTTP {response.status}"
                            logger.error(f"   ❌ {error_msg}")
                            errors.append(error_msg)
                            continue

                        image_bytes = await response.read()
                        image_base64 = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

                # Generate all embeddings (visual, color, texture, style, material)
                logger.info(f"   🎨 Generating embeddings for image {image_id}...")
                embedding_result = await embeddings_service.generate_all_embeddings(
                    entity_id=image_id,
                    entity_type="image",
                    text_content="",
                    image_data=image_base64,
                    material_properties={}
                )

                if not embedding_result or not embedding_result.get('success'):
                    error_msg = f"Failed to generate embeddings for image {image_id}"
                    logger.error(f"   ❌ {error_msg}")
                    errors.append(error_msg)
                    continue

                embeddings = embedding_result.get('embeddings', {})

                # Save visual embedding to VECS (primary collection)
                visual_embedding = embeddings.get('visual_siglip_1152')
                if visual_embedding:
                    await vecs_service.upsert_image_embedding(
                        image_id=image_id,
                        siglip_embedding=visual_embedding,
                        metadata={
                            'document_id': image.get('document_id'),
                            'workspace_id': image.get('workspace_id'),
                            'page_number': image.get('page_number', 1)
                        }
                    )
                    embeddings_generated += 1

                # Save specialized embeddings (color, texture, style, material)
                specialized_embeddings = {}
                for emb_type in ['color_siglip_1152', 'texture_siglip_1152', 'style_siglip_1152', 'material_siglip_1152']:
                    if embeddings.get(emb_type):
                        key = emb_type.replace('_siglip_1152', '')
                        specialized_embeddings[key] = embeddings.get(emb_type)

                if specialized_embeddings:
                    await vecs_service.upsert_specialized_embeddings(
                        image_id=image_id,
                        embeddings=specialized_embeddings,
                        metadata={
                            'document_id': image.get('document_id'),
                            'page_number': image.get('page_number', 1)
                        }
                    )
                    embeddings_generated += len(specialized_embeddings)

                images_processed += 1
                logger.info(f"   ✅ Generated {1 + len(specialized_embeddings)} embeddings for image {image_id}")

                # ✅ Update progress after each image (if job_id provided)
                if request.job_id:
                    progress = int((images_processed / total_images) * 100)
                    supabase.client.table('background_jobs').update({
                        'progress': progress,
                        'last_heartbeat': datetime.utcnow().isoformat(),
                        'updated_at': datetime.utcnow().isoformat(),
                        'metadata': {
                            'total_images': total_images,
                            'images_processed': images_processed,
                            'embeddings_generated': embeddings_generated,
                            'skipped': skipped,
                            'current_image': image_id
                        }
                    }).eq('id', request.job_id).execute()

            except Exception as e:
                error_msg = f"Failed to process image {image.get('id')}: {str(e)}"
                logger.error(f"   ❌ {error_msg}")
                errors.append(error_msg)

        message = f"Regenerated embeddings for {images_processed} images ({embeddings_generated} total embeddings)"
        if skipped > 0:
            message += f", skipped {skipped} images"
        if errors:
            message += f" ({len(errors)} errors)"

        logger.info(f"✅ Image embedding regeneration complete: {images_processed} processed, {embeddings_generated} embeddings, {skipped} skipped")

        # ✅ Mark job as completed (if job_id provided)
        if request.job_id:
            supabase.client.table('background_jobs').update({
                'status': 'completed',
                'progress': 100,
                'completed_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat(),
                'metadata': {
                    'total_images': total_images,
                    'images_processed': images_processed,
                    'embeddings_generated': embeddings_generated,
                    'skipped': skipped,
                    'errors': errors
                }
            }).eq('id', request.job_id).execute()

        return RegenerateImageEmbeddingsResponse(
            success=True,
            message=message,
            images_processed=images_processed,
            embeddings_generated=embeddings_generated,
            skipped=skipped,
            errors=errors
        )

    except Exception as e:
        logger.error(f"❌ Image embedding regeneration failed: {str(e)}")

        # ✅ Mark job as failed (if job_id provided)
        if request.job_id:
            supabase.client.table('background_jobs').update({
                'status': 'failed',
                'error': str(e),
                'failed_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }).eq('id', request.job_id).execute()

        raise HTTPException(status_code=500, detail=f"Failed to regenerate image embeddings: {str(e)}")
