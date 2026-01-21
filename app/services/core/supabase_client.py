"""
Supabase client initialization and configuration.

This module provides a centralized way to initialize and manage the Supabase client
for database operations and storage management.
"""

import logging
import httpx
from datetime import datetime
from typing import Any, Dict, Optional
from supabase import create_client, Client
from app.config import Settings

logger = logging.getLogger(__name__)


class SupabaseClient:
    """Singleton class for managing Supabase client instance."""
    
    _instance: Optional['SupabaseClient'] = None
    _httpx_client: Optional[httpx.Client] = None
    _client: Optional[Client] = None
    
    def __new__(cls) -> 'SupabaseClient':
        """Ensure only one instance of SupabaseClient exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the SupabaseClient (called only once due to singleton pattern)."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._settings: Optional[Settings] = None
    
    
    def _create_httpx_client(self) -> httpx.Client:
        """
        Create httpx client with connection pooling and timeout configuration.
        
        Returns:
            Configured httpx.Client instance
        """
        return httpx.Client(
            limits=httpx.Limits(
                max_connections=50,  # Total connection pool size
                max_keepalive_connections=20,  # Reusable connections
                keepalive_expiry=30.0  # Keep connections alive for 30 seconds
            ),
            timeout=httpx.Timeout(
                connect=10.0,  # Connection timeout
                read=30.0,  # Read timeout
                write=30.0,  # Write timeout
                pool=5.0  # Pool timeout
            ),
            http2=True,
            follow_redirects=True
        )
    def initialize(self, settings: Settings) -> None:
        """
        Initialize the Supabase client with configuration settings.
        
        Args:
            settings: Application settings containing Supabase configuration
            
        Raises:
            ValueError: If required Supabase settings are missing
            Exception: If client initialization fails
        """
        try:
            self._settings = settings
            
            # Validate required settings
            if not settings.supabase_url:
                raise ValueError("SUPABASE_URL is required but not provided")
            
            if not settings.supabase_anon_key:
                raise ValueError("SUPABASE_ANON_KEY is required but not provided")
            
            # Create httpx client with connection pooling
            self._httpx_client = self._create_httpx_client()
            logger.info("✅ Created httpx client with connection pooling (max_connections=50, max_keepalive=20)")

            # Create Supabase client
            # Use service role key if available, otherwise use anon key
            supabase_key = settings.supabase_service_role_key or settings.supabase_anon_key
            self._client = create_client(
                supabase_url=settings.supabase_url,
                supabase_key=supabase_key
            )
            
            logger.info("Supabase client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {str(e)}")
            raise
    
    @property
    def client(self) -> Client:
        """
        Get the Supabase client instance.
        
        Returns:
            Supabase client instance
            
        Raises:
            RuntimeError: If client is not initialized
        """
        if self._client is None:
            raise RuntimeError(
                "Supabase client not initialized. Call initialize() first."
            )
        return self._client
    
    @property
    def settings(self) -> Settings:
        """
        Get the application settings.
        
        Returns:
            Application settings instance
            
        Raises:
            RuntimeError: If settings are not available
        """
        if self._settings is None:
            raise RuntimeError(
                "Settings not available. Call initialize() first."
            )
        return self._settings
    
    def health_check(self) -> bool:
        """
        Perform a health check on the Supabase connection.

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            # Simple query to test connection
            response = self._client.table('processed_documents').select('id').limit(1).execute()
            return True
        except Exception as e:
            logger.warning(f"Supabase health check failed: {str(e)}")
            return False

    def get_connection_pool_stats(self) -> Dict[str, Any]:
        """
        Get current connection pool statistics.

        Returns:
            Dict containing connection pool metrics
        """
        try:
            if not self._httpx_client:
                return {
                    "status": "not_initialized",
                    "max_connections": 0,
                    "max_keepalive": 0,
                    "active_connections": 0,
                    "idle_connections": 0,
                    "pool_utilization_percent": 0
                }

            # Access httpx connection pool stats
            # Note: httpx doesn't expose pool stats directly, so we track via limits
            limits = self._httpx_client._limits

            # Calculate pool utilization (estimate based on configuration)
            max_connections = limits.max_connections or 50
            max_keepalive = limits.max_keepalive_connections or 20

            # Since httpx doesn't expose active connection count,
            # we provide configuration info and health status
            return {
                "status": "healthy" if self.health_check() else "unhealthy",
                "max_connections": max_connections,
                "max_keepalive": max_keepalive,
                "keepalive_expiry_seconds": 30.0,
                "pool_timeout_seconds": 5.0,
                "http2_enabled": True,
                "pool_utilization_percent": 0,  # Not available in httpx
                "note": "httpx does not expose active connection count"
            }
        except Exception as e:
            logger.error(f"Failed to get connection pool stats: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def list_documents(self, limit: int = 100, status_filter: str = None) -> dict:
        """
        List documents from the documents table.

        Args:
            limit: Maximum number of documents to return
            status_filter: Filter by document processing status

        Returns:
            Dictionary containing documents list
        """
        try:
            query = self._client.table('documents').select('*')

            # Filter by processing status if provided
            if status_filter:
                query = query.eq('processing_status', status_filter)

            query = query.limit(limit)
            response = query.execute()

            return {
                "documents": response.data,
                "count": len(response.data)
            }
        except Exception as e:
            logger.error(f"Failed to list documents: {str(e)}")
            return {"documents": [], "count": 0}

    async def get_document_by_id(self, document_id: str) -> dict:
        """
        Get a specific document by ID.

        Args:
            document_id: The document ID to retrieve

        Returns:
            Dictionary containing document data or error information
        """
        try:
            response = self._client.table('processed_documents').select('*').eq('id', document_id).execute()

            if response.data:
                return {
                    "success": True,
                    "data": response.data[0]
                }
            else:
                return {
                    "success": False,
                    "error": f"Document {document_id} not found"
                }
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to retrieve document: {str(e)}"
            }

    async def get_document_images(self, document_id: str) -> list:
        """
        Get images associated with a document.

        Args:
            document_id: The document ID

        Returns:
            List of image records
        """
        try:
            response = self._client.table('document_images').select('*').eq('document_id', document_id).execute()
            return response.data or []
        except Exception as e:
            logger.error(f"Failed to get images for document {document_id}: {str(e)}")
            return []

    async def save_single_image(
        self,
        image_info: Dict[str, Any],
        document_id: str,
        workspace_id: Optional[str] = None,
        image_index: int = 0,
        category: Optional[str] = None,
        job_id: Optional[str] = None,
        extraction_method: str = 'pymupdf',
        bbox: Optional[list] = None,
        detection_confidence: Optional[float] = None,
        product_name: Optional[str] = None,
        # 4-Layer Extraction Metadata
        layer: Optional[int] = None,
        captures_vector_graphics: Optional[bool] = None,
        is_duplicate: Optional[bool] = None,
        duplicate_of: Optional[str] = None,
        perceptual_hash: Optional[str] = None,
        vision_provider: Optional[str] = None,
        vision_model: Optional[str] = None,
        # Material category from upload (tiles, heatpump, wood, etc.)
        material_category: Optional[str] = None
    ) -> Optional[str]:
        """
        Save a single image to document_images table.

        This is a lightweight method for saving images one at a time during processing
        to avoid memory accumulation.

        Args:
            image_info: Image metadata dict with keys like storage_url, page_number, etc.
            document_id: Document UUID
            workspace_id: Workspace UUID (optional)
            image_index: Index of image in processing sequence
            category: Image category (product, certificate, logo, specification, general)
            job_id: Job ID for source tracking (optional)
            extraction_method: Extraction method (pymupdf, manual)
            bbox: Bounding box coordinates [x, y, width, height] normalized to 0-1
            detection_confidence: Confidence score (0.0-1.0)
            product_name: Product name

        Returns:
            Image ID if successful, None otherwise
        """
        try:
            # Extract image URL (try multiple possible keys)
            # Priority: storage_url (cloud) > public_url (cloud) > url > path (local fallback)
            image_url = (
                image_info.get('storage_url') or
                image_info.get('public_url') or
                image_info.get('url') or
                image_info.get('path')
            )

            # Debug logging to track which URL is being used
            if image_info.get('storage_url'):
                logger.debug(f"✅ Using storage_url for image {image_index}: {image_url[:100]}")
            elif image_info.get('public_url'):
                logger.debug(f"✅ Using public_url for image {image_index}: {image_url[:100]}")
            elif image_info.get('path'):
                logger.warning(f"⚠️ Using local path for image {image_index} (upload may have failed): {image_url[:100]}")
                logger.warning(f"   Available keys in image_info: {list(image_info.keys())}")

            if not image_url or image_url.startswith('placeholder_'):
                logger.debug(f"⏭️  Skipping image {image_index} - no valid URL")
                return None

            # Extract metadata
            page_num = image_info.get('page') or image_info.get('page_number')
            if not page_num:
                logger.warning(
                    f"⚠️ Image {image_index} missing page_number - defaulting to 1. "
                    f"Image info keys: {list(image_info.keys())}"
                )
                page_num = 1

            # Extract AI classification results if available
            ai_classification = image_info.get('ai_classification', {})

            # Generate caption: Use AI-generated description/reason if available
            # Priority: explicit caption > explicit description > AI reason > fallback
            caption = image_info.get('caption') or image_info.get('description')
            if not caption:
                # Use AI classification reason as caption if available (it describes what was seen)
                ai_reason = ai_classification.get('reason')
                if ai_reason and ai_reason != 'Unknown' and len(ai_reason) > 10:
                    # Format: "Material image (AI): <reason>"
                    classification_type = ai_classification.get('classification', 'material')
                    caption = f"{classification_type.replace('_', ' ').title()}: {ai_reason}"
                else:
                    caption = f"Image from page {page_num}"
            is_material = ai_classification.get('is_material', False)

            # Determine category: use material_category from upload (tiles, heatpump, etc.)
            # Priority: material_category > explicit category > ai_classification > default 'general'
            # This ensures images are categorized by their extraction category for proper relevancy search
            if material_category:
                final_category = material_category  # e.g., 'tiles', 'heatpump', 'wood'
            elif category:
                final_category = category
            elif is_material:
                final_category = 'product'  # Fallback if no material_category provided
            else:
                final_category = 'general'

            # Determine image_type from AI classification
            # Priority: AI classification type > fallback to 'material_sample'
            image_type = ai_classification.get('classification') or 'material_sample'
            # Valid types: material_closeup, material_in_situ, non_material

            # Prepare image entry (same format as batch save)
            image_entry = {
                'document_id': document_id,
                'image_url': image_url,
                'image_type': image_type,
                'caption': caption,
                'page_number': page_num,
                'confidence': 0.95,
                'processing_status': 'completed',
                'category': final_category,  # ✅ FIXED: Use AI classification to set category
                'source_type': 'pdf_processing',  # ✅ NEW: Track source type
                'source_job_id': job_id,  # ✅ NEW: Track source job
                # Extraction metadata
                'extraction_method': extraction_method,
                'bbox': bbox,
                'detection_confidence': detection_confidence,
                'product_name': product_name,
                # 4-Layer Extraction Metadata
                'layer': layer or image_info.get('layer'),
                'captures_vector_graphics': captures_vector_graphics if captures_vector_graphics is not None else image_info.get('captures_vector_graphics'),
                'is_duplicate': is_duplicate if is_duplicate is not None else image_info.get('is_duplicate'),
                'duplicate_of': duplicate_of or image_info.get('duplicate_of'),
                'perceptual_hash': perceptual_hash or image_info.get('perceptual_hash'),
                'vision_provider': vision_provider or image_info.get('vision_provider'),
                'vision_model': vision_model or image_info.get('vision_model'),
                'metadata': {
                    'source': 'mivaa_pdf_extraction',
                    'image_index': image_index,
                    'extraction_method': extraction_method,
                    'storage_uploaded': image_info.get('storage_uploaded', False),
                    'storage_bucket': image_info.get('storage_bucket', 'material-images'),
                    'storage_path': image_info.get('storage_path'),
                    'width': image_info.get('width'),
                    'height': image_info.get('height'),
                    'format': image_info.get('format'),
                    'quality_score': image_info.get('quality_score'),
                    'file_size': image_info.get('size_bytes'),
                    'extracted_at': datetime.utcnow().isoformat(),
                    # ✅ NEW: Store AI classification results
                    'ai_classification': {
                        'is_material': is_material,
                        'confidence': ai_classification.get('confidence'),
                        'reason': ai_classification.get('reason'),
                        'model': ai_classification.get('model'),
                        'classification': ai_classification.get('classification')
                    } if ai_classification else None,
                    # ✅ NEW: Store vision-guided metadata
                    'vision_guided': {
                        'bbox': bbox,
                        'confidence': detection_confidence,
                        'provider': vision_provider or image_info.get('vision_provider'),
                        'model': vision_model or image_info.get('vision_model'),
                        'product_name': product_name
                    } if extraction_method == 'vision_guided' else None,
                    # ✅ NEW: 4-Layer extraction metadata
                    'layer_info': {
                        'layer': layer or image_info.get('layer'),
                        'captures_vector_graphics': captures_vector_graphics if captures_vector_graphics is not None else image_info.get('captures_vector_graphics'),
                        'is_duplicate': is_duplicate if is_duplicate is not None else image_info.get('is_duplicate'),
                        'duplicate_of': duplicate_of or image_info.get('duplicate_of'),
                        'perceptual_hash': perceptual_hash or image_info.get('perceptual_hash')
                    }
                }
            }

            # Add workspace_id if provided
            if workspace_id:
                image_entry['workspace_id'] = workspace_id

            # Insert into database
            response = self._client.table('document_images').insert(image_entry).execute()

            if response.data and len(response.data) > 0:
                image_id = response.data[0]['id']
                logger.debug(f"✅ Saved image to DB: {image_id} (page {page_num})")
                return image_id
            else:
                logger.warning(f"⚠️ Failed to save image {image_index}: No data returned")
                return None

        except Exception as e:
            logger.error(f"❌ Failed to save image {image_index} to database: {e}")
            return None

    async def save_pdf_processing_result(self, result, original_filename: str = None, file_url: str = None) -> str:
        """
        Save PDF processing result to the database.

        Args:
            result: PDFProcessingResult object
            original_filename: Original filename of the PDF
            file_url: URL of the processed PDF

        Returns:
            ID of the saved record
        """
        try:
            # Get user_id from result metadata or use system user as fallback
            # User ID should be passed from the upload endpoint via result.metadata
            user_id = None
            if hasattr(result, 'metadata') and isinstance(result.metadata, dict):
                user_id = result.metadata.get('user_id')

            # Fallback to system user if no user_id provided
            if not user_id:
                user_id = "00000000-0000-0000-0000-000000000000"  # System user
                logger.warning("No user_id found in result metadata, using system user")

            # Prepare data for insertion
            insert_data = {
                'user_id': user_id,  # Required field
                'original_filename': original_filename or f"{result.document_id}.pdf",
                'file_url': file_url or f"https://example.com/{result.document_id}.pdf",  # Required field
                'processing_status': 'completed',
                'processing_started_at': 'now()',
                'processing_completed_at': 'now()',
                'processing_time_ms': int(result.processing_time * 1000) if result.processing_time else 0,
                'total_pages': result.page_count or 0,
                'total_tiles_extracted': len(result.extracted_images) if result.extracted_images else 0,
                'materials_identified_count': 0,  # Will be updated by material recognition
                'confidence_score_avg': 0.95,  # Default confidence
                'ocr_text_content': result.ocr_text or "",
                'ocr_confidence_avg': 0.90,  # Default OCR confidence
                'ocr_language_detected': 'en',
                'extracted_images': result.extracted_images or [],
                'multimodal_enabled': getattr(result, 'multimodal_enabled', False),
                'python_processor_version': '1.0.0',
                'layout_analysis_version': '1.0.0',
                'document_structure': {
                    'word_count': result.word_count or 0,
                    'character_count': result.character_count or 0,
                    'markdown_content': result.markdown_content or ""
                },
                'image_analysis_results': getattr(result, 'ocr_results', {}),
                'multimodal_metadata': result.metadata or {}
            }

            logger.info(f"💾 Attempting to save PDF processing result to database")
            logger.info(f"   Document ID: {result.document_id}")
            logger.info(f"   Filename: {insert_data['original_filename']}")
            logger.info(f"   Pages: {insert_data['total_pages']}")
            logger.info(f"   Images: {insert_data['total_tiles_extracted']}")

            # Insert into database
            response = self._client.table('pdf_processing_results').insert(insert_data).execute()

            if response.data:
                record_id = response.data[0]['id']
                logger.info(f"✅ PDF processing result saved with ID: {record_id}")
                return record_id
            else:
                logger.error(f"❌ No data returned from insert operation")
                logger.error(f"   Response: {response}")
                raise Exception("No data returned from insert operation")

        except Exception as e:
            logger.error(f"❌ Failed to save PDF processing result: {str(e)}")
            logger.error(f"   Error type: {type(e).__name__}")
            logger.error(f"   Insert data keys: {list(insert_data.keys()) if 'insert_data' in locals() else 'N/A'}")
            raise

    async def save_knowledge_base_entries(self, document_id: str, chunks: list, images: list) -> dict:
        """
        Save extracted chunks and images to knowledge base.

        Args:
            document_id: Document identifier
            chunks: List of text chunks
            images: List of image data

        Returns:
            Dictionary with counts of saved entries
        """
        try:
            saved_chunks = 0
            saved_images = 0

            logger.info(f"💾 Starting knowledge base save for document: {document_id}")
            logger.info(f"   Chunks to save: {len(chunks)}")
            logger.info(f"   Images to save: {len(images)}")

            # First, ensure document exists in documents table and get workspace_id
            workspace_id = None
            try:
                # Check if document already exists and get its workspace_id
                doc_check = self._client.table('documents').select('id, workspace_id').eq('id', document_id).execute()

                if not doc_check.data:
                    # Create document record
                    doc_data = {
                        'id': document_id,
                        'filename': f"{document_id}.pdf",
                        'content_type': 'application/pdf',
                        'content': "",  # Will be updated with markdown content
                        'processing_status': 'completed',
                        'metadata': {
                            'source': 'mivaa_processing',
                            'chunks_count': len(chunks),
                            'images_count': len(images)
                        }
                    }

                    doc_response = self._client.table('documents').insert(doc_data).execute()
                    if doc_response.data:
                        logger.info(f"✅ Created document record: {document_id}")
                        workspace_id = doc_response.data[0].get('workspace_id') if doc_response.data else None
                    else:
                        logger.warning(f"⚠️ Failed to create document record: {document_id}")
                else:
                    logger.info(f"✅ Document already exists: {document_id}")
                    workspace_id = doc_check.data[0].get('workspace_id') if doc_check.data else None
                    logger.info(f"   Workspace ID: {workspace_id}")

            except Exception as doc_error:
                logger.warning(f"⚠️ Document creation failed: {doc_error}")

            # Save text chunks to document_chunks table
            if chunks:
                chunk_entries = []
                for i, chunk in enumerate(chunks):
                    if isinstance(chunk, str) and chunk.strip():  # Only save non-empty string chunks
                        chunk_entry = {
                            'document_id': document_id,
                            'content': chunk,
                            'chunk_index': i,
                            'metadata': {
                                'source': 'mivaa_pdf_extraction',
                                'chunk_length': len(chunk),
                                'chunk_number': i + 1,
                                'page_number': 1  # Default page number
                            }
                        }
                        # Add workspace_id if available
                        if workspace_id:
                            chunk_entry['workspace_id'] = workspace_id
                        chunk_entries.append(chunk_entry)

                if chunk_entries:
                    logger.info(f"💾 Saving {len(chunk_entries)} chunks to document_chunks table (workspace_id: {workspace_id})")
                    response = self._client.table('document_chunks').insert(chunk_entries).execute()
                    saved_chunks = len(response.data) if response.data else 0
                    logger.info(f"✅ Saved {saved_chunks} chunks to database")
                else:
                    logger.warning("⚠️ No valid chunks to save")

            # Save images to document_images table
            if images:
                image_entries = []
                for i, image in enumerate(images):
                    # Handle different image data formats
                    if isinstance(image, dict):
                        # Try multiple possible keys for image URL
                        image_url = (
                            image.get('storage_url') or  # From _process_extracted_image
                            image.get('url') or
                            image.get('path') or
                            image.get('public_url') or
                            f"placeholder_image_{i}.jpg"
                        )
                        page_num = image.get('page') or image.get('page_number') or 1
                        caption = image.get('caption') or image.get('description') or f"Image {i+1}"

                        # Log if we're using a placeholder
                        if image_url.startswith('placeholder_'):
                            logger.warning(f"⚠️ Image {i} has no valid URL. Available keys: {list(image.keys())}")
                            logger.warning(f"   Image data sample: {str(image)[:200]}")
                    else:
                        # Handle string or other formats
                        image_url = str(image) if image else f"placeholder_image_{i}.jpg"
                        page_num = 1
                        caption = f"Image {i+1}"
                        logger.warning(f"⚠️ Image {i} is not a dict, type: {type(image)}")

                    # Only add images with valid URLs (not placeholders)
                    if not image_url.startswith('placeholder_'):
                        image_entry = {
                            'document_id': document_id,
                            'image_url': image_url,
                            'image_type': 'material_sample',
                            'caption': caption,
                            'page_number': page_num,
                            'confidence': 0.95,  # Default confidence
                            'processing_status': 'completed',
                            'metadata': {
                                'source': 'mivaa_pdf_extraction',
                                'image_index': i,
                                'extraction_method': 'pymupdf',
                                'storage_uploaded': image.get('storage_uploaded', False) if isinstance(image, dict) else False,
                                'storage_bucket': image.get('storage_bucket', 'pdf-tiles') if isinstance(image, dict) else 'pdf-tiles',
                                'original_data': image if isinstance(image, dict) else {'url': str(image)}
                            }
                        }
                        # Add workspace_id if available
                        if workspace_id:
                            image_entry['workspace_id'] = workspace_id
                        image_entries.append(image_entry)
                    else:
                        logger.warning(f"⚠️ Skipping image {i} - no valid URL found")

                if image_entries:
                    logger.info(f"💾 Saving {len(image_entries)} images to document_images table (out of {len(images)} total, workspace_id: {workspace_id})")
                    logger.info(f"   Sample image URLs: {[img['image_url'][:100] for img in image_entries[:3]]}")
                    try:
                        response = self._client.table('document_images').insert(image_entries).execute()
                        saved_images = len(response.data) if response.data else 0
                        logger.info(f"✅ Saved {saved_images} images to database")

                        if saved_images < len(image_entries):
                            logger.warning(f"⚠️ Only {saved_images}/{len(image_entries)} images were saved")
                    except Exception as insert_error:
                        logger.error(f"❌ Failed to insert images: {str(insert_error)}")
                        logger.error(f"   Error type: {type(insert_error).__name__}")
                        logger.error(f"   Sample image entry: {image_entries[0] if image_entries else 'N/A'}")
                        import traceback
                        logger.error(f"   Traceback: {traceback.format_exc()}")
                        raise
                else:
                    logger.warning(f"⚠️ No valid images to save (0 out of {len(images)} had valid URLs)")

            logger.info(f"✅ Knowledge base save completed: {saved_chunks} chunks, {saved_images} images")
            return {
                'chunks_saved': saved_chunks,
                'images_saved': saved_images,
                'total_saved': saved_chunks + saved_images
            }

        except Exception as e:
            logger.error(f"❌ Failed to save knowledge base entries: {str(e)}")
            logger.error(f"   Error type: {type(e).__name__}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            return {
                'chunks_saved': 0,
                'images_saved': 0,
                'total_saved': 0,
                'error': str(e)
            }

    async def upload_file(self, bucket_name: str, file_path: str, file_data: bytes,
                         content_type: str = None, upsert: bool = False) -> dict:
        """
        Upload file to Supabase Storage.

        Args:
            bucket_name: Name of the storage bucket
            file_path: Path where the file should be stored
            file_data: File content as bytes
            content_type: MIME type of the file
            upsert: Whether to overwrite existing files

        Returns:
            Dictionary with upload result
        """
        try:
            if not self._client:
                raise Exception("Supabase client not initialized")

            # Debug logging
            logger.info(f"🔍 DEBUG - Uploading file: bucket={bucket_name}, path={file_path}, data_type={type(file_data)}, data_len={len(file_data) if isinstance(file_data, bytes) else 'N/A'}")
            logger.info(f"🔍 DEBUG - Content type: {content_type}, Upsert: {upsert}")

            # Upload file to storage
            response = self._client.storage.from_(bucket_name).upload(
                file_path,
                file_data,
                file_options={
                    "content-type": content_type,
                    "upsert": "true" if upsert else "false"
                }
            )

            logger.info(f"🔍 DEBUG - Upload response type: {type(response)}, hasattr error: {hasattr(response, 'error')}")
            # Check if upload was successful
            # Handle httpx.Response (newer supabase-py versions)
            if hasattr(response, 'status_code'):
                if response.status_code not in [200, 201]:
                    error_msg = response.text if hasattr(response, 'text') else str(response)
                    raise Exception(f"Upload failed with status {response.status_code}: {error_msg}")
                response_data = response.json() if hasattr(response, 'json') else {}
            # Handle old-style response objects
            elif hasattr(response, 'error') and response.error:
                raise Exception(f"Upload failed: {response.error}")
            elif isinstance(response, dict):
                if response.get('error'):
                    raise Exception(f"Upload failed: {response.get('error')}")
                response_data = response
            else:
                response_data = {}

            # Get public URL
            url_response = self._client.storage.from_(bucket_name).get_public_url(file_path)

            logger.info(f"File uploaded successfully to {bucket_name}/{file_path}")
            return {
                "success": True,
                "data": response_data,
                "public_url": url_response,
                "bucket": bucket_name,
                "path": file_path
            }

        except Exception as e:
            logger.error(f"Failed to upload file: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def upload_pdf_file(self, file_data: bytes, filename: str,
                             document_id: str = None) -> dict:
        """
        Upload PDF file to the pdf-documents bucket.

        Args:
            file_data: PDF file content as bytes
            filename: Original filename
            document_id: Optional document ID for organizing files

        Returns:
            Dictionary with upload result including public URL
        """
        try:
            # Generate unique file path
            import uuid
            from datetime import datetime

            if not document_id:
                document_id = str(uuid.uuid4())

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            file_extension = filename.split('.')[-1] if '.' in filename else 'pdf'
            storage_path = f"documents/{document_id}/{timestamp}_{filename}"

            # Upload to pdf-documents bucket
            result = await self.upload_file(
                bucket_name="pdf-documents",
                file_path=storage_path,
                file_data=file_data,
                content_type="application/pdf",
                upsert=False
            )

            if result["success"]:
                result["document_id"] = document_id
                result["storage_path"] = storage_path

            return result

        except Exception as e:
            logger.error(f"Failed to upload PDF file: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def upload_image_file(self, image_data: bytes, filename: str,
                               document_id: str, page_number: int = None) -> dict:
        """
        Upload extracted image to the pdf-tiles bucket.

        Args:
            image_data: Image content as bytes
            filename: Image filename
            document_id: Document ID for organizing images
            page_number: Page number where image was extracted

        Returns:
            Dictionary with upload result including public URL
        """
        try:
            # Generate storage path for extracted images
            from datetime import datetime

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            page_suffix = f"_page{page_number}" if page_number else ""
            file_extension = filename.split('.')[-1] if '.' in filename else 'png'
            storage_path = f"extracted/{document_id}/{timestamp}{page_suffix}_{filename}"

            # Determine content type
            content_type = "image/png"
            if file_extension.lower() in ['jpg', 'jpeg']:
                content_type = "image/jpeg"
            elif file_extension.lower() == 'webp':
                content_type = "image/webp"

            # Upload to pdf-tiles bucket
            result = await self.upload_file(
                bucket_name="pdf-tiles",
                file_path=storage_path,
                file_data=image_data,
                content_type=content_type,
                upsert=False
            )

            if result["success"]:
                result["document_id"] = document_id
                result["page_number"] = page_number
                result["storage_path"] = storage_path

            return result

        except Exception as e:
            logger.error(f"Failed to upload image file: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def delete_file(self, bucket_name: str, file_path: str) -> dict:
        """
        Delete file from Supabase Storage.

        Args:
            bucket_name: Name of the storage bucket
            file_path: Path of the file to delete

        Returns:
            Dictionary with deletion result
        """
        try:
            if not self._client:
                raise Exception("Supabase client not initialized")

            response = self._client.storage.from_(bucket_name).remove([file_path])

            if response.error:
                raise Exception(f"Delete failed: {response.error}")

            logger.info(f"File deleted successfully from {bucket_name}/{file_path}")
            return {
                "success": True,
                "data": response.data
            }

        except Exception as e:
            logger.error(f"Failed to delete file: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def delete_image_file(self, storage_path: str) -> bool:
        """
        Delete image file from Supabase Storage.

        Used to delete non-material images after AI classification.

        Args:
            storage_path: Storage path (e.g., 'material-images/doc_id/image.jpg')

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # Extract bucket name from storage path
            # Format: 'material-images/doc_id/image.jpg'
            bucket_name = 'material-images'

            # Remove bucket name from path if present
            if storage_path.startswith(f'{bucket_name}/'):
                file_path = storage_path[len(f'{bucket_name}/'):]
            else:
                file_path = storage_path

            # Delete from Supabase Storage
            result = self.client.storage.from_(bucket_name).remove([file_path])

            logger.info(f"✅ Deleted image from Supabase: {storage_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to delete {storage_path} from Supabase: {e}")
            return False

    def close(self) -> None:
        """Close the Supabase client connection."""
        if self._client:
            # Supabase client doesn't require explicit closing
            # but we can reset the instance for cleanup
            self._client = None
            logger.info("Supabase client connection closed")


# Global instance
supabase_client = SupabaseClient()


def get_supabase_client() -> SupabaseClient:
    """
    Get the global Supabase client instance.
    
    Returns:
        SupabaseClient instance
    """
    return supabase_client


def initialize_supabase(settings: Settings) -> None:
    """
    Initialize the global Supabase client.
    
    Args:
        settings: Application settings containing Supabase configuration
    """
    supabase_client.initialize(settings)
