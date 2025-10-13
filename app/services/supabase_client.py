"""
Supabase client initialization and configuration.

This module provides a centralized way to initialize and manage the Supabase client
for database operations and storage management.
"""

import logging
from typing import Optional
from supabase import create_client, Client
from app.config import Settings

logger = logging.getLogger(__name__)


class SupabaseClient:
    """Singleton class for managing Supabase client instance."""
    
    _instance: Optional['SupabaseClient'] = None
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

    async def list_documents(self, limit: int = 100, status_filter: str = None) -> dict:
        """
        List documents from the processed_documents table.

        Args:
            limit: Maximum number of documents to return
            status_filter: Filter by document status (ignored for now - column doesn't exist)

        Returns:
            Dictionary containing documents list
        """
        try:
            query = self._client.table('processed_documents').select('*')

            # Note: status column doesn't exist in current schema
            # if status_filter:
            #     query = query.eq('status', status_filter)

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
            # Prepare data for insertion
            insert_data = {
                'original_filename': original_filename or f"{result.document_id}.pdf",
                'file_url': file_url,
                'processing_status': 'completed',
                'processing_started_at': 'now()',
                'processing_completed_at': 'now()',
                'processing_time_ms': int(result.processing_time * 1000),
                'total_pages': result.page_count,
                'total_tiles_extracted': len(result.extracted_images),
                'materials_identified_count': 0,  # Will be updated by material recognition
                'confidence_score_avg': 0.95,  # Default confidence
                'ocr_text_content': result.ocr_text,
                'ocr_confidence_avg': 0.90,  # Default OCR confidence
                'ocr_language_detected': 'en',
                'extracted_images': result.extracted_images,
                'multimodal_enabled': result.multimodal_enabled,
                'python_processor_version': '1.0.0',
                'layout_analysis_version': '1.0.0',
                'document_structure': {
                    'word_count': result.word_count,
                    'character_count': result.character_count,
                    'markdown_content': result.markdown_content
                },
                'image_analysis_results': result.ocr_results,
                'multimodal_metadata': result.metadata
            }

            # Insert into database
            response = self._client.table('pdf_processing_results').insert(insert_data).execute()

            if response.data:
                record_id = response.data[0]['id']
                logger.info(f"PDF processing result saved with ID: {record_id}")
                return record_id
            else:
                raise Exception("No data returned from insert operation")

        except Exception as e:
            logger.error(f"Failed to save PDF processing result: {str(e)}")
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

            # Save text chunks to knowledge_base_entries
            if chunks:
                chunk_entries = []
                for i, chunk in enumerate(chunks):
                    if chunk.strip():  # Only save non-empty chunks
                        chunk_entries.append({
                            'document_id': document_id,
                            'content_type': 'text',
                            'content': chunk,
                            'chunk_index': i,
                            'metadata': {
                                'source': 'pdf_extraction',
                                'chunk_length': len(chunk),
                                'chunk_number': i + 1
                            }
                        })

                if chunk_entries:
                    response = self._client.table('knowledge_base_entries').insert(chunk_entries).execute()
                    saved_chunks = len(response.data) if response.data else 0

            # Save images to document_images
            if images:
                image_entries = []
                for i, image in enumerate(images):
                    image_entries.append({
                        'document_id': document_id,
                        'image_path': image.get('path', ''),
                        'image_url': image.get('url', ''),
                        'page_number': image.get('page', 1),
                        'width': image.get('width', 0),
                        'height': image.get('height', 0),
                        'format': image.get('format', 'unknown'),
                        'metadata': {
                            'source': 'pdf_extraction',
                            'image_index': i,
                            'extraction_method': image.get('extraction_method', 'pymupdf')
                        }
                    })

                if image_entries:
                    response = self._client.table('document_images').insert(image_entries).execute()
                    saved_images = len(response.data) if response.data else 0

            logger.info(f"Saved {saved_chunks} chunks and {saved_images} images to knowledge base")
            return {
                'chunks_saved': saved_chunks,
                'images_saved': saved_images,
                'total_saved': saved_chunks + saved_images
            }

        except Exception as e:
            logger.error(f"Failed to save knowledge base entries: {str(e)}")
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

            # Upload file to storage
            response = self._client.storage.from_(bucket_name).upload(
                file_path,
                file_data,
                file_options={
                    "content-type": content_type,
                    "upsert": upsert
                }
            )

            if response.error:
                raise Exception(f"Upload failed: {response.error}")

            # Get public URL
            url_response = self._client.storage.from_(bucket_name).get_public_url(file_path)

            logger.info(f"File uploaded successfully to {bucket_name}/{file_path}")
            return {
                "success": True,
                "data": response.data,
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