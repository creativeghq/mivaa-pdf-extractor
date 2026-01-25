"""
Visual Metadata Service

Converts visual embeddings to textual metadata and consolidates with AI-extracted metadata.
This service integrates EmbeddingToTextService into the image processing pipeline.
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio

from app.services.embeddings.embedding_to_text_service import EmbeddingToTextService
from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class VisualMetadataService:
    """
    Service for converting visual embeddings to textual metadata.
    Integrates with the image processing pipeline (Stage 3.5).
    """

    def __init__(self, workspace_id: str = None):
        from app.config import get_settings
        self.workspace_id = workspace_id or get_settings().default_workspace_id
        self.embedding_to_text = EmbeddingToTextService(workspace_id=self.workspace_id)
        self.supabase = get_supabase_client()

    async def extract_visual_metadata(
        self,
        image_id: str,
        embeddings: Dict[str, List[float]]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract textual metadata from visual embeddings.

        Args:
            image_id: Image UUID
            embeddings: Dict with specialized embeddings (color_siglip_1152, texture_siglip_1152, etc.)

        Returns:
            Dict with extracted visual metadata or None if failed
        """
        try:
            logger.info(f"🎨 Extracting visual metadata for image {image_id}")

            # Convert embeddings to text metadata using AI
            visual_metadata = await self.embedding_to_text.convert_embeddings_to_metadata(
                image_id=image_id,
                embeddings=embeddings
            )

            if not visual_metadata:
                logger.warning(f"⚠️ No visual metadata extracted for image {image_id}")
                return None

            logger.info(f"✅ Extracted visual metadata for image {image_id}: {list(visual_metadata.keys())}")
            return visual_metadata

        except Exception as e:
            logger.error(f"❌ Failed to extract visual metadata for image {image_id}: {e}")
            return None

    async def save_visual_metadata(
        self,
        image_id: str,
        visual_metadata: Dict[str, Any]
    ) -> bool:
        """
        Save visual metadata to document_images table.

        Args:
            image_id: Image UUID
            visual_metadata: Extracted visual metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            # Update document_images with visual metadata
            # Store in metadata JSONB column under 'visual_analysis' key
            result = self.supabase.client.table('document_images') \
                .select('metadata') \
                .eq('id', image_id) \
                .single() \
                .execute()

            existing_metadata = result.data.get('metadata', {}) if result.data else {}

            # Merge visual metadata into existing metadata
            updated_metadata = {
                **existing_metadata,
                'visual_analysis': visual_metadata
            }

            # Update the record
            self.supabase.client.table('document_images') \
                .update({'metadata': updated_metadata}) \
                .eq('id', image_id) \
                .execute()

            logger.info(f"✅ Saved visual metadata for image {image_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to save visual metadata for image {image_id}: {e}")
            return False

    async def process_image_visual_metadata(
        self,
        image_id: str,
        embeddings: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """
        Complete pipeline: Extract and save visual metadata.

        Args:
            image_id: Image UUID
            embeddings: Dict with specialized embeddings

        Returns:
            Dict with success status and metadata
        """
        try:
            # Extract visual metadata
            visual_metadata = await self.extract_visual_metadata(image_id, embeddings)

            if not visual_metadata:
                return {
                    'success': False,
                    'error': 'Failed to extract visual metadata'
                }

            # Save to database
            saved = await self.save_visual_metadata(image_id, visual_metadata)

            if not saved:
                return {
                    'success': False,
                    'error': 'Failed to save visual metadata'
                }

            return {
                'success': True,
                'visual_metadata': visual_metadata
            }

        except Exception as e:
            logger.error(f"❌ Error processing visual metadata for image {image_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

