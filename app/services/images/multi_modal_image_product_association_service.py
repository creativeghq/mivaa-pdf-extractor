"""
Multi-Modal Image-Product Association Service

Creates intelligent image-product linking using:
- Spatial proximity (40% weight): Same page ±1, spatial distance
- Caption similarity (30% weight): Text similarity between image captions and product descriptions
- CLIP visual similarity (30% weight): Visual-text similarity using existing CLIP embeddings

Replaces random associations with weighted confidence scoring for meaningful relationships.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import math
import re

from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

@dataclass
class AssociationWeights:
    spatial: float = 0.4   # 40% weight for spatial proximity
    caption: float = 0.3   # 30% weight for caption similarity
    clip: float = 0.3      # 30% weight for CLIP visual similarity

@dataclass
class AssociationOptions:
    weights: AssociationWeights = None
    spatial_threshold: float = 0.3
    caption_threshold: float = 0.4
    clip_threshold: float = 0.5
    overall_threshold: float = 0.3
    max_associations_per_image: int = 3
    max_associations_per_product: int = 5

    def __post_init__(self):
        if self.weights is None:
            self.weights = AssociationWeights()

@dataclass
class ImageProductAssociation:
    image_id: str
    product_id: str
    spatial_score: float
    caption_score: float
    clip_score: float
    overall_score: float
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]

class MultiModalImageProductAssociationService:
    """Service for creating intelligent image-product associations using multi-modal analysis."""

    def __init__(self):
        self.supabase = get_supabase_client()
        self.logger = logger

    async def create_document_associations(
        self,
        document_id: str,
        options: Optional[AssociationOptions] = None
    ) -> Dict[str, Any]:
        """
        Create intelligent image-product associations for a document.
        
        Args:
            document_id: The document ID to process
            options: Association configuration options
            
        Returns:
            Dictionary with association results and statistics
        """
        if options is None:
            options = AssociationOptions()

        self.logger.info(f"🎯 Creating multi-modal image-product associations for document: {document_id}")

        try:
            # Get all images and products for the document
            images, products = await asyncio.gather(
                self._get_document_images(document_id),
                self._get_document_products(document_id)
            )

            if not images or not products:
                self.logger.warning(f"⚠️ No images ({len(images)}) or products ({len(products)}) found for document")
                return {
                    "associations_created": 0,
                    "total_evaluated": 0,
                    "average_confidence": 0,
                    "associations": []
                }

            self.logger.info(f"📊 Evaluating {len(images)} images × {len(products)} products = {len(images) * len(products)} potential associations")

            # Evaluate all image-product combinations
            all_associations = []
            total_evaluated = 0

            for image in images:
                for product in products:
                    total_evaluated += 1
                    
                    try:
                        association = await self._evaluate_association(image, product, options)
                        
                        if association.overall_score >= options.overall_threshold:
                            all_associations.append(association)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Error evaluating association between image {image['id']} and product {product['id']}: {e}")

            # Sort by overall score and apply limits
            all_associations.sort(key=lambda x: x.overall_score, reverse=True)
            final_associations = self._apply_association_limits(all_associations, options)

            # Create database relationships
            associations_created = await self._create_database_associations(final_associations)

            average_confidence = (
                sum(assoc.confidence for assoc in final_associations) / len(final_associations)
                if final_associations else 0
            )

            self.logger.info(f"✅ Created {associations_created} intelligent image-product associations")
            self.logger.info(f"📊 Average confidence: {average_confidence * 100:.1f}%")

            return {
                "associations_created": associations_created,
                "total_evaluated": total_evaluated,
                "average_confidence": average_confidence,
                "associations": [self._association_to_dict(assoc) for assoc in final_associations]
            }

        except Exception as e:
            self.logger.error(f"❌ Error creating document associations: {e}")
            raise Exception(f"Failed to create image-product associations: {e}")

    async def _evaluate_association(
        self,
        image: Dict[str, Any],
        product: Dict[str, Any],
        options: AssociationOptions
    ) -> ImageProductAssociation:
        """Evaluate a single image-product association."""
        
        # Calculate individual scores
        spatial_score = await self._calculate_spatial_score(image, product)
        caption_score = await self._calculate_caption_score(image, product)
        clip_score = await self._calculate_clip_score(image, product)

        # Calculate weighted overall score
        weights = options.weights
        overall_score = (
            spatial_score * weights.spatial +
            caption_score * weights.caption +
            clip_score * weights.clip
        )

        # Calculate confidence based on score distribution
        confidence = self._calculate_confidence(spatial_score, caption_score, clip_score, overall_score)

        # Generate reasoning
        reasoning = self._generate_reasoning(spatial_score, caption_score, clip_score, overall_score, weights)

        # Get page info for metadata
        image_page = image.get('page_number', 0)
        product_metadata = product.get('metadata', {})
        product_pages = product_metadata.get('page_range', [])
        if not product_pages and product.get('page_number'):
            product_pages = [product.get('page_number')]

        # Find minimum page difference
        min_page_diff = min(
            abs(image_page - p) for p in product_pages
        ) if product_pages and image_page else None

        return ImageProductAssociation(
            image_id=image['id'],
            product_id=product['id'],
            spatial_score=spatial_score,
            caption_score=caption_score,
            clip_score=clip_score,
            overall_score=overall_score,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "spatial_proximity": {
                    "image_page": image_page,
                    "product_pages": product_pages,
                    "min_page_difference": min_page_diff,
                    "same_page_group": min_page_diff is not None and min_page_diff <= 1
                },
                "caption_similarity": {
                    "image_caption": image.get('caption', '') or image.get('alt_text', ''),
                    "product_name": product.get('name', ''),
                    "text_similarity": caption_score
                },
                "clip_similarity": {
                    "visual_text_similarity": clip_score,
                    "has_image_embedding": bool(image.get('clip_embedding') or image.get('visual_embedding')),
                    "has_product_embedding": bool(product.get('clip_embedding') or product_metadata.get('clip_embedding'))
                }
            }
        )

    async def _calculate_spatial_score(self, image: Dict[str, Any], product: Dict[str, Any]) -> float:
        """Calculate spatial proximity score (0-1).

        Checks all pages in product's page_range and returns the best score.
        """
        image_page = image.get('page_number', 0)
        if not image_page:
            return 0.1  # No image page info, low score

        # Get all product pages from both top-level and metadata.page_range
        product_pages = []

        # Check top-level page_number
        top_level_page = product.get('page_number', 0)
        if top_level_page:
            product_pages.append(top_level_page)

        # Check metadata.page_range (can contain multiple pages)
        metadata = product.get('metadata', {})
        page_range = metadata.get('page_range', [])
        if page_range:
            for page in page_range:
                if isinstance(page, (int, float)) and page not in product_pages:
                    product_pages.append(int(page))

        if not product_pages:
            return 0.1  # No product page info, low score

        # Calculate score for each product page and return the best
        best_score = 0.1
        for product_page in product_pages:
            page_difference = abs(image_page - product_page)

            # Same page = highest score
            if page_difference == 0:
                return 1.0  # Can't get better, return immediately

            # Adjacent pages = high score
            if page_difference == 1:
                score = 0.85
            # Within 2 pages = good score
            elif page_difference <= 2:
                score = 0.7
            # Within 3 pages = medium score
            elif page_difference <= 3:
                score = 0.5
            # Within 5 pages = low-medium score
            elif page_difference <= 5:
                score = 0.3
            # Further apart = very low score
            else:
                score = max(0.1, 0.5 / page_difference)

            best_score = max(best_score, score)

        return best_score

    async def _calculate_caption_score(self, image: Dict[str, Any], product: Dict[str, Any]) -> float:
        """Calculate caption similarity score (0-1).

        Handles generic captions like "Image from page 22" with neutral scores.
        """
        image_text = (image.get('caption', '') or image.get('alt_text', '')).lower()
        product_text = (product.get('description', '') or product.get('name', '')).lower()

        # Check for generic/uninformative captions
        generic_patterns = [
            r'^image\s+(from\s+)?page\s+\d+',
            r'^page\s+\d+\s+image',
            r'^figure\s+\d+',
            r'^img_?\d+',
            r'^extracted\s+image',
            r'^document\s+image',
        ]

        is_generic_caption = False
        if image_text:
            for pattern in generic_patterns:
                if re.match(pattern, image_text.strip()):
                    is_generic_caption = True
                    break

        # If caption is generic, return neutral score (don't penalize or boost)
        if is_generic_caption or not image_text:
            return 0.5  # Neutral - let other signals (spatial, CLIP) decide

        if not product_text:
            return 0.5  # Neutral if no product text

        # Simple text similarity using word overlap
        # Filter out common stopwords and short words
        stopwords = {'the', 'and', 'for', 'with', 'from', 'this', 'that', 'image', 'page'}
        image_words = set(
            word for word in re.split(r'\s+', image_text)
            if len(word) > 2 and word not in stopwords
        )
        product_words = set(
            word for word in re.split(r'\s+', product_text)
            if len(word) > 2 and word not in stopwords
        )

        if not image_words or not product_words:
            return 0.5  # Neutral if no meaningful words

        # Calculate Jaccard similarity
        intersection = image_words.intersection(product_words)
        union = image_words.union(product_words)
        jaccard_similarity = len(intersection) / len(union) if union else 0.0

        # Boost score if product name appears in image caption
        product_name = (product.get('name', '')).lower()
        if product_name and len(product_name) > 2 and product_name in image_text:
            return min(1.0, jaccard_similarity + 0.4)

        # Check for partial product name match (first word of multi-word name)
        product_name_parts = product_name.split()
        if len(product_name_parts) > 0:
            first_part = product_name_parts[0]
            if len(first_part) > 3 and first_part in image_text:
                return min(1.0, jaccard_similarity + 0.25)

        # Scale Jaccard similarity: 0 Jaccard -> 0.3, 0.5 Jaccard -> 0.65, 1.0 Jaccard -> 1.0
        # This ensures even low text overlap doesn't completely tank the score
        return 0.3 + (jaccard_similarity * 0.7)

    async def _calculate_clip_score(self, image: Dict[str, Any], product: Dict[str, Any]) -> float:
        """Calculate CLIP visual similarity score (0-1).

        Uses actual cosine similarity between embeddings when available.
        Falls back to neutral score when embeddings are missing.
        """
        try:
            # Get image embedding (could be under different field names)
            image_embedding = (
                image.get('clip_embedding') or
                image.get('visual_embedding') or
                image.get('embedding')
            )

            # Get product text embedding (if available)
            product_embedding = None
            product_metadata = product.get('metadata', {})
            product_embedding = (
                product.get('clip_embedding') or
                product.get('text_embedding') or
                product_metadata.get('clip_embedding') or
                product_metadata.get('text_embedding')
            )

            # If we have both embeddings, compute actual cosine similarity
            if image_embedding and product_embedding:
                similarity = self._cosine_similarity(image_embedding, product_embedding)
                # CLIP similarities are typically in range [-1, 1], normalize to [0, 1]
                normalized = (similarity + 1) / 2
                return max(0.0, min(1.0, normalized))

            # If we have image embedding but no product embedding, use neutral score
            if image_embedding:
                # Slight boost for having visual data even if no product embedding
                return 0.5

            # No embeddings available - return neutral score
            # This doesn't penalize but doesn't boost either
            return 0.5

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating CLIP score: {e}")
            return 0.5  # Neutral score on error

    def _cosine_similarity(self, vec_a: list, vec_b: list) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0

        try:
            # Compute dot product and magnitudes
            dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
            magnitude_a = math.sqrt(sum(a * a for a in vec_a))
            magnitude_b = math.sqrt(sum(b * b for b in vec_b))

            if magnitude_a == 0 or magnitude_b == 0:
                return 0.0

            return dot_product / (magnitude_a * magnitude_b)
        except (TypeError, ValueError):
            return 0.0

    def _calculate_confidence(
        self,
        spatial_score: float,
        caption_score: float,
        clip_score: float,
        overall_score: float
    ) -> float:
        """Calculate confidence based on score distribution."""
        scores = [spatial_score, caption_score, clip_score]
        
        # Calculate variance
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        
        # Lower variance = higher confidence
        consistency_bonus = max(0, 0.3 - variance)
        
        # Base confidence from overall score
        base_confidence = overall_score
        
        return min(1.0, base_confidence + consistency_bonus)

    def _generate_reasoning(
        self,
        spatial_score: float,
        caption_score: float,
        clip_score: float,
        overall_score: float,
        weights: AssociationWeights
    ) -> str:
        """Generate human-readable reasoning for the association."""
        reasons = []

        # Spatial reasoning
        if spatial_score >= 0.8:
            reasons.append('same/adjacent page')
        elif spatial_score >= 0.6:
            reasons.append('nearby pages')
        elif spatial_score >= 0.4:
            reasons.append('moderate spatial proximity')

        # Caption reasoning
        if caption_score >= 0.7:
            reasons.append('strong text similarity')
        elif caption_score >= 0.5:
            reasons.append('moderate text similarity')
        elif caption_score >= 0.3:
            reasons.append('some text overlap')

        # CLIP reasoning
        if clip_score >= 0.7:
            reasons.append('high visual-text similarity')
        elif clip_score >= 0.5:
            reasons.append('moderate visual relevance')

        # Overall assessment
        if overall_score >= 0.8:
            assessment = 'Strong association'
        elif overall_score >= 0.6:
            assessment = 'Good association'
        elif overall_score >= 0.4:
            assessment = 'Moderate association'
        else:
            assessment = 'Weak association'

        reason_text = f" ({', '.join(reasons)})" if reasons else ""
        return f"{assessment}{reason_text}"

    def _apply_association_limits(
        self,
        associations: List[ImageProductAssociation],
        options: AssociationOptions
    ) -> List[ImageProductAssociation]:
        """Apply per-image and per-product association limits."""
        image_association_counts = {}
        product_association_counts = {}
        final_associations = []

        for association in associations:
            image_count = image_association_counts.get(association.image_id, 0)
            product_count = product_association_counts.get(association.product_id, 0)

            # Check limits
            if (image_count < options.max_associations_per_image and
                product_count < options.max_associations_per_product):

                final_associations.append(association)
                image_association_counts[association.image_id] = image_count + 1
                product_association_counts[association.product_id] = product_count + 1

        return final_associations

    async def _create_database_associations(
        self,
        associations: List[ImageProductAssociation]
    ) -> int:
        """Create database relationships from associations."""
        created = 0

        try:
            if not associations:
                return 0

            # Create product-image relationships
            # ✅ UPDATED: Use image_product_associations schema
            product_image_data = [
                {
                    "product_id": assoc.product_id,
                    "image_id": assoc.image_id,
                    "spatial_score": assoc.spatial_score,
                    "caption_score": assoc.caption_score,
                    "clip_score": assoc.clip_score,
                    "overall_score": assoc.overall_score,
                    "confidence": assoc.confidence,
                    "reasoning": "depicts",  # replaces relationship_type
                    "metadata": {}
                }
                for assoc in associations
            ]

            if product_image_data:
                result = self.supabase.client.table('image_product_associations').upsert(
                    product_image_data,
                    on_conflict='product_id,image_id'
                ).execute()

                if result.data:
                    created = len(result.data)

            # Store detailed association metadata
            association_metadata = [
                {
                    "image_id": assoc.image_id,
                    "product_id": assoc.product_id,
                    "spatial_score": assoc.spatial_score,
                    "caption_score": assoc.caption_score,
                    "clip_score": assoc.clip_score,
                    "overall_score": assoc.overall_score,
                    "confidence": assoc.confidence,
                    "reasoning": assoc.reasoning,
                    "metadata": assoc.metadata
                }
                for assoc in associations
            ]

            if association_metadata:
                self.supabase.client.table('image_product_associations').upsert(
                    association_metadata,
                    on_conflict='image_id,product_id'
                ).execute()

        except Exception as e:
            self.logger.error(f"❌ Error creating database associations: {e}")

        return created

    async def _get_document_images(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all images for a document."""
        try:
            result = self.supabase.client.table('document_images').select('*').eq(
                'document_id', document_id
            ).order('page_number').execute()

            return result.data or []
        except Exception as e:
            self.logger.error(f"❌ Error fetching document images: {e}")
            return []

    async def _get_document_products(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all products for a document."""
        try:
            result = self.supabase.client.table('products').select('*').eq(
                'source_document_id', document_id
            ).order('created_at').execute()

            return result.data or []
        except Exception as e:
            self.logger.error(f"❌ Error fetching document products: {e}")
            return []

    def _association_to_dict(self, association: ImageProductAssociation) -> Dict[str, Any]:
        """Convert association object to dictionary."""
        return {
            "image_id": association.image_id,
            "product_id": association.product_id,
            "spatial_score": association.spatial_score,
            "caption_score": association.caption_score,
            "clip_score": association.clip_score,
            "overall_score": association.overall_score,
            "confidence": association.confidence,
            "reasoning": association.reasoning,
            "metadata": association.metadata
        }

    async def get_document_association_stats(self, document_id: str) -> Dict[str, Any]:
        """Get association statistics for a document."""
        try:
            images = await self._get_document_images(document_id)
            products = await self._get_document_products(document_id)

            image_ids = [img['id'] for img in images]
            product_ids = [prod['id'] for prod in products]

            if not image_ids or not product_ids:
                return {
                    "total_images": len(images),
                    "total_products": len(products),
                    "total_associations": 0,
                    "average_confidence": 0,
                    "associations_by_score": {}
                }

            # Get associations
            result = self.supabase.client.table('image_product_associations').select(
                'overall_score, confidence'
            ).in_('image_id', image_ids).in_('product_id', product_ids).execute()

            associations = result.data or []
            total_associations = len(associations)

            average_confidence = (
                sum(assoc['confidence'] for assoc in associations) / total_associations
                if total_associations > 0 else 0
            )

            # Group associations by score ranges
            associations_by_score = {
                'high (0.8+)': 0,
                'good (0.6-0.8)': 0,
                'moderate (0.4-0.6)': 0,
                'low (<0.4)': 0
            }

            for assoc in associations:
                score = assoc['overall_score']
                if score >= 0.8:
                    associations_by_score['high (0.8+)'] += 1
                elif score >= 0.6:
                    associations_by_score['good (0.6-0.8)'] += 1
                elif score >= 0.4:
                    associations_by_score['moderate (0.4-0.6)'] += 1
                else:
                    associations_by_score['low (<0.4)'] += 1

            return {
                "total_images": len(images),
                "total_products": len(products),
                "total_associations": total_associations,
                "average_confidence": average_confidence,
                "associations_by_score": associations_by_score
            }

        except Exception as e:
            self.logger.error(f"❌ Error getting association stats: {e}")
            raise e

