"""
Multi-Modal Image-Product Association Service

Creates intelligent image-product linking using:
- Spatial proximity (40% weight): Same page ±1, spatial distance
- Caption similarity (30% weight): Text similarity between image captions and product descriptions
- SLIG (SigLIP2) visual similarity (30% weight): Visual-text similarity using existing SLIG embeddings

Replaces random associations with weighted confidence scoring for meaningful relationships.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import math
import re

from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

@dataclass
class AssociationWeights:
    spatial: float = 0.4   # 40% weight for spatial proximity
    caption: float = 0.3   # 30% weight for caption similarity
    clip: float = 0.3      # 30% weight for SLIG (SigLIP2) visual similarity

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
    # Optional: None means the visual signal did not participate, which is not the
    # same claim as a score of 0.0. Persisted as SQL NULL.
    clip_score: Optional[float]
    overall_score: float
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]

class MultiModalImageProductAssociationService:
    """Service for creating intelligent image-product associations using multi-modal analysis."""

    def __init__(self):
        self.supabase = get_supabase_client()
        self.logger = logger
        # One warning per service instance, not one per image x product pair. A
        # per-pair log on a 40-image catalogue is 40 x N lines of the same sentence,
        # which is how a real signal gets tuned out.
        self._logged_visual_skip = False

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

            # The tenant this run belongs to, derived ONCE and verified (#25 M12-4).
            #
            # Images were fetched by document_id and products by source_document_id,
            # and the row written afterwards named only the two ids — no workspace
            # anywhere in the path. `image_product_associations` now carries
            # workspace_id with a composite FK to (products.id, workspace_id), so a
            # cross-tenant association is refused by Postgres; this derivation is what
            # supplies it, and the disagreement check below is what stops a corrupt
            # document silently picking whichever workspace sorted first.
            workspace_ids = {p.get('workspace_id') for p in products if p.get('workspace_id')}
            if len(workspace_ids) != 1:
                raise ValueError(
                    f"document {document_id} resolves to {len(workspace_ids)} workspaces "
                    f"({sorted(workspace_ids)}) — refusing to associate, because writing "
                    "either one would bind these images to a tenant that may not own them"
                )
            workspace_id = workspace_ids.pop()

            # document_images.workspace_id is nullable, so this is checked rather than
            # assumed. An image belonging to another tenant is a bug that must not be
            # resolved by trusting the product side.
            foreign = [
                i['id'] for i in images
                if i.get('workspace_id') and i['workspace_id'] != workspace_id
            ]
            if foreign:
                raise ValueError(
                    f"document {document_id}: {len(foreign)} image(s) belong to a "
                    f"different workspace than its products — first: {foreign[0]}"
                )

            self.logger.info(f"📊 Evaluating {len(images)} images × {len(products)} products = {len(images) * len(products)} potential associations")

            # Evaluate all image-product combinations
            all_associations = []
            total_evaluated = 0

            for image in images:
                for product in products:
                    total_evaluated += 1
                    
                    try:
                        association = await self._evaluate_association(image, product, options)

                        # Spatial is a GATE, not just the heaviest weight.
                        #
                        # _calculate_spatial_score returns 0.0 for an image that is
                        # not on one of the product's declared pages — the hard rule
                        # added 2026-05-02 after a 140-page catalog wrote 12
                        # associations for 4 images. But the weighted sum let that
                        # rule be voted down by two NEUTRAL scores: with spatial 0.0,
                        # overall = 0.3*caption + 0.3*clip, and both default to 0.5
                        # in the ordinary case — a generic "Image from page 22"
                        # caption (which is what the pipeline generates) and a
                        # product with no visual embedding (which is every product,
                        # since products carry only text_embedding_1024 and visual
                        # vectors are DERIVED from these very associations). That is
                        # 0.0 + 0.15 + 0.15 = exactly 0.30, which passed a >= 0.30
                        # threshold. So the wrong-page images the hard cutoff was
                        # written to exclude were being written anyway.
                        #
                        # Two neutral "I don't know"s must not add up to a "yes".
                        if association.spatial_score <= 0.0:
                            continue

                        if association.overall_score >= options.overall_threshold:
                            all_associations.append(association)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Error evaluating association between image {image['id']} and product {product['id']}: {e}")

            # Sort by overall score and apply limits
            all_associations.sort(key=lambda x: x.overall_score, reverse=True)
            final_associations = self._apply_association_limits(all_associations, options)

            # Create database relationships
            associations_created = await self._create_database_associations(
                final_associations, workspace_id
            )

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

        # Weighted over the signals that PARTICIPATED. A signal that could not be
        # computed is dropped and the remaining weights are renormalised, so an absent
        # visual score neither contributes a phantom constant nor silently rescales the
        # threshold everything else is compared against.
        weights = options.weights
        components = [
            (spatial_score, weights.spatial),
            (caption_score, weights.caption),
        ]
        if clip_score is not None:
            components.append((clip_score, weights.clip))
        total_weight = sum(w for _, w in components) or 1.0
        overall_score = sum(score * w for score, w in components) / total_weight

        # Confidence sees the same set. Passing a placeholder here is what let a
        # constant masquerade as agreement between independent signals.
        confidence = self._calculate_confidence(
            [score for score, _ in components], overall_score
        )

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
                # `has_image_embedding` read the same nonexistent columns as the
                # scorer, so it was stored False on every association regardless of what
                # VECS actually held. The `has_*_embedding` booleans are the canonical
                # check; `participated` records whether the signal reached the score at
                # all, which is what a reader of this row actually needs to know.
                "clip_similarity": {
                    "visual_text_similarity": clip_score,
                    "participated": clip_score is not None,
                    "has_image_embedding": bool(
                        image.get('has_slig_embedding') or image.get('has_understanding_embedding')
                    ),
                    "has_product_embedding": bool(product.get('text_embedding_1024'))
                }
            }
        )

    async def _calculate_spatial_score(self, image: Dict[str, Any], product: Dict[str, Any]) -> float:
        """Calculate spatial proximity score (0-1).

        HARD RULE (added 2026-05-02 after audit incident): an image is
        spatially associated with a product **only if its page_number falls
        inside the product's declared page_range**. Adjacent / nearby pages
        score 0.0 — they belong to the *next* product, not this one.

        Previously this method gave 0.85 to adjacent pages, 0.7 to ±2
        pages, etc. With overall_threshold=0.3 that meant every image got
        linked to 2-3 products in dense catalogs. Concrete failure: a 140-
        page catalog (job 184ad4cf) wrote 12 image_product_associations
        for 4 images — every page-34/36 image was linked to VALENOVA (24-
        31), FOLD (32-37), AND PIQUÉ (38-51). Search retrieval was
        contaminated by 3× the correct row count.
        """
        image_page = image.get('page_number', 0)
        if not image_page:
            return 0.0  # No image page info — can't anchor it.

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
            return 0.0  # No product page info — can't anchor it.

        # HARD CUTOFF: image page must be inside the product's page set.
        if int(image_page) not in product_pages:
            return 0.0

        # Inside the range — full proximity score. We no longer score by
        # distance because the binary "is the image on one of this
        # product's pages?" is the only safe signal.
        return 1.0

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

    async def _calculate_clip_score(
        self, image: Dict[str, Any], product: Dict[str, Any]
    ) -> Optional[float]:
        """The visual similarity score, or None when there is no visual signal.

        None is the point of this function (#25 M12-1). What was here read three
        columns that do not exist:

            image.get('clip_embedding') or image.get('visual_embedding') or image.get('embedding')

        `document_images` has none of them — image vectors live in VECS, and the
        canonical O(1) presence check is the `has_*_embedding` boolean. The rows are
        fetched with `select('*')`, so PostgREST returned the columns that DO exist and
        `.get()` answered None for the rest: no KeyError, no warning, the `or` chain
        collapsed, and the documented "neutral score" fallback fired on every call ever
        made. The product side guessed too — `text_embedding`, where the column is
        `text_embedding_1024`.

        A constant 0.5 at 30% weight is not a neutral fallback. It is:

          * a fixed +0.15 on every overall score, so the threshold means something
            different from what it says
          * a third data point in `_calculate_confidence`'s variance, which rewards
            "agreement" with a number that agrees with nothing
          * the string 'moderate visual relevance' in the stored reasoning of every
            association, because 0.5 clears that branch's threshold exactly

        So returning None and renormalising is not a smaller answer than 0.5 — it is
        the difference between an absent signal and a fabricated one.

        The comment that caused it was `# could be under different field names`. The
        author was unsure of the schema and hedged across three guesses; the platform
        rule against exactly that hedge is why `has_slig_embedding` exists.
        """
        try:
            # The canonical O(1) checks. Cheap, and true only when a vector really
            # exists in the matching VECS collection.
            has_visual = bool(
                image.get('has_slig_embedding') or image.get('has_understanding_embedding')
            )
            if not has_visual:
                return None

            # A vector exists but this service does not yet read VECS, so it still
            # cannot participate. Said out loud rather than scored: the honest failure
            # marker (pipeline convention 1) instead of the silent constant that hid
            # this for the lifetime of the service.
            if not self._logged_visual_skip:
                self._logged_visual_skip = True
                self.logger.warning(
                    "⚠️ Visual similarity is not contributing: the image has an embedding "
                    "in VECS but this service does not read it, so associations are "
                    "scored on spatial + caption signal only. Weights are renormalised "
                    "over the signals that actually participated."
                )
            return None

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating visual similarity: {e}")
            return None

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
        scores: List[float],
        overall_score: float
    ) -> float:
        """Confidence from the spread of the signals that actually participated.

        Takes the list rather than three positional scores so an absent signal is
        ABSENT here too. It previously received a constant 0.5 as its third data point,
        which pulled the variance toward zero and handed out a consistency bonus for
        agreeing with a number that measured nothing.
        """
        if not scores:
            return 0.0
        
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
        clip_score: Optional[float],
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

        # Visual reasoning — only when there WAS a visual signal. The constant 0.5
        # cleared the second branch exactly, so 'moderate visual relevance' was written
        # into the stored reasoning of every association this service ever made,
        # describing a comparison that never happened.
        if clip_score is not None:
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
        associations: List[ImageProductAssociation],
        workspace_id: str
    ) -> int:
        """Write the associations. ONE upsert, tenant-bound (#25 M12-4, M12-5).

        This used to upsert the same rows to the same table TWICE: first with
        `reasoning: "depicts"` and `metadata: {}`, then again with the real reasoning
        and the score breakdown. Both were wrapped in one try that logged and swallowed,
        and `created` — taken from the FIRST write — was returned either way.

        So a failure of the second write left a complete-looking association whose
        stated reason was the placeholder and whose metadata was empty, reported as a
        success. That is indistinguishable from a genuine low-information association,
        which is the ambiguity pipeline convention 3 exists to remove: one atomic write,
        not two that can disagree.

        The first write was also pure waste — same table, same conflict target, same
        rows, immediately overwritten.
        """
        if not associations:
            return 0

        rows = [
            {
                "workspace_id": workspace_id,
                "image_id": assoc.image_id,
                "product_id": assoc.product_id,
                "spatial_score": assoc.spatial_score,
                "caption_score": assoc.caption_score,
                # NULL when the visual signal did not participate. The column was made
                # nullable for this: a score of 0.0 would read as "looked and found no
                # resemblance", which is a different claim.
                "clip_score": assoc.clip_score,
                "overall_score": assoc.overall_score,
                "confidence": assoc.confidence,
                "reasoning": assoc.reasoning,
                "metadata": assoc.metadata,
            }
            for assoc in associations
        ]

        try:
            result = self.supabase.client.table('image_product_associations').upsert(
                rows,
                on_conflict='image_id,product_id'
            ).execute()
        except Exception as e:
            # Raised, not swallowed. The caller reports `associations_created` to the
            # orchestrator, and returning 0 after a failed write is the silent-zero
            # shape: a run that wrote nothing looks exactly like a catalogue that
            # earned no associations.
            self.logger.error(f"❌ Error creating database associations: {e}")
            raise

        return len(result.data or [])

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

