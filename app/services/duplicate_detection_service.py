"""
Duplicate Material Detection Service

CRITICAL RULE: Duplicates are ONLY detected when materials are from the SAME factory/manufacturer.
Visual similarity, color, or pattern alone do NOT constitute duplicates.

Detection criteria:
1. MUST have same factory/manufacturer in metadata (REQUIRED)
2. THEN check name similarity
3. THEN check description similarity
4. Visual similarity is supplementary only

If factory/manufacturer differs, materials are NOT duplicates regardless of visual similarity.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import numpy as np

from app.services.supabase_client import SupabaseClient
from app.utils.text_similarity import calculate_string_similarity, calculate_text_similarity

logger = logging.getLogger(__name__)


class DuplicateDetectionService:
    """
    Service for detecting duplicate products across the knowledge base.

    CRITICAL: Only detects duplicates from the SAME factory/manufacturer.
    Different factories = NOT duplicates, even if visually identical.
    """

    # Similarity thresholds (AFTER factory match)
    HIGH_CONFIDENCE_THRESHOLD = 0.85  # 85%+ = very likely duplicate
    MEDIUM_CONFIDENCE_THRESHOLD = 0.70  # 70-85% = possible duplicate
    LOW_CONFIDENCE_THRESHOLD = 0.55  # 55-70% = review needed

    # Weights for overall similarity score (AFTER factory match)
    WEIGHTS = {
        'name': 0.50,           # Name is most important after factory match
        'description': 0.30,    # Description secondary
        'metadata': 0.20        # Other metadata tertiary
        # Visual similarity NOT used for duplicate detection
    }

    # Factory metadata keys to check (in order of priority)
    FACTORY_KEYS = [
        'factory',
        'manufacturer',
        'factory_group',
        'brand',
        'company'
    ]

    def __init__(self, supabase_client: SupabaseClient):
        self.supabase = supabase_client
        self.logger = logger
    
    async def detect_duplicates_for_product(
        self,
        product_id: str,
        workspace_id: str,
        similarity_threshold: float = 0.60
    ) -> List[Dict[str, Any]]:
        """
        Find potential duplicates for a specific product.
        
        Args:
            product_id: Product to check for duplicates
            workspace_id: Workspace context
            similarity_threshold: Minimum similarity score (0.0-1.0)
            
        Returns:
            List of potential duplicate products with similarity scores
        """
        try:
            # Get the target product
            product_response = self.supabase.client.table('products').select('*').eq(
                'id', product_id
            ).eq('workspace_id', workspace_id).single().execute()
            
            if not product_response.data:
                self.logger.error(f"Product {product_id} not found")
                return []
            
            target_product = product_response.data

            # CRITICAL: Get factory/manufacturer from target product
            target_factory = self._extract_factory_info(target_product)

            if not target_factory:
                self.logger.warning(
                    f"Product {product_id} has no factory/manufacturer metadata. "
                    "Cannot detect duplicates without factory information."
                )
                return []

            # Get all other products in workspace
            all_products_response = self.supabase.client.table('products').select('*').eq(
                'workspace_id', workspace_id
            ).neq('id', product_id).execute()

            if not all_products_response.data:
                return []

            candidates = all_products_response.data

            # Calculate similarity for each candidate
            duplicates = []
            for candidate in candidates:
                # CRITICAL: First check if same factory
                candidate_factory = self._extract_factory_info(candidate)

                if not self._is_same_factory(target_factory, candidate_factory):
                    # Different factory = NOT a duplicate, skip
                    continue

                # Same factory - now check other similarities
                similarity_result = await self._calculate_similarity(
                    target_product,
                    candidate
                )

                if similarity_result['overall_score'] >= similarity_threshold:
                    duplicates.append({
                        'product': candidate,
                        'similarity': similarity_result,
                        'confidence_level': self._get_confidence_level(
                            similarity_result['overall_score']
                        )
                    })
            
            # Sort by similarity score (highest first)
            duplicates.sort(
                key=lambda x: x['similarity']['overall_score'],
                reverse=True
            )
            
            return duplicates
            
        except Exception as e:
            self.logger.error(f"Error detecting duplicates for product {product_id}: {e}")
            return []
    
    async def batch_detect_duplicates(
        self,
        workspace_id: str,
        similarity_threshold: float = 0.75,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Scan entire workspace for duplicate products.
        
        Args:
            workspace_id: Workspace to scan
            similarity_threshold: Minimum similarity score
            limit: Optional limit on number of products to check
            
        Returns:
            List of duplicate pairs with similarity scores
        """
        try:
            self.logger.info(f"Starting batch duplicate detection for workspace {workspace_id}")
            
            # Get all products in workspace
            query = self.supabase.client.table('products').select('*').eq(
                'workspace_id', workspace_id
            )
            
            if limit:
                query = query.limit(limit)
            
            products_response = query.execute()
            
            if not products_response.data:
                return []
            
            products = products_response.data
            self.logger.info(f"Checking {len(products)} products for duplicates")
            
            duplicate_pairs = []
            checked_pairs = set()
            
            # Compare each product with every other product
            for i, product1 in enumerate(products):
                for j, product2 in enumerate(products):
                    if i >= j:  # Skip self-comparison and already checked pairs
                        continue
                    
                    # Create unique pair identifier
                    pair_key = tuple(sorted([product1['id'], product2['id']]))
                    if pair_key in checked_pairs:
                        continue

                    checked_pairs.add(pair_key)

                    # CRITICAL: First check if same factory
                    factory1 = self._extract_factory_info(product1)
                    factory2 = self._extract_factory_info(product2)

                    if not self._is_same_factory(factory1, factory2):
                        # Different factory = NOT a duplicate, skip
                        continue

                    # Same factory - now calculate similarity
                    similarity_result = await self._calculate_similarity(
                        product1,
                        product2
                    )

                    if similarity_result['overall_score'] >= similarity_threshold:
                        duplicate_pairs.append({
                            'product1': product1,
                            'product2': product2,
                            'similarity': similarity_result,
                            'confidence_level': self._get_confidence_level(
                                similarity_result['overall_score']
                            ),
                            'factory': factory1  # Include factory info
                        })

                        # Cache the detection result
                        await self._cache_duplicate_detection(
                            workspace_id=workspace_id,
                            product1=product1,
                            product2=product2,
                            similarity_result=similarity_result
                        )
            
            # Sort by similarity score
            duplicate_pairs.sort(
                key=lambda x: x['similarity']['overall_score'],
                reverse=True
            )
            
            self.logger.info(
                f"Found {len(duplicate_pairs)} potential duplicate pairs "
                f"(threshold: {similarity_threshold})"
            )
            
            return duplicate_pairs
            
        except Exception as e:
            self.logger.error(f"Error in batch duplicate detection: {e}")
            return []
    
    def _extract_factory_info(self, product: Dict[str, Any]) -> Optional[str]:
        """
        Extract factory/manufacturer information from product metadata.

        Returns:
            Factory identifier (normalized) or None if not found
        """
        metadata = product.get('metadata', {})

        if not metadata:
            return None

        # Check each factory key in priority order
        for key in self.FACTORY_KEYS:
            value = metadata.get(key)
            if value:
                # Normalize factory name (lowercase, strip whitespace)
                return str(value).lower().strip()

        return None

    def _is_same_factory(
        self,
        factory1: Optional[str],
        factory2: Optional[str]
    ) -> bool:
        """
        Check if two factory identifiers match.

        CRITICAL: This is the primary filter for duplicate detection.
        If factories don't match, products are NOT duplicates.
        """
        if not factory1 or not factory2:
            return False

        # Exact match after normalization
        return factory1 == factory2

    async def _calculate_similarity(
        self,
        product1: Dict[str, Any],
        product2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate similarity between two products from the SAME factory.

        NOTE: This should only be called AFTER confirming same factory.

        Returns:
            Dictionary with individual and overall similarity scores
        """
        # 1. Name similarity (most important)
        name_sim = self._calculate_name_similarity(
            product1.get('name', ''),
            product2.get('name', '')
        )

        # 2. Description similarity
        desc_sim = await self._calculate_description_similarity(
            product1.get('description', ''),
            product2.get('description', '')
        )

        # 3. Metadata similarity (excluding factory since already matched)
        metadata_sim = self._calculate_metadata_similarity(
            product1.get('metadata', {}),
            product2.get('metadata', {})
        )

        # Calculate weighted overall score (NO visual similarity)
        overall_score = (
            self.WEIGHTS['name'] * name_sim +
            self.WEIGHTS['description'] * desc_sim +
            self.WEIGHTS['metadata'] * metadata_sim
        )

        return {
            'overall_score': overall_score,
            'name_similarity': name_sim,
            'description_similarity': desc_sim,
            'metadata_similarity': metadata_sim,
            'factory_matched': True,  # Always true when this method is called
            'breakdown': {
                'name': {'score': name_sim, 'weight': self.WEIGHTS['name']},
                'description': {'score': desc_sim, 'weight': self.WEIGHTS['description']},
                'metadata': {'score': metadata_sim, 'weight': self.WEIGHTS['metadata']}
            },
            'note': 'Factory/manufacturer already matched - this is prerequisite for duplicate detection'
        }
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate name similarity using sequence matching."""
        return calculate_string_similarity(name1, name2, case_sensitive=False)

    async def _calculate_description_similarity(
        self,
        desc1: str,
        desc2: str
    ) -> float:
        """Calculate semantic similarity between descriptions."""
        # Use text similarity with sequence matching
        # TODO: Use semantic embeddings for better accuracy
        return calculate_text_similarity(desc1, desc2, method="sequence")

    # Visual similarity REMOVED - not used for duplicate detection
    # Different materials can look similar but are NOT duplicates if from different factories
    def _calculate_metadata_similarity(
        self,
        metadata1: Dict[str, Any],
        metadata2: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity between product metadata.

        NOTE: Excludes factory keys since factory match is already confirmed.
        """
        if not metadata1 or not metadata2:
            return 0.0

        # Get all unique keys, EXCLUDING factory keys
        all_keys = (set(metadata1.keys()) | set(metadata2.keys())) - set(self.FACTORY_KEYS)

        if not all_keys:
            return 1.0  # No other metadata to compare, consider similar

        matching_keys = 0
        matching_values = 0

        for key in all_keys:
            if key in metadata1 and key in metadata2:
                matching_keys += 1

                # Compare values
                val1 = metadata1[key]
                val2 = metadata2[key]

                if val1 == val2:
                    matching_values += 1
                elif isinstance(val1, str) and isinstance(val2, str):
                    # Fuzzy match for strings
                    similarity = calculate_string_similarity(val1, val2)
                    if similarity > 0.8:
                        matching_values += 0.8

        # Calculate similarity as average of key and value matches
        key_similarity = matching_keys / len(all_keys)
        value_similarity = matching_values / len(all_keys) if all_keys else 0

        return (key_similarity + value_similarity) / 2

    def _get_confidence_level(self, similarity_score: float) -> str:
        """Determine confidence level based on similarity score."""
        if similarity_score >= self.HIGH_CONFIDENCE_THRESHOLD:
            return 'high'
        elif similarity_score >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            return 'medium'
        elif similarity_score >= self.LOW_CONFIDENCE_THRESHOLD:
            return 'low'
        else:
            return 'very_low'

    async def _cache_duplicate_detection(
        self,
        workspace_id: str,
        product1: Dict[str, Any],
        product2: Dict[str, Any],
        similarity_result: Dict[str, Any]
    ) -> None:
        """Cache duplicate detection result for future reference."""
        try:
            cache_data = {
                'workspace_id': workspace_id,
                'product_id_1': product1['id'],
                'product_id_2': product2['id'],
                'overall_similarity_score': similarity_result['overall_score'],
                'name_similarity': similarity_result['name_similarity'],
                'description_similarity': similarity_result['description_similarity'],
                'visual_similarity': None,  # Not used for duplicate detection
                'metadata_similarity': similarity_result['metadata_similarity'],
                'similarity_breakdown': similarity_result['breakdown'],
                'is_duplicate': similarity_result['overall_score'] >= self.HIGH_CONFIDENCE_THRESHOLD,
                'confidence_level': self._get_confidence_level(similarity_result['overall_score']),
                'status': 'pending'
            }

            # Insert or update cache
            # Note: Using insert instead of upsert since unique constraint handles duplicates
            self.supabase.client.table('duplicate_detection_cache').insert(
                cache_data
            ).execute()

        except Exception as e:
            self.logger.error(f"Error caching duplicate detection: {e}")

    async def get_cached_duplicates(
        self,
        workspace_id: str,
        status: Optional[str] = None,
        min_similarity: float = 0.60
    ) -> List[Dict[str, Any]]:
        """
        Get cached duplicate detections.

        Args:
            workspace_id: Workspace to query
            status: Filter by status ('pending', 'reviewed', 'merged', 'dismissed')
            min_similarity: Minimum similarity score

        Returns:
            List of cached duplicate pairs
        """
        try:
            query = self.supabase.client.table('duplicate_detection_cache').select(
                '*'
            ).eq('workspace_id', workspace_id).gte(
                'overall_similarity_score', min_similarity
            )

            if status:
                query = query.eq('status', status)

            response = query.order('overall_similarity_score', desc=True).execute()

            return response.data if response.data else []

        except Exception as e:
            self.logger.error(f"Error getting cached duplicates: {e}")
            return []

    async def update_duplicate_status(
        self,
        cache_id: str,
        status: str,
        user_id: str
    ) -> bool:
        """Update the status of a cached duplicate detection."""
        try:
            update_data = {
                'status': status,
                'reviewed_by': user_id,
                'reviewed_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }

            self.supabase.client.table('duplicate_detection_cache').update(
                update_data
            ).eq('id', cache_id).execute()

            return True

        except Exception as e:
            self.logger.error(f"Error updating duplicate status: {e}")
            return False

