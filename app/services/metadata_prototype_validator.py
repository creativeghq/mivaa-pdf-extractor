"""
Metadata Prototype Validation Service

This service validates AI-extracted metadata against prototype values using CLIP embeddings.
It standardizes free-text metadata to consistent, validated property values.

Architecture:
- Loads prototype embeddings from material_properties table
- Generates CLIP embeddings for extracted values
- Compares using cosine similarity
- Returns validated value if confidence > threshold

Integration:
- Runs AFTER DynamicMetadataExtractor
- Runs BEFORE database storage
- Non-breaking: falls back to original value if validation fails
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np

from app.core.supabase_client import get_supabase_client
from app.services.real_embeddings_service import RealEmbeddingsService

logger = logging.getLogger(__name__)


class MetadataPrototypeValidator:
    """Validates metadata against prototype values using CLIP embeddings."""
    
    def __init__(self, job_id: Optional[str] = None):
        """Initialize validator.
        
        Args:
            job_id: Optional job ID for logging
        """
        self.job_id = job_id
        self.supabase = get_supabase_client()
        self.embeddings_service = RealEmbeddingsService(job_id=job_id)
        self.logger = logging.getLogger(__name__)
        
        # Cache for prototype embeddings (loaded once per instance)
        self._prototype_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_loaded = False
    
    async def load_prototypes(self):
        """Load all prototype embeddings from database into cache."""
        if self._cache_loaded:
            return
        
        try:
            # Fetch all properties with prototypes
            result = self.supabase.client.table('material_properties').select(
                'property_key, name, prototype_descriptions, text_embedding_512'
            ).not_.is_('prototype_descriptions', 'null').not_.is_('text_embedding_512', 'null').execute()
            
            for prop in result.data:
                self._prototype_cache[prop['property_key']] = {
                    'name': prop['name'],
                    'prototypes': prop['prototype_descriptions'],
                    'embedding': np.array(prop['text_embedding_512'])
                }
            
            self._cache_loaded = True
            self.logger.info(f"Loaded {len(self._prototype_cache)} prototype properties")
            
        except Exception as e:
            self.logger.error(f"Failed to load prototypes: {e}")
            raise
    
    async def validate_metadata(
        self,
        extracted_metadata: Dict[str, Any],
        confidence_threshold: float = 0.80
    ) -> Dict[str, Any]:
        """Validate extracted metadata against prototypes.
        
        Args:
            extracted_metadata: Metadata extracted by DynamicMetadataExtractor
            confidence_threshold: Minimum similarity score to accept validation (default: 0.80)
        
        Returns:
            {
                "validated_metadata": {...},  # Validated values
                "validation_info": {...}      # Validation details for each field
            }
        """
        # Ensure prototypes are loaded
        await self.load_prototypes()
        
        validated_metadata = {}
        validation_info = {}
        
        # Flatten nested metadata structure
        flat_metadata = self._flatten_metadata(extracted_metadata)
        
        for field_key, field_value in flat_metadata.items():
            # Skip None values and validation metadata
            if field_value is None or field_key.startswith('_'):
                validated_metadata[field_key] = field_value
                continue
            
            # Check if this field has prototypes
            if field_key in self._prototype_cache:
                # Validate against prototypes
                validated_value, validation_details = await self._validate_field(
                    field_key=field_key,
                    field_value=str(field_value),
                    confidence_threshold=confidence_threshold
                )
                
                validated_metadata[field_key] = validated_value
                validation_info[field_key] = validation_details
            else:
                # No prototypes → keep original value
                validated_metadata[field_key] = field_value
        
        return {
            "validated_metadata": validated_metadata,
            "validation_info": validation_info
        }
    
    async def _validate_field(
        self,
        field_key: str,
        field_value: str,
        confidence_threshold: float
    ) -> Tuple[str, Dict[str, Any]]:
        """Validate a single field against its prototypes.
        
        Returns:
            (validated_value, validation_details)
        """
        try:
            # Generate embedding for extracted value (512D)
            value_embedding = await self.embeddings_service._generate_text_embedding(
                text=field_value,
                job_id=self.job_id,
                dimensions=512
            )
            
            if not value_embedding:
                # Embedding generation failed → keep original
                return field_value, {
                    "original_value": field_value,
                    "validated_value": field_value,
                    "prototype_matched": False,
                    "confidence": 0.0,
                    "reason": "embedding_generation_failed",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Get prototype embedding
            prototype_data = self._prototype_cache[field_key]
            prototype_embedding = prototype_data['embedding']
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(
                np.array(value_embedding),
                prototype_embedding
            )
            
            if similarity >= confidence_threshold:
                # High confidence → find best matching prototype value
                best_match = await self._find_best_prototype_match(
                    field_key=field_key,
                    field_value=field_value,
                    value_embedding=np.array(value_embedding)
                )
                
                return best_match['value'], {
                    "original_value": field_value,
                    "validated_value": best_match['value'],
                    "prototype_matched": True,
                    "confidence": best_match['confidence'],
                    "reason": "semantic_match",
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                # Low confidence → keep original value
                return field_value, {
                    "original_value": field_value,
                    "validated_value": field_value,
                    "prototype_matched": False,
                    "confidence": similarity,
                    "reason": "low_confidence",
                    "timestamp": datetime.utcnow().isoformat()
                }

        except Exception as e:
            self.logger.error(f"Field validation failed for {field_key}: {e}")
            # Fallback to original value
            return field_value, {
                "original_value": field_value,
                "validated_value": field_value,
                "prototype_matched": False,
                "confidence": 0.0,
                "reason": f"validation_error: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _find_best_prototype_match(
        self,
        field_key: str,
        field_value: str,
        value_embedding: np.ndarray
    ) -> Dict[str, Any]:
        """Find the best matching prototype value.

        Args:
            field_key: Property key (e.g., "finish")
            field_value: Extracted value (e.g., "shiny")
            value_embedding: CLIP embedding of extracted value

        Returns:
            {"value": "glossy", "confidence": 0.92}
        """
        prototype_data = self._prototype_cache[field_key]
        prototypes = prototype_data['prototypes']

        best_match = None
        best_similarity = 0.0

        # Check each prototype value
        for prototype_value, variations in prototypes.items():
            # Check exact match first
            if field_value.lower() == prototype_value.lower():
                return {"value": prototype_value, "confidence": 1.0}

            # Check variations
            for variation in variations:
                if field_value.lower() == variation.lower():
                    return {"value": prototype_value, "confidence": 1.0}

            # Calculate semantic similarity
            # Generate embedding for this prototype value
            prototype_embedding = await self.embeddings_service._generate_text_embedding(
                text=prototype_value,
                job_id=self.job_id,
                dimensions=512
            )

            if prototype_embedding:
                similarity = self._cosine_similarity(
                    value_embedding,
                    np.array(prototype_embedding)
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = prototype_value

        return {
            "value": best_match if best_match else field_value,
            "confidence": best_similarity
        }

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score between 0 and 1
        """
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)

        # Calculate cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)

        # Ensure result is between 0 and 1
        return float(max(0.0, min(1.0, similarity)))

    def _flatten_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested metadata structure.

        DynamicMetadataExtractor returns:
        {
            "critical": {"material_category": "ceramic"},
            "discovered": {
                "material_properties": {"finish": "glossy"},
                "performance": {"slip_resistance": "R11"}
            }
        }

        This flattens to:
        {
            "material_category": "ceramic",
            "finish": "glossy",
            "slip_resistance": "R11"
        }
        """
        flat = {}

        # Handle critical metadata
        if "critical" in metadata:
            for key, value in metadata["critical"].items():
                # Extract value from {"value": "...", "confidence": ...} structure
                if isinstance(value, dict) and "value" in value:
                    flat[key] = value["value"]
                else:
                    flat[key] = value

        # Handle discovered metadata (nested by category)
        if "discovered" in metadata:
            for category, fields in metadata["discovered"].items():
                if isinstance(fields, dict):
                    for key, value in fields.items():
                        # Extract value from {"value": "...", "confidence": ...} structure
                        if isinstance(value, dict) and "value" in value:
                            flat[key] = value["value"]
                        else:
                            flat[key] = value

        # Handle direct metadata (already flat)
        for key, value in metadata.items():
            if key not in ["critical", "discovered", "unknown", "metadata"]:
                if isinstance(value, dict) and "value" in value:
                    flat[key] = value["value"]
                else:
                    flat[key] = value

        return flat


# Singleton instance
_validator_instance: Optional[MetadataPrototypeValidator] = None


def get_metadata_validator(job_id: Optional[str] = None) -> MetadataPrototypeValidator:
    """Get singleton validator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = MetadataPrototypeValidator(job_id=job_id)
    return _validator_instance


