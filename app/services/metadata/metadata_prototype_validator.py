"""
Metadata Prototype Validation Service

This service validates AI-extracted metadata against prototype values using Voyage AI embeddings (1024D).
It standardizes free-text metadata to consistent, validated property values.

Architecture:
- Loads prototype embeddings from the material_metadata_fields registry (#347 phase 4.1)
- Generates Voyage AI 1024D embeddings for extracted values
- Compares using cosine similarity
- Returns validated value if confidence > threshold

Integration:
- Runs AFTER DynamicMetadataExtractor
- Runs BEFORE database storage
- Non-breaking: falls back to original value if validation fails
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

from app.services.core.supabase_client import get_supabase_client
from app.services.embeddings.real_embeddings_service import RealEmbeddingsService

logger = logging.getLogger(__name__)


# ── Per-category confidence threshold (audit #217 M3) ─────────────────────────
# Admins tune `material_categories.ai_confidence_threshold` per category bucket
# (e.g. decor/furniture/lighting looser at 0.60, raw materials stricter at 0.80).
# The prototype validator previously ignored that and hardcoded 0.80, so the admin
# control was inert. The ingestion path now resolves the product's category
# threshold here. Cached per-process (5-min TTL); falls back to the default.
import time as _time

_DEFAULT_CONFIDENCE_THRESHOLD = 0.80
_category_threshold_cache: Optional[Dict[str, float]] = None
_category_threshold_ts: float = 0.0
_CATEGORY_THRESHOLD_TTL_SECONDS = 300


async def get_category_confidence_threshold(
    category_key: Optional[str],
    default: float = _DEFAULT_CONFIDENCE_THRESHOLD,
) -> float:
    """Resolve the admin-configured prototype-validation threshold for a material
    category bucket (material_categories.category_key). Returns `default` for an
    unknown/empty category or on DB error."""
    global _category_threshold_cache, _category_threshold_ts
    if not category_key:
        return default
    now = _time.time()
    if _category_threshold_cache is None or (now - _category_threshold_ts) >= _CATEGORY_THRESHOLD_TTL_SECONDS:
        try:
            client = get_supabase_client().client
            res = (
                client.table('material_categories')
                .select('category_key, ai_confidence_threshold')
                .eq('is_active', True)
                .execute()
            )
            cache: Dict[str, float] = {}
            for r in (res.data or []):
                k = (r.get('category_key') or '').strip().lower()
                t = r.get('ai_confidence_threshold')
                if k and t is not None:
                    try:
                        cache[k] = float(t)
                    except (TypeError, ValueError):
                        pass
            _category_threshold_cache = cache
            _category_threshold_ts = now
        except Exception as e:
            logger.warning(f"category threshold load failed (using default {default}): {e}")
            if _category_threshold_cache is None:
                return default
    return _category_threshold_cache.get(category_key.strip().lower(), default)


class MetadataPrototypeValidator:
    """Validates metadata against prototype values using CLIP embeddings."""
    
    def __init__(self, job_id: Optional[str] = None):
        """Initialize validator.
        
        Args:
            job_id: Optional job ID for logging
        """
        self.job_id = job_id
        self.supabase = get_supabase_client()
        self.embeddings_service = RealEmbeddingsService()
        self.logger = logging.getLogger(__name__)
        
        # Cache for prototype embeddings (loaded once per instance)
        self._prototype_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_loaded = False
    
    async def load_prototypes(self):
        """Load the controlled vocabulary for every field that has one.

        #347: this reads `dropdown_options` — the hand-curated allowed values an admin already
        maintains for 51 fields — NOT `prototype_descriptions`, which no row has ever carried.
        That column is why this validator did nothing: it demanded a second, parallel curation
        nobody ever wrote, while the real curation sat in the next column over.

        47 of those 51 fields are NOT canonicalizable, so facet canonicalization never sees
        them. For those, this is the only thing between an extracted value and products.metadata.
        """
        if self._cache_loaded:
            return

        result = self.supabase.client.table("material_metadata_fields").select(
            "field_name, display_name, dropdown_options"
        ).eq("status", "active").not_.is_("dropdown_options", "null").execute()

        for prop in (result.data or []):
            options = [o for o in (prop.get("dropdown_options") or []) if o and str(o).strip()]
            if not options:
                continue
            self._prototype_cache[prop["field_name"]] = {
                "name": prop.get("display_name") or prop["field_name"],
                "options": options,
                # Option embeddings are generated LAZILY, only for a field whose value misses the
                # free exact match. A catalog that already uses the controlled values costs zero
                # embedding calls.
                "option_embeddings": None,
            }

        self._cache_loaded = True

        if not self._prototype_cache:
            self.logger.warning(
                "Metadata validator loaded 0 controlled vocabularies from "
                "material_metadata_fields.dropdown_options - every value passes through "
                "UNVALIDATED. Populate dropdown_options, or stop calling this validator."
            )
        else:
            self.logger.info(
                "Loaded controlled vocabulary for %d fields", len(self._prototype_cache)
            )
    
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
        """Snap one value onto its field's controlled vocabulary.

        Two passes, cheapest first:
          1. normalised exact match against the allowed values - free, and the common case
          2. embedding similarity against each allowed value; best above threshold wins

        A value matching nothing is KEPT AS-IS. Dropping a real value because it is not yet in
        the list would be a worse failure than an un-normalised one.
        """
        entry = self._prototype_cache[field_key]
        options: List[str] = entry["options"]
        normalized = field_value.strip().lower()

        for option in options:
            if normalized == str(option).strip().lower():
                return option, {
                    "original_value": field_value,
                    "validated_value": option,
                    "prototype_matched": True,
                    "confidence": 1.0,
                    "reason": "exact_match",
                    "timestamp": datetime.utcnow().isoformat(),
                }

        try:
            value_embedding = await self.embeddings_service._generate_text_embedding(
                text=field_value, job_id=self.job_id, dimensions=1024
            )
            if not value_embedding:
                return field_value, {
                    "original_value": field_value,
                    "validated_value": field_value,
                    "prototype_matched": False,
                    "confidence": 0.0,
                    "reason": "embedding_generation_failed",
                    "timestamp": datetime.utcnow().isoformat(),
                }

            if entry["option_embeddings"] is None:
                entry["option_embeddings"] = await self._embed_options(field_key, options)

            best_value, best_similarity = None, 0.0
            for option, option_embedding in entry["option_embeddings"].items():
                similarity = self._cosine_similarity(
                    np.array(value_embedding), np.array(option_embedding)
                )
                if similarity > best_similarity:
                    best_similarity, best_value = similarity, option

            if best_value is not None and best_similarity >= confidence_threshold:
                return best_value, {
                    "original_value": field_value,
                    "validated_value": best_value,
                    "prototype_matched": True,
                    "confidence": best_similarity,
                    "reason": "semantic_match",
                    "timestamp": datetime.utcnow().isoformat(),
                }

            return field_value, {
                "original_value": field_value,
                "validated_value": field_value,
                "prototype_matched": False,
                "confidence": best_similarity,
                "reason": "low_confidence",
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            self.logger.error("Field validation failed for %s: %s", field_key, e)
            return field_value, {
                "original_value": field_value,
                "validated_value": field_value,
                "prototype_matched": False,
                "confidence": 0.0,
                "reason": "validation_error: " + str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def _embed_options(
        self, field_key: str, options: List[str]
    ) -> Dict[str, List[float]]:
        """Embed a field's allowed values once, on first miss. Cached for the process."""
        embeddings: Dict[str, List[float]] = {}
        for option in options:
            vec = await self.embeddings_service._generate_text_embedding(
                text=str(option), job_id=self.job_id, dimensions=1024
            )
            if vec:
                embeddings[option] = vec
        self.logger.info(
            "Embedded %d/%d controlled values for %s", len(embeddings), len(options), field_key
        )
        return embeddings


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



