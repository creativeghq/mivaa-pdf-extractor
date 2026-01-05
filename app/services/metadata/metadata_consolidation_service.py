"""
Metadata Consolidation Service

Consolidates metadata from multiple sources:
1. AI text extraction (DynamicMetadataExtractor)
2. Visual metadata (from embeddings)
3. Factory defaults (from catalog)

This service implements Stage 4 metadata consolidation.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MetadataConsolidationService:
    """
    Service for consolidating metadata from multiple sources.
    Implements priority-based merging with confidence tracking.
    """

    def __init__(self):
        pass

    def consolidate_metadata(
        self,
        ai_metadata: Optional[Dict[str, Any]] = None,
        visual_metadata: Optional[Dict[str, Any]] = None,
        factory_defaults: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Consolidate metadata from multiple sources with priority-based merging.

        Priority order (highest to lowest):
        1. AI text extraction (most reliable for text-based fields)
        2. Visual metadata (reliable for visual properties)
        3. Factory defaults (fallback values)

        Args:
            ai_metadata: Metadata from AI text extraction
            visual_metadata: Metadata from visual embeddings
            factory_defaults: Default metadata from catalog

        Returns:
            Consolidated metadata with source tracking
        """
        try:
            logger.info("🔄 Consolidating metadata from multiple sources")

            consolidated = {}
            extraction_metadata = {}

            # Start with factory defaults (lowest priority)
            if factory_defaults:
                for key, value in factory_defaults.items():
                    if value is not None:
                        consolidated[key] = value
                        extraction_metadata[key] = {
                            "source": "factory_default",
                            "confidence": 0.5
                        }

            # Merge visual metadata (medium priority)
            if visual_metadata:
                for key, value_data in visual_metadata.items():
                    if isinstance(value_data, dict) and 'primary' in value_data:
                        # Visual metadata has structure: {"primary": "value", "confidence": 0.88}
                        value = value_data.get('primary')
                        confidence = value_data.get('confidence', 0.8)

                        if value is not None:
                            # Special handling for color field
                            if key == "color":
                                # Don't add to main "color" field (will be handled in AI text section)
                                # Just store as visual_color_detected if no AI text color exists
                                if "colors" not in consolidated:
                                    # No AI text colors, use visual color as fallback
                                    consolidated["visual_color_detected"] = value
                                    extraction_metadata["visual_color_detected"] = {
                                        "source": "visual_embedding",
                                        "confidence": confidence,
                                        "secondary_values": value_data.get('secondary', [])
                                    }
                            else:
                                # Only override if confidence is higher or field doesn't exist
                                existing_confidence = extraction_metadata.get(key, {}).get('confidence', 0.0)
                                if confidence >= existing_confidence:
                                    consolidated[key] = value
                                    extraction_metadata[key] = {
                                        "source": "visual_embedding",
                                        "confidence": confidence,
                                        "secondary_values": value_data.get('secondary', [])
                                    }

            # Merge AI text extraction (highest priority)
            if ai_metadata:
                for key, value in ai_metadata.items():
                    if value is not None and value != "":
                        # Special handling for color/colors field
                        if key in ["color", "colors"]:
                            # Normalize to "colors" array
                            if isinstance(value, list):
                                consolidated["colors"] = value
                            elif isinstance(value, str):
                                consolidated["colors"] = [value]
                            else:
                                consolidated["colors"] = value

                            extraction_metadata["colors"] = {
                                "source": "ai_text_extraction",
                                "confidence": 0.95
                            }

                            # If visual metadata detected a color, add it as visual_color_detected
                            if visual_metadata and "color" in visual_metadata:
                                visual_color_data = visual_metadata["color"]
                                if isinstance(visual_color_data, dict) and 'primary' in visual_color_data:
                                    consolidated["visual_color_detected"] = visual_color_data.get('primary')
                                    extraction_metadata["visual_color_detected"] = {
                                        "source": "visual_embedding",
                                        "confidence": visual_color_data.get('confidence', 0.8),
                                        "secondary_values": visual_color_data.get('secondary', [])
                                    }
                        else:
                            # AI text extraction always wins (highest confidence)
                            consolidated[key] = value
                            extraction_metadata[key] = {
                                "source": "ai_text_extraction",
                                "confidence": 0.95
                            }

            # Add extraction metadata to track sources
            consolidated['_extraction_metadata'] = extraction_metadata
            consolidated['_consolidation_timestamp'] = datetime.utcnow().isoformat()

            # Log color consolidation details
            if "colors" in consolidated or "visual_color_detected" in consolidated:
                colors_info = []
                if "colors" in consolidated:
                    colors_info.append(f"colors={consolidated['colors']}")
                if "visual_color_detected" in consolidated:
                    colors_info.append(f"visual_detected={consolidated['visual_color_detected']}")
                logger.info(f"🎨 Color consolidation: {', '.join(colors_info)}")

            logger.info(f"✅ Consolidated {len(consolidated)} metadata fields from {self._count_sources(extraction_metadata)} sources")
            return consolidated

        except Exception as e:
            logger.error(f"❌ Failed to consolidate metadata: {e}")
            return ai_metadata or visual_metadata or factory_defaults or {}

    def _count_sources(self, extraction_metadata: Dict[str, Any]) -> int:
        """Count unique sources in extraction metadata."""
        sources = set()
        for field_meta in extraction_metadata.values():
            if isinstance(field_meta, dict) and 'source' in field_meta:
                sources.add(field_meta['source'])
        return len(sources)

    def merge_with_priority(
        self,
        existing_metadata: Dict[str, Any],
        new_metadata: Dict[str, Any],
        source: str,
        confidence: float = 0.8
    ) -> Dict[str, Any]:
        """
        Merge new metadata into existing metadata with priority tracking.

        Args:
            existing_metadata: Current metadata
            new_metadata: New metadata to merge
            source: Source of new metadata
            confidence: Confidence score for new metadata

        Returns:
            Merged metadata
        """
        merged = existing_metadata.copy()
        extraction_meta = merged.get('_extraction_metadata', {})

        for key, value in new_metadata.items():
            if value is not None and value != "":
                existing_confidence = extraction_meta.get(key, {}).get('confidence', 0.0)
                
                if confidence >= existing_confidence:
                    merged[key] = value
                    extraction_meta[key] = {
                        "source": source,
                        "confidence": confidence
                    }

        merged['_extraction_metadata'] = extraction_meta
        return merged

