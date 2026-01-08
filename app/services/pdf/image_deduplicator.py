"""
Image Deduplication Service

Uses perceptual hashing (pHash) to detect and remove duplicate images.
Handles duplicates across all extraction layers (embedded, full_render, yolo_crop).

Architecture:
1. Calculate perceptual hash for each image
2. Group images by similar hashes (Hamming distance < threshold)
3. Keep highest quality image from each group
4. Remove duplicates

Perceptual Hash Benefits:
- Detects duplicates even with slight variations (compression, resize, format)
- Fast comparison (Hamming distance on 64-bit hash)
- Works across different image formats
"""

import os
import logging
from typing import List, Dict, Any, Set, Tuple
from PIL import Image
import imagehash

logger = logging.getLogger(__name__)


class ImageDeduplicator:
    """
    Perceptual hash-based image deduplication service.
    
    Uses pHash (perceptual hash) to detect duplicate images across
    all extraction layers with configurable similarity threshold.
    """
    
    def __init__(self, hamming_threshold: int = 5):
        """
        Initialize image deduplicator.
        
        Args:
            hamming_threshold: Maximum Hamming distance for duplicates (default: 5)
                - 0: Exact match only
                - 5: Very similar (recommended for most cases)
                - 10: Similar with more tolerance
                - 15: Loosely similar
        """
        self.hamming_threshold = hamming_threshold
        logger.info(f"✅ ImageDeduplicator initialized (threshold: {hamming_threshold})")
    
    def calculate_phash(self, image_path: str) -> str:
        """
        Calculate perceptual hash for an image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Hex string representation of perceptual hash
        """
        try:
            image = Image.open(image_path)
            phash = imagehash.phash(image)
            return str(phash)
        except Exception as e:
            logger.error(f"Failed to calculate pHash for {image_path}: {e}")
            return ""
    
    def calculate_phash_from_pil(self, pil_image: Image.Image) -> str:
        """
        Calculate perceptual hash from PIL Image object.
        
        Args:
            pil_image: PIL Image object
        
        Returns:
            Hex string representation of perceptual hash
        """
        try:
            phash = imagehash.phash(pil_image)
            return str(phash)
        except Exception as e:
            logger.error(f"Failed to calculate pHash from PIL Image: {e}")
            return ""
    
    def are_duplicates(self, hash1: str, hash2: str) -> bool:
        """
        Check if two perceptual hashes represent duplicate images.
        
        Args:
            hash1: First perceptual hash (hex string)
            hash2: Second perceptual hash (hex string)
        
        Returns:
            True if images are duplicates (Hamming distance <= threshold)
        """
        try:
            h1 = imagehash.hex_to_hash(hash1)
            h2 = imagehash.hex_to_hash(hash2)
            distance = h1 - h2  # Hamming distance
            return distance <= self.hamming_threshold
        except Exception as e:
            logger.error(f"Failed to compare hashes: {e}")
            return False
    
    def deduplicate_images(
        self,
        images: List[Dict[str, Any]],
        keep_strategy: str = "highest_quality"
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Deduplicate images using perceptual hashing.
        
        Args:
            images: List of image dictionaries with 'path' key
            keep_strategy: Strategy for keeping images from duplicate groups
                - 'highest_quality': Keep image with largest file size
                - 'first': Keep first occurrence
                - 'embedded_priority': Prioritize embedded > yolo_crop > full_render
        
        Returns:
            Tuple of (unique_images, duplicate_images)
        """
        if not images:
            return [], []
        
        logger.info(f"🔍 Deduplicating {len(images)} images (threshold: {self.hamming_threshold})...")
        
        # Step 1: Calculate perceptual hashes
        images_with_hash = []
        for img in images:
            phash = self.calculate_phash(img['path'])
            if phash:
                img['perceptual_hash'] = phash
                images_with_hash.append(img)
            else:
                logger.warning(f"Skipping image without hash: {img.get('path')}")
        
        # Step 2: Group by similar hashes
        duplicate_groups = self._group_by_similarity(images_with_hash)
        
        # Step 3: Keep best image from each group
        unique_images = []
        duplicate_images = []
        
        for group in duplicate_groups:
            if len(group) == 1:
                # Not a duplicate
                unique_images.append(group[0])
            else:
                # Duplicate group - keep best one
                best_image = self._select_best_image(group, keep_strategy)
                unique_images.append(best_image)
                
                # Mark others as duplicates
                for img in group:
                    if img != best_image:
                        duplicate_images.append(img)
        
        logger.info(
            f"✅ Deduplication complete: {len(unique_images)} unique, "
            f"{len(duplicate_images)} duplicates removed"
        )
        
        return unique_images, duplicate_images

    def _group_by_similarity(
        self,
        images: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Group images by perceptual hash similarity.

        Args:
            images: List of images with 'perceptual_hash' key

        Returns:
            List of image groups (each group contains similar images)
        """
        groups = []
        processed = set()

        for i, img1 in enumerate(images):
            if i in processed:
                continue

            # Start new group with this image
            group = [img1]
            processed.add(i)

            # Find all similar images
            for j, img2 in enumerate(images[i+1:], start=i+1):
                if j in processed:
                    continue

                if self.are_duplicates(img1['perceptual_hash'], img2['perceptual_hash']):
                    group.append(img2)
                    processed.add(j)

            groups.append(group)

        return groups

    def _select_best_image(
        self,
        group: List[Dict[str, Any]],
        strategy: str
    ) -> Dict[str, Any]:
        """
        Select the best image from a duplicate group.

        Args:
            group: List of duplicate images
            strategy: Selection strategy

        Returns:
            Best image from the group
        """
        if strategy == "first":
            return group[0]

        elif strategy == "highest_quality":
            # Keep image with largest file size (proxy for quality)
            return max(group, key=lambda img: os.path.getsize(img['path']))

        elif strategy == "embedded_priority":
            # Priority: embedded > yolo_crop > full_render
            priority_map = {
                'embedded': 3,
                'yolo_crop': 2,
                'full_render': 1
            }

            # Sort by priority, then by file size
            sorted_group = sorted(
                group,
                key=lambda img: (
                    priority_map.get(img.get('extraction_layer', 'embedded'), 0),
                    os.path.getsize(img['path'])
                ),
                reverse=True
            )
            return sorted_group[0]

        else:
            logger.warning(f"Unknown strategy '{strategy}', using 'first'")
            return group[0]

    def get_duplicate_stats(
        self,
        unique_images: List[Dict[str, Any]],
        duplicate_images: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get statistics about deduplication results.

        Args:
            unique_images: List of unique images
            duplicate_images: List of duplicate images

        Returns:
            Dictionary with deduplication statistics
        """
        total_images = len(unique_images) + len(duplicate_images)

        # Count by extraction layer
        layer_counts = {}
        for img in unique_images:
            layer = img.get('extraction_layer', 'unknown')
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

        # Calculate space saved
        space_saved_bytes = sum(
            os.path.getsize(img['path']) for img in duplicate_images
            if os.path.exists(img['path'])
        )

        return {
            'total_images': total_images,
            'unique_images': len(unique_images),
            'duplicate_images': len(duplicate_images),
            'deduplication_rate': len(duplicate_images) / total_images if total_images > 0 else 0,
            'space_saved_bytes': space_saved_bytes,
            'space_saved_mb': space_saved_bytes / (1024 * 1024),
            'unique_by_layer': layer_counts
        }


