"""
Real Quality Scoring Service - Step 5 Implementation

Calculates real quality scores based on actual data instead of hardcoded values:
1. Image Quality Scores - Based on analysis results, dimensions, format
2. Chunk Quality Scores - Based on content length, coherence, boundaries
3. Product Quality Scores - Based on metadata, properties, embeddings
4. Confidence Scores - Based on model confidence and data completeness

Replaces all hardcoded quality scores (0.85, 0.90, 0.95) with real calculations.
"""

import logging
import math
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class RealQualityScoringService:
    """
    Calculates real quality scores based on actual data.
    
    This service replaces hardcoded quality scores with real calculations:
    - Image quality based on analysis, dimensions, format
    - Chunk quality based on content, coherence, boundaries
    - Product quality based on metadata, properties, embeddings
    - Confidence scores based on model outputs
    """
    
    # Quality score weights
    IMAGE_QUALITY_WEIGHTS = {
        "analysis_completeness": 0.30,  # Has Qwen, Claude, CLIP analysis
        "dimensions_quality": 0.20,      # Image dimensions and format
        "material_properties": 0.25,     # Extracted material properties
        "embedding_coverage": 0.15,      # Has all embedding types
        "confidence_score": 0.10         # Model confidence
    }
    
    CHUNK_QUALITY_WEIGHTS = {
        "content_length": 0.20,           # Appropriate chunk size
        "coherence": 0.25,                # Semantic coherence
        "boundary_quality": 0.20,         # Proper sentence boundaries
        "metadata_richness": 0.15,        # Has metadata
        "embedding_coverage": 0.20        # Has embeddings
    }
    
    PRODUCT_QUALITY_WEIGHTS = {
        "metadata_completeness": 0.25,    # Name, description, properties
        "material_properties": 0.20,      # Color, finish, texture, etc.
        "embedding_coverage": 0.20,       # All 6 embedding types
        "related_images": 0.15,           # Linked images
        "related_products": 0.10,         # Related products
        "confidence_score": 0.10          # Model confidence
    }
    
    def __init__(self, supabase_client=None):
        """Initialize quality scoring service."""
        self.supabase = supabase_client
        self.logger = logger
    
    def calculate_image_quality_score(
        self,
        image_data: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate real image quality score based on actual data.
        
        Args:
            image_data: Image data from database
            
        Returns:
            Tuple of (quality_score, detailed_metrics)
        """
        try:
            metrics = {}
            
            # 1. Analysis Completeness (0-1)
            analysis_completeness = 0.0
            if image_data.get('vision_analysis'):
                analysis_completeness += 0.33
            if image_data.get('claude_validation'):
                analysis_completeness += 0.33
            if image_data.get('clip_embedding'):
                analysis_completeness += 0.34
            metrics['analysis_completeness'] = analysis_completeness
            
            # 2. Dimensions Quality (0-1)
            dimensions_quality = self._calculate_dimensions_quality(image_data)
            metrics['dimensions_quality'] = dimensions_quality
            
            # 3. Material Properties (0-1)
            material_properties = image_data.get('material_properties', {})
            properties_score = min(1.0, len(material_properties) / 5)  # 5 properties = 1.0
            metrics['material_properties'] = properties_score
            
            # 4. Embedding Coverage (0-1)
            embedding_coverage = self._calculate_embedding_coverage(image_data)
            metrics['embedding_coverage'] = embedding_coverage
            
            # 5. Confidence Score (0-1)
            confidence_score = image_data.get('confidence_score', 0.0)
            metrics['confidence_score'] = confidence_score
            
            # Calculate weighted quality score
            quality_score = (
                metrics['analysis_completeness'] * self.IMAGE_QUALITY_WEIGHTS['analysis_completeness'] +
                metrics['dimensions_quality'] * self.IMAGE_QUALITY_WEIGHTS['dimensions_quality'] +
                metrics['material_properties'] * self.IMAGE_QUALITY_WEIGHTS['material_properties'] +
                metrics['embedding_coverage'] * self.IMAGE_QUALITY_WEIGHTS['embedding_coverage'] +
                metrics['confidence_score'] * self.IMAGE_QUALITY_WEIGHTS['confidence_score']
            )
            
            return round(quality_score, 3), metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating image quality score: {e}")
            return 0.0, {}
    
    def calculate_chunk_quality_score(
        self,
        chunk_data: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate real chunk quality score based on actual data.
        
        Args:
            chunk_data: Chunk data from database
            
        Returns:
            Tuple of (quality_score, detailed_metrics)
        """
        try:
            metrics = {}
            content = chunk_data.get('content', '')
            
            # 1. Content Length (0-1)
            # Optimal chunk size: 500-1500 characters
            content_length = len(content)
            if 500 <= content_length <= 1500:
                content_length_score = 1.0
            elif 300 <= content_length < 500 or 1500 < content_length <= 2000:
                content_length_score = 0.8
            elif 200 <= content_length < 300 or 2000 < content_length <= 2500:
                content_length_score = 0.6
            else:
                content_length_score = 0.4
            metrics['content_length'] = content_length_score
            
            # 2. Coherence (0-1)
            coherence_score = chunk_data.get('coherence_score', 0.0)
            if coherence_score == 0:
                # Calculate from content
                coherence_score = self._calculate_coherence(content)
            metrics['coherence'] = coherence_score
            
            # 3. Boundary Quality (0-1)
            boundary_quality = self._calculate_boundary_quality(content)
            metrics['boundary_quality'] = boundary_quality
            
            # 4. Metadata Richness (0-1)
            metadata = chunk_data.get('metadata', {})
            metadata_richness = min(1.0, len(metadata) / 5)  # 5 fields = 1.0
            metrics['metadata_richness'] = metadata_richness

            # 5. Embedding Coverage (0-1)
            embedding_coverage = 1.0 if chunk_data.get('text_embedding_1024') else 0.0
            metrics['embedding_coverage'] = embedding_coverage
            
            # Calculate weighted quality score
            quality_score = (
                metrics['content_length'] * self.CHUNK_QUALITY_WEIGHTS['content_length'] +
                metrics['coherence'] * self.CHUNK_QUALITY_WEIGHTS['coherence'] +
                metrics['boundary_quality'] * self.CHUNK_QUALITY_WEIGHTS['boundary_quality'] +
                metrics['metadata_richness'] * self.CHUNK_QUALITY_WEIGHTS['metadata_richness'] +
                metrics['embedding_coverage'] * self.CHUNK_QUALITY_WEIGHTS['embedding_coverage']
            )
            
            return round(quality_score, 3), metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating chunk quality score: {e}")
            return 0.0, {}
    
    def calculate_product_quality_score(
        self,
        product_data: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        ✅ ENHANCED: Calculate real product quality score with granular metrics.

        Args:
            product_data: Product data from database

        Returns:
            Tuple of (quality_score, detailed_metrics)
        """
        try:
            metrics = {}

            # 1. ✅ ENHANCED: Metadata Completeness with quality assessment (0-1)
            metadata_score = 0.0
            name = product_data.get('name', '')
            description = product_data.get('description', '')
            long_description = product_data.get('long_description', '')
            properties = product_data.get('properties', {})

            # Name quality (0-0.25)
            if name:
                name_length = len(name)
                if name_length >= 10:  # Good descriptive name
                    metadata_score += 0.25
                elif name_length >= 5:  # Acceptable name
                    metadata_score += 0.15
                else:  # Short name
                    metadata_score += 0.10

            # Description quality (0-0.25)
            if description:
                desc_length = len(description)
                if desc_length >= 200:  # Detailed description
                    metadata_score += 0.25
                elif desc_length >= 100:  # Good description
                    metadata_score += 0.20
                elif desc_length >= 50:  # Basic description
                    metadata_score += 0.15
                else:  # Minimal description
                    metadata_score += 0.10

            # Long description quality (0-0.25)
            if long_description:
                long_desc_length = len(long_description)
                if long_desc_length >= 500:  # Comprehensive
                    metadata_score += 0.25
                elif long_desc_length >= 300:  # Detailed
                    metadata_score += 0.20
                elif long_desc_length >= 150:  # Good
                    metadata_score += 0.15
                else:  # Basic
                    metadata_score += 0.10

            # Properties presence (0-0.25)
            if properties and len(properties) > 0:
                metadata_score += 0.25

            metrics['metadata_completeness'] = min(1.0, metadata_score)

            # 2. ✅ ENHANCED: Material Properties with diversity assessment (0-1)
            properties_score = 0.0
            if properties:
                # Count non-empty, non-unknown properties
                valid_properties = [
                    v for v in properties.values()
                    if v and str(v).lower() not in ['unknown', 'n/a', 'none', '']
                ]

                # Score based on number and quality of properties
                num_valid = len(valid_properties)
                if num_valid >= 8:  # Comprehensive properties
                    properties_score = 1.0
                elif num_valid >= 6:  # Good properties
                    properties_score = 0.85
                elif num_valid >= 4:  # Acceptable properties
                    properties_score = 0.70
                elif num_valid >= 2:  # Minimal properties
                    properties_score = 0.50
                elif num_valid >= 1:  # Very minimal
                    properties_score = 0.30

            metrics['material_properties'] = properties_score

            # 3. Embedding Coverage (0-1)
            embedding_coverage = self._calculate_product_embedding_coverage(product_data)
            metrics['embedding_coverage'] = embedding_coverage

            # 4. ✅ ENHANCED: Related Images with diversity assessment (0-1)
            related_images = product_data.get('metadata', {}).get('related_images', [])
            images_score = 0.0
            if related_images:
                num_images = len(related_images)
                if num_images >= 5:  # Excellent image coverage
                    images_score = 1.0
                elif num_images >= 3:  # Good image coverage
                    images_score = 0.80
                elif num_images >= 2:  # Acceptable
                    images_score = 0.60
                elif num_images >= 1:  # Minimal
                    images_score = 0.40

            metrics['related_images'] = images_score

            # 5. ✅ ENHANCED: Related Products (0-1)
            related_products = product_data.get('metadata', {}).get('related_products', [])
            products_score = 0.0
            if related_products:
                num_products = len(related_products)
                if num_products >= 5:  # Excellent relationships
                    products_score = 1.0
                elif num_products >= 3:  # Good relationships
                    products_score = 0.75
                elif num_products >= 2:  # Some relationships
                    products_score = 0.50
                elif num_products >= 1:  # Minimal relationships
                    products_score = 0.30

            metrics['related_products'] = products_score

            # 6. Confidence Score (0-1)
            confidence_score = product_data.get('metadata', {}).get('confidence_score', 0.0)
            # If confidence_score is not set, try to get it from properties
            if confidence_score == 0.0:
                confidence_score = product_data.get('properties', {}).get('confidence', 0.0)
            metrics['confidence_score'] = confidence_score

            # Calculate weighted quality score
            quality_score = (
                metrics['metadata_completeness'] * self.PRODUCT_QUALITY_WEIGHTS['metadata_completeness'] +
                metrics['material_properties'] * self.PRODUCT_QUALITY_WEIGHTS['material_properties'] +
                metrics['embedding_coverage'] * self.PRODUCT_QUALITY_WEIGHTS['embedding_coverage'] +
                metrics['related_images'] * self.PRODUCT_QUALITY_WEIGHTS['related_images'] +
                metrics['related_products'] * self.PRODUCT_QUALITY_WEIGHTS['related_products'] +
                metrics['confidence_score'] * self.PRODUCT_QUALITY_WEIGHTS['confidence_score']
            )

            return round(quality_score, 3), metrics

        except Exception as e:
            self.logger.error(f"Error calculating product quality score: {e}")
            return 0.0, {}
    
    def _calculate_dimensions_quality(self, image_data: Dict[str, Any]) -> float:
        """Calculate quality based on image dimensions."""
        try:
            width = image_data.get('width', 0)
            height = image_data.get('height', 0)
            
            # Optimal dimensions: 800x600 or larger
            if width >= 800 and height >= 600:
                return 1.0
            elif width >= 600 and height >= 400:
                return 0.8
            elif width >= 400 and height >= 300:
                return 0.6
            else:
                return 0.4
        except:
            return 0.5
    
    def _calculate_embedding_coverage(self, image_data: Dict[str, Any]) -> float:
        """Calculate embedding coverage for images."""
        coverage = 0.0
        if image_data.get('clip_embedding'):
            coverage += 1.0
        return min(1.0, coverage)
    
    def _calculate_product_embedding_coverage(self, product_data: Dict[str, Any]) -> float:
        """Calculate embedding coverage for products (all 6 types)."""
        coverage = 0.0
        total = 6

        if product_data.get('text_embedding_1024'):
            coverage += 1
        if product_data.get('visual_clip_embedding_512'):
            coverage += 1
        if product_data.get('multimodal_fusion_embedding_2048'):
            coverage += 1
        if product_data.get('color_embedding_256'):
            coverage += 1
        if product_data.get('texture_embedding_256'):
            coverage += 1
        if product_data.get('application_embedding_512'):
            coverage += 1

        return coverage / total
    
    def _calculate_coherence(self, content: str) -> float:
        """Calculate semantic coherence of content."""
        try:
            # Check for material-related keywords
            keywords = ['material', 'design', 'texture', 'color', 'surface', 'finish', 'pattern']
            keyword_count = sum(1 for kw in keywords if kw.lower() in content.lower())
            
            # Check for sentence structure
            sentences = content.split('.')
            avg_sentence_length = len(content) / max(len(sentences), 1)
            
            # Coherence score based on keywords and structure
            keyword_score = min(1.0, keyword_count / 3)  # 3+ keywords = 1.0
            structure_score = 1.0 if 50 <= avg_sentence_length <= 150 else 0.7
            
            return (keyword_score * 0.6 + structure_score * 0.4)
        except:
            return 0.5
    
    def _calculate_boundary_quality(self, content: str) -> float:
        """Calculate quality of chunk boundaries."""
        try:
            content = content.strip()
            
            # Check if ends with proper punctuation
            ends_with_punctuation = content.endswith(('.', '!', '?', ':', ';'))
            punctuation_score = 1.0 if ends_with_punctuation else 0.6
            
            # Check if starts with capital letter
            starts_with_capital = content[0].isupper() if content else False
            capital_score = 1.0 if starts_with_capital else 0.7
            
            return (punctuation_score * 0.6 + capital_score * 0.4)
        except:
            return 0.5

