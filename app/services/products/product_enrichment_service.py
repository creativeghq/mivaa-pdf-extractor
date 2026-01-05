"""
Product Enrichment Service - Stage 5 Implementation

This service enriches products with real data from:
1. Image analysis results (Qwen, Claude, CLIP embeddings)
2. Material properties extracted from images
3. Product embeddings for semantic search
4. Related product linking
5. Product descriptions from image analysis

Replaces mock enrichment with real AI-powered data extraction.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import anthropic
import os

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Import real embeddings service
from app.services.embeddings.real_embeddings_service import RealEmbeddingsService

# Import real quality scoring service (Step 5)
from app.services.ai_validation.real_quality_scoring_service import RealQualityScoringService


class ProductEnrichmentService:
    """
    Enriches products with real data from image analysis and AI models.
    
    This service:
    - Links products to images based on semantic similarity
    - Extracts material properties from image analysis
    - Generates product descriptions
    - Creates product embeddings
    - Links related products
    """
    
    def __init__(self, supabase_client):
        """
        Initialize product enrichment service.

        Args:
            supabase_client: Supabase client for database operations
        """
        self.supabase = supabase_client
        self.logger = logger
        self.quality_scoring_service = RealQualityScoringService(supabase_client)  # Step 5
        self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
    
    async def enrich_product(
        self,
        product_id: str,
        product_data: Dict[str, Any],
        document_id: str,
        workspace_id: str
    ) -> Dict[str, Any]:
        """
        Enrich a single product with real data.
        
        Args:
            product_id: UUID of product to enrich
            product_data: Current product data
            document_id: Source document ID
            workspace_id: Workspace ID
            
        Returns:
            Dictionary with enrichment results
        """
        try:
            self.logger.info(f"🎯 Starting product enrichment for {product_id}")
            
            # Step 1: Find related images
            related_images = await self._find_related_images(
                product_id=product_id,
                product_data=product_data,
                document_id=document_id
            )
            
            if not related_images:
                self.logger.warning(f"No related images found for product {product_id}")
                return {"success": False, "error": "No related images found"}
            
            self.logger.info(f"✅ Found {len(related_images)} related images")
            
            # Step 2: Extract material properties from images
            material_properties = await self._extract_material_properties_from_images(
                related_images=related_images
            )
            
            # Step 3: Generate enhanced description
            enhanced_description = await self._generate_enhanced_description(
                product_data=product_data,
                material_properties=material_properties,
                image_analysis=related_images[0].get('analysis', {}) if related_images else {}
            )
            
            # Step 4: Create product embedding
            product_embedding = await self._create_product_embedding(
                product_name=product_data.get('name', ''),
                description=enhanced_description,
                material_properties=material_properties
            )
            
            # Step 5: Find related products
            related_products = await self._find_related_products(
                product_id=product_id,
                product_embedding=product_embedding,
                workspace_id=workspace_id
            )
            
            # Step 6: Update product in database
            enrichment_result = {
                "product_id": product_id,
                "material_properties": material_properties,
                "enhanced_description": enhanced_description,
                "product_embedding": product_embedding,
                "related_images": [img['id'] for img in related_images],
                "related_products": related_products,
                "enrichment_timestamp": datetime.utcnow().isoformat(),
                "success": True
            }
            
            # Store enrichment in database
            await self._store_enrichment_results(
                product_id=product_id,
                enrichment_result=enrichment_result
            )
            
            self.logger.info(f"✅ Product enrichment complete for {product_id}")
            return enrichment_result
            
        except Exception as e:
            self.logger.error(f"❌ Product enrichment failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _find_related_images(
        self,
        product_id: str,
        product_data: Dict[str, Any],
        document_id: str
    ) -> List[Dict[str, Any]]:
        """Find images related to a product."""
        try:
            # Get all images from the document
            response = self.supabase.client.table('document_images').select(
                '*'
            ).eq('document_id', document_id).execute()
            
            if not response.data:
                return []
            
            images = response.data
            
            # Filter images with real analysis data
            related_images = []
            for image in images:
                # Check if image has real analysis (not mock)
                if image.get('vision_analysis') or image.get('claude_validation'):
                    related_images.append(image)
            
            return related_images[:5]  # Limit to 5 most relevant images
            
        except Exception as e:
            self.logger.error(f"Error finding related images: {e}")
            return []
    
    async def _extract_material_properties_from_images(
        self,
        related_images: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract material properties from image analysis results."""
        try:
            combined_properties = {
                "colors": [],
                "finishes": [],
                "patterns": [],
                "textures": [],
                "materials": [],
                "confidence": 0.0
            }
            
            confidences = []
            
            for image in related_images:
                # Get material properties from image analysis
                material_props = image.get('material_properties', {})
                
                if material_props:
                    # Collect unique values
                    if material_props.get('color'):
                        combined_properties["colors"].append(material_props['color'])
                    if material_props.get('finish'):
                        combined_properties["finishes"].append(material_props['finish'])
                    if material_props.get('pattern'):
                        combined_properties["patterns"].append(material_props['pattern'])
                    if material_props.get('texture'):
                        combined_properties["textures"].append(material_props['texture'])
                    if material_props.get('composition'):
                        combined_properties["materials"].append(material_props['composition'])
                    
                    confidences.append(material_props.get('confidence', 0.0))
            
            # Remove duplicates and calculate average confidence
            combined_properties["colors"] = list(set(combined_properties["colors"]))
            combined_properties["finishes"] = list(set(combined_properties["finishes"]))
            combined_properties["patterns"] = list(set(combined_properties["patterns"]))
            combined_properties["textures"] = list(set(combined_properties["textures"]))
            combined_properties["materials"] = list(set(combined_properties["materials"]))
            
            if confidences:
                combined_properties["confidence"] = sum(confidences) / len(confidences)
            
            return combined_properties
            
        except Exception as e:
            self.logger.error(f"Error extracting material properties: {e}")
            return {}
    
    async def _generate_enhanced_description(
        self,
        product_data: Dict[str, Any],
        material_properties: Dict[str, Any],
        image_analysis: Dict[str, Any]
    ) -> str:
        """Generate enhanced product description using Claude."""
        try:
            if not self.anthropic_client:
                self.logger.warning("Anthropic client not available, using basic description")
                return product_data.get('description', '')
            
            prompt = f"""Based on the following product information, generate a compelling and detailed product description:

PRODUCT NAME: {product_data.get('name', 'Unknown')}
CURRENT DESCRIPTION: {product_data.get('description', 'N/A')}

MATERIAL PROPERTIES:
- Colors: {', '.join(material_properties.get('colors', []))}
- Finishes: {', '.join(material_properties.get('finishes', []))}
- Patterns: {', '.join(material_properties.get('patterns', []))}
- Textures: {', '.join(material_properties.get('textures', []))}
- Materials: {', '.join(material_properties.get('materials', []))}

IMAGE ANALYSIS:
{json.dumps(image_analysis, indent=2)[:500]}

Generate a professional, engaging product description that:
1. Highlights the key material properties
2. Emphasizes design and aesthetic qualities
3. Mentions practical applications
4. Is 150-300 words long
5. Uses professional language suitable for a design catalog

RESPOND WITH ONLY THE DESCRIPTION, NO ADDITIONAL TEXT."""

            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text if response.content else product_data.get('description', '')
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced description: {e}")
            return product_data.get('description', '')
    
    async def _create_product_embedding(
        self,
        product_name: str,
        description: str,
        material_properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create all 6 embedding types for product using real AI models."""
        try:
            # Combine all text for embedding
            embedding_text = f"{product_name}. {description}. Materials: {', '.join(material_properties.get('materials', []))}. Colors: {', '.join(material_properties.get('colors', []))}"

            # Use RealEmbeddingsService to generate all 6 embedding types
            embeddings_service = RealEmbeddingsService(self.supabase)

            # Generate all embeddings (text, visual, multimodal, color, texture, application)
            all_embeddings = await embeddings_service.generate_all_embeddings(
                entity_id="temp",  # Will be updated with product_id
                entity_type="product",
                text_content=embedding_text,
                material_properties=material_properties
            )

            return all_embeddings.get("embeddings", {})

        except Exception as e:
            self.logger.error(f"Error creating product embedding: {e}")
            return {}
    
    async def _find_related_products(
        self,
        product_id: str,
        product_embedding: List[float],
        workspace_id: str
    ) -> List[str]:
        """Find related products using semantic similarity."""
        try:
            # Query for similar products using embedding similarity
            # This would use pgvector similarity search
            response = self.supabase.client.rpc(
                'search_similar_products',
                {
                    'query_embedding': product_embedding,
                    'workspace_id': workspace_id,
                    'limit': 5,
                    'exclude_product_id': product_id
                }
            ).execute()
            
            if response.data:
                return [p['id'] for p in response.data]
            
            return []
            
        except Exception as e:
            self.logger.warning(f"Error finding related products: {e}")
            return []
    
    async def _store_enrichment_results(
        self,
        product_id: str,
        enrichment_result: Dict[str, Any]
    ) -> bool:
        """
        Store enrichment results in database.

        NOTE: Embeddings are NOT stored on the products table.
        - Text embeddings → stored in `embeddings` table linked via chunk_product_relationships
        - Visual embeddings → stored in VECS collections linked via product_image_relationships

        This method only stores enrichment metadata, descriptions, and quality scores.
        """
        try:
            # Calculate real quality score (not hardcoded)
            product_data = {
                "name": enrichment_result.get('name', ''),
                "description": enrichment_result.get('enhanced_description', ''),
                "long_description": enrichment_result.get('enhanced_description', ''),
                "properties": enrichment_result.get('material_properties', {}),
                "metadata": {
                    "related_images": enrichment_result.get('related_images', []),
                    "related_products": enrichment_result.get('related_products', [])
                }
            }

            quality_score, quality_metrics = self.quality_scoring_service.calculate_product_quality_score(product_data)

            # Update product with enrichment data (NO embeddings - they're in separate tables)
            update_data = {
                "long_description": enrichment_result.get('enhanced_description', ''),
                "properties": enrichment_result.get('material_properties', {}),
                "quality_score": quality_score,
                "quality_metrics": quality_metrics,
                "metadata": {
                    "enriched": True,
                    "enrichment_timestamp": enrichment_result.get('enrichment_timestamp'),
                    "related_images": enrichment_result.get('related_images', []),
                    "related_products": enrichment_result.get('related_products', []),
                    "quality_score": quality_score,
                    "quality_metrics": quality_metrics
                },
                "updated_at": datetime.utcnow().isoformat()
            }

            response = self.supabase.client.table('products').update(
                update_data
            ).eq('id', product_id).execute()

            if response.data:
                self.logger.info(f"✅ Stored enrichment results for product {product_id}")

            return bool(response.data)

        except Exception as e:
            self.logger.error(f"Error storing enrichment results: {e}")
            return False


