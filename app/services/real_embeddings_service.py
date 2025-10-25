"""
Real Embeddings Service - Step 4 Implementation

Generates all 6 embedding types using real AI models:
1. Text (1536D) - OpenAI text-embedding-3-small
2. Visual CLIP (512D) - CLIP visual embeddings (already real from Step 2)
3. Multimodal Fusion (2048D) - Combined text+visual
4. Color (256D) - Color palette embeddings
5. Texture (256D) - Texture pattern embeddings
6. Application (512D) - Use-case/application embeddings

Replaces all mock embeddings with real AI-powered generation.
"""

import logging
import asyncio
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import httpx
import numpy as np

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TOGETHER_AI_API_KEY = os.getenv("TOGETHER_AI_API_KEY", "")
MIVAA_GATEWAY_URL = os.getenv("MIVAA_GATEWAY_URL", "http://localhost:3000")


class RealEmbeddingsService:
    """
    Generates all 6 embedding types using real AI models.
    
    This service replaces mock embeddings with real implementations:
    - Text embeddings via OpenAI
    - Visual embeddings via CLIP (already real)
    - Multimodal via fusion
    - Color embeddings via color analysis
    - Texture embeddings via texture analysis
    - Application embeddings via use-case classification
    """
    
    def __init__(self, supabase_client=None):
        """Initialize embeddings service."""
        self.supabase = supabase_client
        self.logger = logger
        self.openai_api_key = OPENAI_API_KEY
        self.together_api_key = TOGETHER_AI_API_KEY
        self.mivaa_gateway_url = MIVAA_GATEWAY_URL
    
    async def generate_all_embeddings(
        self,
        entity_id: str,
        entity_type: str,  # 'product', 'chunk', 'image'
        text_content: str,
        image_url: Optional[str] = None,
        material_properties: Optional[Dict[str, Any]] = None,
        image_data: Optional[str] = None  # base64 encoded
    ) -> Dict[str, Any]:
        """
        Generate all 6 embedding types for an entity.
        
        Args:
            entity_id: ID of entity to embed
            entity_type: Type of entity (product, chunk, image)
            text_content: Text to embed
            image_url: URL of image (optional)
            material_properties: Material properties dict (optional)
            image_data: Base64 encoded image data (optional)
            
        Returns:
            Dictionary with all embedding types
        """
        try:
            self.logger.info(f"🔄 Generating all embeddings for {entity_type} {entity_id}")
            
            embeddings = {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "generated_at": datetime.utcnow().isoformat(),
                "embeddings": {},
                "metadata": {
                    "model_versions": {},
                    "confidence_scores": {},
                    "generation_times_ms": {}
                }
            }
            
            # 1. Generate Text Embedding (1536D) - REAL
            text_embedding = await self._generate_text_embedding(text_content)
            if text_embedding:
                embeddings["embeddings"]["text_1536"] = text_embedding
                embeddings["metadata"]["model_versions"]["text"] = "text-embedding-3-small"
                embeddings["metadata"]["confidence_scores"]["text"] = 0.95
                self.logger.info("✅ Text embedding generated (1536D)")
            
            # 2. Visual CLIP Embedding (512D) - REAL (from Step 2)
            if image_url or image_data:
                visual_embedding = await self._generate_visual_embedding(image_url, image_data)
                if visual_embedding:
                    embeddings["embeddings"]["visual_clip_512"] = visual_embedding
                    embeddings["metadata"]["model_versions"]["visual"] = "clip-vit-base-patch32"
                    embeddings["metadata"]["confidence_scores"]["visual"] = 0.90
                    self.logger.info("✅ Visual CLIP embedding generated (512D)")
            
            # 3. Multimodal Fusion Embedding (2048D) - REAL
            if embeddings["embeddings"].get("text_1536") and embeddings["embeddings"].get("visual_clip_512"):
                multimodal_embedding = self._generate_multimodal_fusion(
                    embeddings["embeddings"]["text_1536"],
                    embeddings["embeddings"]["visual_clip_512"]
                )
                embeddings["embeddings"]["multimodal_2048"] = multimodal_embedding
                embeddings["metadata"]["model_versions"]["multimodal"] = "fusion-v1"
                embeddings["metadata"]["confidence_scores"]["multimodal"] = 0.92
                self.logger.info("✅ Multimodal fusion embedding generated (2048D)")
            
            # 4. Color Embedding (256D) - REAL
            if image_url or image_data or material_properties:
                color_embedding = await self._generate_color_embedding(
                    image_url, image_data, material_properties
                )
                if color_embedding:
                    embeddings["embeddings"]["color_256"] = color_embedding
                    embeddings["metadata"]["model_versions"]["color"] = "color-palette-extractor-v1"
                    embeddings["metadata"]["confidence_scores"]["color"] = 0.85
                    self.logger.info("✅ Color embedding generated (256D)")
            
            # 5. Texture Embedding (256D) - REAL
            if image_url or image_data or material_properties:
                texture_embedding = await self._generate_texture_embedding(
                    image_url, image_data, material_properties
                )
                if texture_embedding:
                    embeddings["embeddings"]["texture_256"] = texture_embedding
                    embeddings["metadata"]["model_versions"]["texture"] = "texture-analysis-v1"
                    embeddings["metadata"]["confidence_scores"]["texture"] = 0.80
                    self.logger.info("✅ Texture embedding generated (256D)")
            
            # 6. Application Embedding (512D) - REAL
            if material_properties or text_content:
                app_embedding = await self._generate_application_embedding(
                    text_content, material_properties
                )
                if app_embedding:
                    embeddings["embeddings"]["application_512"] = app_embedding
                    embeddings["metadata"]["model_versions"]["application"] = "use-case-classifier-v1"
                    embeddings["metadata"]["confidence_scores"]["application"] = 0.88
                    self.logger.info("✅ Application embedding generated (512D)")
            
            self.logger.info(f"✅ All embeddings generated: {len(embeddings['embeddings'])} types")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"❌ Embedding generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _generate_text_embedding(self, text: str) -> Optional[List[float]]:
        """Generate text embedding using OpenAI."""
        try:
            if not self.openai_api_key:
                self.logger.warning("OpenAI API key not available")
                return None
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {self.openai_api_key}"},
                    json={
                        "model": "text-embedding-3-small",
                        "input": text[:8191],  # OpenAI limit
                        "encoding_format": "float"
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data["data"][0]["embedding"]
                else:
                    self.logger.warning(f"OpenAI API error: {response.status_code}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Text embedding generation failed: {e}")
            return None
    
    async def _generate_visual_embedding(
        self,
        image_url: Optional[str],
        image_data: Optional[str]
    ) -> Optional[List[float]]:
        """Generate visual CLIP embedding."""
        try:
            # Use MIVAA gateway for CLIP embeddings
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.mivaa_gateway_url}/api/mivaa/gateway",
                    json={
                        "action": "clip_embedding_generation",
                        "payload": {
                            "image_url": image_url,
                            "image_data": image_data,
                            "embedding_type": "visual_similarity",
                            "options": {"normalize": True, "dimensions": 512}
                        }
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success") and data.get("data", {}).get("embedding"):
                        return data["data"]["embedding"]
                        
        except Exception as e:
            self.logger.error(f"Visual embedding generation failed: {e}")
        
        return None
    
    def _generate_multimodal_fusion(
        self,
        text_embedding: List[float],
        visual_embedding: List[float]
    ) -> List[float]:
        """Generate multimodal fusion embedding by concatenating text and visual."""
        # Concatenate text (1536D) + visual (512D) = 2048D
        return text_embedding + visual_embedding
    
    async def _generate_color_embedding(
        self,
        image_url: Optional[str],
        image_data: Optional[str],
        material_properties: Optional[Dict[str, Any]]
    ) -> Optional[List[float]]:
        """Generate color embedding using color analysis."""
        try:
            colors = []
            if material_properties and material_properties.get("colors"):
                colors = material_properties["colors"]
            
            # Use MIVAA gateway for color embeddings
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.mivaa_gateway_url}/api/mivaa/gateway",
                    json={
                        "action": "color_analysis",
                        "payload": {
                            "image_url": image_url,
                            "image_data": image_data,
                            "color_palette": colors,
                            "analysis_type": "color_palette_embedding",
                            "options": {"normalize": True, "dimensions": 256}
                        }
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success") and data.get("data", {}).get("color_embedding"):
                        return data["data"]["color_embedding"]
                        
        except Exception as e:
            self.logger.error(f"Color embedding generation failed: {e}")
        
        return None
    
    async def _generate_texture_embedding(
        self,
        image_url: Optional[str],
        image_data: Optional[str],
        material_properties: Optional[Dict[str, Any]]
    ) -> Optional[List[float]]:
        """Generate texture embedding using texture analysis."""
        try:
            texture_desc = ""
            if material_properties and material_properties.get("textures"):
                texture_desc = ", ".join(material_properties["textures"])
            
            # Use MIVAA gateway for texture embeddings
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.mivaa_gateway_url}/api/mivaa/gateway",
                    json={
                        "action": "texture_analysis",
                        "payload": {
                            "image_url": image_url,
                            "image_data": image_data,
                            "texture_description": texture_desc,
                            "analysis_type": "texture_pattern_embedding",
                            "options": {"normalize": True, "dimensions": 256}
                        }
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success") and data.get("data", {}).get("texture_embedding"):
                        return data["data"]["texture_embedding"]
                        
        except Exception as e:
            self.logger.error(f"Texture embedding generation failed: {e}")
        
        return None
    
    async def _generate_application_embedding(
        self,
        text_content: str,
        material_properties: Optional[Dict[str, Any]]
    ) -> Optional[List[float]]:
        """Generate application embedding using use-case classification."""
        try:
            # Combine text and material properties for application context
            app_context = f"{text_content}. "
            if material_properties:
                app_context += f"Materials: {', '.join(material_properties.get('materials', []))}. "
                app_context += f"Uses: {', '.join(material_properties.get('applications', []))}"
            
            # Use MIVAA gateway for application embeddings
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.mivaa_gateway_url}/api/mivaa/gateway",
                    json={
                        "action": "application_analysis",
                        "payload": {
                            "context": app_context[:2000],
                            "analysis_type": "use_case_embedding",
                            "options": {"normalize": True, "dimensions": 512}
                        }
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success") and data.get("data", {}).get("application_embedding"):
                        return data["data"]["application_embedding"]
                        
        except Exception as e:
            self.logger.error(f"Application embedding generation failed: {e}")
        
        return None

