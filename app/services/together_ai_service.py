"""
TogetherAI Service for LLaMA Vision integration with MIVAA platform.

This service provides semantic analysis capabilities using TogetherAI's LLaMA Vision model
(meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo) for material identification and analysis.
"""

import asyncio
import base64
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from ..config import get_settings
from ..core.exceptions import ServiceError, ConfigurationError, ExternalServiceError

logger = logging.getLogger(__name__)


@dataclass
class TogetherAIConfig:
    """Configuration for TogetherAI service."""
    api_key: str
    base_url: str = "https://api.together.xyz/v1/chat/completions"
    model: str = "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"
    max_tokens: int = 1024
    temperature: float = 0.1
    timeout: int = 120
    max_retries: int = 3
    rate_limit_requests_per_minute: int = 10
    rate_limit_burst: int = 5


@dataclass
class SemanticAnalysisRequest:
    """Request model for semantic analysis."""
    image_url: str
    image_base64: Optional[str] = None
    analysis_type: str = "material_identification"
    context: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SemanticAnalysisResult:
    """Result model for semantic analysis."""
    description: str
    confidence: float
    material_properties: Dict[str, Any]
    categories: List[str]
    processing_time: float
    model_used: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "description": self.description,
            "confidence": self.confidence,
            "material_properties": self.material_properties,
            "categories": self.categories,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "timestamp": self.timestamp.isoformat()
        }


class TogetherAIService:
    """
    Service for interfacing with TogetherAI's LLaMA Vision model.
    
    Provides semantic analysis capabilities for material images with rate limiting,
    caching, and robust error handling.
    """
    
    def __init__(self, config: Optional[TogetherAIConfig] = None):
        """Initialize TogetherAI service with configuration."""
        if config is None:
            settings = get_settings()
            together_config = settings.get_together_ai_config()
            config = TogetherAIConfig(
                api_key=together_config["api_key"],
                base_url=together_config["base_url"],
                model=together_config["model"],
                max_tokens=together_config["max_tokens"],
                temperature=together_config["temperature"],
                timeout=together_config["timeout"],
                max_retries=together_config["max_retries"],
                rate_limit_requests_per_minute=together_config["rate_limit_requests_per_minute"],
                rate_limit_burst=together_config["rate_limit_burst"]
            )
        
        self.config = config
        self._validate_config()
        
        # Initialize HTTP client with timeout
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
        )
        
        # Simple in-memory cache for semantic analysis results
        self._cache: Dict[str, tuple[SemanticAnalysisResult, datetime]] = {}
        self._cache_ttl = timedelta(hours=24)  # Cache results for 24 hours
        
        # Rate limiting state
        self._request_times: List[datetime] = []
        
        logger.info(f"TogetherAI service initialized with model: {self.config.model}")
    
    def _validate_config(self) -> None:
        """Validate the TogetherAI configuration."""
        if not self.config.api_key:
            raise ConfigurationError("TogetherAI API key is required")
        
        if not self.config.base_url:
            raise ConfigurationError("TogetherAI base URL is required")
        
        if not self.config.model:
            raise ConfigurationError("TogetherAI model is required")
        
        logger.debug("TogetherAI configuration validated successfully")
    
    def _generate_cache_key(self, request: SemanticAnalysisRequest) -> str:
        """Generate cache key for a semantic analysis request."""
        # Create a hash of the image URL/base64 and analysis parameters
        content = f"{request.image_url}:{request.image_base64}:{request.analysis_type}:{request.context}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _cleanup_cache(self) -> None:
        """Remove expired entries from cache."""
        current_time = datetime.utcnow()
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if current_time - timestamp > self._cache_ttl
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _check_rate_limit(self) -> None:
        """Check if request is within rate limits."""
        current_time = datetime.utcnow()
        minute_ago = current_time - timedelta(minutes=1)
        
        # Remove requests older than 1 minute
        self._request_times = [t for t in self._request_times if t > minute_ago]
        
        # Check if we're within rate limits
        if len(self._request_times) >= self.config.rate_limit_requests_per_minute:
            raise ExternalServiceError(
                f"Rate limit exceeded: {len(self._request_times)} requests in the last minute"
            )
        
        # Add current request time
        self._request_times.append(current_time)
    
    def _get_material_analysis_prompt(self, context: Optional[str] = None) -> str:
        """Generate the prompt for material analysis."""
        base_prompt = """Analyze this material image and provide a comprehensive semantic description. 

Focus on:
1. Material type and composition
2. Visual characteristics (texture, color, finish)
3. Potential applications and use cases
4. Physical properties that can be inferred
5. Manufacturing or production methods
6. Quality and condition assessment

Provide your response as a structured analysis with clear categorization."""
        
        if context:
            return f"{base_prompt}\n\nAdditional context: {context}"
        
        return base_prompt
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def _make_api_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request to TogetherAI with retry logic."""
        try:
            response = await self.client.post(
                self.config.base_url,
                json=request_data
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            logger.error(f"TogetherAI API HTTP error: {e.response.status_code} - {e.response.text}")
            raise ExternalServiceError(
                f"TogetherAI API error: {e.response.status_code} - {e.response.text}"
            )
        except httpx.TimeoutException as e:
            logger.error(f"TogetherAI API timeout: {e}")
            raise ExternalServiceError("TogetherAI API request timed out")
        except Exception as e:
            logger.error(f"Unexpected error in TogetherAI API request: {e}")
            raise ExternalServiceError(f"Unexpected TogetherAI API error: {str(e)}")
    
    async def analyze_material_semantics(
        self, 
        request: SemanticAnalysisRequest
    ) -> SemanticAnalysisResult:
        """
        Analyze material image for semantic properties.
        
        Args:
            request: Semantic analysis request containing image data and parameters
            
        Returns:
            SemanticAnalysisResult with analysis results
            
        Raises:
            ServiceError: If analysis fails
            ExternalServiceError: If TogetherAI API fails
        """
        start_time = time.time()
        
        try:
            # Check rate limits
            self._check_rate_limit()
            
            # Check cache first
            cache_key = self._generate_cache_key(request)
            self._cleanup_cache()
            
            if cache_key in self._cache:
                cached_result, _ = self._cache[cache_key]
                logger.info(f"Returning cached semantic analysis for key: {cache_key[:12]}...")
                return cached_result
            
            # Prepare image data
            image_content = None
            if request.image_base64:
                image_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{request.image_base64}"
                    }
                }
            elif request.image_url:
                image_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": request.image_url
                    }
                }
            else:
                raise ServiceError("Either image_url or image_base64 must be provided")
            
            # Prepare API request
            prompt = self._get_material_analysis_prompt(request.context)
            
            api_request = {
                "model": self.config.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            image_content
                        ]
                    }
                ],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "stream": False
            }
            
            logger.info(f"Making TogetherAI API request for material analysis...")
            
            # Make API request
            response_data = await self._make_api_request(api_request)
            
            # Parse response
            if "choices" not in response_data or not response_data["choices"]:
                raise ExternalServiceError("Invalid response from TogetherAI API: no choices")
            
            analysis_text = response_data["choices"][0]["message"]["content"]
            
            # Extract structured information from the analysis
            # This is a simplified extraction - in production, you might want more sophisticated parsing
            result = self._parse_analysis_response(analysis_text, start_time)
            
            # Cache the result
            self._cache[cache_key] = (result, datetime.utcnow())
            
            logger.info(f"Semantic analysis completed in {result.processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Semantic analysis failed after {processing_time:.2f}s: {e}")
            raise
    
    def _parse_analysis_response(self, analysis_text: str, start_time: float) -> SemanticAnalysisResult:
        """Parse the analysis response from TogetherAI into structured result."""
        processing_time = time.time() - start_time
        
        # Basic parsing - extract key information from the response
        # In a production system, you might want more sophisticated NLP parsing
        lines = analysis_text.split('\n')
        description = analysis_text.strip()
        
        # Extract basic material properties and categories
        # This is a simplified approach - you might want to enhance this
        material_properties = {}
        categories = []
        confidence = 0.8  # Default confidence
        
        # Look for common material indicators in the text
        text_lower = analysis_text.lower()
        
        # Basic material type detection
        if any(word in text_lower for word in ['metal', 'steel', 'aluminum', 'iron']):
            categories.append('metal')
            material_properties['material_family'] = 'metal'
        elif any(word in text_lower for word in ['wood', 'timber', 'lumber']):
            categories.append('organic')
            material_properties['material_family'] = 'wood'
        elif any(word in text_lower for word in ['plastic', 'polymer', 'synthetic']):
            categories.append('synthetic')
            material_properties['material_family'] = 'plastic'
        elif any(word in text_lower for word in ['fabric', 'textile', 'cloth']):
            categories.append('textile')
            material_properties['material_family'] = 'textile'
        elif any(word in text_lower for word in ['ceramic', 'porcelain', 'clay']):
            categories.append('ceramic')
            material_properties['material_family'] = 'ceramic'
        
        # Extract color information
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'gray', 'silver']
        detected_colors = [color for color in colors if color in text_lower]
        if detected_colors:
            material_properties['primary_colors'] = detected_colors
        
        # Extract texture information
        textures = ['smooth', 'rough', 'textured', 'glossy', 'matte', 'shiny']
        detected_textures = [texture for texture in textures if texture in text_lower]
        if detected_textures:
            material_properties['surface_textures'] = detected_textures
        
        return SemanticAnalysisResult(
            description=description,
            confidence=confidence,
            material_properties=material_properties,
            categories=categories,
            processing_time=processing_time,
            model_used=self.config.model
        )
    
    async def analyze_image_from_url(
        self, 
        image_url: str, 
        context: Optional[str] = None
    ) -> SemanticAnalysisResult:
        """
        Analyze material image from URL.
        
        Args:
            image_url: URL of the image to analyze
            context: Optional context for the analysis
            
        Returns:
            SemanticAnalysisResult with analysis results
        """
        request = SemanticAnalysisRequest(
            image_url=image_url,
            context=context,
            analysis_type="material_identification"
        )
        
        return await self.analyze_material_semantics(request)
    
    async def analyze_image_from_base64(
        self, 
        image_base64: str, 
        context: Optional[str] = None
    ) -> SemanticAnalysisResult:
        """
        Analyze material image from base64 data.
        
        Args:
            image_base64: Base64 encoded image data
            context: Optional context for the analysis
            
        Returns:
            SemanticAnalysisResult with analysis results
        """
        request = SemanticAnalysisRequest(
            image_url="",  # Not used when base64 is provided
            image_base64=image_base64,
            context=context,
            analysis_type="material_identification"
        )
        
        return await self.analyze_material_semantics(request)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check for TogetherAI service.
        
        Returns:
            Health status information
        """
        try:
            # Simple test request to check API availability
            test_request = {
                "model": self.config.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Test message for health check"
                            }
                        ]
                    }
                ],
                "max_tokens": 10,
                "temperature": 0.1
            }
            
            start_time = time.time()
            response = await self.client.post(
                self.config.base_url,
                json=test_request
            )
            response_time = time.time() - start_time
            
            response.raise_for_status()
            
            return {
                "status": "healthy",
                "service": "together_ai",
                "model": self.config.model,
                "response_time": response_time,
                "cache_size": len(self._cache),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"TogetherAI health check failed: {e}")
            return {
                "status": "unhealthy",
                "service": "together_ai",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """
        Get service statistics.
        
        Returns:
            Service statistics and metrics
        """
        return {
            "service": "together_ai",
            "model": self.config.model,
            "cache_size": len(self._cache),
            "rate_limit_config": {
                "requests_per_minute": self.config.rate_limit_requests_per_minute,
                "burst": self.config.rate_limit_burst
            },
            "recent_requests": len(self._request_times),
            "cache_ttl_hours": self._cache_ttl.total_seconds() / 3600,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def clear_cache(self) -> Dict[str, Any]:
        """
        Clear the analysis cache.
        
        Returns:
            Cache clearing result
        """
        cache_size = len(self._cache)
        self._cache.clear()
        
        logger.info(f"Cleared TogetherAI service cache ({cache_size} entries)")
        
        return {
            "status": "success",
            "cleared_entries": cache_size,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()


# Service factory function
def create_together_ai_service() -> TogetherAIService:
    """Create a TogetherAI service instance with default configuration."""
    return TogetherAIService()


# Global service instance (lazy initialization)
_together_ai_service: Optional[TogetherAIService] = None


async def get_together_ai_service() -> TogetherAIService:
    """
    Get the global TogetherAI service instance.
    
    Returns:
        TogetherAIService instance
    """
    global _together_ai_service
    
    if _together_ai_service is None:
        _together_ai_service = create_together_ai_service()
    
    return _together_ai_service


# Convenience functions for direct usage
async def analyze_material_image_url(
    image_url: str, 
    context: Optional[str] = None
) -> SemanticAnalysisResult:
    """
    Convenience function to analyze material image from URL.
    
    Args:
        image_url: URL of the image to analyze
        context: Optional context for the analysis
        
    Returns:
        SemanticAnalysisResult with analysis results
    """
    service = await get_together_ai_service()
    return await service.analyze_image_from_url(image_url, context)


async def analyze_material_image_base64(
    image_base64: str, 
    context: Optional[str] = None
) -> SemanticAnalysisResult:
    """
    Convenience function to analyze material image from base64.
    
    Args:
        image_base64: Base64 encoded image data
        context: Optional context for the analysis
        
    Returns:
        SemanticAnalysisResult with analysis results
    """
    service = await get_together_ai_service()
    return await service.analyze_image_from_base64(image_base64, context)