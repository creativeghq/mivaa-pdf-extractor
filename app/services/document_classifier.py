"""
Document Classifier Service

Two-stage classification system for PDF content:
1. Fast text classification (product/supporting/administrative/transitional)
2. Deep enrichment with metadata extraction

Uses Qwen3-VL for fast classification and Claude for deep analysis.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from app.services.together_ai_service import TogetherAIService
from app.services.ai_call_logger import AICallLogger
from app.config.confidence_thresholds import ConfidenceThresholds

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """
    Two-stage document classification system.
    
    Stage 1: Fast text classification using Qwen
    Stage 2: Deep enrichment using Claude (for product content)
    """
    
    # Content type definitions
    CONTENT_TYPES = {
        "product": "Product information (specifications, features, images)",
        "supporting": "Supporting content (technical details, certifications, installation guides)",
        "administrative": "Administrative content (company info, contact details, legal)",
        "transitional": "Transitional content (table of contents, page numbers, headers/footers)",
    }
    
    def __init__(self, ai_logger: Optional[AICallLogger] = None):
        """
        Initialize document classifier.
        
        Args:
            ai_logger: AI call logger instance
        """
        self.together_ai = TogetherAIService()
        self.ai_logger = ai_logger or AICallLogger()
    
    async def classify_content(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Classify content using two-stage approach.
        
        Args:
            content: Text content to classify
            context: Optional context (page number, surrounding content, etc.)
            job_id: Optional job ID for logging
            
        Returns:
            Classification result with type, confidence, metadata
        """
        # Stage 1: Fast classification
        stage1_result = await self._fast_classify(content, context, job_id)
        
        content_type = stage1_result["content_type"]
        confidence = stage1_result["confidence"]
        
        # Stage 2: Deep enrichment (only for product content)
        if content_type == "product" and confidence >= 0.7:
            stage2_result = await self._deep_enrich(content, context, job_id)
            
            return {
                "content_type": content_type,
                "confidence": max(confidence, stage2_result.get("confidence", 0.0)),
                "metadata": stage2_result.get("metadata", {}),
                "is_product": True,
                "enrichment_applied": True,
                "stage1_confidence": confidence,
                "stage2_confidence": stage2_result.get("confidence", 0.0),
            }
        else:
            return {
                "content_type": content_type,
                "confidence": confidence,
                "metadata": {},
                "is_product": content_type == "product",
                "enrichment_applied": False,
                "stage1_confidence": confidence,
            }
    
    async def _fast_classify(
        self,
        content: str,
        context: Optional[Dict[str, Any]],
        job_id: Optional[str],
    ) -> Dict[str, Any]:
        """
        Stage 1: Fast classification using Qwen.
        
        Args:
            content: Text content
            context: Optional context
            job_id: Optional job ID
            
        Returns:
            Classification result
        """
        # Build context-aware prompt
        page_num = context.get("page_number", "unknown") if context else "unknown"
        has_images = context.get("has_images", False) if context else False
        
        prompt = f"""Classify the following content into one of these categories:

1. PRODUCT: Product information, specifications, features, product images
2. SUPPORTING: Technical details, certifications, installation guides, warranties
3. ADMINISTRATIVE: Company information, contact details, legal text, disclaimers
4. TRANSITIONAL: Table of contents, page numbers, headers, footers, navigation

Content (from page {page_num}, has_images: {has_images}):
{content[:1000]}

Respond with ONLY the category name (PRODUCT, SUPPORTING, ADMINISTRATIVE, or TRANSITIONAL) and a confidence score (0.0-1.0).
Format: CATEGORY|CONFIDENCE

Example: PRODUCT|0.85"""
        
        try:
            start_time = datetime.now()
            
            # Call Qwen for fast classification
            response = await self.together_ai.generate_completion(
                prompt=prompt,
                model="Qwen/Qwen3-VL-8B-Instruct",
                max_tokens=50,
                temperature=0.1,
            )
            
            end_time = datetime.now()
            latency_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Parse response
            response_text = response.get("text", "").strip()
            parts = response_text.split("|")
            
            if len(parts) >= 2:
                category = parts[0].strip().lower()
                try:
                    confidence = float(parts[1].strip())
                except ValueError:
                    confidence = 0.5
            else:
                # Fallback parsing
                response_lower = response_text.lower()
                if "product" in response_lower:
                    category = "product"
                elif "supporting" in response_lower:
                    category = "supporting"
                elif "administrative" in response_lower:
                    category = "administrative"
                else:
                    category = "transitional"
                confidence = 0.6
            
            # Adjust confidence based on context
            if has_images and category == "product":
                confidence = min(1.0, confidence + 0.1)  # Boost confidence for products with images
            
            # Log AI call
            if self.ai_logger and job_id:
                await self.ai_logger.log_ai_call({
                    "job_id": job_id,
                    "task": "document_classification_stage1",
                    "model": "Qwen/Qwen3-VL-8B-Instruct",
                    "input_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                    "output_tokens": response.get("usage", {}).get("completion_tokens", 0),
                    "latency_ms": latency_ms,
                    "confidence_score": confidence,
                    "response_data": {"category": category, "confidence": confidence},
                })
            
            logger.info(
                f"✅ Fast classification: {category} (confidence: {confidence:.2f}, "
                f"latency: {latency_ms}ms)"
            )
            
            return {
                "content_type": category,
                "confidence": confidence,
                "latency_ms": latency_ms,
            }
            
        except Exception as e:
            logger.error(f"❌ Fast classification failed: {str(e)}")
            
            # Fallback to simple heuristics
            content_lower = content.lower()
            
            if any(word in content_lower for word in ["product", "specification", "features", "dimensions"]):
                return {"content_type": "product", "confidence": 0.5}
            elif any(word in content_lower for word in ["technical", "installation", "warranty", "certificate"]):
                return {"content_type": "supporting", "confidence": 0.5}
            elif any(word in content_lower for word in ["company", "contact", "legal", "copyright"]):
                return {"content_type": "administrative", "confidence": 0.5}
            else:
                return {"content_type": "transitional", "confidence": 0.4}
    
    async def _deep_enrich(
        self,
        content: str,
        context: Optional[Dict[str, Any]],
        job_id: Optional[str],
    ) -> Dict[str, Any]:
        """
        Stage 2: Deep enrichment for product content.
        
        Args:
            content: Text content
            context: Optional context
            job_id: Optional job ID
            
        Returns:
            Enrichment result with metadata
        """
        # For now, return basic metadata
        # This will be enhanced with Claude in Phase 3
        
        metadata = {
            "has_specifications": "specification" in content.lower() or "spec" in content.lower(),
            "has_dimensions": any(word in content.lower() for word in ["dimension", "size", "width", "height", "depth"]),
            "has_materials": any(word in content.lower() for word in ["material", "composition", "finish"]),
            "has_pricing": any(word in content.lower() for word in ["price", "cost", "$", "€", "£"]),
            "content_length": len(content),
        }
        
        # Calculate enrichment confidence based on metadata completeness
        metadata_count = sum(1 for v in metadata.values() if isinstance(v, bool) and v)
        confidence = min(0.95, 0.6 + (metadata_count * 0.1))
        
        return {
            "metadata": metadata,
            "confidence": confidence,
        }
    
    async def classify_batch(
        self,
        contents: List[str],
        contexts: Optional[List[Dict[str, Any]]] = None,
        job_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Classify multiple content items in parallel.
        
        Args:
            contents: List of text contents
            contexts: Optional list of contexts (same length as contents)
            job_id: Optional job ID
            
        Returns:
            List of classification results
        """
        if contexts is None:
            contexts = [None] * len(contents)
        
        tasks = [
            self.classify_content(content, context, job_id)
            for content, context in zip(contents, contexts)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Classification failed for item {i}: {str(result)}")
                processed_results.append({
                    "content_type": "unknown",
                    "confidence": 0.0,
                    "error": str(result),
                })
            else:
                processed_results.append(result)
        
        return processed_results


