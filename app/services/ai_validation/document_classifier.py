"""
Document Classifier Service

Two-stage classification system for PDF content:
1. Fast text classification (product/supporting/administrative/transitional)
2. Deep enrichment with metadata extraction

Uses Claude Haiku 4.5 for fast classification and Claude Opus for deep analysis.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from app.services.core.ai_call_logger import AICallLogger

from app.services.utilities.prompt_registry import load_prompt, render

logger = logging.getLogger(__name__)


#: Invariant 9: a classifier whose verdict drives downstream behaviour states that
#: verdict through a real Anthropic tool schema, never as free text a parser has to
#: guess at. The enum is the contract -- the model cannot answer "productish", and
#: there is nothing left to salvage-parse.
_VALID_CATEGORIES = {"product", "supporting", "administrative", "transitional"}

DOCUMENT_CLASSIFICATION_TOOL = {
    "name": "record_page_classification",
    "description": "Record the classification of one catalog page.",
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["product", "supporting", "administrative", "transitional"],
                "description": (
                    "product: product information, specifications, features. "
                    "supporting: technical details, certifications, guides. "
                    "administrative: company info, legal, contact. "
                    "transitional: TOC, headers, footers, navigation."
                ),
            },
            "confidence": {
                "type": "number",
                "description": "Confidence in the category, 0.0 to 1.0.",
            },
        },
        "required": ["category", "confidence"],
    },
}


class DocumentClassifier:
    """
    Two-stage document classification system.
    
    Stage 1: Fast text classification using Claude Haiku 4.5
    Stage 2: Deep enrichment using Claude Opus (for product content)
    """
    
    # Content type definitions
    CONTENT_TYPES = {
        "product": "Product information (specifications, features, images)",
        "supporting": "Supporting content (technical details, certifications, installation guides)",
        "administrative": "Administrative content (company info, contact details, legal)",
        "transitional": "Transitional content (table of contents, page numbers, headers/footers)",
    }
    
    def __init__(self, ai_logger: Optional[AICallLogger] = None):
        """Initialize document classifier (Anthropic Claude Opus)."""
        import os
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
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
        
        # A failed classification is not a category. Return it as-is so the caller
        # sees the marker rather than a page silently treated as "not a product".
        if content_type == "classification_failed":
            return {
                "content_type": "classification_failed",
                "confidence": 0.0,
                "metadata": {},
                "is_product": False,
                "enrichment_applied": False,
                "error": stage1_result.get("error"),
            }

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
        Stage 1: Fast classification using Claude Haiku 4.5.
        
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
        
        prompt = render(await load_prompt("classification", "document_page", stage="discovery"),
            page_num=page_num, has_images=has_images, content=content[:1000])
        
        try:
            start_time = datetime.now()

            # Forced tool_choice, via the tracked helper so the cost lands in
            # ai_usage_logs automatically (pipeline convention 10) instead of the
            # hand-rolled httpx POST + manual log_ai_call this replaced.
            from app.services.core.claude_helper import tracked_claude_call_async

            response = await tracked_claude_call_async(
                task="document_classification_stage1",
                model="claude-haiku-4-5",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
                job_id=job_id,
                extra_kwargs={
                    "tools": [DOCUMENT_CLASSIFICATION_TOOL],
                    "tool_choice": {
                        "type": "tool",
                        "name": DOCUMENT_CLASSIFICATION_TOOL["name"],
                    },
                },
            )

            end_time = datetime.now()
            latency_ms = int((end_time - start_time).total_seconds() * 1000)

            tool_input = None
            for block in (response.content or []):
                if getattr(block, "type", None) == "tool_use":
                    tool_input = getattr(block, "input", None)
                    break
            if not isinstance(tool_input, dict):
                # tool_choice was forced, so this means the API contract broke.
                # There is nothing to salvage and guessing would be worse than
                # admitting it.
                raise ValueError("Claude returned no tool_use block for a forced tool_choice")

            category = str(tool_input.get("category", "")).strip().lower()
            if category not in _VALID_CATEGORIES:
                raise ValueError(f"Classifier returned an out-of-enum category: {category!r}")
            try:
                confidence = float(tool_input.get("confidence"))
            except (TypeError, ValueError):
                raise ValueError("Classifier returned a non-numeric confidence")
            confidence = max(0.0, min(1.0, confidence))
                        
            # Adjust confidence based on context
            if has_images and category == "product":
                confidence = min(1.0, confidence + 0.1)  # Boost confidence for products with images
            
            # (No manual log_ai_call: tracked_claude_call_async already recorded
            # this call's tokens and cost in ai_usage_logs.)

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
            # No keyword-heuristic fallback. It used to return e.g.
            # {"content_type": "product", "confidence": 0.5} from a substring match,
            # which is indistinguishable at every call site from a real verdict at
            # middling confidence — so a dead API key or a broken model silently
            # reclassified an entire catalog by keyword. `classification_failed` is
            # the explicit failure marker (pipeline convention 1): consumers check
            # the marker, because emptiness and low confidence are both ambiguous.
            logger.error(f"❌ Fast classification failed: {str(e)}")
            return {
                "content_type": "classification_failed",
                "confidence": 0.0,
                "error": str(e)[:300],
            }
    
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


