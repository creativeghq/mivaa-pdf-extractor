"""
Embedding-to-Text Service

Converts specialized visual embeddings to textual metadata using AI with database prompts.
Follows the platform's prompt-based architecture.
"""

import logging
import os
import json
import re
from typing import Dict, List, Any, Optional
import anthropic

from app.services.core.supabase_client import get_supabase_client
from app.services.core.ai_call_logger import AICallLogger
from app.services.core.anthropic_error_reporter import report_anthropic_failure

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


class EmbeddingToTextService:
    """
    Convert visual embeddings to textual metadata using AI interpretation.
    Uses database prompts for vocabulary and extraction logic.
    """

    def __init__(self, workspace_id: str = None):
        from app.config import get_settings
        self.supabase = get_supabase_client()
        self.workspace_id = workspace_id or get_settings().default_workspace_id
        self.ai_logger = AICallLogger()
        self.prompt = None
        self._load_prompt()

    def _load_prompt(self):
        """Load embedding-to-text prompt from database."""
        try:
            result = self.supabase.client.table('prompts') \
                .select('prompt_text') \
                .eq('workspace_id', self.workspace_id) \
                .eq('prompt_type', 'extraction') \
                .eq('stage', 'image_analysis') \
                .eq('category', 'embedding_to_text') \
                .eq('is_active', True) \
                .order('version', desc=True) \
                .limit(1) \
                .execute()

            if result.data:
                self.prompt = result.data[0]['prompt_text']
                logger.info("✅ Loaded embedding-to-text prompt from database")
            else:
                logger.error("❌ No embedding-to-text prompt found in database")
                self.prompt = None

        except Exception as e:
            logger.error(f"Error loading prompt: {e}")
            self.prompt = None

    async def convert_embeddings_to_metadata(
        self,
        image_id: str,
        embeddings: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """
        Convert specialized embeddings to textual metadata using AI.

        Args:
            image_id: Image UUID
            embeddings: Dict carrying per-aspect embeddings. Both naming
                conventions are accepted during the v2 rollout:
                  - v2 (post-2026-05-04): color_aspect_1024 / texture_aspect_1024
                    / style_aspect_1024 / material_aspect_1024 (1024D Voyage)
                  - legacy: color_slig_768 / texture_slig_768 / style_slig_768 /
                    material_slig_768 (768D SLIG-blend)

        Returns:
            Dict with extracted metadata and confidence scores
        """
        if not self.prompt:
            logger.error("No prompt available for embedding-to-text conversion")
            return {}

        try:
            # Pick whichever aspect key the caller provided (v2 first, legacy fallback).
            # The dict-fallback chain returns the first non-empty value for each
            # aspect; downstream code only sees a single `embeddings.<aspect>` per call.
            def _pick(*keys: str) -> List[float]:
                for k in keys:
                    val = embeddings.get(k)
                    if val:
                        return val
                return []

            color_vec = _pick("color_aspect_1024", "color_slig_768")
            texture_vec = _pick("texture_aspect_1024", "texture_slig_768")
            material_vec = _pick("material_aspect_1024", "material_slig_768")
            style_vec = _pick("style_aspect_1024", "style_slig_768")

            # Build context for AI
            embedding_context = {
                "image_id": image_id,
                "embeddings": {
                    "color_embedding": {
                        "dimension": len(color_vec),
                        "sample": color_vec[:10] if color_vec else [],
                    },
                    "texture_embedding": {
                        "dimension": len(texture_vec),
                        "sample": texture_vec[:10] if texture_vec else [],
                    },
                    "material_embedding": {
                        "dimension": len(material_vec),
                        "sample": material_vec[:10] if material_vec else [],
                    },
                    "style_embedding": {
                        "dimension": len(style_vec),
                        "sample": style_vec[:10] if style_vec else [],
                    },
                },
            }

            # Build full prompt
            full_prompt = f"{self.prompt}\n\n**Embedding Data:**\n\n```json\n{json.dumps(embedding_context, indent=2)}\n```\n\nAnalyze these embeddings and extract textual metadata. Return ONLY valid JSON."

            # Call AI with timing
            import time
            start_time = time.time()
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            response = client.messages.create(
                model="claude-opus-4-7",
                max_tokens=2048,
                messages=[{"role": "user", "content": full_prompt}]
            )
            latency_ms = int((time.time() - start_time) * 1000)

            # Log AI call with correct signature
            await self.ai_logger.log_ai_call(
                task="embedding_to_text_conversion",
                model="claude-opus-4-7",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cost=self._calculate_cost(response.usage),
                latency_ms=latency_ms,
                confidence_score=0.85,
                confidence_breakdown={
                    "model_confidence": 0.90,
                    "completeness": 0.85,
                    "consistency": 0.80,
                    "validation": 0.85
                },
                action="use_ai_result",
                job_id=image_id
            )

            # Parse response
            response_text = response.content[0].text.strip()
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

            if json_match:
                result = json.loads(json_match.group(0))
                logger.info(f"✅ Converted embeddings to text for image {image_id}")
                return result
            else:
                logger.error("Failed to parse AI response as JSON")
                return {}

        except Exception as e:
            report_anthropic_failure(
                e,
                service="embedding_to_text_service",
                context={"image_id": image_id},
            )
            logger.error(f"Error converting embeddings to text: {e}")
            return {}

    def _calculate_cost(self, usage) -> float:
        """Calculate cost for Claude API call."""
        input_cost = (usage.input_tokens / 1_000_000) * 3.00  # $3 per 1M input tokens
        output_cost = (usage.output_tokens / 1_000_000) * 15.00  # $15 per 1M output tokens
        return input_cost + output_cost


