"""
Metadata Consolidation Service

Merges metadata from multiple sources with AI-powered conflict resolution.
Follows the platform's prompt-based architecture.
"""

import logging
import os
import json
import re
from typing import Dict, List, Any, Optional
import anthropic

from app.services.supabase_client import get_supabase_client
from app.services.ai_call_logger import AICallLogger

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


class MetadataConsolidationService:
    """
    Consolidate metadata from multiple sources using AI with database prompts.
    """

    def __init__(self, workspace_id: str = "ffafc28b-1b8b-4b0d-b226-9f9a6154004e"):
        self.supabase = get_supabase_client()
        self.workspace_id = workspace_id
        self.ai_logger = AICallLogger()
        self.prompt = None
        self._load_prompt()

    def _load_prompt(self):
        """Load metadata consolidation prompt from database."""
        try:
            result = self.supabase.table('prompts') \
                .select('prompt_text') \
                .eq('workspace_id', self.workspace_id) \
                .eq('prompt_type', 'extraction') \
                .eq('stage', 'entity_creation') \
                .eq('category', 'metadata_consolidation') \
                .eq('is_active', True) \
                .order('version', desc=True) \
                .limit(1) \
                .execute()

            if result.data:
                self.prompt = result.data[0]['prompt_text']
                logger.info("✅ Loaded metadata consolidation prompt from database")
            else:
                logger.error("❌ No metadata consolidation prompt found in database")
                self.prompt = None

        except Exception as e:
            logger.error(f"Error loading prompt: {e}")
            self.prompt = None

    async def consolidate_metadata(
        self,
        product_id: str,
        sources: Dict[str, Dict[str, Any]],
        existing_metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Consolidate metadata from multiple sources using AI.

        Args:
            product_id: Product UUID
            sources: Dict of metadata sources:
                {
                    "manual_overrides": {...},
                    "ai_text_extraction": {...},
                    "visual_embeddings": {...},
                    "pattern_matching": {...},
                    "factory_defaults": {...}
                }
            existing_metadata: Current product metadata (optional)

        Returns:
            Consolidated metadata with extraction tracking
        """
        if not self.prompt:
            logger.error("No prompt available for metadata consolidation")
            return existing_metadata or {}

        try:
            # Build context for AI
            consolidation_context = {
                "product_id": product_id,
                "sources": sources,
                "existing_metadata": existing_metadata or {}
            }

            # Build full prompt
            full_prompt = f"{self.prompt}\n\n**Metadata Sources:**\n\n```json\n{json.dumps(consolidation_context, indent=2)}\n```\n\nConsolidate all metadata sources intelligently. Return ONLY valid JSON."

            # Call AI
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                messages=[{"role": "user", "content": full_prompt}]
            )

            # Log AI call
            await self.ai_logger.log_ai_call(
                workspace_id=self.workspace_id,
                model="claude-3-5-sonnet-20241022",
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                cost=self._calculate_cost(response.usage),
                purpose="metadata_consolidation",
                metadata={"product_id": product_id}
            )

            # Parse response
            response_text = response.content[0].text.strip()
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

            if json_match:
                result = json.loads(json_match.group(0))
                
                # Extract consolidated metadata
                consolidated = result.get('consolidated_metadata', {})
                extraction_metadata = result.get('extraction_metadata', {})
                
                # Add extraction tracking to metadata
                consolidated['_extraction_metadata'] = extraction_metadata
                consolidated['_sources_used'] = result.get('sources_used', [])
                consolidated['_overall_confidence'] = result.get('overall_confidence', 0.0)
                consolidated['_completeness_score'] = result.get('completeness_score', 0.0)
                
                logger.info(f"✅ Consolidated metadata for product {product_id} (confidence: {consolidated['_overall_confidence']:.2f})")
                return consolidated
            else:
                logger.error("Failed to parse AI response as JSON")
                return existing_metadata or {}

        except Exception as e:
            logger.error(f"Error consolidating metadata: {e}")
            return existing_metadata or {}

    def _calculate_cost(self, usage) -> float:
        """Calculate cost for Claude API call."""
        input_cost = (usage.input_tokens / 1_000_000) * 3.00  # $3 per 1M input tokens
        output_cost = (usage.output_tokens / 1_000_000) * 15.00  # $15 per 1M output tokens
        return input_cost + output_cost

