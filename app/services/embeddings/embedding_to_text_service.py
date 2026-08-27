"""
Embedding-to-Text Service

Converts specialized visual embeddings to textual metadata using AI with database prompts.
Follows the platform's prompt-based architecture.
"""

import logging
import os
import json
from typing import Dict, List, Any

from app.services.core.supabase_client import get_supabase_client
from app.services.core.anthropic_error_reporter import report_anthropic_failure
from app.services.utilities.prompt_registry import workspace_scope, prefer_workspace

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
        self.prompt = None
        self._load_prompt()

    def _load_prompt(self):
        """Load embedding-to-text prompt from database."""
        try:
            result = self.supabase.client.table('prompts') \
                .select('prompt_text, workspace_id') \
                .in_('workspace_id', workspace_scope(self.workspace_id)) \
                .eq('prompt_type', 'extraction') \
                .eq('stage', 'image_analysis') \
                .eq('category', 'embedding_to_text') \
                .eq('is_active', True) \
                .order('version', desc=True) \
                .execute()

            # `.limit(1)` is gone on purpose: the query now spans two workspaces, so taking the
            # first row would pick the platform default over the tenant's own customisation
            # whenever the database happened to return it first.
            row = prefer_workspace(result.data or [], self.workspace_id)
            if row:
                self.prompt = row['prompt_text']
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
            embeddings: Dict carrying per-aspect embeddings under the v2 keys
                color_aspect_1024 / texture_aspect_1024 / style_aspect_1024 /
                material_aspect_1024 (1024D Voyage of VisionAnalysis text).

        Returns:
            Dict with extracted metadata and confidence scores
        """
        if not self.prompt:
            logger.error("No prompt available for embedding-to-text conversion")
            return {}

        try:
            color_vec = embeddings.get("color_aspect_1024") or []
            texture_vec = embeddings.get("texture_aspect_1024") or []
            material_vec = embeddings.get("material_aspect_1024") or []
            style_vec = embeddings.get("style_aspect_1024") or []

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

            # Through the tracked helper (#33 item 2). What was here did four things
            # wrong at once, none of which failed:
            #
            #   * priced itself from a constant — `_calculate_cost` charged $3/$15 per
            #     million while calling `claude-opus-4-8`. Those are Sonnet rates, so
            #     every conversion was booked at roughly a fifth of what it cost.
            #     `ai_model_pricing` is the one USD source and the helper resolves
            #     against it.
            #   * called the SYNC `messages.create` from an `async def`, blocking the
            #     event loop for a whole Opus round-trip per image.
            #   * logged `job_id=image_id` — an image id in the job column, so the spend
            #     joined to no job and the helper's job -> billable-user resolution had
            #     nothing to work with. `image_id` and `workspace_id` are the columns
            #     that actually describe this call.
            #   * recorded nothing at all when the call raised, since the log came after
            #     it. Anthropic bills a request it accepted regardless.
            from app.services.core.claude_helper import tracked_claude_call_async

            # Forced tool (#32). The parse below was `re.search(r'\{.*\}', ...)` — a
            # greedy match for anything between the first { and the last }, which
            # silently swallows prose on either side and produces a dict from whatever
            # happened to be in between.
            #
            # The schema is OPEN: the metadata shape is described inside `self.prompt`,
            # loaded from the database. Restating those keys here would create a second
            # source, and because the model is forced to satisfy the schema, an admin's
            # edit would silently stop taking effect.
            _METADATA_TOOL = {
                "name": "emit_visual_metadata",
                "description": (
                    "Emit the textual metadata derived from these embeddings, as the "
                    "JSON object described in the instructions."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                },
            }

            response = await tracked_claude_call_async(
                task="embedding_to_text_conversion",
                model="claude-opus-4-8",
                max_tokens=2048,
                messages=[{"role": "user", "content": full_prompt}],
                extra_kwargs={
                    "tools": [_METADATA_TOOL],
                    "tool_choice": {"type": "tool", "name": _METADATA_TOOL["name"]},
                },
                confidence_score=0.85,
                confidence_breakdown={
                    "model_confidence": 0.90,
                    "completeness": 0.85,
                    "consistency": 0.80,
                    "validation": 0.85
                },
                action="use_ai_result",
                image_id=image_id,
                workspace_id=self.workspace_id,
            )

            from app.services.core.claude_tool_call import (
                ToolCallNotReturned,
                extract_tool_input,
            )

            try:
                result = extract_tool_input(response, _METADATA_TOOL["name"])
            except ToolCallNotReturned as e:
                # `{}` is preserved as the caller's "no metadata" signal, but it is now
                # only reachable from a genuinely broken reply rather than from prose
                # the regex could not find braces in.
                logger.error(f"No tool call in embedding-to-text response: {e}")
                return {}

            logger.info(f"✅ Converted embeddings to text for image {image_id}")
            return result

        except Exception as e:
            report_anthropic_failure(
                e,
                service="embedding_to_text_service",
                context={"image_id": image_id},
            )
            logger.error(f"Error converting embeddings to text: {e}")
            return {}

