"""
Prompt Templates Service - Database-Driven

ALL prompts are now stored in the database and managed via /admin/ai-configs.
This file provides the get_prompt_template function that fetches from the database.

NO HARDCODED PROMPTS - All prompts must exist in the database.
"""

import logging
from typing import Optional

from app.services.core.supabase_client import get_supabase_client
from app.services.utilities.prompt_registry import workspace_scope, prefer_workspace

logger = logging.getLogger(__name__)


async def get_prompt_template_from_db(
    workspace_id: str,
    stage: str,
    category: str
) -> Optional[str]:
    """
    Fetch prompt template from database.

    Args:
        workspace_id: Workspace ID
        stage: Extraction stage (discovery, chunking, image_analysis, entity_creation)
        category: Content category (products, certificates, logos, specifications, default)

    Returns:
        Prompt template string or None if not found

    Raises:
        ValueError: If prompt is not found in database
    """
    try:
        supabase = get_supabase_client()

        # Try exact match first
        # workspace row first, platform default second (#347) — a default living under the
        # global workspace was previously invisible to every tenant.
        result = supabase.client.table('prompts')\
            .select('prompt_text, workspace_id')\
            .eq('prompt_type', 'extraction')\
            .in_('workspace_id', workspace_scope(workspace_id))\
            .eq('stage', stage)\
            .eq('category', category)\
            .eq('is_active', True)\
            .order('version', desc=True)\
            .execute()

        row = prefer_workspace(result.data or [], workspace_id)
        if row:
            logger.info(f"Found prompt for {stage}/{category} in database")
            return row['prompt_text']

        # Try default for this stage
        result = supabase.client.table('prompts')\
            .select('prompt_text, workspace_id')\
            .eq('prompt_type', 'extraction')\
            .in_('workspace_id', workspace_scope(workspace_id))\
            .eq('stage', stage)\
            .eq('category', 'default')\
            .eq('is_active', True)\
            .order('version', desc=True)\
            .execute()

        row = prefer_workspace(result.data or [], workspace_id)
        if row:
            logger.info(f"Found default prompt for {stage} in database")
            return row['prompt_text']

        # No prompt found - this is an error
        error_msg = f"CRITICAL: No prompt found in database for stage='{stage}', category='{category}'. Please add it via /admin/ai-configs."
        logger.error(error_msg)
        raise ValueError(error_msg)

    except Exception as e:
        logger.error(f"Error fetching prompt template: {str(e)}")
        raise
