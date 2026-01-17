"""
Prompt Templates Service - Database-Driven

ALL prompts are now stored in the database and managed via /admin/ai-configs.
This file provides the get_prompt_template function that fetches from the database.

NO HARDCODED PROMPTS - All prompts must exist in the database.
"""

import logging
from typing import Optional

from app.services.core.supabase_client import get_supabase_client

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
        result = supabase.client.table('prompts')\
            .select('prompt_text')\
            .eq('prompt_type', 'extraction')\
            .eq('workspace_id', workspace_id)\
            .eq('stage', stage)\
            .eq('category', category)\
            .eq('is_active', True)\
            .order('version', desc=True)\
            .limit(1)\
            .execute()

        if result.data and len(result.data) > 0:
            logger.info(f"Found prompt for {stage}/{category} in database")
            return result.data[0]['prompt_text']

        # Try default for this stage
        result = supabase.client.table('prompts')\
            .select('prompt_text')\
            .eq('prompt_type', 'extraction')\
            .eq('workspace_id', workspace_id)\
            .eq('stage', stage)\
            .eq('category', 'default')\
            .eq('is_active', True)\
            .order('version', desc=True)\
            .limit(1)\
            .execute()

        if result.data and len(result.data) > 0:
            logger.info(f"Found default prompt for {stage} in database")
            return result.data[0]['prompt_text']

        # No prompt found - this is an error
        error_msg = f"CRITICAL: No prompt found in database for stage='{stage}', category='{category}'. Please add it via /admin/ai-configs."
        logger.error(error_msg)
        raise ValueError(error_msg)

    except Exception as e:
        logger.error(f"Error fetching prompt template: {str(e)}")
        raise


def get_prompt_template(stage: str, category: str) -> str:
    """
    DEPRECATED: This synchronous function is deprecated.
    Use get_prompt_template_from_db() instead.

    This function raises an error to ensure all callers migrate to the async database version.

    Args:
        stage: Extraction stage
        category: Content category

    Raises:
        NotImplementedError: Always - use get_prompt_template_from_db instead
    """
    raise NotImplementedError(
        "get_prompt_template() is deprecated. "
        "Use await get_prompt_template_from_db(workspace_id, stage, category) instead. "
        "All prompts must be fetched from the database. "
        "Add prompts via /admin/ai-configs."
    )
