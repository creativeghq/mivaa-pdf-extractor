"""
Unified Prompt Service

Single service for accessing all prompts from the unified 'prompts' table.
Replaces: admin_prompt_service, prompt_template_service, search_prompt_service
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from uuid import UUID

from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class UnifiedPromptService:
    """Service for managing all prompts from unified prompts table"""

    # Prompt types
    AGENT = "agent"
    EXTRACTION = "extraction"
    TEMPLATE = "template"
    SEARCH = "search"

    def __init__(self):
        self.supabase = get_supabase_client()
        self._cache = {}
        self._cache_ttl = timedelta(minutes=5)

    def _get_cache_key(self, **kwargs) -> str:
        """Generate cache key from parameters"""
        return ":".join(f"{k}={v}" for k, v in sorted(kwargs.items()) if v is not None)

    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get from cache if not expired"""
        if cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            if datetime.now() - timestamp < self._cache_ttl:
                return data
        return None

    def _set_cache(self, cache_key: str, data: Any):
        """Set cache with timestamp"""
        self._cache[cache_key] = (data, datetime.now())

    async def get_agent_prompts(
        self,
        category: Optional[str] = None,
        is_active: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get agent prompts (pdf-processor, search, product, interior-designer)

        Args:
            category: Optional agent category filter
            is_active: Filter by active status

        Returns:
            List of agent prompts
        """
        try:
            cache_key = self._get_cache_key(type=self.AGENT, category=category, active=is_active)
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached

            query = self.supabase.client.table('prompts')\
                .select('*')\
                .eq('prompt_type', self.AGENT)\
                .eq('is_active', is_active)

            if category:
                query = query.eq('category', category)

            result = query.order('category').execute()
            data = result.data if result.data else []

            self._set_cache(cache_key, data)
            return data

        except Exception as e:
            logger.error(f"Error fetching agent prompts: {str(e)}")
            return []

    async def get_extraction_prompts(
        self,
        workspace_id: str,
        stage: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get extraction prompts for PDF processing

        Args:
            workspace_id: Workspace ID
            stage: Optional stage filter (discovery, entity_creation, image_analysis)
            category: Optional category filter (products, material_properties)

        Returns:
            List of extraction prompts
        """
        try:
            cache_key = self._get_cache_key(type=self.EXTRACTION, workspace=workspace_id, stage=stage, category=category)
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached

            query = self.supabase.client.table('prompts')\
                .select('*')\
                .eq('prompt_type', self.EXTRACTION)\
                .eq('workspace_id', workspace_id)\
                .eq('is_active', True)

            if stage:
                query = query.eq('stage', stage)
            if category:
                query = query.eq('category', category)

            result = query.order('stage').order('category').order('version', desc=True).execute()
            data = result.data if result.data else []

            self._set_cache(cache_key, data)
            return data

        except Exception as e:
            logger.error(f"Error fetching extraction prompts: {str(e)}")
            return []

    async def get_extraction_prompt(
        self,
        workspace_id: str,
        stage: str,
        category: str
    ) -> Optional[Dict[str, Any]]:
        """Get specific extraction prompt (latest version)"""
        try:
            result = self.supabase.client.table('prompts')\
                .select('*')\
                .eq('prompt_type', self.EXTRACTION)\
                .eq('workspace_id', workspace_id)\
                .eq('stage', stage)\
                .eq('category', category)\
                .eq('is_active', True)\
                .order('version', desc=True)\
                .limit(1)\
                .execute()

            if result.data and len(result.data) > 0:
                return result.data[0]
            return None

        except Exception as e:
            logger.error(f"Error fetching extraction prompt: {str(e)}")
            return None

    async def get_template_prompts(
        self,
        workspace_id: str,
        stage: Optional[str] = None,
        industry: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get template prompts for customizable AI prompts

        Args:
            workspace_id: Workspace ID
            stage: Optional stage filter
            industry: Optional industry filter

        Returns:
            List of template prompts
        """
        try:
            cache_key = self._get_cache_key(type=self.TEMPLATE, workspace=workspace_id, stage=stage, industry=industry)
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached

            query = self.supabase.client.table('prompts')\
                .select('*')\
                .eq('prompt_type', self.TEMPLATE)\
                .eq('workspace_id', workspace_id)\
                .eq('is_active', True)

            if stage:
                query = query.eq('stage', stage)
            if industry:
                query = query.eq('industry', industry)

            result = query.order('stage').execute()
            data = result.data if result.data else []

            self._set_cache(cache_key, data)
            return data

        except Exception as e:
            logger.error(f"Error fetching template prompts: {str(e)}")
            return []

    async def get_template_prompt(
        self,
        workspace_id: str,
        stage: str,
        category: Optional[str] = None,
        industry: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get best matching template prompt with priority:
        1. Custom template for specific industry + category
        2. Custom template for specific industry (any category)
        3. Custom template for specific category (any industry)
        4. Default template for stage
        """
        try:
            # Try exact match: industry + category
            if industry and category:
                result = self.supabase.client.table('prompts')\
                    .select('*')\
                    .eq('prompt_type', self.TEMPLATE)\
                    .eq('workspace_id', workspace_id)\
                    .eq('stage', stage)\
                    .eq('industry', industry)\
                    .eq('category', category)\
                    .eq('is_active', True)\
                    .order('updated_at', desc=True)\
                    .limit(1)\
                    .execute()
                if result.data:
                    return result.data[0]

            # Try industry match (any category)
            if industry:
                result = self.supabase.client.table('prompts')\
                    .select('*')\
                    .eq('prompt_type', self.TEMPLATE)\
                    .eq('workspace_id', workspace_id)\
                    .eq('stage', stage)\
                    .eq('industry', industry)\
                    .is_('category', 'null')\
                    .eq('is_active', True)\
                    .order('updated_at', desc=True)\
                    .limit(1)\
                    .execute()
                if result.data:
                    return result.data[0]

            # Try category match (any industry)
            if category:
                result = self.supabase.client.table('prompts')\
                    .select('*')\
                    .eq('prompt_type', self.TEMPLATE)\
                    .eq('workspace_id', workspace_id)\
                    .eq('stage', stage)\
                    .eq('category', category)\
                    .is_('industry', 'null')\
                    .eq('is_active', True)\
                    .order('updated_at', desc=True)\
                    .limit(1)\
                    .execute()
                if result.data:
                    return result.data[0]

            # Default template for stage
            result = self.supabase.client.table('prompts')\
                .select('*')\
                .eq('prompt_type', self.TEMPLATE)\
                .eq('workspace_id', workspace_id)\
                .eq('stage', stage)\
                .eq('is_default', True)\
                .eq('is_active', True)\
                .limit(1)\
                .execute()

            if result.data:
                return result.data[0]
            return None

        except Exception as e:
            logger.error(f"Error fetching template prompt: {str(e)}")
            return None

    async def get_search_prompts(
        self,
        workspace_id: str,
        prompt_subtype: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get search prompts (enhancement, formatting, filtering, enrichment)

        Args:
            workspace_id: Workspace ID
            prompt_subtype: Optional filter (enhancement, formatting, filtering, enrichment)

        Returns:
            List of search prompts
        """
        try:
            cache_key = self._get_cache_key(type=self.SEARCH, workspace=workspace_id, subtype=prompt_subtype)
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached

            query = self.supabase.client.table('prompts')\
                .select('*')\
                .eq('prompt_type', self.SEARCH)\
                .eq('workspace_id', workspace_id)\
                .eq('is_active', True)

            if prompt_subtype:
                query = query.eq('subcategory', prompt_subtype)

            result = query.order('created_at').execute()
            data = result.data if result.data else []

            self._set_cache(cache_key, data)
            return data

        except Exception as e:
            logger.error(f"Error fetching search prompts: {str(e)}")
            return []

    def clear_cache(self):
        """Clear all cached prompts"""
        self._cache.clear()
        logger.info("Prompt cache cleared")

