"""
Unified Prompt Service

Single service for accessing all prompts from the unified 'prompts' table.
Replaces: prompt_template_service, search_prompt_service.

NOT admin_prompt_service (#347 phase 3P.3, decided 2026-08-14). That module is the admin
CRUD and audit surface — update_prompt, get_prompt_history, test_prompt, _create_audit_entry —
behind seven endpoints in api/admin_prompts.py. Nothing here replaces any of it: this service
only READS. Deleting it on the strength of this line would remove the ability to edit prompts
at all, which is the entire point of prompts living in the database.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from app.services.core.supabase_client import get_supabase_client
from app.services.utilities.prompt_registry import (
    prefer_workspace,
    prefer_workspace_rows,
    workspace_scope,
)

from app.services.utilities.prompt_registry import PromptStoreUnavailable

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
            # Raised, not collapsed (#28 M14-5). Returning an empty result here made a
            # database outage indistinguishable from "this workspace has no prompts" —
            # the exact ambiguity `PromptNotConfigured` vs `PromptStoreUnavailable`
            # exists to remove, and the reason the six loaders this registry replaced
            # could not be reacted to correctly by any caller.
            raise PromptStoreUnavailable(str(e)) from e

    async def get_extraction_prompt(
        self,
        workspace_id: str,
        stage: str,
        category: str
    ) -> Optional[Dict[str, Any]]:
        """Get specific extraction prompt (latest version)"""
        try:
            # Widened to [workspace, global] so a platform default is reachable, and
            # `.limit(1)` dropped with it: on a two-scope query it returns whichever row the
            # database happened to order first, which could be the default over the tenant's
            # own customisation (#347).
            result = self.supabase.client.table('prompts')\
                .select('*')\
                .eq('prompt_type', self.EXTRACTION)\
                .in_('workspace_id', workspace_scope(workspace_id))\
                .eq('stage', stage)\
                .eq('category', category)\
                .eq('is_active', True)\
                .order('version', desc=True)\
                .execute()

            return prefer_workspace(result.data or [], workspace_id)

        except Exception as e:
            logger.error(f"Error fetching extraction prompt: {str(e)}")
            # Raised, not collapsed (#28 M14-5). Returning an empty result here made a
            # database outage indistinguishable from "this workspace has no prompts" —
            # the exact ambiguity `PromptNotConfigured` vs `PromptStoreUnavailable`
            # exists to remove, and the reason the six loaders this registry replaced
            # could not be reacted to correctly by any caller.
            raise PromptStoreUnavailable(str(e)) from e

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
            if not workspace_id:
                return []

            cache_key = self._get_cache_key(type=self.TEMPLATE, workspace=workspace_id, stage=stage, industry=industry)
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached

            query = self.supabase.client.table('prompts')\
                .select('*')\
                .eq('prompt_type', self.TEMPLATE)\
                .in_('workspace_id', workspace_scope(workspace_id))\
                .eq('is_active', True)

            if stage:
                query = query.eq('stage', stage)
            if industry:
                query = query.eq('industry', industry)

            result = query.order('stage').execute()
            # A tenant that has customised a template would otherwise see it twice — its own
            # row and the platform default — with no guarantee which one a caller picks.
            data = prefer_workspace_rows(
                result.data or [], workspace_id,
                lambda r: (r.get('stage'), r.get('category'), r.get('industry')),
            )

            self._set_cache(cache_key, data)
            return data

        except Exception as e:
            logger.error(f"Error fetching template prompts: {str(e)}")
            # Raised, not collapsed (#28 M14-5). Returning an empty result here made a
            # database outage indistinguishable from "this workspace has no prompts" —
            # the exact ambiguity `PromptNotConfigured` vs `PromptStoreUnavailable`
            # exists to remove, and the reason the six loaders this registry replaced
            # could not be reacted to correctly by any caller.
            raise PromptStoreUnavailable(str(e)) from e

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
                    .in_('workspace_id', workspace_scope(workspace_id))\
                    .eq('stage', stage)\
                    .eq('industry', industry)\
                    .eq('category', category)\
                    .eq('is_active', True)\
                    .order('updated_at', desc=True)\
                    .execute()
                row = prefer_workspace(result.data or [], workspace_id)
                if row:
                    return row

            # Try industry match (any category)
            if industry:
                result = self.supabase.client.table('prompts')\
                    .select('*')\
                    .eq('prompt_type', self.TEMPLATE)\
                    .in_('workspace_id', workspace_scope(workspace_id))\
                    .eq('stage', stage)\
                    .eq('industry', industry)\
                    .is_('category', 'null')\
                    .eq('is_active', True)\
                    .order('updated_at', desc=True)\
                    .execute()
                row = prefer_workspace(result.data or [], workspace_id)
                if row:
                    return row

            # Try category match (any industry)
            if category:
                result = self.supabase.client.table('prompts')\
                    .select('*')\
                    .eq('prompt_type', self.TEMPLATE)\
                    .in_('workspace_id', workspace_scope(workspace_id))\
                    .eq('stage', stage)\
                    .eq('category', category)\
                    .is_('industry', 'null')\
                    .eq('is_active', True)\
                    .order('updated_at', desc=True)\
                    .execute()
                row = prefer_workspace(result.data or [], workspace_id)
                if row:
                    return row

            # Default template for stage
            result = self.supabase.client.table('prompts')\
                .select('*')\
                .eq('prompt_type', self.TEMPLATE)\
                .in_('workspace_id', workspace_scope(workspace_id))\
                .eq('stage', stage)\
                .eq('is_default', True)\
                .eq('is_active', True)\
                .execute()

            return prefer_workspace(result.data or [], workspace_id)

        except Exception as e:
            logger.error(f"Error fetching template prompt: {str(e)}")
            # Raised, not collapsed (#28 M14-5). Returning an empty result here made a
            # database outage indistinguishable from "this workspace has no prompts" —
            # the exact ambiguity `PromptNotConfigured` vs `PromptStoreUnavailable`
            # exists to remove, and the reason the six loaders this registry replaced
            # could not be reacted to correctly by any caller.
            raise PromptStoreUnavailable(str(e)) from e

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
            if not workspace_id:
                return []

            cache_key = self._get_cache_key(type=self.SEARCH, workspace=workspace_id, subtype=prompt_subtype)
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached

            query = self.supabase.client.table('prompts')\
                .select('*')\
                .eq('prompt_type', self.SEARCH)\
                .in_('workspace_id', workspace_scope(workspace_id))\
                .eq('is_active', True)

            if prompt_subtype:
                query = query.eq('subcategory', prompt_subtype)

            result = query.order('created_at').execute()
            data = prefer_workspace_rows(
                result.data or [], workspace_id,
                lambda r: (r.get('category'), r.get('subcategory')),
            )

            self._set_cache(cache_key, data)
            return data

        except Exception as e:
            logger.error(f"Error fetching search prompts: {str(e)}")
            # Raised, not collapsed (#28 M14-5). Returning an empty result here made a
            # database outage indistinguishable from "this workspace has no prompts" —
            # the exact ambiguity `PromptNotConfigured` vs `PromptStoreUnavailable`
            # exists to remove, and the reason the six loaders this registry replaced
            # could not be reacted to correctly by any caller.
            raise PromptStoreUnavailable(str(e)) from e

    def clear_cache(self):
        """Clear all cached prompts"""
        self._cache.clear()
        logger.info("Prompt cache cleared")

