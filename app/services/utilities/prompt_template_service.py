"""
Prompt Template Service

Manages customizable AI prompts for different extraction stages and industries.
UPDATED: Now uses UnifiedPromptService for all prompt operations.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.services.core.supabase_client import get_supabase_client
from app.services.utilities.unified_prompt_service import UnifiedPromptService

from app.services.utilities.prompt_registry import PromptStoreUnavailable

logger = logging.getLogger(__name__)


class PromptTemplateService:
    """Service for managing prompt templates."""

    def __init__(self):
        self.prompt_service = UnifiedPromptService()

    async def get_template(
        self,
        workspace_id: str,
        stage: str,
        category: Optional[str] = None,
        industry: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best matching prompt template for a stage/category/industry.

        Priority:
        1. Custom template for specific industry + category
        2. Custom template for specific industry (any category)
        3. Custom template for specific category (any industry)
        4. Default template for stage

        Args:
            workspace_id: Workspace UUID
            stage: Processing stage (metadata_extraction, discovery, classification, chunking)
            category: Optional category (products, certificates, etc.)
            industry: Optional industry (construction, interior_design, etc.)

        Returns:
            Template dict or None if not found
        """
        try:
            return await self.prompt_service.get_template_prompt(
                workspace_id=workspace_id,
                stage=stage,
                category=category,
                industry=industry
            )
        except Exception as e:
            logger.error(f"Failed to get template: {str(e)}")
            # Raised, not collapsed (#28 M14-5). Returning an empty result here made a
            # database outage indistinguishable from "this workspace has no prompts" —
            # the exact ambiguity `PromptNotConfigured` vs `PromptStoreUnavailable`
            # exists to remove, and the reason the six loaders this registry replaced
            # could not be reacted to correctly by any caller.
            raise PromptStoreUnavailable(str(e)) from e
    
    async def list_templates(
        self,
        workspace_id: str,
        stage: Optional[str] = None,
        category: Optional[str] = None,
        industry: Optional[str] = None,
        include_inactive: bool = False
    ) -> List[Dict[str, Any]]:
        """List all prompt templates with optional filtering."""
        try:
            rows = await self.prompt_service.get_template_prompts(
                workspace_id=workspace_id,
                stage=stage,
                industry=industry
            )
            # get_template_prompts returns raw `prompts` table rows (prompt_text,
            # configuration, subcategory, ...). The admin API contract (and the frontend)
            # expect the legacy prompt_templates shape (prompt_template, model_preference,
            # temperature, max_tokens). Reshape here so response validation doesn't 500
            # (Sentry MIVAA-5JG).
            return [self._to_template_response(r) for r in rows]
        except Exception as e:
            logger.error(f"Failed to list templates: {str(e)}")
            # Raised, not collapsed (#28 M14-5). Returning an empty result here made a
            # database outage indistinguishable from "this workspace has no prompts" —
            # the exact ambiguity `PromptNotConfigured` vs `PromptStoreUnavailable`
            # exists to remove, and the reason the six loaders this registry replaced
            # could not be reacted to correctly by any caller.
            raise PromptStoreUnavailable(str(e)) from e

    @staticmethod
    def _to_template_response(row: Dict[str, Any]) -> Dict[str, Any]:
        """Map a raw `prompts` row onto the PromptTemplateResponse contract."""
        config = row.get('configuration') or {}
        if not isinstance(config, dict):
            config = {}

        def _num(value, default):
            try:
                return type(default)(value)
            except (TypeError, ValueError):
                return default

        return {
            'id': str(row.get('id') or ''),
            'workspace_id': str(row.get('workspace_id') or ''),
            'name': row.get('name') or '',
            'description': row.get('description'),
            'industry': row.get('industry'),
            'stage': row.get('stage') or '',
            'category': row.get('category'),
            # `prompts` stores the body in prompt_text; the admin API calls it prompt_template.
            'prompt_template': row.get('prompt_text') or row.get('prompt_template') or '',
            'system_prompt': row.get('system_prompt'),
            'model_preference': config.get('model_preference') or config.get('model'),
            'temperature': _num(config.get('temperature', 0.1), 0.1),
            'max_tokens': _num(config.get('max_tokens', 4096), 4096),
            'is_default': bool(row.get('is_default', False)),
            'is_active': bool(row.get('is_active', True)),
            'version': _num(row.get('version', 1), 1),
            'created_by': row.get('created_by'),
            'created_at': str(row.get('created_at') or ''),
            'updated_at': str(row.get('updated_at') or ''),
        }

    #: The unified `prompts` table stores the body in prompt_text and the model
    #: knobs in a `configuration` jsonb; the admin API contract (and the frontend)
    #: still speak the legacy prompt_templates vocabulary. _to_template_response
    #: maps rows outwards; this maps the same fields inwards.
    @staticmethod
    def _configuration_from(
        model_preference: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        base: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        config = dict(base or {})
        if model_preference is not None:
            config['model_preference'] = model_preference
        if temperature is not None:
            config['temperature'] = temperature
        if max_tokens is not None:
            config['max_tokens'] = max_tokens
        return config

    async def create_template(
        self,
        workspace_id: str,
        name: str,
        stage: str,
        prompt_template: str,
        description: Optional[str] = None,
        industry: Optional[str] = None,
        category: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model_preference: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        created_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new prompt template in the unified `prompts` table."""
        try:
            supabase = get_supabase_client()

            data = {
                'prompt_type': 'template',
                'workspace_id': workspace_id,
                'name': name,
                'stage': stage,
                # NOT NULL in `prompts`; the legacy table allowed it to be absent.
                'category': category or 'general',
                'prompt_text': prompt_template,
                'description': description,
                'industry': industry,
                'system_prompt': system_prompt,
                'configuration': self._configuration_from(
                    model_preference, temperature, max_tokens
                ),
                'is_custom': True,
                'created_by': created_by,
            }

            response = supabase.client.table('prompts').insert(data).execute()

            if response.data:
                logger.info(f"✅ Created prompt template: {name} (ID: {response.data[0]['id']})")
                return self._to_template_response(response.data[0])

            raise Exception("Failed to create template")

        except Exception as e:
            logger.error(f"Failed to create template: {str(e)}")
            raise

    async def update_template(
        self,
        template_id: str,
        workspace_id: str,
        prompt_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        model_preference: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        is_active: Optional[bool] = None,
        changed_by: Optional[str] = None,
        change_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update a prompt template in the unified `prompts` table."""
        try:
            supabase = get_supabase_client()

            current_response = supabase.client.table('prompts')\
                .select('prompt_text, system_prompt, version, configuration')\
                .eq('id', template_id)\
                .eq('prompt_type', 'template')\
                .eq('workspace_id', workspace_id)\
                .execute()

            if not current_response.data:
                raise ValueError(f"Template {template_id} not found")

            current = current_response.data[0]

            update_data: Dict[str, Any] = {}

            if prompt_template is not None:
                update_data['prompt_text'] = prompt_template

            if system_prompt is not None:
                update_data['system_prompt'] = system_prompt

            if name is not None:
                update_data['name'] = name

            if description is not None:
                update_data['description'] = description

            if any(v is not None for v in (model_preference, temperature, max_tokens)):
                base = current.get('configuration')
                update_data['configuration'] = self._configuration_from(
                    model_preference, temperature, max_tokens,
                    base=base if isinstance(base, dict) else {},
                )

            if is_active is not None:
                update_data['is_active'] = is_active

            if not update_data:
                raise ValueError("No fields to update")

            update_data['version'] = (current.get('version') or 1) + 1
            update_data['updated_at'] = datetime.utcnow().isoformat()

            response = supabase.client.table('prompts')\
                .update(update_data)\
                .eq('id', template_id)\
                .eq('prompt_type', 'template')\
                .eq('workspace_id', workspace_id)\
                .execute()

            if not response.data:
                raise Exception("Failed to update template")

            # Audit trail goes to `prompt_history`, the same table admin_prompt_service
            # writes for extraction prompts — one history for one prompts table.
            if prompt_template is not None or system_prompt is not None:
                try:
                    supabase.client.table('prompt_history').insert({
                        'prompt_id': template_id,
                        'old_prompt_text': current.get('prompt_text') or '',
                        'new_prompt_text': prompt_template or current.get('prompt_text') or '',
                        'old_system_prompt': current.get('system_prompt'),
                        'new_system_prompt': system_prompt or current.get('system_prompt'),
                        'changed_by': changed_by,
                        'change_reason': change_reason,
                    }).execute()
                except Exception as hist_err:
                    # The edit itself succeeded; losing the audit row must not
                    # present as a failed save, but it must not be silent either.
                    logger.warning(
                        f"Prompt template {template_id} updated but history write failed: {hist_err}"
                    )

            logger.info(
                f"✅ Updated prompt template: {response.data[0].get('name')} "
                f"(version {response.data[0].get('version')})"
            )
            return self._to_template_response(response.data[0])

        except Exception as e:
            logger.error(f"Failed to update template: {str(e)}")
            raise

    async def delete_template(self, template_id: str, workspace_id: str) -> bool:
        """Soft-delete a prompt template (is_active=False)."""
        try:
            supabase = get_supabase_client()

            response = supabase.client.table('prompts')\
                .update({'is_active': False, 'updated_at': datetime.utcnow().isoformat()})\
                .eq('id', template_id)\
                .eq('prompt_type', 'template')\
                .eq('workspace_id', workspace_id)\
                .execute()

            return bool(response.data)

        except Exception as e:
            logger.error(f"Failed to delete template: {str(e)}")
            # Raised, not collapsed (#28 M14-5). Returning an empty result here made a
            # database outage indistinguishable from "this workspace has no prompts" —
            # the exact ambiguity `PromptNotConfigured` vs `PromptStoreUnavailable`
            # exists to remove, and the reason the six loaders this registry replaced
            # could not be reacted to correctly by any caller.
            raise PromptStoreUnavailable(str(e)) from e

    async def get_template_history(self, template_id: str) -> List[Dict[str, Any]]:
        """Get change history for a template from the unified `prompt_history`."""
        try:
            supabase = get_supabase_client()

            response = supabase.client.table('prompt_history')\
                .select('*')\
                .eq('prompt_id', template_id)\
                .order('changed_at', desc=True)\
                .execute()

            return response.data if response.data else []

        except Exception as e:
            logger.error(f"Failed to get template history: {str(e)}")
            # Raised, not collapsed (#28 M14-5). Returning an empty result here made a
            # database outage indistinguishable from "this workspace has no prompts" —
            # the exact ambiguity `PromptNotConfigured` vs `PromptStoreUnavailable`
            # exists to remove, and the reason the six loaders this registry replaced
            # could not be reacted to correctly by any caller.
            raise PromptStoreUnavailable(str(e)) from e
