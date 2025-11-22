"""
Prompt Template Service

Manages customizable AI prompts for different extraction stages and industries.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.services.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class PromptTemplateService:
    """Service for managing prompt templates."""

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
            supabase = get_supabase_client().client

            # Try exact match first: industry + category
            if industry and category:
                response = supabase.table('prompt_templates')\
                    .select('*')\
                    .eq('workspace_id', workspace_id)\
                    .eq('stage', stage)\
                    .eq('industry', industry)\
                    .eq('category', category)\
                    .eq('is_active', True)\
                    .order('updated_at', desc=True)\
                    .limit(1)\
                    .execute()

                if response.data:
                    return response.data[0]

            # Try industry match (any category)
            if industry:
                response = supabase.table('prompt_templates')\
                    .select('*')\
                    .eq('workspace_id', workspace_id)\
                    .eq('stage', stage)\
                    .eq('industry', industry)\
                    .is_('category', 'null')\
                    .eq('is_active', True)\
                    .order('updated_at', desc=True)\
                    .limit(1)\
                    .execute()

                if response.data:
                    return response.data[0]

            # Try category match (any industry)
            if category:
                response = supabase.table('prompt_templates')\
                    .select('*')\
                    .eq('workspace_id', workspace_id)\
                    .eq('stage', stage)\
                    .eq('category', category)\
                    .is_('industry', 'null')\
                    .eq('is_active', True)\
                    .order('updated_at', desc=True)\
                    .limit(1)\
                    .execute()

                if response.data:
                    return response.data[0]

            # Try default template
            response = supabase.table('prompt_templates')\
                .select('*')\
                .eq('workspace_id', workspace_id)\
                .eq('stage', stage)\
                .eq('is_default', True)\
                .eq('is_active', True)\
                .order('updated_at', desc=True)\
                .limit(1)\
                .execute()

            if response.data:
                return response.data[0]

            return None

        except Exception as e:
            logger.error(f"Failed to get template: {str(e)}")
            return None
    
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
            supabase = get_supabase_client().client

            query = supabase.table('prompt_templates').select('*').eq('workspace_id', workspace_id)

            if stage:
                query = query.eq('stage', stage)

            if category:
                query = query.eq('category', category)

            if industry:
                query = query.eq('industry', industry)

            if not include_inactive:
                query = query.eq('is_active', True)

            # Order by default first, then by name
            query = query.order('is_default', desc=True).order('name')

            response = query.execute()
            return response.data if response.data else []

        except Exception as e:
            logger.error(f"Failed to list templates: {str(e)}")
            return []

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
        """Create a new prompt template."""
        try:
            supabase = get_supabase_client().client

            data = {
                'workspace_id': workspace_id,
                'name': name,
                'stage': stage,
                'prompt_template': prompt_template,
                'description': description,
                'industry': industry,
                'category': category,
                'system_prompt': system_prompt,
                'model_preference': model_preference,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'created_by': created_by
            }

            response = supabase.table('prompt_templates').insert(data).execute()

            if response.data:
                logger.info(f"✅ Created prompt template: {name} (ID: {response.data[0]['id']})")
                return response.data[0]

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
        """Update an existing prompt template."""
        try:
            supabase = get_supabase_client().client

            # Get current template for history
            current_response = supabase.table('prompt_templates')\
                .select('prompt_template, system_prompt, version')\
                .eq('id', template_id)\
                .eq('workspace_id', workspace_id)\
                .execute()

            if not current_response.data:
                raise ValueError(f"Template {template_id} not found")

            current = current_response.data[0]

            # Build update data
            update_data = {}

            if prompt_template is not None:
                update_data['prompt_template'] = prompt_template

            if system_prompt is not None:
                update_data['system_prompt'] = system_prompt

            if name is not None:
                update_data['name'] = name

            if description is not None:
                update_data['description'] = description

            if model_preference is not None:
                update_data['model_preference'] = model_preference

            if temperature is not None:
                update_data['temperature'] = temperature

            if max_tokens is not None:
                update_data['max_tokens'] = max_tokens

            if is_active is not None:
                update_data['is_active'] = is_active

            if not update_data:
                raise ValueError("No fields to update")

            # Increment version
            update_data['version'] = current['version'] + 1
            update_data['updated_at'] = datetime.utcnow().isoformat()

            # Update template
            response = supabase.table('prompt_templates')\
                .update(update_data)\
                .eq('id', template_id)\
                .eq('workspace_id', workspace_id)\
                .execute()

            if not response.data:
                raise Exception("Failed to update template")

            # Record history if prompt changed
            if prompt_template is not None or system_prompt is not None:
                history_data = {
                    'template_id': template_id,
                    'old_prompt': current['prompt_template'],
                    'new_prompt': prompt_template or current['prompt_template'],
                    'old_system_prompt': current.get('system_prompt'),
                    'new_system_prompt': system_prompt or current.get('system_prompt'),
                    'changed_by': changed_by,
                    'change_reason': change_reason
                }

                supabase.table('prompt_template_history').insert(history_data).execute()

            logger.info(f"✅ Updated prompt template: {response.data[0]['name']} (version {response.data[0]['version']})")
            return response.data[0]

        except Exception as e:
            logger.error(f"Failed to update template: {str(e)}")
            raise

    async def delete_template(self, template_id: str, workspace_id: str) -> bool:
        """Delete a prompt template (soft delete by setting is_active=False)."""
        try:
            supabase = get_supabase_client().client

            response = supabase.table('prompt_templates')\
                .update({'is_active': False})\
                .eq('id', template_id)\
                .eq('workspace_id', workspace_id)\
                .execute()

            return len(response.data) > 0

        except Exception as e:
            logger.error(f"Failed to delete template: {str(e)}")
            return False

    async def get_template_history(self, template_id: str) -> List[Dict[str, Any]]:
        """Get change history for a template."""
        try:
            supabase = get_supabase_client().client

            response = supabase.table('prompt_template_history')\
                .select('*')\
                .eq('template_id', template_id)\
                .order('changed_at', desc=True)\
                .execute()

            return response.data if response.data else []

        except Exception as e:
            logger.error(f"Failed to get template history: {str(e)}")
            return []

