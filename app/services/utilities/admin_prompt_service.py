"""
Admin Prompt Service

Manages extraction prompts for admins including CRUD operations and audit trail.
UPDATED: Now uses UnifiedPromptService for all prompt operations.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import UUID

from app.services.core.supabase_client import get_supabase_client
from app.services.utilities.unified_prompt_service import UnifiedPromptService

logger = logging.getLogger(__name__)


class AdminPromptService:
    """Service for managing extraction prompts"""

    def __init__(self):
        self.supabase = get_supabase_client()
        self.prompt_service = UnifiedPromptService()
    
    async def get_prompts(
        self,
        workspace_id: str,
        stage: Optional[str] = None,
        category: Optional[str] = None,
        prompt_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all prompts from unified table, optionally filtered.

        Args:
            workspace_id: Workspace ID
            stage: Optional stage filter
            category: Optional category filter
            prompt_type: Optional prompt_type filter (agent, extraction, template, search)

        Returns:
            List of prompts with used_in field
        """
        try:
            # Query unified prompts table directly to get ALL prompts
            query = self.supabase.client.table('prompts')\
                .select('*')\
                .eq('is_active', True)

            if stage and stage != 'all':
                query = query.eq('stage', stage)
            if category and category != 'all':
                query = query.eq('category', category)
            if prompt_type and prompt_type != 'all':
                query = query.eq('prompt_type', prompt_type)

            result = query.order('prompt_type').order('stage').order('category').execute()

            # Transform to include prompt_template for backward compatibility
            prompts = []
            for p in (result.data or []):
                prompts.append({
                    'id': p.get('id'),
                    'workspace_id': p.get('workspace_id') or workspace_id,
                    'stage': p.get('stage') or 'general',
                    'category': p.get('category'),
                    'name': p.get('name'),
                    'prompt_type': p.get('prompt_type'),
                    'prompt_template': p.get('prompt_text') or p.get('system_prompt') or '',
                    'system_prompt': p.get('system_prompt'),
                    'is_custom': p.get('is_custom', False),
                    'version': p.get('version', 1),
                    'created_by': p.get('created_by'),
                    'created_at': p.get('created_at'),
                    'updated_at': p.get('updated_at'),
                    'used_in': p.get('used_in') or []
                })

            return prompts
        except Exception as e:
            logger.error(f"Error fetching prompts: {str(e)}")
            return []

    async def get_prompt(
        self,
        workspace_id: str,
        stage: str,
        category: str
    ) -> Optional[Dict[str, Any]]:
        """Get specific prompt"""
        try:
            return await self.prompt_service.get_extraction_prompt(
                workspace_id=workspace_id,
                stage=stage,
                category=category
            )
        except Exception as e:
            logger.error(f"Error fetching prompt: {str(e)}")
            return None

    async def update_prompt(
        self,
        workspace_id: str,
        stage: str,
        category: str,
        prompt_template: str,
        system_prompt: Optional[str],
        changed_by: str,
        change_reason: str
    ) -> Dict[str, Any]:
        """
        Update prompt with audit trail
        
        Args:
            workspace_id: Workspace ID
            stage: Extraction stage
            category: Content category
            prompt_template: New prompt template
            system_prompt: Optional system prompt
            changed_by: User ID making the change
            change_reason: Reason for change
            
        Returns:
            Updated prompt record
        """
        try:
            # Get current prompt
            current = await self.get_prompt(workspace_id, stage, category)
            
            if current:
                # Create audit trail entry
                await self._create_audit_entry(
                    prompt_id=current['id'],
                    old_prompt=current['prompt_template'],
                    new_prompt=prompt_template,
                    changed_by=changed_by,
                    change_reason=change_reason
                )
                
                # Increment version
                new_version = current['version'] + 1
                
                # Update existing prompt
                result = self.supabase.client.table('prompts')\
                    .update({
                        'prompt_text': prompt_template,
                        'system_prompt': system_prompt,
                        'version': new_version,
                        'is_custom': True,
                        'updated_at': datetime.utcnow().isoformat()
                    })\
                    .eq('id', current['id'])\
                    .execute()

                return result.data[0] if result.data else {}
            else:
                # Create new prompt
                result = self.supabase.client.table('prompts')\
                    .insert({
                        'workspace_id': workspace_id,
                        'prompt_type': 'extraction',
                        'stage': stage,
                        'category': category,
                        'prompt_text': prompt_template,
                        'system_prompt': system_prompt,
                        'is_custom': True,
                        'version': 1,
                        'created_by': changed_by
                    })\
                    .execute()

                return result.data[0] if result.data else {}
                
        except Exception as e:
            logger.error(f"Error updating prompt: {str(e)}")
            raise
    
    async def get_prompt_history(
        self,
        prompt_id: str
    ) -> List[Dict[str, Any]]:
        """Get change history for a prompt - now uses unified prompt_history table"""
        try:
            result = self.supabase.client.table('prompt_history')\
                .select('*')\
                .eq('prompt_id', prompt_id)\
                .order('changed_at', desc=True)\
                .limit(5)\
                .execute()

            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Error fetching prompt history: {str(e)}")
            return []
    
    async def test_prompt(
        self,
        stage: str,
        category: str,
        prompt_template: str,
        test_content: str
    ) -> Dict[str, Any]:
        """
        Test a prompt before saving
        
        Args:
            stage: Extraction stage
            category: Content category
            prompt_template: Prompt to test
            test_content: Sample content to test with
            
        Returns:
            Test results including execution time and output
        """
        try:
            start_time = datetime.utcnow()
            
            # TODO: Implement actual AI model testing
            # For now, return mock results
            
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            return {
                'success': True,
                'result': 'Test execution successful',
                'execution_time_ms': execution_time,
                'prompt_length': len(prompt_template),
                'test_content_length': len(test_content)
            }
            
        except Exception as e:
            logger.error(f"Error testing prompt: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_extraction_config(
        self,
        workspace_id: str
    ) -> Dict[str, Any]:
        """Get extraction configuration for workspace"""
        try:
            result = self.supabase.client.table('extraction_config')\
                .select('*')\
                .eq('workspace_id', workspace_id)\
                .limit(1)\
                .execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0]
            
            # Return defaults
            return {
                'workspace_id': workspace_id,
                'enabled_categories': ['products'],
                'default_categories': ['products'],
                'discovery_model': 'claude',
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'enable_prompt_enhancement': True,
                'quality_threshold': 0.7
            }
            
        except Exception as e:
            logger.error(f"Error fetching extraction config: {str(e)}")
            return {}
    
    async def update_extraction_config(
        self,
        workspace_id: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update extraction configuration"""
        try:
            # Check if config exists
            existing = await self.get_extraction_config(workspace_id)
            
            if existing and 'id' in existing:
                # Update existing
                result = self.supabase.client.table('extraction_config')\
                    .update({
                        **config,
                        'updated_at': datetime.utcnow().isoformat()
                    })\
                    .eq('workspace_id', workspace_id)\
                    .execute()
            else:
                # Create new
                result = self.supabase.client.table('extraction_config')\
                    .insert({
                        'workspace_id': workspace_id,
                        **config
                    })\
                    .execute()
            
            return result.data[0] if result.data else {}
            
        except Exception as e:
            logger.error(f"Error updating extraction config: {str(e)}")
            raise
    
    async def _create_audit_entry(
        self,
        prompt_id: str,
        old_prompt: str,
        new_prompt: str,
        changed_by: str,
        change_reason: str
    ) -> None:
        """Create audit trail entry - now uses unified prompt_history table"""
        try:
            self.supabase.client.table('prompt_history')\
                .insert({
                    'prompt_id': prompt_id,
                    'old_prompt_text': old_prompt,
                    'new_prompt_text': new_prompt,
                    'changed_by': changed_by,
                    'change_reason': change_reason
                })\
                .execute()

            logger.info(f"Created audit entry for prompt {prompt_id}")

        except Exception as e:
            logger.error(f"Error creating audit entry: {str(e)}")
            # Don't raise - audit trail failure shouldn't block the update


