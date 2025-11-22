"""
Prompt Template Service

Manages customizable AI prompts for different extraction stages and industries.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncpg

from app.database import get_db_pool

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
        pool = await get_db_pool()
        
        async with pool.acquire() as conn:
            # Try to find best match
            query = """
                SELECT 
                    id, name, description, industry, stage, category,
                    prompt_template, system_prompt, model_preference,
                    temperature, max_tokens, version, created_at, updated_at
                FROM prompt_templates
                WHERE workspace_id = $1 
                    AND stage = $2
                    AND is_active = TRUE
                    AND (
                        -- Exact match: industry + category
                        (industry = $3 AND category = $4)
                        OR
                        -- Industry match, any category
                        (industry = $3 AND category IS NULL)
                        OR
                        -- Category match, any industry
                        (category = $4 AND industry IS NULL)
                        OR
                        -- Default template
                        (is_default = TRUE AND industry IS NULL AND category IS NULL)
                    )
                ORDER BY 
                    -- Prioritize exact matches
                    CASE 
                        WHEN industry = $3 AND category = $4 THEN 1
                        WHEN industry = $3 AND category IS NULL THEN 2
                        WHEN category = $4 AND industry IS NULL THEN 3
                        WHEN is_default = TRUE THEN 4
                        ELSE 5
                    END,
                    updated_at DESC
                LIMIT 1
            """
            
            row = await conn.fetchrow(query, workspace_id, stage, industry, category)
            
            if row:
                return dict(row)
            
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
        pool = await get_db_pool()
        
        async with pool.acquire() as conn:
            conditions = ["workspace_id = $1"]
            params: List[Any] = [workspace_id]
            param_count = 1
            
            if stage:
                param_count += 1
                conditions.append(f"stage = ${param_count}")
                params.append(stage)
            
            if category:
                param_count += 1
                conditions.append(f"category = ${param_count}")
                params.append(category)
            
            if industry:
                param_count += 1
                conditions.append(f"industry = ${param_count}")
                params.append(industry)
            
            if not include_inactive:
                conditions.append("is_active = TRUE")
            
            query = f"""
                SELECT 
                    id, name, description, industry, stage, category,
                    prompt_template, system_prompt, model_preference,
                    temperature, max_tokens, is_default, is_active,
                    version, created_by, created_at, updated_at
                FROM prompt_templates
                WHERE {' AND '.join(conditions)}
                ORDER BY is_default DESC, industry NULLS LAST, category NULLS LAST, name
            """
            
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]
    
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
        pool = await get_db_pool()
        
        async with pool.acquire() as conn:
            query = """
                INSERT INTO prompt_templates (
                    workspace_id, name, description, industry, stage, category,
                    prompt_template, system_prompt, model_preference,
                    temperature, max_tokens, created_by
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                RETURNING id, name, created_at
            """
            
            row = await conn.fetchrow(
                query,
                workspace_id, name, description, industry, stage, category,
                prompt_template, system_prompt, model_preference,
                temperature, max_tokens, created_by
            )
            
            logger.info(f"✅ Created prompt template: {name} (ID: {row['id']})")
            return dict(row)

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
        pool = await get_db_pool()

        async with pool.acquire() as conn:
            # Get current template for history
            current = await conn.fetchrow(
                "SELECT prompt_template, system_prompt FROM prompt_templates WHERE id = $1 AND workspace_id = $2",
                template_id, workspace_id
            )

            if not current:
                raise ValueError(f"Template {template_id} not found")

            # Build update query dynamically
            updates = []
            params = []
            param_count = 0

            if prompt_template is not None:
                param_count += 1
                updates.append(f"prompt_template = ${param_count}")
                params.append(prompt_template)

            if system_prompt is not None:
                param_count += 1
                updates.append(f"system_prompt = ${param_count}")
                params.append(system_prompt)

            if name is not None:
                param_count += 1
                updates.append(f"name = ${param_count}")
                params.append(name)

            if description is not None:
                param_count += 1
                updates.append(f"description = ${param_count}")
                params.append(description)

            if model_preference is not None:
                param_count += 1
                updates.append(f"model_preference = ${param_count}")
                params.append(model_preference)

            if temperature is not None:
                param_count += 1
                updates.append(f"temperature = ${param_count}")
                params.append(temperature)

            if max_tokens is not None:
                param_count += 1
                updates.append(f"max_tokens = ${param_count}")
                params.append(max_tokens)

            if is_active is not None:
                param_count += 1
                updates.append(f"is_active = ${param_count}")
                params.append(is_active)

            if not updates:
                raise ValueError("No fields to update")

            # Increment version
            param_count += 1
            updates.append(f"version = version + 1")
            updates.append(f"updated_at = NOW()")

            # Add WHERE clause params
            param_count += 1
            params.append(template_id)
            param_count += 1
            params.append(workspace_id)

            query = f"""
                UPDATE prompt_templates
                SET {', '.join(updates)}
                WHERE id = ${param_count - 1} AND workspace_id = ${param_count}
                RETURNING id, name, version, updated_at
            """

            row = await conn.fetchrow(query, *params)

            # Record history if prompt changed
            if prompt_template is not None or system_prompt is not None:
                await conn.execute(
                    """
                    INSERT INTO prompt_template_history (
                        template_id, old_prompt, new_prompt, old_system_prompt, new_system_prompt,
                        changed_by, change_reason
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    template_id,
                    current['prompt_template'],
                    prompt_template or current['prompt_template'],
                    current['system_prompt'],
                    system_prompt or current['system_prompt'],
                    changed_by,
                    change_reason
                )

            logger.info(f"✅ Updated prompt template: {row['name']} (version {row['version']})")
            return dict(row)

    async def delete_template(self, template_id: str, workspace_id: str) -> bool:
        """Delete a prompt template (soft delete by setting is_active=False)."""
        pool = await get_db_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE prompt_templates SET is_active = FALSE WHERE id = $1 AND workspace_id = $2",
                template_id, workspace_id
            )

            return result == "UPDATE 1"

    async def get_template_history(self, template_id: str) -> List[Dict[str, Any]]:
        """Get change history for a template."""
        pool = await get_db_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, old_prompt, new_prompt, old_system_prompt, new_system_prompt,
                       changed_by, change_reason, changed_at
                FROM prompt_template_history
                WHERE template_id = $1
                ORDER BY changed_at DESC
                """,
                template_id
            )

            return [dict(row) for row in rows]

