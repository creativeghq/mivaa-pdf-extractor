"""
Prompt Templates API

REST endpoints for managing customizable AI prompts.
"""

import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.services.prompt_template_service import PromptTemplateService
from app.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/prompt-templates", tags=["Admin - Prompt Templates"])


# Request/Response Models
class PromptTemplateResponse(BaseModel):
    """Prompt template response"""
    id: str
    workspace_id: str
    name: str
    description: Optional[str]
    industry: Optional[str]
    stage: str
    category: Optional[str]
    prompt_template: str
    system_prompt: Optional[str]
    model_preference: Optional[str]
    temperature: float
    max_tokens: int
    is_default: bool
    is_active: bool
    version: int
    created_by: Optional[str]
    created_at: str
    updated_at: str


class CreatePromptTemplateRequest(BaseModel):
    """Create prompt template request"""
    workspace_id: str
    name: str
    stage: str
    prompt_template: str
    description: Optional[str] = None
    industry: Optional[str] = None
    category: Optional[str] = None
    system_prompt: Optional[str] = None
    model_preference: Optional[str] = Field(None, description="claude, gpt, or auto")
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, ge=256, le=16384)


class UpdatePromptTemplateRequest(BaseModel):
    """Update prompt template request"""
    prompt_template: Optional[str] = None
    system_prompt: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    model_preference: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=256, le=16384)
    is_active: Optional[bool] = None
    change_reason: Optional[str] = None


class PromptTemplateHistoryResponse(BaseModel):
    """Prompt template history response"""
    id: str
    old_prompt: Optional[str]
    new_prompt: Optional[str]
    old_system_prompt: Optional[str]
    new_system_prompt: Optional[str]
    changed_by: Optional[str]
    change_reason: Optional[str]
    changed_at: str


# Endpoints
@router.get("", response_model=List[PromptTemplateResponse])
async def list_prompt_templates(
    workspace_id: str,
    stage: Optional[str] = None,
    category: Optional[str] = None,
    industry: Optional[str] = None,
    include_inactive: bool = False,
    current_user: dict = Depends(get_current_user)
):
    """
    **📋 List Prompt Templates**
    
    List all customizable AI prompts for a workspace.
    
    ## Query Parameters
    - **workspace_id** (required): Workspace UUID
    - **stage** (optional): Filter by stage (metadata_extraction, discovery, classification, chunking)
    - **category** (optional): Filter by category (products, certificates, logos, specifications)
    - **industry** (optional): Filter by industry (construction, interior_design, general)
    - **include_inactive** (optional): Include inactive templates (default: false)
    """
    try:
        service = PromptTemplateService()
        templates = await service.list_templates(
            workspace_id=workspace_id,
            stage=stage,
            category=category,
            industry=industry,
            include_inactive=include_inactive
        )
        return templates
    except Exception as e:
        logger.error(f"Failed to list prompt templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=dict)
async def create_prompt_template(
    request: CreatePromptTemplateRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    **➕ Create Prompt Template**
    
    Create a new customizable AI prompt template.
    
    ## Stages
    - **metadata_extraction**: Extract product metadata from PDFs
    - **discovery**: Discover products in PDFs
    - **classification**: Classify images
    - **chunking**: Semantic text chunking
    
    ## Industries
    - **general**: Default for all material types
    - **construction**: Tiles, flooring, construction materials
    - **interior_design**: Furniture, decor, design products
    """
    try:
        service = PromptTemplateService()
        result = await service.create_template(
            workspace_id=request.workspace_id,
            name=request.name,
            stage=request.stage,
            prompt_template=request.prompt_template,
            description=request.description,
            industry=request.industry,
            category=request.category,
            system_prompt=request.system_prompt,
            model_preference=request.model_preference,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            created_by=current_user.get('id')
        )
        return result
    except Exception as e:
        logger.error(f"Failed to create prompt template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{template_id}", response_model=dict)
async def update_prompt_template(
    template_id: str,
    workspace_id: str,
    request: UpdatePromptTemplateRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    **✏️ Update Prompt Template**

    Update an existing prompt template.
    """
    try:
        service = PromptTemplateService()
        result = await service.update_template(
            template_id=template_id,
            workspace_id=workspace_id,
            prompt_template=request.prompt_template,
            system_prompt=request.system_prompt,
            name=request.name,
            description=request.description,
            model_preference=request.model_preference,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            is_active=request.is_active,
            changed_by=current_user.get('id'),
            change_reason=request.change_reason
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update prompt template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{template_id}")
async def delete_prompt_template(
    template_id: str,
    workspace_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    **🗑️ Delete Prompt Template**

    Soft delete a prompt template (sets is_active=false).
    """
    try:
        service = PromptTemplateService()
        success = await service.delete_template(template_id, workspace_id)

        if not success:
            raise HTTPException(status_code=404, detail="Template not found")

        return {"success": True, "message": "Template deleted"}
    except Exception as e:
        logger.error(f"Failed to delete prompt template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{template_id}/history", response_model=List[PromptTemplateHistoryResponse])
async def get_prompt_template_history(
    template_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    **📜 Get Prompt Template History**

    Get change history for a prompt template.
    """
    try:
        service = PromptTemplateService()
        history = await service.get_template_history(template_id)
        return history
    except Exception as e:
        logger.error(f"Failed to get prompt template history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

