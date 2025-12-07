"""
Admin Prompts API

REST endpoints for managing extraction prompts and configuration.
"""

import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends, Body
from pydantic import BaseModel, Field

from app.services.admin_prompt_service import AdminPromptService
from app.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/extraction-prompts", tags=["Admin - Prompts"])


# Request/Response Models
class PromptResponse(BaseModel):
    """Prompt response model"""
    id: str
    workspace_id: str
    stage: Optional[str] = None
    category: str
    name: Optional[str] = None
    prompt_type: Optional[str] = None
    prompt_template: str
    system_prompt: Optional[str] = None
    is_custom: bool = False
    version: int = 1
    created_by: Optional[str] = None
    created_at: str
    updated_at: str
    used_in: Optional[List[str]] = None


class UpdatePromptRequest(BaseModel):
    """Update prompt request"""
    prompt_template: str = Field(..., description="New prompt template")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt")
    change_reason: str = Field(..., description="Reason for the change")


class PromptHistoryResponse(BaseModel):
    """Prompt history response"""
    id: str
    prompt_id: str
    old_prompt: str
    new_prompt: str
    changed_by: Optional[str]
    change_reason: Optional[str]
    changed_at: str


class TestPromptRequest(BaseModel):
    """Test prompt request"""
    stage: str
    category: str
    prompt_template: str
    test_content: str


class TestPromptResponse(BaseModel):
    """Test prompt response"""
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None


class ExtractionConfigResponse(BaseModel):
    """Extraction config response"""
    id: Optional[str] = None
    workspace_id: str
    enabled_categories: List[str]
    default_categories: List[str]
    discovery_model: str
    chunk_size: int
    chunk_overlap: int
    enable_prompt_enhancement: bool
    quality_threshold: float


class UpdateExtractionConfigRequest(BaseModel):
    """Update extraction config request"""
    enabled_categories: Optional[List[str]] = None
    default_categories: Optional[List[str]] = None
    discovery_model: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    enable_prompt_enhancement: Optional[bool] = None
    quality_threshold: Optional[float] = None


# Endpoints
@router.get("", response_model=List[PromptResponse])
async def list_prompts(
    workspace_id: str,
    stage: Optional[str] = None,
    category: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    **📋 List Extraction Prompts - Admin Configuration**

    List all AI extraction prompts configured for a workspace with optional filtering.

    ## 🎯 Use Cases

    - View all configured extraction prompts
    - Filter prompts by processing stage
    - Filter prompts by extraction category
    - Audit prompt configurations

    ## 📝 Query Parameters

    - **workspace_id** (required): Workspace UUID
    - **stage** (optional): Filter by stage (discovery, extraction, validation)
    - **category** (optional): Filter by category (products, certificates, logos, specifications)

    ## ✅ Response Example

    ```json
    [
      {
        "id": "prompt-uuid-1",
        "workspace_id": "workspace-uuid",
        "stage": "discovery",
        "category": "products",
        "prompt_template": "Identify all products in this PDF...",
        "system_prompt": "You are an expert at identifying products...",
        "is_custom": true,
        "version": 2,
        "created_by": "user-uuid",
        "created_at": "2025-11-08T10:00:00Z",
        "updated_at": "2025-11-08T12:00:00Z"
      }
    ]
    ```

    ## ⚠️ Error Codes

    - **400 Bad Request**: Invalid workspace_id
    - **401 Unauthorized**: Authentication required
    - **403 Forbidden**: Insufficient permissions
    - **500 Internal Server Error**: Database error
    """
    try:
        service = AdminPromptService()
        prompts = await service.get_prompts(workspace_id, stage, category)
        return prompts
    except Exception as e:
        logger.error(f"Error listing prompts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{stage}/{category}", response_model=PromptResponse)
async def get_prompt(
    stage: str,
    category: str,
    workspace_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get specific prompt by stage and category"""
    try:
        service = AdminPromptService()
        prompt = await service.get_prompt(workspace_id, stage, category)
        
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        
        return prompt
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{stage}/{category}", response_model=PromptResponse)
async def update_prompt(
    stage: str,
    category: str,
    workspace_id: str,
    request: UpdatePromptRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Update prompt with audit trail
    
    Path Parameters:
    - stage: Extraction stage (discovery, chunking, image_analysis, entity_creation)
    - category: Content category (products, certificates, logos, specifications)
    
    Query Parameters:
    - workspace_id: Workspace ID
    
    Body:
    - prompt_template: New prompt template
    - system_prompt: Optional system prompt
    - change_reason: Reason for the change
    """
    try:
        service = AdminPromptService()
        
        # Get user ID from current_user
        user_id = current_user.get('id', 'unknown')
        
        updated_prompt = await service.update_prompt(
            workspace_id=workspace_id,
            stage=stage,
            category=category,
            prompt_template=request.prompt_template,
            system_prompt=request.system_prompt,
            changed_by=user_id,
            change_reason=request.change_reason
        )
        
        return updated_prompt
    except Exception as e:
        logger.error(f"Error updating prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{prompt_id}", response_model=List[PromptHistoryResponse])
async def get_prompt_history(
    prompt_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get change history for a prompt"""
    try:
        service = AdminPromptService()
        history = await service.get_prompt_history(prompt_id)
        return history
    except Exception as e:
        logger.error(f"Error getting prompt history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test", response_model=TestPromptResponse)
async def test_prompt(
    request: TestPromptRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Test a prompt before saving
    
    Body:
    - stage: Extraction stage
    - category: Content category
    - prompt_template: Prompt to test
    - test_content: Sample content to test with
    """
    try:
        service = AdminPromptService()
        result = await service.test_prompt(
            stage=request.stage,
            category=request.category,
            prompt_template=request.prompt_template,
            test_content=request.test_content
        )
        return result
    except Exception as e:
        logger.error(f"Error testing prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Extraction Config Endpoints
config_router = APIRouter(prefix="/admin/extraction-config", tags=["Admin - Config"])


@config_router.get("", response_model=ExtractionConfigResponse)
async def get_extraction_config(
    workspace_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get extraction configuration for workspace"""
    try:
        service = AdminPromptService()
        config = await service.get_extraction_config(workspace_id)
        return config
    except Exception as e:
        logger.error(f"Error getting extraction config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@config_router.put("", response_model=ExtractionConfigResponse)
async def update_extraction_config(
    workspace_id: str,
    request: UpdateExtractionConfigRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Update extraction configuration
    
    Query Parameters:
    - workspace_id: Workspace ID
    
    Body: Configuration fields to update
    """
    try:
        service = AdminPromptService()
        
        # Build config dict from request (only include non-None values)
        config = {}
        if request.enabled_categories is not None:
            config['enabled_categories'] = request.enabled_categories
        if request.default_categories is not None:
            config['default_categories'] = request.default_categories
        if request.discovery_model is not None:
            config['discovery_model'] = request.discovery_model
        if request.chunk_size is not None:
            config['chunk_size'] = request.chunk_size
        if request.chunk_overlap is not None:
            config['chunk_overlap'] = request.chunk_overlap
        if request.enable_prompt_enhancement is not None:
            config['enable_prompt_enhancement'] = request.enable_prompt_enhancement
        if request.quality_threshold is not None:
            config['quality_threshold'] = request.quality_threshold
        
        updated_config = await service.update_extraction_config(workspace_id, config)
        return updated_config
    except Exception as e:
        logger.error(f"Error updating extraction config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

