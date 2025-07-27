"""
Central dependency injection module for the Mivaa PDF Extractor service.

This module provides centralized dependency injection functions for:
- Service instances (Supabase, LlamaIndex, MaterialKai, PDF Processor)
- Authentication and authorization
- Request context and workspace validation

Security Note: Authentication dependencies enforce JWT validation and workspace isolation.
"""

from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import get_settings
from app.middleware.jwt_auth import JWTAuthMiddleware, WorkspaceContext
from app.services.supabase_client import get_supabase_client as _get_supabase_client
from app.services.material_kai_service import get_material_kai_service as _get_material_kai_service
from app.services.llamaindex_service import LlamaIndexService
from app.services.pdf_processor import PDFProcessor

# Initialize security scheme
security = HTTPBearer()

# Initialize settings
settings = get_settings()

# ============================================================================
# Service Dependencies
# ============================================================================

def get_supabase_client():
    """Dependency to get Supabase client instance."""
    return _get_supabase_client()


async def get_material_kai_service():
    """Dependency to get Material Kai service instance."""
    return await _get_material_kai_service()


async def get_llamaindex_service() -> LlamaIndexService:
    """Dependency to get LlamaIndex service instance."""
    # Import here to avoid circular imports
    from app.services.llamaindex_service import LlamaIndexService
    return LlamaIndexService()


def get_pdf_processor():
    """Dependency to get PDF processor instance."""
    return PDFProcessor()


# ============================================================================
# Authentication Dependencies
# ============================================================================

async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """
    Extract and validate the current authenticated user from JWT token.
    
    Args:
        request: FastAPI request object
        credentials: HTTP Bearer token credentials
        
    Returns:
        Dict containing user information and claims
        
    Raises:
        HTTPException: If token is invalid or user is not authenticated
    """
    try:
        # Get JWT middleware instance
        jwt_middleware = JWTAuthMiddleware()
        
        # Validate token and extract user info
        user_info = await jwt_middleware.validate_token(credentials.credentials)
        
        if not user_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        return user_info
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_workspace_context(
    request: Request,
    user: Dict[str, Any] = Depends(get_current_user)
) -> WorkspaceContext:
    """
    Extract and validate workspace context from authenticated user.
    
    Args:
        request: FastAPI request object
        user: Authenticated user information
        
    Returns:
        WorkspaceContext with validated workspace information
        
    Raises:
        HTTPException: If workspace context is invalid or missing
    """
    try:
        # Get JWT middleware instance
        jwt_middleware = JWTAuthMiddleware()
        
        # Extract workspace context
        workspace_context = await jwt_middleware.extract_workspace_context(user)
        
        if not workspace_context:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid or missing workspace context"
            )
            
        return workspace_context
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Workspace validation failed: {str(e)}"
        )


# ============================================================================
# Permission Dependencies
# ============================================================================

def require_permission(permission: str):
    """
    Create a dependency that requires a specific permission.
    
    Args:
        permission: Required permission string (e.g., 'pdf:read', 'pdf:write')
        
    Returns:
        Dependency function that validates the permission
    """
    async def permission_dependency(
        workspace_context: WorkspaceContext = Depends(get_workspace_context)
    ) -> WorkspaceContext:
        """Validate that the user has the required permission."""
        if not workspace_context.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {permission}"
            )
        return workspace_context
    
    return permission_dependency


# ============================================================================
# Convenience Permission Dependencies
# ============================================================================

# PDF processing permissions
require_pdf_read = require_permission("pdf:read")
require_pdf_write = require_permission("pdf:write")
require_pdf_delete = require_permission("pdf:delete")

# Document management permissions
require_document_read = require_permission("document:read")
require_document_write = require_permission("document:write")
require_document_delete = require_permission("document:delete")

# Search permissions
require_search_read = require_permission("search:read")

# Image processing permissions
require_image_read = require_permission("image:read")
require_image_write = require_permission("image:write")

# Admin permissions
require_admin = require_permission("admin:all")


# ============================================================================
# Optional Authentication Dependencies
# ============================================================================

async def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[Dict[str, Any]]:
    """
    Extract user information if authentication is provided, but don't require it.
    
    Args:
        request: FastAPI request object
        credentials: Optional HTTP Bearer token credentials
        
    Returns:
        User information if authenticated, None otherwise
    """
    if not credentials:
        return None
        
    try:
        return await get_current_user(request, credentials)
    except HTTPException:
        return None


async def get_optional_workspace_context(
    request: Request,
    user: Optional[Dict[str, Any]] = Depends(get_optional_user)
) -> Optional[WorkspaceContext]:
    """
    Extract workspace context if user is authenticated, but don't require it.
    
    Args:
        request: FastAPI request object
        user: Optional authenticated user information
        
    Returns:
        WorkspaceContext if authenticated, None otherwise
    """
    if not user:
        return None
        
    try:
        return await get_workspace_context(request, user)
    except HTTPException:
        return None