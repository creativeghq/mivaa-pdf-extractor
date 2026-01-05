"""
Central dependency injection module for the Mivaa PDF Extractor service.

This module provides centralized dependency injection functions for:
- Service instances (Supabase, RAG Service, MaterialKai, PDF Processor)
- Authentication and authorization
- Request context and workspace validation

Security Note: Authentication dependencies enforce JWT validation and workspace isolation.
"""

from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import get_settings
from app.middleware.jwt_auth import JWTAuthMiddleware
from app.schemas.auth import WorkspaceContext, User
from app.services.core.supabase_client import get_supabase_client as _get_supabase_client
from app.services.integrations.material_kai_service import get_material_kai_service as _get_material_kai_service
from app.services.search.rag_service import RAGService
from app.services.pdf.pdf_processor import PDFProcessor

# Initialize security scheme
security = HTTPBearer()

# Initialize settings
settings = get_settings()

# ============================================================================
# Service Dependencies (Using app.state for consistency)
# ============================================================================

def get_supabase_client():
    """
    Get Supabase client instance.

    Uses the global singleton instance initialized at startup.

    Returns:
        SupabaseClient instance

    Raises:
        HTTPException: If Supabase client is not initialized
    """
    client = _get_supabase_client()

    if not client.client:
        raise HTTPException(
            status_code=503,
            detail="Database service is not available. Please check configuration."
        )

    return client


async def get_material_kai_service(request: Request):
    """
    Get Material Kai service instance from app state.

    Args:
        request: FastAPI request object to access app.state

    Returns:
        MaterialKaiService instance

    Raises:
        HTTPException: If Material Kai service is not available
    """
    if hasattr(request.app.state, 'material_kai_service') and request.app.state.material_kai_service:
        return request.app.state.material_kai_service

    raise HTTPException(
        status_code=503,
        detail="Material Kai service is not available. Please check service configuration."
    )


async def get_rag_service(request: Request) -> RAGService:
    """
    Get RAG service instance from app state (lazy-loaded).

    The RAG service is initialized on first use through the component manager.
    This ensures efficient resource usage and proper lifecycle management.

    Args:
        request: FastAPI request object to access app.state

    Returns:
        RAGService instance

    Raises:
        HTTPException: If RAG service is not available
    """
    # Check if RAG service is already loaded in app state
    if hasattr(request.app.state, 'rag_service') and request.app.state.rag_service:
        service = request.app.state.rag_service
        if service.available:
            return service

    # Try to lazy load via component manager
    if hasattr(request.app.state, 'component_manager') and request.app.state.component_manager:
        try:
            service = await request.app.state.component_manager.get("rag_service")
            if service and service.available:
                request.app.state.rag_service = service
                return service
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to lazy load RAG service: {e}")

    raise HTTPException(
        status_code=503,
        detail="RAG service is not available. Please check service configuration."
    )


def get_pdf_processor():
    """
    Get PDF processor instance.

    Creates a new instance for each request to avoid state sharing issues.
    PDFProcessor is lightweight and stateless, so this is acceptable.

    Returns:
        PDFProcessor instance
    """
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
        jwt_middleware = JWTAuthMiddleware(None)
        
        # Validate token and extract claims
        claims = await jwt_middleware._validate_token(credentials.credentials)
        
        if not claims:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        return claims
        
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
        user: Authenticated user information (JWT claims)
        
    Returns:
        WorkspaceContext with validated workspace information
        
    Raises:
        HTTPException: If workspace context is invalid or missing
    """
    try:
        # Get JWT middleware instance
        jwt_middleware = JWTAuthMiddleware(None)
        
        # Extract workspace context
        workspace_context = await jwt_middleware._extract_workspace_context(user)
        
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
