"""
Authentication and authorization schemas for the Mivaa PDF Extractor service.

This module contains Pydantic models for:
- User authentication and profile information
- Workspace context and permissions
- JWT token claims and validation
- Role-based access control (RBAC) structures
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

try:
    # Try Pydantic v2 first
    from pydantic import BaseModel, Field, field_validator as validator
except ImportError:
    # Fall back to Pydantic v1
    from pydantic import BaseModel, Field, validator


class UserRole(str, Enum):
    """User roles within a workspace."""
    
    MEMBER = "member"
    ADMIN = "admin"
    OWNER = "owner"


class Permission(str, Enum):
    """Available permissions for workspace operations."""
    
    # PDF processing permissions
    PDF_READ = "pdf:read"
    PDF_WRITE = "pdf:write"
    PDF_DELETE = "pdf:delete"
    
    # Document management permissions
    DOCUMENT_READ = "document:read"
    DOCUMENT_WRITE = "document:write"
    DOCUMENT_DELETE = "document:delete"
    
    # Search permissions
    SEARCH_READ = "search:read"
    
    # Image processing permissions
    IMAGE_READ = "image:read"
    IMAGE_WRITE = "image:write"
    
    # Admin permissions
    ADMIN_ALL = "admin:all"
    
    # Workspace management permissions
    WORKSPACE_READ = "workspace:read"
    WORKSPACE_WRITE = "workspace:write"
    WORKSPACE_DELETE = "workspace:delete"


class User(BaseModel):
    """User model for authentication and profile information."""
    
    id: str = Field(..., description="Unique user identifier")
    email: str = Field(..., description="User email address")
    name: Optional[str] = Field(None, description="User display name")
    avatar_url: Optional[str] = Field(None, description="User avatar image URL")
    created_at: datetime = Field(..., description="User creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    is_active: bool = Field(True, description="Whether user account is active")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        json_schema_extra = {
            "example": {
                "id": "user_123",
                "email": "user@example.com",
                "name": "John Doe",
                "avatar_url": "https://example.com/avatar.jpg",
                "created_at": "2024-07-26T18:00:00Z",
                "updated_at": "2024-07-26T18:00:00Z",
                "is_active": True
            }
        }


class WorkspaceContext(BaseModel):
    """Workspace context for authenticated requests."""
    
    workspace_id: str = Field(..., description="Unique workspace identifier")
    user_id: str = Field(..., description="User ID within the workspace")
    role: UserRole = Field(..., description="User role in the workspace")
    permissions: List[Permission] = Field(default_factory=list, description="User permissions")
    
    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "workspace_id": "workspace_123",
                "user_id": "user_123",
                "role": "admin",
                "permissions": [
                    "pdf:read",
                    "pdf:write",
                    "document:read",
                    "document:write",
                    "search:read"
                ]
            }
        }
    
    def has_permission(self, permission: str) -> bool:
        """
        Check if the user has a specific permission.
        
        Args:
            permission: Permission string to check
            
        Returns:
            True if user has the permission, False otherwise
        """
        # Convert string to Permission enum if needed
        if isinstance(permission, str):
            try:
                permission_enum = Permission(permission)
            except ValueError:
                return False
        else:
            permission_enum = permission
        
        # Check if user has the specific permission
        if permission_enum in self.permissions:
            return True
        
        # Check for admin override
        if Permission.ADMIN_ALL in self.permissions:
            return True
        
        return False
    
    def has_role(self, min_role: UserRole) -> bool:
        """
        Check if the user has at least the minimum required role.
        
        Args:
            min_role: Minimum required role
            
        Returns:
            True if user has sufficient role, False otherwise
        """
        role_hierarchy = {
            UserRole.MEMBER: 1,
            UserRole.ADMIN: 2,
            UserRole.OWNER: 3
        }
        
        user_level = role_hierarchy.get(self.role, 0)
        required_level = role_hierarchy.get(min_role, 999)
        
        return user_level >= required_level
    
    def can_access_workspace(self) -> bool:
        """
        Check if the user can access the workspace.
        
        Returns:
            True if user can access workspace, False otherwise
        """
        return bool(self.workspace_id and self.user_id)


class JWTClaims(BaseModel):
    """JWT token claims structure."""
    
    sub: str = Field(..., description="Subject (user ID)")
    exp: int = Field(..., description="Expiration timestamp")
    iat: int = Field(..., description="Issued at timestamp")
    workspace_id: str = Field(..., description="Workspace ID")
    role: UserRole = Field(..., description="User role")
    permissions: List[Permission] = Field(default_factory=list, description="User permissions")
    email: Optional[str] = Field(None, description="User email")
    
    class Config:
        use_enum_values = True


class AuthResponse(BaseModel):
    """Authentication response with token and user information."""
    
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user: User = Field(..., description="User information")
    workspace_context: WorkspaceContext = Field(..., description="Workspace context")
    
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 3600,
                "user": {
                    "id": "user_123",
                    "email": "user@example.com",
                    "name": "John Doe"
                },
                "workspace_context": {
                    "workspace_id": "workspace_123",
                    "user_id": "user_123",
                    "role": "admin",
                    "permissions": ["pdf:read", "pdf:write"]
                }
            }
        }


class TokenValidationRequest(BaseModel):
    """Request model for token validation."""
    
    token: str = Field(..., description="JWT token to validate")


class TokenValidationResponse(BaseModel):
    """Response model for token validation."""
    
    valid: bool = Field(..., description="Whether token is valid")
    claims: Optional[JWTClaims] = Field(None, description="Token claims if valid")
    error: Optional[str] = Field(None, description="Error message if invalid")


class WorkspaceMember(BaseModel):
    """Workspace member information."""
    
    user_id: str = Field(..., description="User ID")
    workspace_id: str = Field(..., description="Workspace ID")
    role: UserRole = Field(..., description="User role in workspace")
    permissions: List[Permission] = Field(default_factory=list, description="User permissions")
    joined_at: datetime = Field(..., description="When user joined workspace")
    invited_by: Optional[str] = Field(None, description="ID of user who invited this member")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
