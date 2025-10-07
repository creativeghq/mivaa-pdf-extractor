"""
JWT Authentication Middleware for Mivaa PDF Extractor Service

This module implements JWT authentication middleware with workspace-aware security
following the patterns defined in the JWT integration architecture document.

Key Features:
- JWT token validation with required claims checking
- Workspace isolation and context extraction
- Permission-based access control
- Token blacklist support (optional Redis integration)
- Security headers and error handling
"""

import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone, timedelta

import jwt
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..config import get_settings
from ..database import get_supabase_client

# Import auth schemas for type consistency
try:
    from ..schemas.auth import WorkspaceContext, User, UserRole, Permission
except ImportError:
    # Fallback if schemas not available yet - define minimal classes
    from enum import Enum
    try:
        from pydantic import BaseModel
    except ImportError:
        # Minimal BaseModel fallback
        class BaseModel:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
    
    class UserRole(str, Enum):
        MEMBER = "member"
        ADMIN = "admin"
        OWNER = "owner"
    
    class Permission(str, Enum):
        ADMIN_ALL = "admin:all"
    
    class WorkspaceContext(BaseModel):
        workspace_id: str
        user_id: str
        role: UserRole
        permissions: List[str] = []
        
        def has_permission(self, permission: str) -> bool:
            return permission in self.permissions or "admin:all" in self.permissions
    
    class User(BaseModel):
        id: str
        email: str
        name: Optional[str] = None


logger = logging.getLogger(__name__)
security = HTTPBearer()


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """
    JWT Authentication Middleware for workspace-aware authentication.
    
    Validates JWT tokens, extracts workspace context, and enforces
    permission-based access control for all protected endpoints.
    """
    
    def __init__(self, app, exclude_paths: Optional[List[str]] = None):
        """
        Initialize JWT Authentication Middleware.
        
        Args:
            app: FastAPI application instance
            exclude_paths: List of paths to exclude from authentication
        """
        super().__init__(app)
        self.settings = get_settings()
        self.exclude_paths = exclude_paths or [
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc"
        ]
        
        # Initialize Supabase client for token validation (lazy initialization)
        self.supabase_wrapper = None
        self.supabase = None
        
        logger.info("JWT Authentication Middleware initialized")

    def _ensure_supabase_client(self):
        """Ensure Supabase client is initialized (lazy initialization)."""
        if self.supabase is None:
            try:
                self.supabase_wrapper = get_supabase_client()
                self.supabase = self.supabase_wrapper.client
            except RuntimeError:
                # Supabase client not yet initialized, will retry later
                pass

    async def dispatch(self, request: Request, call_next):
        """
        Process incoming requests with JWT authentication.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain
            
        Returns:
            HTTP response with security headers
        """
        # Skip authentication for excluded paths
        if self._is_excluded_path(request.url.path):
            response = await call_next(request)
            return self._add_security_headers(response)
        
        try:
            # Extract and validate JWT token
            token = await self._extract_token(request)
            if not token:
                return self._unauthorized_response("Missing authentication token")
            
            # Validate token and extract claims
            claims = await self._validate_token(token)
            if not claims:
                return self._unauthorized_response("Invalid authentication token")
            
            # Check if token is blacklisted (optional Redis check)
            if await self._is_token_blacklisted(token):
                return self._unauthorized_response("Token has been revoked")
            
            # Extract workspace context
            workspace_context = await self._extract_workspace_context(claims)
            if not workspace_context:
                return self._forbidden_response("Invalid workspace context")
            
            # Add authentication context to request state
            request.state.auth_user_id = claims.get("sub")
            request.state.workspace_id = workspace_context.workspace_id
            request.state.workspace_role = workspace_context.role
            request.state.permissions = workspace_context.permissions
            # Store serializable claims (convert any datetime objects to strings)
            serializable_claims = {}
            for key, value in claims.items():
                if isinstance(value, datetime):
                    serializable_claims[key] = value.isoformat()
                else:
                    serializable_claims[key] = value
            request.state.jwt_claims = serializable_claims
            
            # Log authentication success
            logger.info(
                f"Authentication successful for user {claims.get('sub')} "
                f"in workspace {workspace_context.workspace_id}"
            )
            
            # Continue to next middleware/handler
            response = await call_next(request)
            return self._add_security_headers(response)
            
        except HTTPException as e:
            logger.warning(f"Authentication failed: {e.detail}")
            return JSONResponse(
                status_code=e.status_code,
                content={"error": e.detail, "type": "authentication_error"}
            )
        except Exception as e:
            import traceback
            logger.error(f"Authentication middleware error: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Internal authentication error", "type": "server_error"}
            )
    
    def _is_excluded_path(self, path: str) -> bool:
        """Check if path should be excluded from authentication."""
        return any(path.startswith(excluded) for excluded in self.exclude_paths)
    
    async def _extract_token(self, request: Request) -> Optional[str]:
        """
        Extract JWT token from request headers.
        
        Args:
            request: HTTP request object
            
        Returns:
            JWT token string or None if not found
        """
        # Check Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header.split(" ")[1]
        
        # Check for token in cookies (fallback)
        token = request.cookies.get("access_token")
        if token:
            return token
        
        return None
    
    async def _validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate JWT token or simple API key and extract claims.

        Args:
            token: JWT token string or simple API key

        Returns:
            Token claims dictionary or None if invalid
        """
        try:
            # First, check if this is a simple API key (20 characters, starts with mk_)
            if self._is_simple_api_key(token):
                return await self._validate_simple_api_key(token)

            # Otherwise, try to decode as JWT token
            claims = jwt.decode(
                token,
                self.settings.jwt_secret_key,
                algorithms=[self.settings.jwt_algorithm],
                options={"verify_exp": True, "verify_iat": True}
            )

            # Validate required claims
            required_claims = ["sub", "exp", "iat"]
            for claim in required_claims:
                if claim not in claims:
                    logger.warning(f"Missing required claim: {claim}")
                    return None

            # Check token expiration
            exp_timestamp = claims.get("exp")
            if exp_timestamp and datetime.fromtimestamp(exp_timestamp, tz=timezone.utc) < datetime.now(timezone.utc):
                logger.warning("Token has expired")
                return None

            return claims
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Token validation error: {str(e)}")
            return None

    def _is_simple_api_key(self, token: str) -> bool:
        """
        Check if the token is a simple API key format.

        Args:
            token: Token string to check

        Returns:
            True if token appears to be a simple API key
        """
        # Simple API key: starts with mk_ and is 18-20 characters
        return (
            token.startswith("mk_") and
            len(token) >= 18 and
            len(token) <= 20 and
            all(c.isalnum() or c == '_' for c in token)
        )

    async def _validate_simple_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Validate simple API key against configured keys.

        Args:
            api_key: Simple API key string

        Returns:
            Claims dictionary if valid, None otherwise
        """
        try:
            # Check against configured Material Kai API key
            if api_key == self.settings.material_kai_api_key:
                logger.info(f"Valid Material Kai API key authenticated")
                return {
                    "sub": "material-kai-platform",
                    "api_key": api_key,
                    "service": "mivaa",
                    "permissions": ["admin:all", "pdf:read", "pdf:write", "document:read", "document:write", "search:read", "image:read", "image:write"],
                    "user_id": "material-kai-platform",
                    "organization": "material-kai-vision-platform",
                    "workspace_id": self.settings.material_kai_workspace_id,
                    "role": "admin",
                    "iat": int(datetime.now(timezone.utc).timestamp()),
                    "exp": int((datetime.now(timezone.utc) + timedelta(hours=24)).timestamp())
                }

            logger.warning(f"Invalid API key: {api_key}")
            return None

        except Exception as e:
            logger.error(f"API key validation error: {str(e)}")
            return None
    
    async def _is_token_blacklisted(self, token: str) -> bool:
        """
        Check if token is blacklisted (optional Redis integration).
        
        Args:
            token: JWT token string
            
        Returns:
            True if token is blacklisted, False otherwise
        """
        # TODO: Implement Redis blacklist checking if Redis is configured
        # For now, return False (no blacklist check)
        return False
    
    async def _extract_workspace_context(self, claims: Dict[str, Any]) -> Optional[WorkspaceContext]:
        """
        Extract workspace context from JWT claims.
        
        Args:
            claims: JWT token claims
            
        Returns:
            WorkspaceContext object or None if invalid
        """
        try:
            # Extract workspace information from claims
            workspace_id = claims.get("workspace_id")
            if not workspace_id:
                logger.warning("Missing workspace_id in token claims")
                return None
            
            # Extract user role and permissions
            role_str = claims.get("role", "member")
            try:
                role = UserRole(role_str)
            except ValueError:
                logger.warning(f"Invalid role in token claims: {role_str}")
                role = UserRole.MEMBER
            
            permissions = claims.get("permissions", [])
            
            # Validate workspace access via Supabase
            user_id = claims.get("sub")
            if not await self._validate_workspace_access(user_id, workspace_id):
                logger.warning(f"User {user_id} does not have access to workspace {workspace_id}")
                return None
            
            return WorkspaceContext(
                workspace_id=workspace_id,
                user_id=user_id,
                role=role,
                permissions=permissions
            )
            
        except Exception as e:
            logger.error(f"Workspace context extraction error: {str(e)}")
            return None
    
    async def _validate_workspace_access(self, user_id: str, workspace_id: str) -> bool:
        """
        Validate user access to workspace via Supabase.
        
        Args:
            user_id: User ID from JWT claims
            workspace_id: Workspace ID from JWT claims
            
        Returns:
            True if user has access, False otherwise
        """
        try:
            # Special case: Material Kai platform service always has access
            if user_id == "material-kai-platform":
                logger.info(f"Granting workspace access to Material Kai platform service for workspace {workspace_id}")
                return True

            # Ensure Supabase client is initialized
            self._ensure_supabase_client()

            if self.supabase is None:
                logger.warning("Supabase client not available for workspace validation")
                return False

            # Query workspace membership from Supabase
            response = self.supabase.table("workspace_members").select("*").eq(
                "user_id", user_id
            ).eq("workspace_id", workspace_id).execute()

            # Check if user is a member of the workspace
            return len(response.data) > 0
            
        except Exception as e:
            logger.error(f"Workspace access validation error: {str(e)}")
            return False
    
    def _unauthorized_response(self, message: str) -> JSONResponse:
        """Return standardized unauthorized response."""
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": message,
                "type": "authentication_error",
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    def _forbidden_response(self, message: str) -> JSONResponse:
        """Return standardized forbidden response."""
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "error": message,
                "type": "authorization_error",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    def _add_security_headers(self, response) -> Any:
        """
        Add security headers to response.
        
        Args:
            response: HTTP response object
            
        Returns:
            Response with added security headers
        """
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response


# Permission decorator functions for endpoint protection
def require_permission(permission: str):
    """
    Decorator to require specific permission for endpoint access.
    
    Args:
        permission: Required permission string (e.g., 'pdf:process', 'workspace:read')
        
    Returns:
        Decorator function
    """
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            # Check if user has required permission
            user_permissions = getattr(request.state, "permissions", [])
            if permission not in user_permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing required permission: {permission}"
                )
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator


def require_workspace_role(min_role: str):
    """
    Decorator to require minimum workspace role for endpoint access.
    
    Args:
        min_role: Minimum required role ('member', 'admin', 'owner')
        
    Returns:
        Decorator function
    """
    role_hierarchy = {"member": 1, "admin": 2, "owner": 3}
    
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            user_role = getattr(request.state, "workspace_role", "member")
            user_level = role_hierarchy.get(user_role, 0)
            required_level = role_hierarchy.get(min_role, 999)
            
            if user_level < required_level:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient role. Required: {min_role}, Current: {user_role}"
                )
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator


def get_current_user(request: Request) -> Dict[str, Any]:
    """
    Extract current user context from authenticated request.
    
    Args:
        request: Authenticated HTTP request
        
    Returns:
        User context dictionary
    """
    return {
        "user_id": getattr(request.state, "auth_user_id", None),
        "workspace_id": getattr(request.state, "workspace_id", None),
        "workspace_role": getattr(request.state, "workspace_role", None),
        "permissions": getattr(request.state, "permissions", []),
        "jwt_claims": getattr(request.state, "jwt_claims", {})
    }


def get_current_workspace_context(request: Request) -> Optional[WorkspaceContext]:
    """
    Extract current workspace context from authenticated request.
    
    Args:
        request: Authenticated HTTP request
        
    Returns:
        WorkspaceContext object if available, None otherwise
    """
    workspace_id = getattr(request.state, "workspace_id", None)
    user_id = getattr(request.state, "auth_user_id", None)
    role = getattr(request.state, "workspace_role", None)
    permissions = getattr(request.state, "permissions", [])
    
    if not (workspace_id and user_id and role):
        return None
    
    try:
        return WorkspaceContext(
            workspace_id=workspace_id,
            user_id=user_id,
            role=UserRole(role) if isinstance(role, str) else role,
            permissions=permissions
        )
    except Exception:
        return None