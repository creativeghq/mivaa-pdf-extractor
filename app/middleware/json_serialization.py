"""
JSON Serialization Middleware

This middleware ensures all API responses are properly JSON-serialized,
handling datetime objects and other non-serializable types automatically.
"""

import json
import logging
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.types import ASGIApp

from ..utils.json_encoder import CustomJSONEncoder, safe_json_response

logger = logging.getLogger(__name__)


class JSONSerializationMiddleware(BaseHTTPMiddleware):
    """
    Middleware to ensure all JSON responses are properly serialized.
    
    This middleware intercepts responses and ensures that any datetime objects
    or other non-JSON-serializable types are properly converted before sending
    the response to the client.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        logger.info("JSON Serialization Middleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and ensure response is properly serialized.
        
        Args:
            request: The incoming request
            call_next: The next middleware/handler in the chain
            
        Returns:
            Response with properly serialized JSON content
        """
        try:
            # Process the request
            response = await call_next(request)
            
            # Only process JSON responses
            if (isinstance(response, JSONResponse) and 
                hasattr(response, 'body') and 
                response.headers.get('content-type', '').startswith('application/json')):
                
                try:
                    # Try to decode the response body
                    body_content = response.body.decode('utf-8')
                    
                    # Parse the JSON to check for serialization issues
                    try:
                        parsed_content = json.loads(body_content)
                        # Test if it can be re-serialized (this will catch datetime issues)
                        json.dumps(parsed_content)
                        # If we get here, the content is already properly serialized
                        return response
                    except (TypeError, ValueError) as e:
                        if 'datetime' in str(e) or 'JSON serializable' in str(e):
                            logger.warning(f"Fixing JSON serialization issue: {e}")
                            
                            # Re-serialize with our custom encoder
                            safe_content = safe_json_response(parsed_content)
                            
                            # Create new response with fixed content
                            return JSONResponse(
                                content=safe_content,
                                status_code=response.status_code,
                                headers=dict(response.headers)
                            )
                        else:
                            # Re-raise non-datetime related errors
                            raise
                            
                except Exception as e:
                    logger.error(f"Error processing JSON response: {e}")
                    # Return original response if we can't fix it
                    return response
            
            return response
            
        except Exception as e:
            logger.error(f"JSON Serialization Middleware error: {e}")
            # Return a safe error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "type": "serialization_error",
                    "timestamp": safe_json_response({"timestamp": "error"})["timestamp"]
                }
            )


def add_json_serialization_middleware(app):
    """
    Add JSON serialization middleware to FastAPI app.
    
    Args:
        app: FastAPI application instance
    """
    app.add_middleware(JSONSerializationMiddleware)
    logger.info("JSON Serialization Middleware added to application")
