"""
Material Kai Vision Platform Integration Service

This service provides integration with the Material Kai Vision Platform,
enabling seamless communication, data exchange, and workflow coordination
between the MIVAA PDF Extractor and the broader platform ecosystem.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, BinaryIO
from pathlib import Path
import aiohttp
import aiofiles
from pydantic import BaseModel, Field
import base64
import mimetypes
from functools import wraps
import time
import random

from ..config import get_settings
from ..utils.exceptions import MaterialKaiIntegrationError

logger = logging.getLogger(__name__)


# Retry decorator for robust error handling
def retry_on_failure(max_retries: int = 3, backoff_factor: float = 1.0,
                    retry_exceptions: tuple = (aiohttp.ClientError, asyncio.TimeoutError)):
    """
    Decorator to retry async functions on specific exceptions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Base delay multiplier for exponential backoff
        retry_exceptions: Tuple of exceptions that should trigger a retry
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise MaterialKaiIntegrationError(f"Operation failed after {max_retries} retries: {e}")
                    
                    # Calculate delay with exponential backoff and jitter
                    delay = backoff_factor * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.2f}s...")
                    await asyncio.sleep(delay)
                except Exception as e:
                    # Don't retry on non-retryable exceptions
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise MaterialKaiIntegrationError(f"Non-retryable error: {e}")
            
            # This should never be reached, but just in case
            raise MaterialKaiIntegrationError(f"Unexpected error in retry logic for {func.__name__}")
        
        return wrapper
    return decorator


def handle_http_errors(func):
    """
    Decorator to handle and categorize HTTP errors from Material Kai Vision Platform.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except aiohttp.ClientResponseError as e:
            if e.status == 400:
                raise MaterialKaiIntegrationError(f"Bad request to Material Kai API: {e.message}")
            elif e.status == 401:
                raise MaterialKaiIntegrationError("Authentication failed with Material Kai API")
            elif e.status == 403:
                raise MaterialKaiIntegrationError("Access forbidden to Material Kai API resource")
            elif e.status == 404:
                raise MaterialKaiIntegrationError("Material Kai API endpoint not found")
            elif e.status == 429:
                raise MaterialKaiIntegrationError("Rate limit exceeded for Material Kai API")
            elif e.status >= 500:
                raise MaterialKaiIntegrationError(f"Material Kai API server error: {e.message}")
            else:
                raise MaterialKaiIntegrationError(f"HTTP error {e.status}: {e.message}")
        except aiohttp.ClientConnectionError as e:
            raise MaterialKaiIntegrationError(f"Connection error to Material Kai API: {e}")
        except aiohttp.ClientTimeout as e:
            raise MaterialKaiIntegrationError(f"Timeout connecting to Material Kai API: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise MaterialKaiIntegrationError(f"Unexpected error: {e}")
    
    return wrapper


class MaterialKaiDocument(BaseModel):
    """Material Kai document model for platform integration."""
    
    id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    source_type: str = Field(default="pdf", description="Source document type")
    processing_status: str = Field(default="pending", description="Processing status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    tags: List[str] = Field(default_factory=list, description="Document tags")
    platform_id: Optional[str] = Field(None, description="Platform-specific identifier")


class MaterialKaiWorkflow(BaseModel):
    """Material Kai workflow model for process coordination."""
    
    id: str = Field(..., description="Workflow identifier")
    name: str = Field(..., description="Workflow name")
    status: str = Field(default="active", description="Workflow status")
    steps: List[Dict[str, Any]] = Field(default_factory=list, description="Workflow steps")
    document_ids: List[str] = Field(default_factory=list, description="Associated document IDs")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Workflow metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class MaterialKaiIntegrationError(Exception):
    """Custom exception for Material Kai integration errors."""
    pass


class MaterialKaiService:
    """
    Material Kai Vision Platform Integration Service.
    
    Provides comprehensive integration capabilities including:
    - Document synchronization
    - Workflow coordination
    - Real-time communication
    - Data exchange protocols
    - Platform authentication
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Material Kai service with configuration."""
        self.settings = get_settings()
        self.config = config or self._get_default_config()
        
        # Platform connection settings
        self.platform_url = self.config.get("platform_url", "")
        self.api_key = self.config.get("api_key", "")
        self.workspace_id = self.config.get("workspace_id", "")
        self.service_name = self.config.get("service_name", "mivaa-pdf-extractor")

        # Integration settings
        self.sync_enabled = self.config.get("sync_enabled", False)
        self.real_time_enabled = self.config.get("real_time_enabled", False)
        self.batch_size = self.config.get("batch_size", 10)
        self.retry_attempts = self.config.get("retry_attempts", 3)
        self.timeout = self.config.get("timeout", 30)

        # Internal state
        self._session: Optional[aiohttp.ClientSession] = None
        self._websocket: Optional[aiohttp.ClientWebSocketResponse] = None
        self._is_connected = False
        self._last_sync = None

        # Check if platform integration is enabled
        self.platform_enabled = bool(self.platform_url and self.api_key)

        if self.platform_enabled:
            logger.info(f"Material Kai service initialized for workspace: {self.workspace_id}")
        else:
            logger.info("Material Kai platform integration disabled (no platform URL or API key configured)")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration from settings."""
        return {
            "platform_url": getattr(self.settings, "material_kai_platform_url", ""),  # Disabled by default
            "api_key": getattr(self.settings, "material_kai_api_key", ""),
            "workspace_id": getattr(self.settings, "material_kai_workspace_id", ""),
            "service_name": getattr(self.settings, "material_kai_service_name", "mivaa-pdf-extractor"),
            "sync_enabled": getattr(self.settings, "material_kai_sync_enabled", False),  # Disabled by default
            "real_time_enabled": getattr(self.settings, "material_kai_real_time_enabled", False),  # Disabled by default
            "batch_size": getattr(self.settings, "material_kai_batch_size", 10),
            "retry_attempts": getattr(self.settings, "material_kai_retry_attempts", 3),
            "timeout": getattr(self.settings, "material_kai_timeout", 30),
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self) -> bool:
        """
        Establish connection to Material Kai Vision Platform.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Skip connection if platform is disabled
            if not self.platform_enabled:
                logger.info("Material Kai platform integration is disabled, skipping connection")
                return True

            if not self.api_key or not self.workspace_id:
                logger.warning("Material Kai API key or workspace ID not configured")
                return False
            
            # Create HTTP session
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": f"{self.service_name}/1.0.0",
                    "X-Workspace-ID": self.workspace_id
                }
            )
            
            # Test connection
            health_check = await self.health_check()
            if health_check["status"] == "healthy":
                self._is_connected = True
                logger.info("Successfully connected to Material Kai Vision Platform")
                
                # Initialize real-time connection if enabled
                if self.real_time_enabled:
                    await self._initialize_websocket()
                
                return True
            else:
                logger.error(f"Material Kai platform health check failed: {health_check}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to Material Kai platform: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Material Kai Vision Platform."""
        try:
            if self._websocket and not self._websocket.closed:
                await self._websocket.close()
                self._websocket = None
            
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None
            
            self._is_connected = False
            logger.info("Disconnected from Material Kai Vision Platform")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check with Material Kai platform.

        Returns:
            Dict containing health status and details
        """
        try:
            # Return healthy status if platform is disabled
            if not self.platform_enabled:
                return {
                    "status": "healthy",
                    "message": "Platform integration disabled",
                    "timestamp": datetime.utcnow().isoformat()
                }

            if not self._session:
                return {
                    "status": "unhealthy",
                    "error": "No active session",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            async with self._session.get(f"{self.platform_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "healthy",
                        "platform_status": data.get("status", "unknown"),
                        "version": data.get("version", "unknown"),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "error": f"HTTP {response.status}",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def register_service(self) -> Dict[str, Any]:
        """
        Register this service with the Material Kai platform.
        
        Returns:
            Dict containing registration result
        """
        try:
            if not self._session:
                raise MaterialKaiIntegrationError("Not connected to platform")
            
            registration_data = {
                "service_name": self.service_name,
                "service_type": "pdf_processor",
                "version": "1.0.0",
                "capabilities": [
                    "pdf_extraction",
                    "text_processing",
                    "metadata_extraction",
                    "rag_integration",
                    "batch_processing"
                ],
                "endpoints": {
                    "health": "/health",
                    "process": "/api/v1/pdf/process",
                    "extract": "/api/v1/pdf/extract",
                    "query": "/api/v1/rag/query"
                },
                "workspace_id": self.workspace_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            async with self._session.post(
                f"{self.platform_url}/services/register",
                json=registration_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Service registered successfully: {result.get('service_id')}")
                    return result
                else:
                    error_text = await response.text()
                    raise MaterialKaiIntegrationError(f"Registration failed: {error_text}")
                    
        except Exception as e:
            logger.error(f"Service registration failed: {e}")
            raise MaterialKaiIntegrationError(f"Registration error: {e}")
    
    async def sync_document(self, document: MaterialKaiDocument) -> Dict[str, Any]:
        """
        Synchronize a document with the Material Kai platform.
        
        Args:
            document: Document to synchronize
            
        Returns:
            Dict containing sync result
        """
        try:
            if not self._session:
                raise MaterialKaiIntegrationError("Not connected to platform")
            
            document_data = document.dict()
            document_data["service_source"] = self.service_name
            document_data["workspace_id"] = self.workspace_id
            
            async with self._session.post(
                f"{self.platform_url}/documents/sync",
                json=document_data
            ) as response:
                if response.status in [200, 201]:
                    result = await response.json()
                    logger.info(f"Document synced successfully: {document.id}")
                    return result
                else:
                    error_text = await response.text()
                    raise MaterialKaiIntegrationError(f"Document sync failed: {error_text}")
                    
        except Exception as e:
            logger.error(f"Document sync failed for {document.id}: {e}")
            raise MaterialKaiIntegrationError(f"Sync error: {e}")
    
    async def batch_sync_documents(self, documents: List[MaterialKaiDocument]) -> Dict[str, Any]:
        """
        Synchronize multiple documents in batch.
        
        Args:
            documents: List of documents to synchronize
            
        Returns:
            Dict containing batch sync results
        """
        try:
            if not self._session:
                raise MaterialKaiIntegrationError("Not connected to platform")
            
            # Process in batches
            results = []
            failed = []
            
            for i in range(0, len(documents), self.batch_size):
                batch = documents[i:i + self.batch_size]
                batch_data = {
                    "documents": [doc.dict() for doc in batch],
                    "service_source": self.service_name,
                    "workspace_id": self.workspace_id,
                    "batch_id": f"batch_{i//self.batch_size + 1}"
                }
                
                try:
                    async with self._session.post(
                        f"{self.platform_url}/documents/batch-sync",
                        json=batch_data
                    ) as response:
                        if response.status in [200, 201]:
                            batch_result = await response.json()
                            results.extend(batch_result.get("results", []))
                        else:
                            error_text = await response.text()
                            failed.extend([doc.id for doc in batch])
                            logger.error(f"Batch sync failed: {error_text}")
                            
                except Exception as e:
                    failed.extend([doc.id for doc in batch])
                    logger.error(f"Batch sync error: {e}")
            
            return {
                "total_documents": len(documents),
                "successful": len(results),
                "failed": len(failed),
                "failed_ids": failed,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Batch sync failed: {e}")
            raise MaterialKaiIntegrationError(f"Batch sync error: {e}")
    
    async def create_workflow(self, workflow: MaterialKaiWorkflow) -> Dict[str, Any]:
        """
        Create a workflow in the Material Kai platform.
        
        Args:
            workflow: Workflow to create
            
        Returns:
            Dict containing creation result
        """
        try:
            if not self._session:
                raise MaterialKaiIntegrationError("Not connected to platform")
            
            workflow_data = workflow.dict()
            workflow_data["service_source"] = self.service_name
            workflow_data["workspace_id"] = self.workspace_id
            
            async with self._session.post(
                f"{self.platform_url}/workflows/create",
                json=workflow_data
            ) as response:
                if response.status in [200, 201]:
                    result = await response.json()
                    logger.info(f"Workflow created successfully: {workflow.id}")
                    return result
                else:
                    error_text = await response.text()
                    raise MaterialKaiIntegrationError(f"Workflow creation failed: {error_text}")
                    
        except Exception as e:
            logger.error(f"Workflow creation failed for {workflow.id}: {e}")
            raise MaterialKaiIntegrationError(f"Workflow creation error: {e}")
    
    async def update_workflow_status(self, workflow_id: str, status: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update workflow status in the Material Kai platform.
        
        Args:
            workflow_id: Workflow identifier
            status: New status
            metadata: Optional metadata update
            
        Returns:
            Dict containing update result
        """
        try:
            if not self._session:
                raise MaterialKaiIntegrationError("Not connected to platform")
            
            update_data = {
                "workflow_id": workflow_id,
                "status": status,
                "updated_at": datetime.utcnow().isoformat(),
                "service_source": self.service_name
            }
            
            if metadata:
                update_data["metadata"] = metadata
            
            async with self._session.patch(
                f"{self.platform_url}/workflows/{workflow_id}/status",
                json=update_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Workflow status updated: {workflow_id} -> {status}")
                    return result
                else:
                    error_text = await response.text()
                    raise MaterialKaiIntegrationError(f"Workflow update failed: {error_text}")
                    
        except Exception as e:
            logger.error(f"Workflow status update failed for {workflow_id}: {e}")
            raise MaterialKaiIntegrationError(f"Workflow update error: {e}")
    
    async def query_documents(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query documents from the Material Kai platform.
        
        Args:
            query: Search query
            filters: Optional filters
            
        Returns:
            Dict containing query results
        """
        try:
            if not self._session:
                raise MaterialKaiIntegrationError("Not connected to platform")
            
            query_data = {
                "query": query,
                "workspace_id": self.workspace_id,
                "service_source": self.service_name
            }
            
            if filters:
                query_data["filters"] = filters
            
            async with self._session.post(
                f"{self.platform_url}/documents/query",
                json=query_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Document query executed: {len(result.get('documents', []))} results")
                    return result
                else:
                    error_text = await response.text()
                    raise MaterialKaiIntegrationError(f"Document query failed: {error_text}")
                    
        except Exception as e:
            logger.error(f"Document query failed: {e}")
            raise MaterialKaiIntegrationError(f"Query error: {e}")
    
    async def _initialize_websocket(self):
        """Initialize WebSocket connection for real-time communication."""
        try:
            if not self._session:
                return
            
            ws_url = self.platform_url.replace("http", "ws") + "/ws"
            self._websocket = await self._session.ws_connect(
                ws_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "X-Workspace-ID": self.workspace_id,
                    "X-Service-Name": self.service_name
                }
            )
            
            logger.info("WebSocket connection established")
            
            # Start listening for messages
            asyncio.create_task(self._websocket_listener())
            
        except Exception as e:
            logger.error(f"WebSocket initialization failed: {e}")
    
    async def _websocket_listener(self):
        """Listen for WebSocket messages."""
        try:
            if not self._websocket:
                return
            
            async for msg in self._websocket:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_websocket_message(data)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in WebSocket message: {msg.data}")
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self._websocket.exception()}")
                    break
                    
        except Exception as e:
            logger.error(f"WebSocket listener error: {e}")
    
    async def _handle_websocket_message(self, data: Dict[str, Any]):
        """Handle incoming WebSocket messages."""
        try:
            message_type = data.get("type")
            
            if message_type == "workflow_update":
                logger.info(f"Received workflow update: {data.get('workflow_id')}")
            elif message_type == "document_update":
                logger.info(f"Received document update: {data.get('document_id')}")
            elif message_type == "platform_notification":
                logger.info(f"Received platform notification: {data.get('message')}")
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    async def send_real_time_update(self, update_type: str, data: Dict[str, Any]):
        """
        Send real-time update via WebSocket.
        
        Args:
            update_type: Type of update
            data: Update data
        """
        try:
            if not self._websocket or self._websocket.closed:
                logger.warning("WebSocket not available for real-time update")
                return
            
            message = {
                "type": update_type,
                "data": data,
                "timestamp": datetime.utcnow().isoformat(),
                "service_source": self.service_name
            }
            
            await self._websocket.send_str(json.dumps(message))
            logger.debug(f"Sent real-time update: {update_type}")
            
        except Exception as e:
            logger.error(f"Failed to send real-time update: {e}")
    
    def is_connected(self) -> bool:
        """Check if service is connected to platform."""
        return self._is_connected and self._session and not self._session.closed
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get detailed connection status."""
        return {
            "connected": self.is_connected(),
            "websocket_active": self._websocket and not self._websocket.closed,
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "platform_url": self.platform_url,
            "workspace_id": self.workspace_id,
            "service_name": self.service_name
        }
    
    # Image Processing Methods for Material Kai Vision Platform Integration
    
    @retry_on_failure(max_retries=3, backoff_factor=1.5)
    @handle_http_errors
    async def upload_image(
        self,
        image_data: Union[bytes, BinaryIO],
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Upload an image to Material Kai Vision Platform for analysis.
        
        Args:
            image_data: Image data as bytes or file-like object
            filename: Original filename of the image
            metadata: Optional metadata about the image
            
        Returns:
            Dict containing upload result with image_id and analysis_url
        """
        try:
            if not self._session:
                raise MaterialKaiIntegrationError("Not connected to platform")
            
            # Prepare image data
            if hasattr(image_data, 'read'):
                image_bytes = image_data.read()
            else:
                image_bytes = image_data
            
            # Encode image as base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Detect MIME type
            mime_type, _ = mimetypes.guess_type(filename)
            if not mime_type or not mime_type.startswith('image/'):
                mime_type = 'image/jpeg'  # Default fallback
            
            upload_data = {
                "image_data": image_b64,
                "filename": filename,
                "mime_type": mime_type,
                "size": len(image_bytes),
                "workspace_id": self.workspace_id,
                "service_source": self.service_name,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            async with self._session.post(
                f"{self.platform_url}/vision/upload",
                json=upload_data
            ) as response:
                if response.status in [200, 201]:
                    result = await response.json()
                    logger.info(f"Image uploaded successfully: {result.get('image_id')}")
                    return result
                else:
                    error_text = await response.text()
                    raise MaterialKaiIntegrationError(f"Image upload failed: {error_text}")
                    
        except Exception as e:
            logger.error(f"Image upload failed: {e}")
            raise MaterialKaiIntegrationError(f"Upload error: {e}")
    
    async def analyze_image(
        self,
        image_id: str,
        analysis_types: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze an uploaded image using Material Kai Vision Platform.
        
        Args:
            image_id: ID of the uploaded image
            analysis_types: List of analysis types to perform
            options: Additional analysis options
            
        Returns:
            Dict containing analysis results
        """
        try:
            if not self._session:
                raise MaterialKaiIntegrationError("Not connected to platform")
            
            # Default analysis types if not specified
            if analysis_types is None:
                analysis_types = [
                    "object_detection",
                    "scene_classification",
                    "text_extraction",
                    "visual_features"
                ]
            
            analysis_data = {
                "image_id": image_id,
                "analysis_types": analysis_types,
                "workspace_id": self.workspace_id,
                "service_source": self.service_name,
                "options": options or {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            async with self._session.post(
                f"{self.platform_url}/vision/analyze",
                json=analysis_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Image analysis completed: {image_id}")
                    return result
                else:
                    error_text = await response.text()
                    raise MaterialKaiIntegrationError(f"Image analysis failed: {error_text}")
                    
        except Exception as e:
            logger.error(f"Image analysis failed for {image_id}: {e}")
            raise MaterialKaiIntegrationError(f"Analysis error: {e}")
    
    @retry_on_failure(max_retries=3, backoff_factor=1.5)
    @handle_http_errors
    async def generate_image_embeddings(
        self,
        image_id: str,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate vector embeddings for an image using vision models.
        
        Args:
            image_id: ID of the uploaded image
            model_name: Specific model to use for embeddings
            
        Returns:
            Dict containing embedding vectors and metadata
        """
        try:
            if not self._session:
                raise MaterialKaiIntegrationError("Not connected to platform")
            
            embedding_data = {
                "image_id": image_id,
                "model_name": model_name or "material-kai-vision-v2",
                "workspace_id": self.workspace_id,
                "service_source": self.service_name,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            async with self._session.post(
                f"{self.platform_url}/vision/embeddings",
                json=embedding_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Image embeddings generated: {image_id}")
                    return result
                else:
                    error_text = await response.text()
                    raise MaterialKaiIntegrationError(f"Embedding generation failed: {error_text}")
                    
        except Exception as e:
            logger.error(f"Embedding generation failed for {image_id}: {e}")
            raise MaterialKaiIntegrationError(f"Embedding error: {e}")
    
    async def search_similar_images(
        self,
        query_image_id: str,
        limit: int = 10,
        similarity_threshold: float = 0.75,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for visually similar images in the platform.
        
        Args:
            query_image_id: ID of the query image
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score
            filters: Additional search filters
            
        Returns:
            Dict containing similar images and similarity scores
        """
        try:
            if not self._session:
                raise MaterialKaiIntegrationError("Not connected to platform")
            
            search_data = {
                "query_image_id": query_image_id,
                "limit": limit,
                "similarity_threshold": similarity_threshold,
                "workspace_id": self.workspace_id,
                "service_source": self.service_name,
                "filters": filters or {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            async with self._session.post(
                f"{self.platform_url}/vision/search",
                json=search_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Visual search completed: {len(result.get('results', []))} results")
                    return result
                else:
                    error_text = await response.text()
                    raise MaterialKaiIntegrationError(f"Visual search failed: {error_text}")
                    
        except Exception as e:
            logger.error(f"Visual search failed for {query_image_id}: {e}")
            raise MaterialKaiIntegrationError(f"Search error: {e}")
    
    async def enhance_image(
        self,
        image_id: str,
        enhancement_types: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enhance an image using Material Kai Vision Platform capabilities.
        
        Args:
            image_id: ID of the uploaded image
            enhancement_types: List of enhancement operations to perform
            options: Enhancement-specific options
            
        Returns:
            Dict containing enhanced image data and metadata
        """
        try:
            if not self._session:
                raise MaterialKaiIntegrationError("Not connected to platform")
            
            # Default enhancement types if not specified
            if enhancement_types is None:
                enhancement_types = [
                    "noise_reduction",
                    "sharpening",
                    "contrast_enhancement"
                ]
            
            enhancement_data = {
                "image_id": image_id,
                "enhancement_types": enhancement_types,
                "workspace_id": self.workspace_id,
                "service_source": self.service_name,
                "options": options or {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            async with self._session.post(
                f"{self.platform_url}/vision/enhance",
                json=enhancement_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Image enhancement completed: {image_id}")
                    return result
                else:
                    error_text = await response.text()
                    raise MaterialKaiIntegrationError(f"Image enhancement failed: {error_text}")
                    
        except Exception as e:
            logger.error(f"Image enhancement failed for {image_id}: {e}")
            raise MaterialKaiIntegrationError(f"Enhancement error: {e}")
    
    async def batch_process_images(
        self,
        image_operations: List[Dict[str, Any]],
        batch_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process multiple images in a single batch operation.
        
        Args:
            image_operations: List of image operations to perform
            batch_options: Batch processing options
            
        Returns:
            Dict containing batch processing results
        """
        try:
            if not self._session:
                raise MaterialKaiIntegrationError("Not connected to platform")
            
            batch_data = {
                "operations": image_operations,
                "workspace_id": self.workspace_id,
                "service_source": self.service_name,
                "batch_options": batch_options or {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            async with self._session.post(
                f"{self.platform_url}/vision/batch",
                json=batch_data
            ) as response:
                if response.status in [200, 202]:
                    result = await response.json()
                    logger.info(f"Batch processing initiated: {len(image_operations)} operations")
                    return result
                else:
                    error_text = await response.text()
                    raise MaterialKaiIntegrationError(f"Batch processing failed: {error_text}")
                    
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise MaterialKaiIntegrationError(f"Batch error: {e}")
    
    @retry_on_failure(max_retries=2, backoff_factor=1.0)  # Status checks need fewer retries
    @handle_http_errors
    @retry_on_failure(max_retries=2, backoff_factor=1.0)  # Status checks need fewer retries
    @handle_http_errors
    async def get_image_analysis_status(self, analysis_id: str) -> Dict[str, Any]:
        """
        Get the status of an image analysis operation.
        
        Args:
            analysis_id: ID of the analysis operation
            
        Returns:
            Dict containing analysis status and results if completed
        """
        try:
            if not self._session:
                raise MaterialKaiIntegrationError("Not connected to platform")
            
            async with self._session.get(
                f"{self.platform_url}/vision/analysis/{analysis_id}/status"
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.debug(f"Analysis status retrieved: {analysis_id}")
                    return result
                else:
                    error_text = await response.text()
                    raise MaterialKaiIntegrationError(f"Status check failed: {error_text}")
                    
        except Exception as e:
            logger.error(f"Status check failed for {analysis_id}: {e}")
            raise MaterialKaiIntegrationError(f"Status error: {e}")


# Global service instance
_material_kai_service: Optional[MaterialKaiService] = None


async def get_material_kai_service() -> MaterialKaiService:
    """Get or create Material Kai service instance."""
    global _material_kai_service

    if _material_kai_service is None:
        _material_kai_service = MaterialKaiService()
        try:
            connection_success = await _material_kai_service.connect()
            if not connection_success:
                logger.warning("Material Kai service connection failed, service will use fallback mode")
        except Exception as e:
            logger.warning(f"Material Kai service connection error: {e}, service will use fallback mode")

    return _material_kai_service


async def cleanup_material_kai_service():
    """Cleanup Material Kai service instance."""
    global _material_kai_service
    
    if _material_kai_service:
        await _material_kai_service.disconnect()
        _material_kai_service = None