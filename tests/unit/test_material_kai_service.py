"""
Unit tests for Material Kai Vision Platform service.

Tests the MaterialKaiService class in isolation using mocks for external dependencies.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List
import json
import asyncio

from app.services.material_kai_service import MaterialKaiService


class TestMaterialKaiService:
    """Test suite for MaterialKaiService class."""

    @pytest.fixture
    def mock_http_client(self):
        """Create a mocked HTTP client."""
        mock_client = AsyncMock()
        
        # Mock successful responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success", "data": {}}
        mock_response.text = '{"status": "success"}'
        
        mock_client.get.return_value = mock_response
        mock_client.post.return_value = mock_response
        mock_client.put.return_value = mock_response
        mock_client.delete.return_value = mock_response
        
        return mock_client

    @pytest.fixture
    def mock_websocket(self):
        """Create a mocked WebSocket connection."""
        mock_ws = AsyncMock()
        mock_ws.send.return_value = None
        mock_ws.recv.return_value = '{"type": "ack", "message": "received"}'
        mock_ws.close.return_value = None
        return mock_ws

    @pytest.fixture
    def material_kai_service(self, mock_http_client):
        """Create a MaterialKaiService instance with mocked dependencies."""
        with patch('app.services.material_kai_service.httpx.AsyncClient', return_value=mock_http_client):
            service = MaterialKaiService()
            service.http_client = mock_http_client
            return service

    @pytest.mark.asyncio
    async def test_initialization(self, material_kai_service):
        """Test MaterialKaiService initialization."""
        assert material_kai_service.base_url is not None
        assert material_kai_service.api_key is not None
        assert material_kai_service.http_client is not None

    @pytest.mark.asyncio
    async def test_health_check_success(self, material_kai_service):
        """Test successful health check."""
        # Mock successful health check response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "healthy",
            "version": "1.0.0",
            "services": ["vision", "analysis", "storage"]
        }
        
        material_kai_service.http_client.get.return_value = mock_response
        
        result = await material_kai_service.health_check()
        
        assert result["healthy"] is True
        assert result["status"] == "healthy"
        assert "version" in result

    @pytest.mark.asyncio
    async def test_health_check_failure(self, material_kai_service):
        """Test health check failure."""
        # Mock health check failure
        material_kai_service.http_client.get.side_effect = Exception("Connection failed")
        
        result = await material_kai_service.health_check()
        
        assert result["healthy"] is False
        assert "Connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_register_service_success(self, material_kai_service):
        """Test successful service registration."""
        service_config = {
            "name": "pdf-processor",
            "version": "1.0.0",
            "endpoints": ["/process", "/analyze"],
            "capabilities": ["pdf_processing", "text_extraction"]
        }
        
        # Mock successful registration
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "service_id": "svc_123",
            "status": "registered",
            "webhook_url": "https://platform.com/webhook/svc_123"
        }
        
        material_kai_service.http_client.post.return_value = mock_response
        
        result = await material_kai_service.register_service(service_config)
        
        assert result["success"] is True
        assert result["service_id"] == "svc_123"
        assert "webhook_url" in result

    @pytest.mark.asyncio
    async def test_register_service_failure(self, material_kai_service):
        """Test service registration failure."""
        service_config = {"name": "test-service"}
        
        # Mock registration failure
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Invalid configuration"}
        
        material_kai_service.http_client.post.return_value = mock_response
        
        result = await material_kai_service.register_service(service_config)
        
        assert result["success"] is False
        assert "Invalid configuration" in result["error"]

    @pytest.mark.asyncio
    async def test_sync_document_success(self, material_kai_service):
        """Test successful document synchronization."""
        document_data = {
            "id": "doc_123",
            "title": "Test Document",
            "content": "Document content",
            "metadata": {"type": "pdf", "pages": 5}
        }
        
        # Mock successful sync
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "sync_id": "sync_456",
            "status": "synced",
            "platform_document_id": "platform_doc_789"
        }
        
        material_kai_service.http_client.post.return_value = mock_response
        
        result = await material_kai_service.sync_document(document_data)
        
        assert result["success"] is True
        assert result["sync_id"] == "sync_456"
        assert result["platform_document_id"] == "platform_doc_789"

    @pytest.mark.asyncio
    async def test_sync_document_failure(self, material_kai_service):
        """Test document synchronization failure."""
        document_data = {"id": "doc_123", "title": "Test Document"}
        
        # Mock sync failure
        material_kai_service.http_client.post.side_effect = Exception("Sync failed")
        
        result = await material_kai_service.sync_document(document_data)
        
        assert result["success"] is False
        assert "Sync failed" in result["error"]

    @pytest.mark.asyncio
    async def test_analyze_image_success(self, material_kai_service):
        """Test successful image analysis."""
        image_data = b"fake image data"
        analysis_options = {
            "detect_objects": True,
            "extract_text": True,
            "detect_faces": False
        }
        
        # Mock successful analysis
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "analysis_id": "analysis_123",
            "objects": [
                {"label": "document", "confidence": 0.95, "bbox": [10, 10, 100, 100]},
                {"label": "text", "confidence": 0.87, "bbox": [20, 20, 80, 80]}
            ],
            "extracted_text": "Sample extracted text",
            "faces": [],
            "metadata": {"processing_time": 1.2}
        }
        
        material_kai_service.http_client.post.return_value = mock_response
        
        result = await material_kai_service.analyze_image(image_data, analysis_options)
        
        assert result["success"] is True
        assert result["analysis_id"] == "analysis_123"
        assert len(result["objects"]) == 2
        assert result["extracted_text"] == "Sample extracted text"

    @pytest.mark.asyncio
    async def test_analyze_image_failure(self, material_kai_service):
        """Test image analysis failure."""
        image_data = b"fake image data"
        
        # Mock analysis failure
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Analysis service unavailable"}
        
        material_kai_service.http_client.post.return_value = mock_response
        
        result = await material_kai_service.analyze_image(image_data)
        
        assert result["success"] is False
        assert "Analysis service unavailable" in result["error"]

    @pytest.mark.asyncio
    async def test_batch_analyze_images_success(self, material_kai_service):
        """Test successful batch image analysis."""
        images = [
            {"id": "img_1", "data": b"image 1 data"},
            {"id": "img_2", "data": b"image 2 data"},
            {"id": "img_3", "data": b"image 3 data"}
        ]
        
        # Mock successful batch analysis
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.json.return_value = {
            "batch_id": "batch_456",
            "status": "processing",
            "total_images": 3,
            "estimated_completion": "2023-12-01T12:30:00Z"
        }
        
        material_kai_service.http_client.post.return_value = mock_response
        
        result = await material_kai_service.batch_analyze_images(images)
        
        assert result["success"] is True
        assert result["batch_id"] == "batch_456"
        assert result["total_images"] == 3

    @pytest.mark.asyncio
    async def test_get_batch_status_success(self, material_kai_service):
        """Test successful batch status retrieval."""
        batch_id = "batch_456"
        
        # Mock successful status retrieval
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "batch_id": "batch_456",
            "status": "completed",
            "progress": 100,
            "completed_images": 3,
            "failed_images": 0,
            "results": [
                {"image_id": "img_1", "status": "success", "analysis_id": "analysis_1"},
                {"image_id": "img_2", "status": "success", "analysis_id": "analysis_2"},
                {"image_id": "img_3", "status": "success", "analysis_id": "analysis_3"}
            ]
        }
        
        material_kai_service.http_client.get.return_value = mock_response
        
        result = await material_kai_service.get_batch_status(batch_id)
        
        assert result["success"] is True
        assert result["status"] == "completed"
        assert result["progress"] == 100
        assert len(result["results"]) == 3

    @pytest.mark.asyncio
    async def test_search_similar_images_success(self, material_kai_service):
        """Test successful similar image search."""
        query_image_data = b"query image data"
        search_options = {
            "top_k": 5,
            "similarity_threshold": 0.8,
            "include_metadata": True
        }
        
        # Mock successful search
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "search_id": "search_789",
            "results": [
                {
                    "image_id": "img_similar_1",
                    "similarity_score": 0.95,
                    "metadata": {"source": "document_1", "page": 1}
                },
                {
                    "image_id": "img_similar_2",
                    "similarity_score": 0.87,
                    "metadata": {"source": "document_2", "page": 3}
                }
            ],
            "total_matches": 2
        }
        
        material_kai_service.http_client.post.return_value = mock_response
        
        result = await material_kai_service.search_similar_images(query_image_data, search_options)
        
        assert result["success"] is True
        assert result["search_id"] == "search_789"
        assert len(result["results"]) == 2
        assert result["results"][0]["similarity_score"] == 0.95

    @pytest.mark.asyncio
    async def test_send_notification_success(self, material_kai_service):
        """Test successful notification sending."""
        notification_data = {
            "type": "document_processed",
            "document_id": "doc_123",
            "status": "completed",
            "metadata": {"processing_time": 5.2}
        }
        
        # Mock successful notification
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "notification_id": "notif_456",
            "status": "sent",
            "delivery_time": "2023-12-01T12:00:00Z"
        }
        
        material_kai_service.http_client.post.return_value = mock_response
        
        result = await material_kai_service.send_notification(notification_data)
        
        assert result["success"] is True
        assert result["notification_id"] == "notif_456"
        assert result["status"] == "sent"

    @pytest.mark.asyncio
    async def test_create_workflow_success(self, material_kai_service):
        """Test successful workflow creation."""
        workflow_config = {
            "name": "PDF Processing Workflow",
            "steps": [
                {"type": "extract_text", "config": {}},
                {"type": "analyze_images", "config": {"detect_objects": True}},
                {"type": "generate_summary", "config": {}}
            ],
            "triggers": ["document_upload"],
            "outputs": ["processed_document", "analysis_report"]
        }
        
        # Mock successful workflow creation
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "workflow_id": "workflow_789",
            "status": "created",
            "version": "1.0",
            "webhook_url": "https://platform.com/webhook/workflow_789"
        }
        
        material_kai_service.http_client.post.return_value = mock_response
        
        result = await material_kai_service.create_workflow(workflow_config)
        
        assert result["success"] is True
        assert result["workflow_id"] == "workflow_789"
        assert "webhook_url" in result

    @pytest.mark.asyncio
    async def test_execute_workflow_success(self, material_kai_service):
        """Test successful workflow execution."""
        workflow_id = "workflow_789"
        input_data = {
            "document_id": "doc_123",
            "document_url": "https://storage.com/doc_123.pdf",
            "metadata": {"priority": "high"}
        }
        
        # Mock successful workflow execution
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.json.return_value = {
            "execution_id": "exec_456",
            "workflow_id": "workflow_789",
            "status": "running",
            "started_at": "2023-12-01T12:00:00Z",
            "estimated_completion": "2023-12-01T12:05:00Z"
        }
        
        material_kai_service.http_client.post.return_value = mock_response
        
        result = await material_kai_service.execute_workflow(workflow_id, input_data)
        
        assert result["success"] is True
        assert result["execution_id"] == "exec_456"
        assert result["status"] == "running"

    @pytest.mark.asyncio
    async def test_get_workflow_status_success(self, material_kai_service):
        """Test successful workflow status retrieval."""
        execution_id = "exec_456"
        
        # Mock successful status retrieval
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "execution_id": "exec_456",
            "workflow_id": "workflow_789",
            "status": "completed",
            "progress": 100,
            "started_at": "2023-12-01T12:00:00Z",
            "completed_at": "2023-12-01T12:04:30Z",
            "results": {
                "extracted_text": "Document text content",
                "analysis_report": {"objects": 5, "text_blocks": 12},
                "summary": "Document summary"
            }
        }
        
        material_kai_service.http_client.get.return_value = mock_response
        
        result = await material_kai_service.get_workflow_status(execution_id)
        
        assert result["success"] is True
        assert result["status"] == "completed"
        assert result["progress"] == 100
        assert "results" in result

    @pytest.mark.asyncio
    async def test_connect_websocket_success(self, material_kai_service, mock_websocket):
        """Test successful WebSocket connection."""
        with patch('app.services.material_kai_service.websockets.connect', return_value=mock_websocket):
            result = await material_kai_service.connect_websocket()
            
            assert result["success"] is True
            assert "connected" in result["status"].lower()

    @pytest.mark.asyncio
    async def test_connect_websocket_failure(self, material_kai_service):
        """Test WebSocket connection failure."""
        with patch('app.services.material_kai_service.websockets.connect', side_effect=Exception("Connection failed")):
            result = await material_kai_service.connect_websocket()
            
            assert result["success"] is False
            assert "Connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_send_websocket_message_success(self, material_kai_service, mock_websocket):
        """Test successful WebSocket message sending."""
        message = {
            "type": "document_update",
            "document_id": "doc_123",
            "status": "processing"
        }
        
        # Set up WebSocket connection
        material_kai_service.websocket = mock_websocket
        
        result = await material_kai_service.send_websocket_message(message)
        
        assert result["success"] is True
        mock_websocket.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_websocket_message_no_connection(self, material_kai_service):
        """Test WebSocket message sending without connection."""
        message = {"type": "test"}
        
        # No WebSocket connection
        material_kai_service.websocket = None
        
        result = await material_kai_service.send_websocket_message(message)
        
        assert result["success"] is False
        assert "not connected" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_platform_statistics_success(self, material_kai_service):
        """Test successful platform statistics retrieval."""
        # Mock successful statistics retrieval
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total_documents": 1500,
            "total_images": 5000,
            "active_workflows": 25,
            "processing_queue_size": 12,
            "storage_usage": "2.5GB",
            "api_calls_today": 850,
            "uptime": "99.9%"
        }
        
        material_kai_service.http_client.get.return_value = mock_response
        
        result = await material_kai_service.get_platform_statistics()
        
        assert result["success"] is True
        assert result["statistics"]["total_documents"] == 1500
        assert result["statistics"]["uptime"] == "99.9%"

    @pytest.mark.asyncio
    async def test_update_service_config_success(self, material_kai_service):
        """Test successful service configuration update."""
        service_id = "svc_123"
        config_updates = {
            "max_concurrent_requests": 10,
            "timeout_seconds": 30,
            "retry_attempts": 3
        }
        
        # Mock successful config update
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "service_id": "svc_123",
            "status": "updated",
            "config": config_updates,
            "updated_at": "2023-12-01T12:00:00Z"
        }
        
        material_kai_service.http_client.put.return_value = mock_response
        
        result = await material_kai_service.update_service_config(service_id, config_updates)
        
        assert result["success"] is True
        assert result["status"] == "updated"
        assert result["config"]["max_concurrent_requests"] == 10

    @pytest.mark.asyncio
    async def test_delete_document_success(self, material_kai_service):
        """Test successful document deletion."""
        document_id = "doc_123"
        
        # Mock successful deletion
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "document_id": "doc_123",
            "status": "deleted",
            "deleted_at": "2023-12-01T12:00:00Z"
        }
        
        material_kai_service.http_client.delete.return_value = mock_response
        
        result = await material_kai_service.delete_document(document_id)
        
        assert result["success"] is True
        assert result["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_get_service_logs_success(self, material_kai_service):
        """Test successful service logs retrieval."""
        service_id = "svc_123"
        log_options = {
            "level": "INFO",
            "limit": 100,
            "since": "2023-12-01T00:00:00Z"
        }
        
        # Mock successful logs retrieval
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "service_id": "svc_123",
            "logs": [
                {
                    "timestamp": "2023-12-01T12:00:00Z",
                    "level": "INFO",
                    "message": "Service started",
                    "metadata": {}
                },
                {
                    "timestamp": "2023-12-01T12:01:00Z",
                    "level": "INFO",
                    "message": "Document processed",
                    "metadata": {"document_id": "doc_123"}
                }
            ],
            "total_logs": 2,
            "has_more": False
        }
        
        material_kai_service.http_client.get.return_value = mock_response
        
        result = await material_kai_service.get_service_logs(service_id, log_options)
        
        assert result["success"] is True
        assert len(result["logs"]) == 2
        assert result["total_logs"] == 2

    def test_configuration_validation(self, material_kai_service):
        """Test service configuration validation."""
        config = material_kai_service.get_configuration()
        
        assert "base_url" in config
        assert "api_key" in config
        assert "timeout" in config
        assert isinstance(config["max_retries"], int)

    @pytest.mark.asyncio
    async def test_close_connections_success(self, material_kai_service, mock_websocket):
        """Test successful connection cleanup."""
        # Set up connections
        material_kai_service.websocket = mock_websocket
        
        await material_kai_service.close_connections()
        
        # Verify WebSocket was closed
        mock_websocket.close.assert_called_once()
        assert material_kai_service.websocket is None

    @pytest.mark.asyncio
    async def test_retry_mechanism_success(self, material_kai_service):
        """Test retry mechanism on temporary failures."""
        # Mock temporary failure followed by success
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 503
        mock_response_fail.json.return_value = {"error": "Service temporarily unavailable"}
        
        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"status": "healthy"}
        
        material_kai_service.http_client.get.side_effect = [
            mock_response_fail,
            mock_response_success
        ]
        
        result = await material_kai_service.health_check()
        
        assert result["healthy"] is True
        assert material_kai_service.http_client.get.call_count == 2