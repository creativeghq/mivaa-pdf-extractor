"""
Integration tests for external service integrations.

This module tests the integration with external services including
Material Kai Vision Platform, OpenAI API, and other third-party services,
ensuring proper communication, error handling, and data flow.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import json
from datetime import datetime
import aiohttp
from aioresponses import aioresponses

from app.services.material_kai_service import MaterialKaiService
from app.services.openai_service import OpenAIService
from app.core.config import get_settings


@pytest.mark.integration
@pytest.mark.external
class TestMaterialKaiIntegration:
    """Integration tests for Material Kai Vision Platform service."""
    
    @pytest.fixture
    async def material_kai_service(self):
        """Create Material Kai service instance."""
        service = MaterialKaiService()
        await service.initialize()
        return service
    
    @pytest.fixture
    def mock_material_kai_responses(self):
        """Mock responses from Material Kai API."""
        return {
            "register_service": {
                "service_id": "pdf_processor_123",
                "status": "registered",
                "api_key": "mk_test_key_123",
                "webhook_url": "https://api.materialkai.com/webhooks/pdf_processor_123"
            },
            "sync_document": {
                "sync_id": "sync_456",
                "status": "synced",
                "document_id": "doc_123",
                "material_kai_id": "mk_doc_789"
            },
            "analyze_image": {
                "analysis_id": "analysis_789",
                "results": {
                    "objects": [
                        {"name": "document", "confidence": 0.95, "bbox": [10, 10, 100, 100]},
                        {"name": "text", "confidence": 0.88, "bbox": [20, 20, 80, 80]}
                    ],
                    "text_regions": [
                        {"text": "Sample document text", "bbox": [15, 15, 85, 85], "confidence": 0.92}
                    ],
                    "quality_score": 0.87,
                    "categories": ["document", "business", "text"]
                }
            },
            "batch_process": {
                "batch_id": "batch_101",
                "status": "processing",
                "total_items": 5,
                "processed_items": 0,
                "estimated_completion": "2024-01-01T12:30:00Z"
            },
            "get_workflow_status": {
                "workflow_id": "workflow_202",
                "status": "completed",
                "progress": 100,
                "results": {
                    "processed_documents": 3,
                    "extracted_images": 12,
                    "analysis_results": [
                        {"document_id": "doc_1", "confidence": 0.95},
                        {"document_id": "doc_2", "confidence": 0.88},
                        {"document_id": "doc_3", "confidence": 0.91}
                    ]
                }
            }
        }
    
    async def test_service_registration_integration(
        self, 
        material_kai_service, 
        mock_material_kai_responses
    ):
        """Test service registration with Material Kai platform."""
        with aioresponses() as m:
            # Mock registration endpoint
            m.post(
                'https://api.materialkai.com/v1/services/register',
                payload=mock_material_kai_responses["register_service"],
                status=200
            )
            
            registration_data = {
                "service_name": "PDF Processor",
                "service_type": "document_processing",
                "capabilities": ["pdf_extraction", "image_analysis", "text_recognition"],
                "webhook_url": "https://our-service.com/webhooks/material-kai"
            }
            
            result = await material_kai_service.register_service(registration_data)
            
            assert result["service_id"] == "pdf_processor_123"
            assert result["status"] == "registered"
            assert "api_key" in result
            
            # Verify the service stored the API key
            assert material_kai_service.api_key == "mk_test_key_123"
    
    async def test_document_synchronization_integration(
        self, 
        material_kai_service, 
        mock_material_kai_responses
    ):
        """Test document synchronization with Material Kai."""
        # Set up authenticated service
        material_kai_service.api_key = "test_api_key"
        
        with aioresponses() as m:
            # Mock document sync endpoint
            m.post(
                'https://api.materialkai.com/v1/documents/sync',
                payload=mock_material_kai_responses["sync_document"],
                status=200
            )
            
            document_data = {
                "document_id": "doc_123",
                "title": "Test Document",
                "content": "Document content for analysis",
                "metadata": {
                    "page_count": 5,
                    "file_size": 1024,
                    "file_type": "pdf"
                },
                "images": [
                    {"image_id": "img_1", "url": "https://storage.com/img1.jpg"},
                    {"image_id": "img_2", "url": "https://storage.com/img2.jpg"}
                ]
            }
            
            result = await material_kai_service.sync_document(document_data)
            
            assert result["sync_id"] == "sync_456"
            assert result["status"] == "synced"
            assert result["material_kai_id"] == "mk_doc_789"
    
    async def test_image_analysis_integration(
        self, 
        material_kai_service, 
        mock_material_kai_responses
    ):
        """Test image analysis through Material Kai."""
        material_kai_service.api_key = "test_api_key"
        
        with aioresponses() as m:
            # Mock image analysis endpoint
            m.post(
                'https://api.materialkai.com/v1/images/analyze',
                payload=mock_material_kai_responses["analyze_image"],
                status=200
            )
            
            image_data = {
                "image_url": "https://storage.com/test_image.jpg",
                "analysis_options": {
                    "detect_objects": True,
                    "extract_text": True,
                    "quality_assessment": True,
                    "categorization": True
                }
            }
            
            result = await material_kai_service.analyze_image(image_data)
            
            assert result["analysis_id"] == "analysis_789"
            assert len(result["results"]["objects"]) == 2
            assert result["results"]["quality_score"] == 0.87
            assert "document" in result["results"]["categories"]
    
    async def test_batch_processing_integration(
        self, 
        material_kai_service, 
        mock_material_kai_responses
    ):
        """Test batch processing workflow with Material Kai."""
        material_kai_service.api_key = "test_api_key"
        
        with aioresponses() as m:
            # Mock batch processing endpoints
            m.post(
                'https://api.materialkai.com/v1/batch/process',
                payload=mock_material_kai_responses["batch_process"],
                status=200
            )
            
            # Mock status check
            m.get(
                'https://api.materialkai.com/v1/batch/batch_101/status',
                payload={
                    **mock_material_kai_responses["batch_process"],
                    "status": "completed",
                    "processed_items": 5
                },
                status=200
            )
            
            batch_data = {
                "items": [
                    {"type": "document", "id": "doc_1", "url": "https://storage.com/doc1.pdf"},
                    {"type": "document", "id": "doc_2", "url": "https://storage.com/doc2.pdf"},
                    {"type": "image", "id": "img_1", "url": "https://storage.com/img1.jpg"},
                    {"type": "image", "id": "img_2", "url": "https://storage.com/img2.jpg"},
                    {"type": "image", "id": "img_3", "url": "https://storage.com/img3.jpg"}
                ],
                "processing_options": {
                    "priority": "normal",
                    "webhook_url": "https://our-service.com/webhooks/batch-complete"
                }
            }
            
            # Start batch processing
            batch_result = await material_kai_service.start_batch_processing(batch_data)
            assert batch_result["batch_id"] == "batch_101"
            assert batch_result["total_items"] == 5
            
            # Check batch status
            status_result = await material_kai_service.get_batch_status("batch_101")
            assert status_result["status"] == "completed"
            assert status_result["processed_items"] == 5
    
    async def test_webhook_handling_integration(self, material_kai_service):
        """Test webhook handling from Material Kai."""
        # Mock webhook payload
        webhook_payload = {
            "event_type": "analysis_complete",
            "timestamp": "2024-01-01T12:00:00Z",
            "data": {
                "analysis_id": "analysis_789",
                "document_id": "doc_123",
                "status": "completed",
                "results": {
                    "confidence": 0.95,
                    "categories": ["document", "business"],
                    "extracted_text": "Sample extracted text",
                    "objects_detected": 5
                }
            }
        }
        
        # Process webhook
        result = await material_kai_service.handle_webhook(webhook_payload)
        
        assert result["status"] == "processed"
        assert result["event_type"] == "analysis_complete"
        assert result["analysis_id"] == "analysis_789"
    
    async def test_error_handling_integration(self, material_kai_service):
        """Test error handling in Material Kai integration."""
        material_kai_service.api_key = "test_api_key"
        
        with aioresponses() as m:
            # Mock API error responses
            m.post(
                'https://api.materialkai.com/v1/documents/sync',
                payload={"error": "Invalid document format", "code": "INVALID_FORMAT"},
                status=400
            )
            
            m.post(
                'https://api.materialkai.com/v1/images/analyze',
                payload={"error": "Service temporarily unavailable", "code": "SERVICE_UNAVAILABLE"},
                status=503
            )
            
            # Test document sync error
            document_data = {"document_id": "invalid_doc", "content": ""}
            
            with pytest.raises(Exception) as exc_info:
                await material_kai_service.sync_document(document_data)
            assert "Invalid document format" in str(exc_info.value)
            
            # Test image analysis error
            image_data = {"image_url": "invalid_url"}
            
            with pytest.raises(Exception) as exc_info:
                await material_kai_service.analyze_image(image_data)
            assert "Service temporarily unavailable" in str(exc_info.value)
    
    async def test_rate_limiting_integration(self, material_kai_service):
        """Test rate limiting handling with Material Kai API."""
        material_kai_service.api_key = "test_api_key"
        
        with aioresponses() as m:
            # Mock rate limit response
            m.post(
                'https://api.materialkai.com/v1/documents/sync',
                payload={"error": "Rate limit exceeded", "retry_after": 60},
                status=429,
                headers={"Retry-After": "60"}
            )
            
            document_data = {"document_id": "doc_123", "content": "test"}
            
            with pytest.raises(Exception) as exc_info:
                await material_kai_service.sync_document(document_data)
            
            assert "Rate limit exceeded" in str(exc_info.value)


@pytest.mark.integration
@pytest.mark.external
class TestOpenAIIntegration:
    """Integration tests for OpenAI API service."""
    
    @pytest.fixture
    async def openai_service(self):
        """Create OpenAI service instance."""
        with patch('app.services.openai_service.OpenAI') as mock_openai:
            service = OpenAIService()
            await service.initialize()
            return service
    
    async def test_embeddings_generation_integration(self, openai_service):
        """Test embeddings generation with OpenAI API."""
        # Mock OpenAI client response
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3] * 256)  # 768-dimensional embedding
        ]
        
        with patch.object(openai_service.client.embeddings, 'create', return_value=mock_response):
            texts = ["This is a test document.", "Another test sentence."]
            
            embeddings = await openai_service.generate_embeddings(texts)
            
            assert len(embeddings) == 2
            assert len(embeddings[0]) == 768  # text-embedding-3-large dimension
            assert isinstance(embeddings[0][0], float)
            
            # Verify API call
            openai_service.client.embeddings.create.assert_called_once_with(
                model="text-embedding-3-large",
                input=texts,
                dimensions=768
            )
    
    async def test_text_completion_integration(self, openai_service):
        """Test text completion with OpenAI API."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="This is a generated response."))
        ]
        
        with patch.object(openai_service.client.chat.completions, 'create', return_value=mock_response):
            prompt = "Summarize the following document: [document content]"
            
            response = await openai_service.generate_completion(prompt)
            
            assert response == "This is a generated response."
            
            # Verify API call
            openai_service.client.chat.completions.create.assert_called_once()
            call_args = openai_service.client.chat.completions.create.call_args
            assert call_args[1]["model"] == "gpt-4"
            assert len(call_args[1]["messages"]) == 1
            assert call_args[1]["messages"][0]["content"] == prompt
    
    async def test_batch_embeddings_integration(self, openai_service):
        """Test batch embeddings generation."""
        # Mock multiple API calls for large batches
        mock_responses = []
        for i in range(3):  # 3 batches
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[0.1 + i * 0.1, 0.2 + i * 0.1, 0.3 + i * 0.1] * 256)
                for _ in range(100)  # 100 embeddings per batch
            ]
            mock_responses.append(mock_response)
        
        with patch.object(openai_service.client.embeddings, 'create', side_effect=mock_responses):
            # Generate 250 texts (should be split into 3 batches)
            texts = [f"Text chunk {i}" for i in range(250)]
            
            embeddings = await openai_service.generate_embeddings_batch(texts, batch_size=100)
            
            assert len(embeddings) == 250
            assert len(embeddings[0]) == 768
            
            # Verify 3 API calls were made
            assert openai_service.client.embeddings.create.call_count == 3
    
    async def test_openai_error_handling_integration(self, openai_service):
        """Test error handling for OpenAI API failures."""
        from openai import RateLimitError, APIError
        
        # Test rate limit error
        with patch.object(openai_service.client.embeddings, 'create', side_effect=RateLimitError("Rate limit exceeded", response=None, body=None)):
            texts = ["Test text"]
            
            with pytest.raises(Exception) as exc_info:
                await openai_service.generate_embeddings(texts)
            assert "Rate limit exceeded" in str(exc_info.value)
        
        # Test API error
        with patch.object(openai_service.client.embeddings, 'create', side_effect=APIError("API error", response=None, body=None)):
            texts = ["Test text"]
            
            with pytest.raises(Exception) as exc_info:
                await openai_service.generate_embeddings(texts)
            assert "API error" in str(exc_info.value)
    
    async def test_token_usage_tracking_integration(self, openai_service):
        """Test token usage tracking."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3] * 256)]
        mock_response.usage = MagicMock(total_tokens=150, prompt_tokens=150)
        
        with patch.object(openai_service.client.embeddings, 'create', return_value=mock_response):
            texts = ["This is a test document for token counting."]
            
            embeddings = await openai_service.generate_embeddings(texts)
            
            # Check that usage was tracked
            usage_stats = await openai_service.get_usage_stats()
            assert usage_stats["total_tokens"] >= 150
            assert usage_stats["embeddings_requests"] >= 1


@pytest.mark.integration
@pytest.mark.external
class TestExternalServiceResilience:
    """Integration tests for external service resilience and fallback mechanisms."""
    
    @pytest.fixture
    async def services(self):
        """Create service instances."""
        material_kai = MaterialKaiService()
        openai_service = OpenAIService()
        
        await material_kai.initialize()
        await openai_service.initialize()
        
        return {
            "material_kai": material_kai,
            "openai": openai_service
        }
    
    async def test_service_health_checks(self, services):
        """Test health checks for external services."""
        with aioresponses() as m:
            # Mock Material Kai health check
            m.get(
                'https://api.materialkai.com/v1/health',
                payload={"status": "healthy", "version": "1.0.0"},
                status=200
            )
            
            # Test Material Kai health
            mk_health = await services["material_kai"].health_check()
            assert mk_health["status"] == "healthy"
        
        # Test OpenAI health (mock a simple API call)
        with patch.object(services["openai"].client.embeddings, 'create') as mock_create:
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1] * 768)]
            mock_create.return_value = mock_response
            
            openai_health = await services["openai"].health_check()
            assert openai_health["status"] == "healthy"
    
    async def test_circuit_breaker_pattern(self, services):
        """Test circuit breaker pattern for failing services."""
        # Simulate multiple failures to trigger circuit breaker
        with aioresponses() as m:
            # Mock multiple failures
            for _ in range(5):
                m.post(
                    'https://api.materialkai.com/v1/documents/sync',
                    status=500
                )
            
            material_kai = services["material_kai"]
            
            # Make multiple failing requests
            for i in range(5):
                try:
                    await material_kai.sync_document({"document_id": f"doc_{i}"})
                except Exception:
                    pass  # Expected to fail
            
            # Circuit breaker should now be open
            circuit_status = await material_kai.get_circuit_breaker_status()
            assert circuit_status["state"] in ["open", "half_open"]
    
    async def test_retry_mechanism(self, services):
        """Test retry mechanism for transient failures."""
        with aioresponses() as m:
            # Mock initial failures followed by success
            m.post(
                'https://api.materialkai.com/v1/documents/sync',
                status=503  # Service unavailable
            )
            m.post(
                'https://api.materialkai.com/v1/documents/sync',
                status=503  # Service unavailable
            )
            m.post(
                'https://api.materialkai.com/v1/documents/sync',
                payload={"sync_id": "sync_123", "status": "synced"},
                status=200  # Success on third try
            )
            
            material_kai = services["material_kai"]
            
            # Should succeed after retries
            result = await material_kai.sync_document_with_retry(
                {"document_id": "doc_123"},
                max_retries=3
            )
            
            assert result["sync_id"] == "sync_123"
            assert result["status"] == "synced"
    
    async def test_fallback_mechanisms(self, services):
        """Test fallback mechanisms when primary services fail."""
        # Test OpenAI fallback to local embeddings
        with patch.object(services["openai"].client.embeddings, 'create', side_effect=Exception("OpenAI unavailable")):
            # Should fallback to local embedding service
            with patch('app.services.local_embeddings.LocalEmbeddingService.generate_embeddings') as mock_local:
                mock_local.return_value = [[0.1, 0.2, 0.3] * 256]
                
                texts = ["Test text for fallback"]
                embeddings = await services["openai"].generate_embeddings_with_fallback(texts)
                
                assert len(embeddings) == 1
                assert len(embeddings[0]) == 768
                mock_local.assert_called_once_with(texts)
    
    async def test_concurrent_external_requests(self, services):
        """Test handling of concurrent external service requests."""
        with aioresponses() as m:
            # Mock multiple successful responses
            for i in range(10):
                m.post(
                    'https://api.materialkai.com/v1/documents/sync',
                    payload={"sync_id": f"sync_{i}", "status": "synced"},
                    status=200
                )
        
        material_kai = services["material_kai"]
        
        # Make concurrent requests
        tasks = []
        for i in range(10):
            task = material_kai.sync_document({"document_id": f"doc_{i}"})
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # All requests should succeed
        assert len(results) == 10
        for i, result in enumerate(results):
            assert result["sync_id"] == f"sync_{i}"
            assert result["status"] == "synced"
    
    async def test_timeout_handling(self, services):
        """Test timeout handling for slow external services."""
        with aioresponses() as m:
            # Mock slow response (will timeout)
            async def slow_callback(url, **kwargs):
                await asyncio.sleep(10)  # Simulate slow response
                return aioresponses.CallbackResult(
                    status=200,
                    payload={"sync_id": "sync_123", "status": "synced"}
                )
            
            m.post(
                'https://api.materialkai.com/v1/documents/sync',
                callback=slow_callback
            )
        
        material_kai = services["material_kai"]
        
        # Should timeout and raise exception
        with pytest.raises(asyncio.TimeoutError):
            await material_kai.sync_document_with_timeout(
                {"document_id": "doc_123"},
                timeout=2.0  # 2 second timeout
            )
    
    async def test_service_degradation_handling(self, services):
        """Test handling of degraded service performance."""
        response_times = []
        
        with aioresponses() as m:
            # Mock responses with increasing delay
            async def delayed_callback(url, **kwargs):
                delay = len(response_times) * 0.5  # Increasing delay
                await asyncio.sleep(delay)
                response_times.append(delay)
                return aioresponses.CallbackResult(
                    status=200,
                    payload={"sync_id": f"sync_{len(response_times)}", "status": "synced"}
                )
            
            for _ in range(5):
                m.post(
                    'https://api.materialkai.com/v1/documents/sync',
                    callback=delayed_callback
                )
        
        material_kai = services["material_kai"]
        
        # Make requests and monitor performance
        for i in range(5):
            start_time = asyncio.get_event_loop().time()
            await material_kai.sync_document({"document_id": f"doc_{i}"})
            end_time = asyncio.get_event_loop().time()
            
            # Service should detect degradation and potentially switch to degraded mode
            if end_time - start_time > 2.0:  # If response time > 2 seconds
                degraded_status = await material_kai.get_service_status()
                assert degraded_status["performance"] == "degraded"
                break