"""
Performance Tests for MIVAA PDF Extractor

This module contains performance tests that verify the system meets
performance requirements under various load conditions and scenarios.
"""

import pytest
import asyncio
import time
import tempfile
import os
import statistics
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from httpx import AsyncClient
import psutil
import concurrent.futures
from typing import List, Dict, Any

from app.main import app
from app.config import get_settings


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmark tests for the PDF processing system."""

    @pytest.fixture
    def performance_pdf_content(self):
        """Create PDF content of various sizes for performance testing."""
        sizes = {
            "small": b"%PDF-1.4\n" + b"Small content " * 100 + b"\nendobj\n",
            "medium": b"%PDF-1.4\n" + b"Medium content " * 1000 + b"\nendobj\n",
            "large": b"%PDF-1.4\n" + b"Large content " * 10000 + b"\nendobj\n",
            "xlarge": b"%PDF-1.4\n" + b"XLarge content " * 50000 + b"\nendobj\n"
        }
        return sizes

    @pytest.fixture
    def performance_pdf_files(self, performance_pdf_content):
        """Create temporary PDF files of various sizes."""
        files = {}
        temp_files = []
        
        for size_name, content in performance_pdf_content.items():
            tmp_file = tempfile.NamedTemporaryFile(suffix=f"_{size_name}.pdf", delete=False)
            tmp_file.write(content)
            tmp_file.flush()
            files[size_name] = tmp_file.name
            temp_files.append(tmp_file.name)
        
        yield files
        
        # Cleanup
        for file_path in temp_files:
            try:
                os.unlink(file_path)
            except FileNotFoundError:
                pass

    def measure_execution_time(self, func, *args, **kwargs):
        """Measure execution time of a function."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return result, execution_time

    async def measure_async_execution_time(self, coro):
        """Measure execution time of an async coroutine."""
        start_time = time.perf_counter()
        result = await coro
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return result, execution_time

    def measure_memory_usage(self, func, *args, **kwargs):
        """Measure memory usage during function execution."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        result = func(*args, **kwargs)
        
        peak_memory = process.memory_info().rss
        memory_delta = peak_memory - initial_memory
        
        return result, memory_delta

    @pytest.mark.asyncio
    async def test_single_pdf_upload_performance(
        self,
        performance_pdf_files,
        mock_supabase_client,
        mock_llamaindex_service
    ):
        """Test performance of single PDF upload for different file sizes."""
        
        # Mock responses
        mock_supabase_client.table().insert().execute.return_value = MagicMock(
            data=[{"id": "perf-test-id", "filename": "test.pdf", "status": "uploaded"}]
        )
        
        mock_llamaindex_service.process_document.return_value = {
            "chunks": 10,
            "embeddings_created": True,
            "index_updated": True
        }

        performance_results = {}

        async with AsyncClient(app=app, base_url="http://test") as client:
            for size_name, file_path in performance_pdf_files.items():
                with open(file_path, "rb") as pdf_file:
                    upload_coro = client.post(
                        "/api/v1/upload",
                        files={"file": (f"test_{size_name}.pdf", pdf_file, "application/pdf")}
                    )
                    
                    response, execution_time = await self.measure_async_execution_time(upload_coro)
                    
                    performance_results[size_name] = {
                        "execution_time": execution_time,
                        "status_code": response.status_code,
                        "file_size": os.path.getsize(file_path)
                    }
                    
                    # Assert performance requirements
                    if size_name == "small":
                        assert execution_time < 1.0, f"Small file upload took {execution_time:.2f}s, should be < 1.0s"
                    elif size_name == "medium":
                        assert execution_time < 3.0, f"Medium file upload took {execution_time:.2f}s, should be < 3.0s"
                    elif size_name == "large":
                        assert execution_time < 10.0, f"Large file upload took {execution_time:.2f}s, should be < 10.0s"
                    
                    assert response.status_code == 200

        # Log performance results for analysis
        print("\n=== Single PDF Upload Performance Results ===")
        for size_name, results in performance_results.items():
            print(f"{size_name.upper()}: {results['execution_time']:.3f}s "
                  f"({results['file_size']} bytes)")

    @pytest.mark.asyncio
    async def test_concurrent_upload_performance(
        self,
        performance_pdf_files,
        mock_supabase_client,
        mock_llamaindex_service
    ):
        """Test performance under concurrent upload load."""
        
        # Mock responses
        mock_supabase_client.table().insert().execute.side_effect = [
            MagicMock(data=[{"id": f"concurrent-{i}", "filename": f"test{i}.pdf", "status": "uploaded"}])
            for i in range(20)
        ]
        
        mock_llamaindex_service.process_document.return_value = {
            "chunks": 5,
            "embeddings_created": True,
            "index_updated": True
        }

        async def upload_file(client, file_path, index):
            """Upload a single file and measure time."""
            with open(file_path, "rb") as pdf_file:
                start_time = time.perf_counter()
                response = await client.post(
                    "/api/v1/upload",
                    files={"file": (f"concurrent_test_{index}.pdf", pdf_file, "application/pdf")}
                )
                end_time = time.perf_counter()
                return {
                    "index": index,
                    "execution_time": end_time - start_time,
                    "status_code": response.status_code
                }

        async with AsyncClient(app=app, base_url="http://test") as client:
            # Test with 10 concurrent uploads
            concurrent_uploads = 10
            file_path = performance_pdf_files["medium"]  # Use medium-sized files
            
            start_time = time.perf_counter()
            
            # Create concurrent upload tasks
            tasks = [
                upload_file(client, file_path, i) 
                for i in range(concurrent_uploads)
            ]
            
            results = await asyncio.gather(*tasks)
            
            total_time = time.perf_counter() - start_time
            
            # Analyze results
            execution_times = [r["execution_time"] for r in results]
            successful_uploads = [r for r in results if r["status_code"] == 200]
            
            avg_time = statistics.mean(execution_times)
            max_time = max(execution_times)
            min_time = min(execution_times)
            
            # Performance assertions
            assert len(successful_uploads) == concurrent_uploads, "All uploads should succeed"
            assert total_time < 15.0, f"Total time for {concurrent_uploads} concurrent uploads: {total_time:.2f}s"
            assert avg_time < 5.0, f"Average upload time: {avg_time:.2f}s should be < 5.0s"
            assert max_time < 10.0, f"Max upload time: {max_time:.2f}s should be < 10.0s"

        print(f"\n=== Concurrent Upload Performance ({concurrent_uploads} files) ===")
        print(f"Total time: {total_time:.3f}s")
        print(f"Average time: {avg_time:.3f}s")
        print(f"Min time: {min_time:.3f}s")
        print(f"Max time: {max_time:.3f}s")
        print(f"Throughput: {concurrent_uploads/total_time:.2f} uploads/second")

    @pytest.mark.asyncio
    async def test_query_performance(
        self,
        mock_llamaindex_service
    ):
        """Test RAG query performance with different query complexities."""
        
        # Mock different response times for different query types
        def mock_query_response(query_text):
            if "simple" in query_text.lower():
                time.sleep(0.1)  # Simulate fast query
                return {
                    "answer": "Simple answer",
                    "sources": ["chunk_1"],
                    "confidence": 0.9
                }
            elif "complex" in query_text.lower():
                time.sleep(0.5)  # Simulate slower complex query
                return {
                    "answer": "Complex detailed answer with multiple aspects",
                    "sources": ["chunk_1", "chunk_2", "chunk_3"],
                    "confidence": 0.85
                }
            else:
                time.sleep(0.2)  # Default query time
                return {
                    "answer": "Standard answer",
                    "sources": ["chunk_1", "chunk_2"],
                    "confidence": 0.88
                }
        
        mock_llamaindex_service.query_documents.side_effect = lambda query, **kwargs: mock_query_response(query)

        query_types = {
            "simple": "What is this document about?",
            "medium": "Explain the main concepts discussed in the document",
            "complex": "Provide a detailed analysis of the relationships between different concepts and their implications"
        }

        async with AsyncClient(app=app, base_url="http://test") as client:
            performance_results = {}
            
            for query_type, query_text in query_types.items():
                query_coro = client.post(
                    "/api/v1/query",
                    json={
                        "query": query_text,
                        "document_ids": ["test-doc-1"]
                    }
                )
                
                response, execution_time = await self.measure_async_execution_time(query_coro)
                
                performance_results[query_type] = {
                    "execution_time": execution_time,
                    "status_code": response.status_code,
                    "query_length": len(query_text)
                }
                
                # Performance assertions
                if query_type == "simple":
                    assert execution_time < 1.0, f"Simple query took {execution_time:.2f}s"
                elif query_type == "medium":
                    assert execution_time < 2.0, f"Medium query took {execution_time:.2f}s"
                elif query_type == "complex":
                    assert execution_time < 3.0, f"Complex query took {execution_time:.2f}s"
                
                assert response.status_code == 200

        print("\n=== Query Performance Results ===")
        for query_type, results in performance_results.items():
            print(f"{query_type.upper()}: {results['execution_time']:.3f}s")

    @pytest.mark.asyncio
    async def test_health_check_performance(self):
        """Test health check endpoint performance."""
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Test basic health check
            health_coro = client.get("/health")
            response, execution_time = await self.measure_async_execution_time(health_coro)
            
            assert response.status_code == 200
            assert execution_time < 0.5, f"Health check took {execution_time:.2f}s, should be < 0.5s"
            
            # Test detailed health check
            detailed_health_coro = client.get("/health/detailed")
            detailed_response, detailed_time = await self.measure_async_execution_time(detailed_health_coro)
            
            assert detailed_response.status_code == 200
            assert detailed_time < 2.0, f"Detailed health check took {detailed_time:.2f}s, should be < 2.0s"

        print(f"\n=== Health Check Performance ===")
        print(f"Basic health check: {execution_time:.3f}s")
        print(f"Detailed health check: {detailed_time:.3f}s")

    @pytest.mark.asyncio
    async def test_memory_usage_during_processing(
        self,
        performance_pdf_files,
        mock_supabase_client,
        mock_llamaindex_service
    ):
        """Test memory usage during PDF processing."""
        
        # Mock responses
        mock_supabase_client.table().insert().execute.return_value = MagicMock(
            data=[{"id": "memory-test-id", "filename": "test.pdf", "status": "uploaded"}]
        )
        
        mock_llamaindex_service.process_document.return_value = {
            "chunks": 20,
            "embeddings_created": True,
            "index_updated": True
        }

        process = psutil.Process()
        initial_memory = process.memory_info().rss

        async with AsyncClient(app=app, base_url="http://test") as client:
            # Process a large file and monitor memory
            large_file_path = performance_pdf_files["large"]
            
            with open(large_file_path, "rb") as pdf_file:
                response = await client.post(
                    "/api/v1/upload",
                    files={"file": ("large_test.pdf", pdf_file, "application/pdf")}
                )
            
            peak_memory = process.memory_info().rss
            memory_increase = peak_memory - initial_memory
            memory_increase_mb = memory_increase / (1024 * 1024)
            
            assert response.status_code == 200
            # Memory increase should be reasonable (less than 500MB for test)
            assert memory_increase_mb < 500, f"Memory increase: {memory_increase_mb:.2f}MB"

        print(f"\n=== Memory Usage Test ===")
        print(f"Initial memory: {initial_memory / (1024*1024):.2f}MB")
        print(f"Peak memory: {peak_memory / (1024*1024):.2f}MB")
        print(f"Memory increase: {memory_increase_mb:.2f}MB")

    @pytest.mark.asyncio
    async def test_api_response_time_consistency(self):
        """Test API response time consistency over multiple requests."""
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response_times = []
            num_requests = 20
            
            for i in range(num_requests):
                start_time = time.perf_counter()
                response = await client.get("/health")
                end_time = time.perf_counter()
                
                response_times.append(end_time - start_time)
                assert response.status_code == 200
            
            # Calculate statistics
            avg_time = statistics.mean(response_times)
            std_dev = statistics.stdev(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
            
            # Consistency assertions
            assert avg_time < 0.5, f"Average response time: {avg_time:.3f}s"
            assert std_dev < 0.1, f"Response time std dev: {std_dev:.3f}s (should be consistent)"
            assert max_time < 1.0, f"Max response time: {max_time:.3f}s"

        print(f"\n=== API Response Time Consistency ({num_requests} requests) ===")
        print(f"Average: {avg_time:.3f}s")
        print(f"Std Dev: {std_dev:.3f}s")
        print(f"Min: {min_time:.3f}s")
        print(f"Max: {max_time:.3f}s")

    @pytest.mark.asyncio
    async def test_throughput_under_load(
        self,
        performance_pdf_files,
        mock_supabase_client,
        mock_llamaindex_service
    ):
        """Test system throughput under sustained load."""
        
        # Mock responses for batch processing
        mock_supabase_client.table().insert().execute.side_effect = [
            MagicMock(data=[{"id": f"load-test-{i}", "filename": f"load{i}.pdf", "status": "uploaded"}])
            for i in range(50)
        ]
        
        mock_llamaindex_service.process_document.return_value = {
            "chunks": 5,
            "embeddings_created": True,
            "index_updated": True
        }

        async def sustained_load_test(client, file_path, duration_seconds=10):
            """Run sustained load for specified duration."""
            start_time = time.perf_counter()
            end_time = start_time + duration_seconds
            completed_requests = 0
            
            while time.perf_counter() < end_time:
                try:
                    with open(file_path, "rb") as pdf_file:
                        response = await client.post(
                            "/api/v1/upload",
                            files={"file": (f"load_test_{completed_requests}.pdf", pdf_file, "application/pdf")}
                        )
                    
                    if response.status_code == 200:
                        completed_requests += 1
                    
                    # Small delay to prevent overwhelming
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    print(f"Request failed: {e}")
                    continue
            
            actual_duration = time.perf_counter() - start_time
            return completed_requests, actual_duration

        async with AsyncClient(app=app, base_url="http://test") as client:
            file_path = performance_pdf_files["small"]  # Use small files for throughput test
            test_duration = 5  # 5 seconds of sustained load
            
            completed_requests, actual_duration = await sustained_load_test(
                client, file_path, test_duration
            )
            
            throughput = completed_requests / actual_duration
            
            # Throughput assertions
            assert completed_requests > 0, "Should complete at least some requests"
            assert throughput > 1.0, f"Throughput: {throughput:.2f} requests/second should be > 1.0"

        print(f"\n=== Throughput Under Load ===")
        print(f"Duration: {actual_duration:.2f}s")
        print(f"Completed requests: {completed_requests}")
        print(f"Throughput: {throughput:.2f} requests/second")

    @pytest.mark.asyncio
    async def test_database_query_performance(
        self,
        mock_supabase_client
    ):
        """Test database query performance."""
        
        # Mock database responses with simulated delays
        def mock_db_query(*args, **kwargs):
            time.sleep(0.05)  # Simulate 50ms database query
            return MagicMock(data=[{"id": "test", "status": "completed"}])
        
        mock_supabase_client.table().select().execute.side_effect = mock_db_query

        async with AsyncClient(app=app, base_url="http://test") as client:
            query_times = []
            
            # Test multiple database queries
            for i in range(10):
                start_time = time.perf_counter()
                response = await client.get(f"/api/v1/documents/test-doc-{i}/status")
                end_time = time.perf_counter()
                
                query_times.append(end_time - start_time)
                assert response.status_code == 200
            
            avg_query_time = statistics.mean(query_times)
            max_query_time = max(query_times)
            
            # Database performance assertions
            assert avg_query_time < 1.0, f"Average DB query time: {avg_query_time:.3f}s"
            assert max_query_time < 2.0, f"Max DB query time: {max_query_time:.3f}s"

        print(f"\n=== Database Query Performance ===")
        print(f"Average query time: {avg_query_time:.3f}s")
        print(f"Max query time: {max_query_time:.3f}s")

    @pytest.mark.asyncio
    async def test_error_handling_performance(self):
        """Test that error handling doesn't significantly impact performance."""
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Test error response times
            error_response_times = []
            
            for i in range(5):
                start_time = time.perf_counter()
                # Request non-existent endpoint to trigger 404
                response = await client.get(f"/api/v1/nonexistent-endpoint-{i}")
                end_time = time.perf_counter()
                
                error_response_times.append(end_time - start_time)
                assert response.status_code == 404
            
            avg_error_time = statistics.mean(error_response_times)
            
            # Error handling should be fast
            assert avg_error_time < 0.5, f"Average error response time: {avg_error_time:.3f}s"

        print(f"\n=== Error Handling Performance ===")
        print(f"Average error response time: {avg_error_time:.3f}s")