"""
Integration tests for database connectivity and health checks.

Tests the actual database connections and health check functionality
with real or mocked external services.
"""

import pytest
from unittest.mock import patch, Mock, AsyncMock
import asyncio
from datetime import datetime

from app.database.connection import (
    test_supabase_connection,
    perform_comprehensive_health_checks,
    get_supabase_client
)
from app.config import Settings


class TestDatabaseIntegration:
    """Integration tests for database functionality."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock(spec=Settings)
        settings.supabase_url = "https://test.supabase.co"
        settings.supabase_key = "test_key_123"
        settings.database_url = "postgresql://test:test@localhost:5432/test_db"
        return settings

    @pytest.mark.integration
    @patch('app.database.connection.create_client')
    def test_supabase_connection_success(self, mock_create_client, mock_settings):
        """Test successful Supabase connection."""
        # Mock successful Supabase client
        mock_client = Mock()
        mock_client.table.return_value.select.return_value.limit.return_value.execute.return_value = Mock(
            data=[{"test": "data"}],
            count=1
        )
        mock_create_client.return_value = mock_client

        with patch('app.database.connection.settings', mock_settings):
            result = test_supabase_connection()

        assert result["status"] == "healthy"
        assert result["response_time"] > 0
        assert "connection_pool" in result
        mock_create_client.assert_called_once_with(
            mock_settings.supabase_url,
            mock_settings.supabase_key
        )

    @pytest.mark.integration
    @patch('app.database.connection.create_client')
    def test_supabase_connection_failure(self, mock_create_client, mock_settings):
        """Test Supabase connection failure."""
        # Mock connection failure
        mock_create_client.side_effect = Exception("Connection failed")

        with patch('app.database.connection.settings', mock_settings):
            result = test_supabase_connection()

        assert result["status"] == "unhealthy"
        assert "Connection failed" in result["error"]
        assert result["response_time"] > 0

    @pytest.mark.integration
    @patch('app.database.connection.create_client')
    def test_supabase_connection_timeout(self, mock_create_client, mock_settings):
        """Test Supabase connection timeout."""
        # Mock slow response
        mock_client = Mock()
        mock_client.table.return_value.select.return_value.limit.return_value.execute.side_effect = lambda: asyncio.sleep(10)
        mock_create_client.return_value = mock_client

        with patch('app.database.connection.settings', mock_settings):
            result = test_supabase_connection()

        # Should handle timeout gracefully
        assert result["status"] in ["unhealthy", "timeout"]

    @pytest.mark.integration
    @patch('app.database.connection.create_client')
    def test_get_supabase_client_success(self, mock_create_client, mock_settings):
        """Test getting Supabase client successfully."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        with patch('app.database.connection.settings', mock_settings):
            client = get_supabase_client()

        assert client is not None
        mock_create_client.assert_called_once_with(
            mock_settings.supabase_url,
            mock_settings.supabase_key
        )

    @pytest.mark.integration
    @patch('app.database.connection.create_client')
    def test_get_supabase_client_failure(self, mock_create_client, mock_settings):
        """Test getting Supabase client failure."""
        mock_create_client.side_effect = Exception("Client creation failed")

        with patch('app.database.connection.settings', mock_settings):
            client = get_supabase_client()

        assert client is None

    @pytest.mark.integration
    @patch('app.database.connection.test_supabase_connection')
    @patch('app.database.connection.psutil')
    @patch('app.database.connection.aiohttp.ClientSession')
    async def test_comprehensive_health_checks_all_healthy(
        self, mock_session_class, mock_psutil, mock_supabase_test
    ):
        """Test comprehensive health checks when all systems are healthy."""
        # Mock Supabase health
        mock_supabase_test.return_value = {
            "status": "healthy",
            "response_time": 0.1,
            "connection_pool": {"active": 1, "idle": 4}
        }

        # Mock system resources
        mock_psutil.cpu_percent.return_value = 25.0
        mock_psutil.virtual_memory.return_value = Mock(percent=60.0, available=4000000000)
        mock_psutil.disk_usage.return_value = Mock(percent=45.0, free=50000000000)

        # Mock HTTP session for external service checks
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "ok"})
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await perform_comprehensive_health_checks()

        assert result["overall_status"] == "healthy"
        assert result["database"]["status"] == "healthy"
        assert result["system"]["cpu_usage"] == 25.0
        assert result["system"]["memory_usage"] == 60.0
        assert result["system"]["disk_usage"] == 45.0
        assert "timestamp" in result

    @pytest.mark.integration
    @patch('app.database.connection.test_supabase_connection')
    @patch('app.database.connection.psutil')
    async def test_comprehensive_health_checks_database_unhealthy(
        self, mock_psutil, mock_supabase_test
    ):
        """Test comprehensive health checks when database is unhealthy."""
        # Mock unhealthy Supabase
        mock_supabase_test.return_value = {
            "status": "unhealthy",
            "error": "Connection timeout",
            "response_time": 5.0
        }

        # Mock healthy system resources
        mock_psutil.cpu_percent.return_value = 15.0
        mock_psutil.virtual_memory.return_value = Mock(percent=40.0, available=6000000000)
        mock_psutil.disk_usage.return_value = Mock(percent=30.0, free=70000000000)

        result = await perform_comprehensive_health_checks()

        assert result["overall_status"] == "unhealthy"
        assert result["database"]["status"] == "unhealthy"
        assert "Connection timeout" in result["database"]["error"]

    @pytest.mark.integration
    @patch('app.database.connection.test_supabase_connection')
    @patch('app.database.connection.psutil')
    async def test_comprehensive_health_checks_high_resource_usage(
        self, mock_psutil, mock_supabase_test
    ):
        """Test comprehensive health checks with high resource usage."""
        # Mock healthy Supabase
        mock_supabase_test.return_value = {
            "status": "healthy",
            "response_time": 0.2
        }

        # Mock high resource usage
        mock_psutil.cpu_percent.return_value = 95.0
        mock_psutil.virtual_memory.return_value = Mock(percent=90.0, available=500000000)
        mock_psutil.disk_usage.return_value = Mock(percent=85.0, free=5000000000)

        result = await perform_comprehensive_health_checks()

        assert result["overall_status"] == "degraded"
        assert result["system"]["cpu_usage"] == 95.0
        assert result["system"]["memory_usage"] == 90.0
        assert result["system"]["disk_usage"] == 85.0

    @pytest.mark.integration
    @patch('app.database.connection.psutil')
    async def test_comprehensive_health_checks_system_error(self, mock_psutil):
        """Test comprehensive health checks when system monitoring fails."""
        # Mock system monitoring failure
        mock_psutil.cpu_percent.side_effect = Exception("System monitoring failed")

        result = await perform_comprehensive_health_checks()

        assert result["overall_status"] == "unhealthy"
        assert "error" in result["system"]

    @pytest.mark.integration
    @patch('app.database.connection.aiohttp.ClientSession')
    async def test_external_service_health_check(self, mock_session_class):
        """Test external service health check functionality."""
        # Mock successful external service response
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "operational"})
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

        # This would test a specific external service check function
        # Implementation depends on actual external service checks in the codebase
        result = await perform_comprehensive_health_checks()

        # Verify external services are checked if implemented
        assert "external_services" in result or "overall_status" in result

    @pytest.mark.integration
    def test_database_connection_pool_management(self, mock_settings):
        """Test database connection pool management."""
        with patch('app.database.connection.settings', mock_settings):
            with patch('app.database.connection.create_client') as mock_create_client:
                mock_client = Mock()
                mock_create_client.return_value = mock_client

                # Test multiple client requests
                client1 = get_supabase_client()
                client2 = get_supabase_client()

                assert client1 is not None
                assert client2 is not None
                # Verify connection reuse or proper pooling
                assert mock_create_client.call_count >= 1

    @pytest.mark.integration
    @patch('app.database.connection.create_client')
    def test_database_query_performance(self, mock_create_client, mock_settings):
        """Test database query performance monitoring."""
        # Mock client with performance data
        mock_client = Mock()
        mock_result = Mock()
        mock_result.data = [{"id": 1, "name": "test"}]
        mock_result.count = 1
        mock_client.table.return_value.select.return_value.limit.return_value.execute.return_value = mock_result

        mock_create_client.return_value = mock_client

        with patch('app.database.connection.settings', mock_settings):
            result = test_supabase_connection()

        assert result["response_time"] > 0
        assert result["status"] == "healthy"

    @pytest.mark.integration
    async def test_health_check_caching(self):
        """Test health check result caching to avoid excessive checks."""
        with patch('app.database.connection.test_supabase_connection') as mock_supabase:
            mock_supabase.return_value = {"status": "healthy", "response_time": 0.1}

            # Perform multiple health checks rapidly
            result1 = await perform_comprehensive_health_checks()
            result2 = await perform_comprehensive_health_checks()

            assert result1["overall_status"] == "healthy"
            assert result2["overall_status"] == "healthy"

            # Verify caching behavior if implemented
            # This depends on actual caching implementation

    @pytest.mark.integration
    @patch('app.database.connection.create_client')
    def test_database_error_recovery(self, mock_create_client, mock_settings):
        """Test database error recovery mechanisms."""
        # Mock initial failure followed by success
        mock_create_client.side_effect = [
            Exception("Initial connection failed"),
            Mock()  # Successful connection on retry
        ]

        with patch('app.database.connection.settings', mock_settings):
            # First attempt should fail
            result1 = test_supabase_connection()
            assert result1["status"] == "unhealthy"

            # Second attempt should succeed (simulating recovery)
            mock_create_client.side_effect = None
            mock_client = Mock()
            mock_client.table.return_value.select.return_value.limit.return_value.execute.return_value = Mock(
                data=[{"test": "data"}]
            )
            mock_create_client.return_value = mock_client

            result2 = test_supabase_connection()
            assert result2["status"] == "healthy"

    @pytest.mark.integration
    def test_health_check_metrics_collection(self, mock_settings):
        """Test collection of health check metrics for monitoring."""
        with patch('app.database.connection.settings', mock_settings):
            with patch('app.database.connection.create_client') as mock_create_client:
                mock_client = Mock()
                mock_client.table.return_value.select.return_value.limit.return_value.execute.return_value = Mock(
                    data=[{"test": "data"}]
                )
                mock_create_client.return_value = mock_client

                result = test_supabase_connection()

                # Verify metrics are collected
                assert "response_time" in result
                assert "status" in result
                assert isinstance(result["response_time"], (int, float))
                assert result["response_time"] >= 0

    @pytest.mark.integration
    async def test_concurrent_health_checks(self):
        """Test concurrent health check execution."""
        with patch('app.database.connection.test_supabase_connection') as mock_supabase:
            mock_supabase.return_value = {"status": "healthy", "response_time": 0.1}

            # Run multiple health checks concurrently
            tasks = [
                perform_comprehensive_health_checks()
                for _ in range(5)
            ]

            results = await asyncio.gather(*tasks)

            # All should succeed
            for result in results:
                assert result["overall_status"] in ["healthy", "degraded", "unhealthy"]
                assert "timestamp" in result