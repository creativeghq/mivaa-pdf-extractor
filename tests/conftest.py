"""
Shared pytest configuration and fixtures for MIVAA PDF Extractor tests.

This module provides common test fixtures, configuration, and utilities
used across all test modules in the test suite.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Import application components
from app.main import create_app
from app.config import get_settings
from app.database.connection import get_supabase_client


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings():
    """Provide test-specific settings configuration."""
    # Override settings for testing
    os.environ.update({
        "DEBUG": "true",
        "TESTING": "true",
        "DATABASE_URL": "postgresql://test:test@localhost:5432/test_db",
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_ANON_KEY": "test_anon_key",
        "SUPABASE_SERVICE_ROLE_KEY": "test_service_key",
        "OPENAI_API_KEY": "test_openai_key",
        "SENTRY_ENABLED": "false",  # Disable Sentry in tests
        "LLAMAINDEX_ENABLED": "false",  # Disable by default for unit tests
        "MATERIAL_KAI_ENABLED": "false",  # Disable by default for unit tests
    })
    return get_settings()


@pytest.fixture
def app(test_settings):
    """Create FastAPI application instance for testing."""
    return create_app()


@pytest.fixture
def client(app) -> Generator[TestClient, None, None]:
    """Create synchronous test client for FastAPI application."""
    with TestClient(app) as test_client:
        yield test_client


@pytest_asyncio.fixture
async def async_client(app) -> AsyncGenerator[AsyncClient, None]:
    """Create asynchronous test client for FastAPI application."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_supabase_client():
    """Provide a mocked Supabase client for testing."""
    mock_client = MagicMock()
    
    # Mock common Supabase operations
    mock_client.table.return_value.select.return_value.execute.return_value = MagicMock(
        data=[], count=0
    )
    mock_client.table.return_value.insert.return_value.execute.return_value = MagicMock(
        data=[{"id": 1}], count=1
    )
    mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock(
        data=[{"id": 1}], count=1
    )
    mock_client.table.return_value.delete.return_value.eq.return_value.execute.return_value = MagicMock(
        data=[], count=1
    )
    
    return mock_client


@pytest.fixture
def mock_openai_client():
    """Provide a mocked OpenAI client for testing."""
    mock_client = AsyncMock()
    
    # Mock embeddings response
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1536)]
    )
    
    # Mock chat completions response
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Test response"))]
    )
    
    return mock_client


@pytest.fixture
def sample_pdf_file():
    """Create a temporary PDF file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        # Create a minimal PDF content (this is a very basic PDF structure)
        pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Test PDF Content) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000204 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
297
%%EOF"""
        tmp_file.write(pdf_content)
        tmp_file.flush()
        
        yield tmp_file.name
        
        # Cleanup
        try:
            os.unlink(tmp_file.name)
        except OSError:
            pass


@pytest.fixture
def sample_markdown_content():
    """Provide sample markdown content for testing."""
    return """# Test Document

This is a test document with various elements.

## Section 1

Some text content with **bold** and *italic* formatting.

### Subsection 1.1

- List item 1
- List item 2
- List item 3

## Section 2

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
| Data 4   | Data 5   | Data 6   |

### Code Example

```python
def hello_world():
    print("Hello, World!")
```

## Conclusion

This is the end of the test document.
"""


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing file operations."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_llamaindex_service():
    """Provide a mocked LlamaIndex service for testing."""
    mock_service = AsyncMock()
    
    # Mock service methods
    mock_service.index_document.return_value = {"status": "success", "document_id": "test_doc_1"}
    mock_service.search_documents.return_value = {
        "results": [
            {"content": "Test result 1", "score": 0.95},
            {"content": "Test result 2", "score": 0.87}
        ]
    }
    mock_service.ask_question.return_value = {
        "answer": "Test answer",
        "sources": ["doc1", "doc2"]
    }
    mock_service.summarize_document.return_value = {
        "summary": "Test summary of the document"
    }
    
    return mock_service


@pytest.fixture
def mock_material_kai_service():
    """Provide a mocked Material Kai service for testing."""
    mock_service = AsyncMock()
    
    # Mock service methods
    mock_service.sync_document.return_value = {"status": "synced", "sync_id": "sync_123"}
    mock_service.register_service.return_value = {"status": "registered", "service_id": "service_456"}
    mock_service.send_notification.return_value = {"status": "sent", "notification_id": "notif_789"}
    
    return mock_service


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically cleanup any test files created during testing."""
    yield
    
    # Cleanup logic can be added here if needed
    # For now, we rely on temporary file/directory fixtures for cleanup


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "external: mark test as requiring external services"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)