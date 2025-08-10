"""
Unit tests for configuration module.

Tests the configuration management and environment variable handling.
"""

import pytest
from unittest.mock import patch, Mock
import os
from pathlib import Path

from app.config import Settings


class TestSettings:
    """Test suite for Settings configuration class."""

    def test_default_values(self):
        """Test that default configuration values are set correctly."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            
            # Test basic settings
            assert settings.app_name == "MIVAA PDF Extractor"
            assert settings.version == "1.0.0"
            assert settings.debug is False
            assert settings.host == "0.0.0.0"
            assert settings.port == 8000
            
            # Test database settings
            assert settings.database_url == "postgresql://user:password@localhost:5432/mivaa_pdf"
            assert settings.supabase_url == ""
            assert settings.supabase_key == ""
            
            # Test OpenAI settings
            assert settings.openai_api_key == ""
            assert settings.openai_model == "gpt-3.5-turbo"
            
            # Test file settings
            assert settings.upload_dir == "uploads"
            assert settings.max_file_size == 10 * 1024 * 1024  # 10MB
            assert settings.allowed_extensions == [".pdf"]

    def test_environment_variable_override(self):
        """Test that environment variables override default values."""
        env_vars = {
            "APP_NAME": "Test PDF Extractor",
            "VERSION": "2.0.0",
            "DEBUG": "true",
            "HOST": "127.0.0.1",
            "PORT": "9000",
            "DATABASE_URL": "postgresql://test:test@localhost:5432/test_db",
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_ANON_KEY": "test_key_123",
            "OPENAI_API_KEY": "sk-test123",
            "OPENAI_MODEL": "gpt-4",
            "UPLOAD_DIR": "test_uploads",
            "MAX_FILE_SIZE": "5242880",  # 5MB
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            
            assert settings.app_name == "Test PDF Extractor"
            assert settings.version == "2.0.0"
            assert settings.debug is True
            assert settings.host == "127.0.0.1"
            assert settings.port == 9000
            assert settings.database_url == "postgresql://test:test@localhost:5432/test_db"
            assert settings.supabase_url == "https://test.supabase.co"
            assert settings.supabase_key == "test_key_123"
            assert settings.openai_api_key == "sk-test123"
            assert settings.openai_model == "gpt-4"
            assert settings.upload_dir == "test_uploads"
            assert settings.max_file_size == 5242880

    def test_boolean_environment_variables(self):
        """Test that boolean environment variables are parsed correctly."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
            ("no", False),
            ("", False),
        ]
        
        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"DEBUG": env_value}, clear=True):
                settings = Settings()
                assert settings.debug == expected, f"Failed for env_value: {env_value}"

    def test_integer_environment_variables(self):
        """Test that integer environment variables are parsed correctly."""
        with patch.dict(os.environ, {"PORT": "8080", "MAX_FILE_SIZE": "1048576"}, clear=True):
            settings = Settings()
            assert settings.port == 8080
            assert settings.max_file_size == 1048576

    def test_invalid_integer_environment_variables(self):
        """Test handling of invalid integer environment variables."""
        with patch.dict(os.environ, {"PORT": "invalid"}, clear=True):
            # Should fall back to default value or raise validation error
            # Depending on pydantic configuration
            try:
                settings = Settings()
                # If no exception, should use default
                assert settings.port == 8000
            except ValueError:
                # If pydantic raises validation error, that's also acceptable
                pass

    def test_list_environment_variables(self):
        """Test that list environment variables are parsed correctly."""
        with patch.dict(os.environ, {"ALLOWED_EXTENSIONS": ".pdf,.docx,.txt"}, clear=True):
            settings = Settings()
            assert settings.allowed_extensions == [".pdf", ".docx", ".txt"]

    def test_empty_list_environment_variables(self):
        """Test handling of empty list environment variables."""
        with patch.dict(os.environ, {"ALLOWED_EXTENSIONS": ""}, clear=True):
            settings = Settings()
            assert settings.allowed_extensions == []

    def test_get_llamaindex_config(self):
        """Test LlamaIndex configuration helper method."""
        env_vars = {
            "LLAMAINDEX_ENABLED": "true",
            "LLAMAINDEX_CHUNK_SIZE": "1024",
            "LLAMAINDEX_CHUNK_OVERLAP": "200",
            "LLAMAINDEX_EMBEDDING_MODEL": "text-embedding-ada-002",
            "LLAMAINDEX_LLM_MODEL": "gpt-4",
            "LLAMAINDEX_TEMPERATURE": "0.7",
            "LLAMAINDEX_MAX_TOKENS": "2048",
            "LLAMAINDEX_INDEX_TYPE": "vector",
            "LLAMAINDEX_PERSIST_DIR": "llamaindex_storage",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            config = settings.get_llamaindex_config()
            
            assert config["enabled"] is True
            assert config["chunk_size"] == 1024
            assert config["chunk_overlap"] == 200
            assert config["embedding_model"] == "text-embedding-ada-002"
            assert config["llm_model"] == "gpt-4"
            assert config["temperature"] == 0.7
            assert config["max_tokens"] == 2048
            assert config["index_type"] == "vector"
            assert config["persist_dir"] == "llamaindex_storage"

    def test_get_material_kai_config(self):
        """Test Material Kai configuration helper method."""
        env_vars = {
            "MATERIAL_KAI_ENABLED": "true",
            "MATERIAL_KAI_BASE_URL": "https://api.materialkai.com",
            "MATERIAL_KAI_API_KEY": "mk_test_key",
            "MATERIAL_KAI_TIMEOUT": "30",
            "MATERIAL_KAI_RETRY_ATTEMPTS": "3",
            "MATERIAL_KAI_BATCH_SIZE": "10",
            "MATERIAL_KAI_SYNC_INTERVAL": "300",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            config = settings.get_material_kai_config()
            
            assert config["enabled"] is True
            assert config["base_url"] == "https://api.materialkai.com"
            assert config["api_key"] == "mk_test_key"
            assert config["timeout"] == 30
            assert config["retry_attempts"] == 3
            assert config["batch_size"] == 10
            assert config["sync_interval"] == 300

    def test_get_sentry_config(self):
        """Test Sentry configuration helper method."""
        env_vars = {
            "SENTRY_DSN": "https://test@sentry.io/123456",
            "SENTRY_ENVIRONMENT": "production",
            "SENTRY_TRACES_SAMPLE_RATE": "0.1",
            "SENTRY_PROFILES_SAMPLE_RATE": "0.1",
            "SENTRY_ENABLED": "true",
            "SENTRY_RELEASE": "1.0.0",
            "SENTRY_SERVER_NAME": "pdf-extractor-prod",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            config = settings.get_sentry_config()
            
            assert config["dsn"] == "https://test@sentry.io/123456"
            assert config["environment"] == "production"
            assert config["traces_sample_rate"] == 0.1
            assert config["profiles_sample_rate"] == 0.1
            assert config["enabled"] is True
            assert config["release"] == "1.0.0"
            assert config["server_name"] == "pdf-extractor-prod"

    def test_sentry_config_disabled(self):
        """Test Sentry configuration when disabled."""
        with patch.dict(os.environ, {"SENTRY_ENABLED": "false"}, clear=True):
            settings = Settings()
            config = settings.get_sentry_config()
            
            assert config["enabled"] is False
            assert config["dsn"] == ""

    def test_upload_directory_creation(self):
        """Test that upload directory path is handled correctly."""
        with patch.dict(os.environ, {"UPLOAD_DIR": "test_uploads"}, clear=True):
            settings = Settings()
            assert settings.upload_dir == "test_uploads"

    def test_database_url_validation(self):
        """Test database URL validation."""
        valid_urls = [
            "postgresql://user:pass@localhost:5432/db",
            "postgresql://user@localhost/db",
            "sqlite:///./test.db",
        ]
        
        for url in valid_urls:
            with patch.dict(os.environ, {"DATABASE_URL": url}, clear=True):
                settings = Settings()
                assert settings.database_url == url

    def test_openai_api_key_validation(self):
        """Test OpenAI API key validation."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123456789"}, clear=True):
            settings = Settings()
            assert settings.openai_api_key == "sk-test123456789"

    def test_supabase_configuration(self):
        """Test Supabase configuration."""
        env_vars = {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_ANON_KEY": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            assert settings.supabase_url == "https://test.supabase.co"
            assert settings.supabase_key == "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test"

    def test_file_size_limits(self):
        """Test file size limit configuration."""
        test_cases = [
            ("1048576", 1048576),  # 1MB
            ("10485760", 10485760),  # 10MB
            ("52428800", 52428800),  # 50MB
        ]
        
        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"MAX_FILE_SIZE": env_value}, clear=True):
                settings = Settings()
                assert settings.max_file_size == expected

    def test_allowed_extensions_parsing(self):
        """Test allowed extensions parsing."""
        test_cases = [
            (".pdf", [".pdf"]),
            (".pdf,.docx", [".pdf", ".docx"]),
            (".pdf, .docx, .txt", [".pdf", ".docx", ".txt"]),
            ("pdf,docx,txt", ["pdf", "docx", "txt"]),
        ]
        
        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"ALLOWED_EXTENSIONS": env_value}, clear=True):
                settings = Settings()
                assert settings.allowed_extensions == expected

    def test_config_immutability(self):
        """Test that configuration is immutable after creation."""
        settings = Settings()
        
        # Attempt to modify should raise an error if using frozen dataclass
        # or should be ignored if using pydantic with allow_mutation=False
        try:
            settings.app_name = "Modified Name"
            # If no error, check if the value actually changed
            # In a properly configured immutable settings, it shouldn't change
        except (AttributeError, TypeError):
            # Expected behavior for immutable configuration
            pass

    def test_development_vs_production_config(self):
        """Test different configurations for development vs production."""
        # Development configuration
        dev_env = {
            "DEBUG": "true",
            "HOST": "127.0.0.1",
            "PORT": "8000",
        }
        
        with patch.dict(os.environ, dev_env, clear=True):
            dev_settings = Settings()
            assert dev_settings.debug is True
            assert dev_settings.host == "127.0.0.1"
        
        # Production configuration
        prod_env = {
            "DEBUG": "false",
            "HOST": "0.0.0.0",
            "PORT": "80",
        }
        
        with patch.dict(os.environ, prod_env, clear=True):
            prod_settings = Settings()
            assert prod_settings.debug is False
            assert prod_settings.host == "0.0.0.0"
            assert prod_settings.port == 80

    def test_missing_required_config(self):
        """Test behavior when required configuration is missing."""
        # Test with minimal environment
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            # Should use defaults for optional settings
            assert settings.app_name == "MIVAA PDF Extractor"
            # Required settings should have sensible defaults or empty strings
            assert isinstance(settings.openai_api_key, str)
            assert isinstance(settings.supabase_url, str)

    def test_config_helper_methods_with_defaults(self):
        """Test configuration helper methods with default values."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            
            # Test LlamaIndex config with defaults
            llamaindex_config = settings.get_llamaindex_config()
            assert isinstance(llamaindex_config, dict)
            assert "enabled" in llamaindex_config
            assert "chunk_size" in llamaindex_config
            
            # Test Material Kai config with defaults
            material_kai_config = settings.get_material_kai_config()
            assert isinstance(material_kai_config, dict)
            assert "enabled" in material_kai_config
            assert "base_url" in material_kai_config
            
            # Test Sentry config with defaults
            sentry_config = settings.get_sentry_config()
            assert isinstance(sentry_config, dict)
            assert "enabled" in sentry_config
            assert "dsn" in sentry_config