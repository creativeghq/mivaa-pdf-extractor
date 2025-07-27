"""
Unit tests for Configuration Manager service.

Tests the ConfigManager class in isolation using mocks for external dependencies.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from typing import Dict, Any, Optional
import json
import os
import tempfile

from app.services.config_manager import ConfigManager


class TestConfigManager:
    """Test suite for ConfigManager class."""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration data for testing."""
        return {
            "database": {
                "supabase_url": "https://test.supabase.co",
                "supabase_key": "test_key",
                "connection_pool_size": 10,
                "timeout": 30
            },
            "openai": {
                "api_key": "test_openai_key",
                "model": "gpt-4",
                "embedding_model": "text-embedding-3-large",
                "embedding_dimensions": 768,
                "max_tokens": 4000
            },
            "material_kai": {
                "base_url": "https://api.materialkai.com",
                "api_key": "test_mk_key",
                "timeout": 60,
                "max_retries": 3
            },
            "processing": {
                "max_file_size": 50 * 1024 * 1024,  # 50MB
                "supported_formats": ["pdf", "png", "jpg", "jpeg"],
                "concurrent_jobs": 5,
                "temp_dir": "/tmp/processing"
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "app.log",
                "max_size": "10MB",
                "backup_count": 5
            },
            "security": {
                "secret_key": "test_secret_key",
                "algorithm": "HS256",
                "access_token_expire_minutes": 30,
                "allowed_origins": ["http://localhost:3000"]
            }
        }

    @pytest.fixture
    def config_manager(self, sample_config):
        """Create a ConfigManager instance with mocked dependencies."""
        with patch('app.services.config_manager.os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(sample_config))):
            manager = ConfigManager()
            return manager

    def test_initialization_success(self, config_manager):
        """Test successful ConfigManager initialization."""
        assert config_manager.config is not None
        assert "database" in config_manager.config
        assert "openai" in config_manager.config
        assert config_manager.config["database"]["supabase_url"] == "https://test.supabase.co"

    def test_initialization_file_not_found(self):
        """Test initialization when config file doesn't exist."""
        with patch('app.services.config_manager.os.path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                ConfigManager()

    def test_initialization_invalid_json(self):
        """Test initialization with invalid JSON config file."""
        with patch('app.services.config_manager.os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data="invalid json")):
            with pytest.raises(json.JSONDecodeError):
                ConfigManager()

    def test_get_config_section_success(self, config_manager):
        """Test successful retrieval of config section."""
        db_config = config_manager.get_config_section("database")
        
        assert db_config is not None
        assert db_config["supabase_url"] == "https://test.supabase.co"
        assert db_config["connection_pool_size"] == 10

    def test_get_config_section_not_found(self, config_manager):
        """Test retrieval of non-existent config section."""
        result = config_manager.get_config_section("nonexistent")
        assert result is None

    def test_get_config_value_success(self, config_manager):
        """Test successful retrieval of specific config value."""
        value = config_manager.get_config_value("database.supabase_url")
        assert value == "https://test.supabase.co"
        
        value = config_manager.get_config_value("openai.embedding_dimensions")
        assert value == 768

    def test_get_config_value_with_default(self, config_manager):
        """Test retrieval of config value with default fallback."""
        value = config_manager.get_config_value("nonexistent.key", default="default_value")
        assert value == "default_value"

    def test_get_config_value_nested_not_found(self, config_manager):
        """Test retrieval of non-existent nested config value."""
        value = config_manager.get_config_value("database.nonexistent_key")
        assert value is None

    def test_set_config_value_success(self, config_manager):
        """Test successful setting of config value."""
        config_manager.set_config_value("database.timeout", 60)
        
        value = config_manager.get_config_value("database.timeout")
        assert value == 60

    def test_set_config_value_new_section(self, config_manager):
        """Test setting config value in new section."""
        config_manager.set_config_value("new_section.new_key", "new_value")
        
        value = config_manager.get_config_value("new_section.new_key")
        assert value == "new_value"

    def test_update_config_section_success(self, config_manager):
        """Test successful update of entire config section."""
        new_db_config = {
            "supabase_url": "https://new.supabase.co",
            "supabase_key": "new_key",
            "connection_pool_size": 20
        }
        
        config_manager.update_config_section("database", new_db_config)
        
        db_config = config_manager.get_config_section("database")
        assert db_config["supabase_url"] == "https://new.supabase.co"
        assert db_config["connection_pool_size"] == 20

    def test_validate_config_success(self, config_manager):
        """Test successful config validation."""
        result = config_manager.validate_config()
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_config_missing_required(self, config_manager):
        """Test config validation with missing required fields."""
        # Remove required field
        del config_manager.config["database"]["supabase_url"]
        
        result = config_manager.validate_config()
        
        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert any("supabase_url" in error for error in result["errors"])

    def test_validate_config_invalid_type(self, config_manager):
        """Test config validation with invalid data types."""
        # Set invalid type
        config_manager.config["database"]["connection_pool_size"] = "invalid"
        
        result = config_manager.validate_config()
        
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_get_database_config(self, config_manager):
        """Test database configuration retrieval."""
        db_config = config_manager.get_database_config()
        
        assert db_config["url"] == "https://test.supabase.co"
        assert db_config["key"] == "test_key"
        assert db_config["pool_size"] == 10

    def test_get_openai_config(self, config_manager):
        """Test OpenAI configuration retrieval."""
        openai_config = config_manager.get_openai_config()
        
        assert openai_config["api_key"] == "test_openai_key"
        assert openai_config["model"] == "gpt-4"
        assert openai_config["embedding_dimensions"] == 768

    def test_get_material_kai_config(self, config_manager):
        """Test Material Kai configuration retrieval."""
        mk_config = config_manager.get_material_kai_config()
        
        assert mk_config["base_url"] == "https://api.materialkai.com"
        assert mk_config["api_key"] == "test_mk_key"
        assert mk_config["timeout"] == 60

    def test_get_processing_config(self, config_manager):
        """Test processing configuration retrieval."""
        proc_config = config_manager.get_processing_config()
        
        assert proc_config["max_file_size"] == 50 * 1024 * 1024
        assert "pdf" in proc_config["supported_formats"]
        assert proc_config["concurrent_jobs"] == 5

    def test_get_logging_config(self, config_manager):
        """Test logging configuration retrieval."""
        log_config = config_manager.get_logging_config()
        
        assert log_config["level"] == "INFO"
        assert log_config["file"] == "app.log"
        assert log_config["backup_count"] == 5

    def test_get_security_config(self, config_manager):
        """Test security configuration retrieval."""
        sec_config = config_manager.get_security_config()
        
        assert sec_config["secret_key"] == "test_secret_key"
        assert sec_config["algorithm"] == "HS256"
        assert sec_config["access_token_expire_minutes"] == 30

    def test_is_development_mode(self, config_manager):
        """Test development mode detection."""
        # Test default (should be False)
        assert config_manager.is_development_mode() is False
        
        # Set development mode
        config_manager.set_config_value("environment", "development")
        assert config_manager.is_development_mode() is True

    def test_is_production_mode(self, config_manager):
        """Test production mode detection."""
        # Test default (should be True)
        assert config_manager.is_production_mode() is True
        
        # Set development mode
        config_manager.set_config_value("environment", "development")
        assert config_manager.is_production_mode() is False

    def test_get_environment_variables(self, config_manager):
        """Test environment variables retrieval."""
        with patch.dict(os.environ, {
            'SUPABASE_URL': 'env_supabase_url',
            'OPENAI_API_KEY': 'env_openai_key'
        }):
            env_vars = config_manager.get_environment_variables()
            
            assert 'SUPABASE_URL' in env_vars
            assert env_vars['SUPABASE_URL'] == 'env_supabase_url'

    def test_merge_environment_variables(self, config_manager):
        """Test merging environment variables with config."""
        with patch.dict(os.environ, {
            'SUPABASE_URL': 'env_supabase_url',
            'OPENAI_API_KEY': 'env_openai_key'
        }):
            config_manager.merge_environment_variables()
            
            # Environment variables should override config values
            assert config_manager.get_config_value("database.supabase_url") == "env_supabase_url"

    def test_save_config_success(self, config_manager):
        """Test successful config saving."""
        with patch('builtins.open', mock_open()) as mock_file:
            result = config_manager.save_config()
            
            assert result["success"] is True
            mock_file.assert_called_once()

    def test_save_config_failure(self, config_manager):
        """Test config saving failure."""
        with patch('builtins.open', side_effect=IOError("Permission denied")):
            result = config_manager.save_config()
            
            assert result["success"] is False
            assert "Permission denied" in result["error"]

    def test_backup_config_success(self, config_manager):
        """Test successful config backup."""
        with patch('builtins.open', mock_open()) as mock_file, \
             patch('app.services.config_manager.shutil.copy2') as mock_copy:
            
            result = config_manager.backup_config()
            
            assert result["success"] is True
            assert "backup_path" in result
            mock_copy.assert_called_once()

    def test_restore_config_success(self, config_manager, sample_config):
        """Test successful config restoration."""
        backup_path = "/tmp/config_backup.json"
        
        with patch('app.services.config_manager.os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(sample_config))):
            
            result = config_manager.restore_config(backup_path)
            
            assert result["success"] is True

    def test_restore_config_file_not_found(self, config_manager):
        """Test config restoration with missing backup file."""
        backup_path = "/tmp/nonexistent_backup.json"
        
        with patch('app.services.config_manager.os.path.exists', return_value=False):
            result = config_manager.restore_config(backup_path)
            
            assert result["success"] is False
            assert "not found" in result["error"].lower()

    def test_reset_to_defaults(self, config_manager):
        """Test resetting config to default values."""
        # Modify some values
        config_manager.set_config_value("database.timeout", 999)
        config_manager.set_config_value("openai.max_tokens", 999)
        
        # Reset to defaults
        result = config_manager.reset_to_defaults()
        
        assert result["success"] is True
        # Values should be reset (assuming defaults are different)
        assert config_manager.get_config_value("database.timeout") != 999

    def test_get_config_schema(self, config_manager):
        """Test config schema retrieval."""
        schema = config_manager.get_config_schema()
        
        assert "database" in schema
        assert "openai" in schema
        assert "required" in schema["database"]
        assert "properties" in schema["database"]

    def test_validate_against_schema(self, config_manager):
        """Test config validation against schema."""
        result = config_manager.validate_against_schema()
        
        assert "valid" in result
        assert "errors" in result

    def test_get_sensitive_keys(self, config_manager):
        """Test retrieval of sensitive configuration keys."""
        sensitive_keys = config_manager.get_sensitive_keys()
        
        assert "database.supabase_key" in sensitive_keys
        assert "openai.api_key" in sensitive_keys
        assert "security.secret_key" in sensitive_keys

    def test_mask_sensitive_values(self, config_manager):
        """Test masking of sensitive configuration values."""
        masked_config = config_manager.mask_sensitive_values()
        
        # Sensitive values should be masked
        assert masked_config["database"]["supabase_key"] == "***"
        assert masked_config["openai"]["api_key"] == "***"
        
        # Non-sensitive values should remain
        assert masked_config["database"]["supabase_url"] == "https://test.supabase.co"

    def test_export_config(self, config_manager):
        """Test config export functionality."""
        with patch('builtins.open', mock_open()) as mock_file:
            result = config_manager.export_config("/tmp/export.json", include_sensitive=False)
            
            assert result["success"] is True
            mock_file.assert_called_once()

    def test_import_config(self, config_manager, sample_config):
        """Test config import functionality."""
        import_path = "/tmp/import.json"
        
        with patch('app.services.config_manager.os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(sample_config))):
            
            result = config_manager.import_config(import_path)
            
            assert result["success"] is True

    def test_get_config_diff(self, config_manager, sample_config):
        """Test configuration difference detection."""
        # Create a modified config
        modified_config = sample_config.copy()
        modified_config["database"]["timeout"] = 60
        modified_config["new_section"] = {"new_key": "new_value"}
        
        diff = config_manager.get_config_diff(modified_config)
        
        assert "changes" in diff
        assert "additions" in diff
        assert "deletions" in diff

    def test_apply_config_patch(self, config_manager):
        """Test applying configuration patches."""
        patch_data = {
            "database.timeout": 60,
            "openai.max_tokens": 8000,
            "new_section.new_key": "new_value"
        }
        
        result = config_manager.apply_config_patch(patch_data)
        
        assert result["success"] is True
        assert config_manager.get_config_value("database.timeout") == 60
        assert config_manager.get_config_value("new_section.new_key") == "new_value"

    def test_get_config_history(self, config_manager):
        """Test configuration change history."""
        # Make some changes to create history
        config_manager.set_config_value("database.timeout", 60)
        config_manager.set_config_value("openai.max_tokens", 8000)
        
        history = config_manager.get_config_history()
        
        assert "changes" in history
        assert len(history["changes"]) > 0

    def test_rollback_config(self, config_manager):
        """Test configuration rollback functionality."""
        original_timeout = config_manager.get_config_value("database.timeout")
        
        # Make a change
        config_manager.set_config_value("database.timeout", 999)
        
        # Rollback
        result = config_manager.rollback_config(steps=1)
        
        assert result["success"] is True
        assert config_manager.get_config_value("database.timeout") == original_timeout

    def test_watch_config_changes(self, config_manager):
        """Test configuration change watching."""
        callback_called = False
        
        def test_callback(key, old_value, new_value):
            nonlocal callback_called
            callback_called = True
        
        config_manager.watch_config_changes("database.timeout", test_callback)
        config_manager.set_config_value("database.timeout", 999)
        
        assert callback_called is True

    def test_get_config_statistics(self, config_manager):
        """Test configuration statistics retrieval."""
        stats = config_manager.get_config_statistics()
        
        assert "total_sections" in stats
        assert "total_keys" in stats
        assert "sensitive_keys_count" in stats
        assert "last_modified" in stats

    def test_validate_config_types(self, config_manager):
        """Test configuration type validation."""
        # Test valid types
        result = config_manager.validate_config_types()
        assert result["valid"] is True
        
        # Set invalid type
        config_manager.config["database"]["connection_pool_size"] = "not_a_number"
        result = config_manager.validate_config_types()
        assert result["valid"] is False

    def test_get_config_documentation(self, config_manager):
        """Test configuration documentation retrieval."""
        docs = config_manager.get_config_documentation()
        
        assert "sections" in docs
        assert "database" in docs["sections"]
        assert "description" in docs["sections"]["database"]

    def test_singleton_pattern(self):
        """Test that ConfigManager follows singleton pattern."""
        with patch('app.services.config_manager.os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data='{"test": "config"}')):
            
            manager1 = ConfigManager()
            manager2 = ConfigManager()
            
            assert manager1 is manager2

    def test_thread_safety(self, config_manager):
        """Test thread safety of configuration operations."""
        import threading
        import time
        
        results = []
        
        def worker():
            for i in range(10):
                config_manager.set_config_value(f"test.key_{threading.current_thread().ident}", i)
                time.sleep(0.001)  # Small delay to encourage race conditions
                value = config_manager.get_config_value(f"test.key_{threading.current_thread().ident}")
                results.append(value)
        
        threads = [threading.Thread(target=worker) for _ in range(5)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All operations should complete without errors
        assert len(results) == 50

    def test_config_encryption(self, config_manager):
        """Test configuration encryption/decryption."""
        sensitive_value = "very_secret_key"
        
        # Encrypt
        encrypted = config_manager.encrypt_value(sensitive_value)
        assert encrypted != sensitive_value
        
        # Decrypt
        decrypted = config_manager.decrypt_value(encrypted)
        assert decrypted == sensitive_value

    def test_config_validation_rules(self, config_manager):
        """Test custom configuration validation rules."""
        # Add custom validation rule
        def validate_url(value):
            return value.startswith(('http://', 'https://'))
        
        config_manager.add_validation_rule("database.supabase_url", validate_url)
        
        # Test valid URL
        config_manager.set_config_value("database.supabase_url", "https://valid.url")
        result = config_manager.validate_config()
        assert result["valid"] is True
        
        # Test invalid URL
        config_manager.set_config_value("database.supabase_url", "invalid_url")
        result = config_manager.validate_config()
        assert result["valid"] is False