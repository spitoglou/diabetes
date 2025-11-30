"""
Tests for configuration management.
"""

import os
from unittest.mock import patch

import pytest

from src.config import Config, get_config, reset_config, set_config


class TestConfig:
    """Tests for Config class."""

    def test_default_values(self):
        """Test that Config has sensible defaults."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.glucose_low == 70
            assert config.glucose_high == 180
            assert config.valid_patient_ids == (559, 563, 570, 575, 588, 591)
            assert config.log_level == "INFO"

    def test_env_override(self):
        """Test that environment variables override defaults."""
        with patch.dict(
            os.environ,
            {
                "DATABASE_NAME": "custom_db",
                "LOG_LEVEL": "DEBUG",
                "DEBUG": "true",
            },
        ):
            config = Config()
            assert config.database_name == "custom_db"
            assert config.log_level == "DEBUG"
            assert config.debug is True

    def test_is_valid_patient_id(self):
        """Test patient ID validation."""
        config = Config()

        # Valid IDs
        assert config.is_valid_patient_id("559") is True
        assert config.is_valid_patient_id(563) is True
        assert config.is_valid_patient_id("570") is True

        # Invalid IDs
        assert config.is_valid_patient_id("999") is False
        assert config.is_valid_patient_id("invalid") is False
        assert config.is_valid_patient_id("") is False

    def test_validate_missing_mongo_uri(self):
        """Test validation fails without MONGO_URI."""
        with patch.dict(os.environ, {"MONGO_URI": ""}, clear=True):
            config = Config()
            with pytest.raises(ValueError, match="MONGO_URI"):
                config.validate()

    def test_validate_neptune_without_token(self):
        """Test validation fails when Neptune enabled without token."""
        with patch.dict(
            os.environ,
            {
                "MONGO_URI": "mongodb://localhost",
                "ENABLE_NEPTUNE": "true",
                "NEPTUNE_API_TOKEN": "",
            },
        ):
            config = Config()
            with pytest.raises(ValueError, match="NEPTUNE_API_TOKEN"):
                config.validate()

    def test_validate_success(self):
        """Test validation passes with required values."""
        with patch.dict(
            os.environ,
            {
                "MONGO_URI": "mongodb://localhost",
                "ENABLE_NEPTUNE": "false",
            },
        ):
            config = Config()
            config.validate()  # Should not raise


class TestConfigSingleton:
    """Tests for global config management."""

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_get_config_returns_instance(self):
        """Test get_config returns a Config instance."""
        config = get_config()
        assert isinstance(config, Config)

    def test_get_config_singleton(self):
        """Test get_config returns same instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_set_config(self):
        """Test setting custom config."""
        custom_config = Config()
        custom_config.database_name = "custom_test_db"

        set_config(custom_config)
        retrieved = get_config()

        assert retrieved.database_name == "custom_test_db"

    def test_reset_config(self):
        """Test resetting config."""
        config1 = get_config()
        reset_config()
        config2 = get_config()

        # After reset, should be a new instance
        assert config1 is not config2
