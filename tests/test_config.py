"""Tests for configuration loading."""

import pytest


class TestSettings:
    """Test suite for config.settings module."""

    def test_settings_loads(self):
        """Test that settings can be imported and loaded."""
        from config.settings import settings

        assert settings is not None

    def test_settings_has_required_fields(self):
        """Test that settings has all required fields."""
        from config.settings import settings

        # MongoDB settings
        assert hasattr(settings, "MONGO_URI")
        assert hasattr(settings, "MONGO_DATABASE")
        assert hasattr(settings, "MONGO_COLLECTION")

        # Patient settings
        assert hasattr(settings, "OHIO_ID")
        assert hasattr(settings, "WINDOW_STEPS")
        assert hasattr(settings, "PREDICTION_HORIZON")

        # Server settings
        assert hasattr(settings, "HOST")
        assert hasattr(settings, "PORT")
        assert hasattr(settings, "DEBUG")

        # Simulation settings
        assert hasattr(settings, "INTERVAL")

    def test_settings_types(self):
        """Test that settings have correct types."""
        from config.settings import settings

        assert isinstance(settings.OHIO_ID, str)
        assert isinstance(settings.WINDOW_STEPS, int)
        assert isinstance(settings.PREDICTION_HORIZON, int)
        assert isinstance(settings.PORT, int)
        assert isinstance(settings.DEBUG, bool)
        assert isinstance(settings.INTERVAL, int)

    def test_settings_defaults(self):
        """Test that settings have sensible default values."""
        from config.settings import settings

        assert settings.PORT > 0
        assert settings.PORT < 65536
        assert settings.WINDOW_STEPS > 0
        assert settings.PREDICTION_HORIZON > 0

    def test_neptune_enabled_property(self):
        """Test neptune_enabled property logic."""
        from config.settings import settings

        # neptune_enabled should be bool
        assert isinstance(settings.neptune_enabled, bool)

        # If both project and token are set, should be True
        if settings.NEPTUNE_PROJECT and settings.NEPTUNE_API_TOKEN:
            assert settings.neptune_enabled is True
