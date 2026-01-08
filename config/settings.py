"""
Unified configuration management using Pydantic Settings.
All configuration values are loaded from environment variables.
"""

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized application settings.

    All settings are loaded from environment variables or .env file.
    Defaults are provided for local development convenience.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
    )

    # MongoDB Configuration
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DATABASE: str = "test_database_1"
    MONGO_COLLECTION: str = "sandalphon1"

    # Patient & Model Configuration
    OHIO_ID: str = "559"
    WINDOW_STEPS: int = 6
    PREDICTION_HORIZON: int = 6

    # Server Configuration
    DEBUG: bool = False
    DATABASE: str = (
        "test_database_1"  # Alias for MONGO_DATABASE for backward compatibility
    )
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Simulation Configuration
    INTERVAL: int = 20

    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = False
    LOG_FILE: Optional[str] = None

    # Neptune ML Tracking (Optional)
    NEPTUNE_PROJECT: Optional[str] = None
    NEPTUNE_API_TOKEN: Optional[str] = None

    @property
    def neptune_enabled(self) -> bool:
        """Check if Neptune tracking is properly configured."""
        return bool(self.NEPTUNE_PROJECT and self.NEPTUNE_API_TOKEN)


# Global settings instance
settings = Settings()
