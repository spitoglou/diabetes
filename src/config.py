"""
Centralized configuration management.
All settings are loaded from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Tuple

from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


def _get_env(key: str, default: str = "") -> str:
    """Get environment variable with default."""
    return os.getenv(key, default)


def _get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes")


def _get_env_int(key: str, default: int) -> int:
    """Get integer environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # MongoDB settings
    mongo_uri: str = field(default_factory=lambda: _get_env("MONGO_URI"))
    database_name: str = field(
        default_factory=lambda: _get_env("DATABASE_NAME", "test_database_1")
    )

    # Neptune ML tracking
    neptune_api_token: str = field(
        default_factory=lambda: _get_env("NEPTUNE_API_TOKEN")
    )
    neptune_project: str = field(
        default_factory=lambda: _get_env("NEPTUNE_PROJECT", "spitoglou/intermediate")
    )
    enable_neptune: bool = field(
        default_factory=lambda: _get_env_bool("ENABLE_NEPTUNE", False)
    )

    # Application settings
    debug: bool = field(default_factory=lambda: _get_env_bool("DEBUG", False))
    log_level: str = field(default_factory=lambda: _get_env("LOG_LEVEL", "INFO"))

    # Prediction settings
    default_patient_id: str = field(
        default_factory=lambda: _get_env("DEFAULT_PATIENT_ID", "559")
    )
    window_steps: int = field(default_factory=lambda: _get_env_int("WINDOW_STEPS", 12))
    prediction_horizon: int = field(
        default_factory=lambda: _get_env_int("PREDICTION_HORIZON", 6)
    )

    # Domain constants - glucose thresholds (mg/dL)
    glucose_low: int = 70
    glucose_high: int = 180

    # Valid patient IDs from Ohio T1DM dataset
    valid_patient_ids: Tuple[int, ...] = (559, 563, 570, 575, 588, 591)

    # Data paths
    data_path: str = field(default_factory=lambda: _get_env("DATA_PATH", "data"))
    models_path: str = field(default_factory=lambda: _get_env("MODELS_PATH", "models"))
    dataframes_path: str = field(
        default_factory=lambda: _get_env("DATAFRAMES_PATH", "dataframes")
    )

    # Time of day ranges (hour boundaries)
    time_ranges: dict = field(
        default_factory=lambda: {
            "morning": (7, 11),
            "afternoon": (12, 16),
            "evening": (17, 20),
            "night": (21, 23),
            "late_night": (0, 6),
        }
    )

    def validate(self) -> None:
        """Validate required configuration values."""
        errors = []
        if not self.mongo_uri:
            errors.append("MONGO_URI environment variable is required")
        if self.enable_neptune and not self.neptune_api_token:
            errors.append("NEPTUNE_API_TOKEN is required when Neptune is enabled")
        if errors:
            raise ValueError(
                "Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    def is_valid_patient_id(self, patient_id: str | int) -> bool:
        """Check if patient ID is in the valid whitelist."""
        try:
            return int(patient_id) in self.valid_patient_ids
        except (ValueError, TypeError):
            return False


# Global config instance - can be overridden for testing
_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set a custom configuration (useful for testing)."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset configuration to reload from environment."""
    global _config
    _config = None
