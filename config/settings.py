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

    # Streaming Interval Configuration
    # These control the delay between sending CGM readings to the server.
    # Realistic values should match the CGM sensor's actual sample time.
    #
    # Ohio dataset clients (client, synced-client):
    #   - Data collected with Medtronic Guardian sensors (5-minute intervals)
    #   - Realistic: 300 seconds
    #
    # Simglucose client (simglucose-client):
    #   - Uses Dexcom CGM sensor model (3-minute intervals)
    #   - Realistic: 180 seconds
    #
    # Default values are set for fast demo/testing. Use realistic values
    # for production or accurate simulation timing.
    INTERVAL: int = 300  # Ohio clients: 300s realistic, 20s for fast demo
    SIMGLUCOSE_INTERVAL: int = 180  # Simglucose: 180s realistic (Dexcom 3-min)

    # Simglucose Configuration
    SIMGLUCOSE_PATIENT: str = "adult#001"
    SIMGLUCOSE_SEED: Optional[int] = None
    # Default meal schedule: (hour, carbs_grams)
    # Breakfast 7am (45g), Lunch 12pm (70g), Snack 4pm (15g), Dinner 6pm (80g), Bedtime 11pm (10g)
    SIMGLUCOSE_MEALS: str = "7:45,12:70,16:15,18:80,23:10"
    # Insulin mode: "none" (open-loop), "basal" (basal only), "basal-bolus" (full control)
    SIMGLUCOSE_INSULIN_MODE: str = "basal"

    # Configurable Insulin Parameters (None = use patient-specific defaults)
    # Basal rate in U/hr - continuous background insulin (typical: 0.5-2.0 U/hr for adults)
    SIMGLUCOSE_BASAL_RATE: Optional[float] = None
    # Target glucose in mg/dL for correction boluses (typical: 100-150 mg/dL)
    SIMGLUCOSE_TARGET_GLUCOSE: float = 140.0
    # Carb ratio in g/U - grams of carbs covered by 1 unit insulin (typical: 5-25 g/U)
    SIMGLUCOSE_CARB_RATIO: Optional[float] = None
    # Correction factor in mg/dL/U - glucose drop per 1 unit insulin (typical: 20-100 mg/dL/U)
    SIMGLUCOSE_CORRECTION_FACTOR: Optional[float] = None
    # Pre-bolus time in minutes - deliver meal bolus before eating (typical: 0-30 min)
    SIMGLUCOSE_PRE_BOLUS_MINUTES: int = 0

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
