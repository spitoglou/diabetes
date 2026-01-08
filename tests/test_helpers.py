"""Tests for helper modules."""

import json

import pytest


class TestMiscHelpers:
    """Test suite for src.helpers.misc module."""

    def test_get_part_of_day_morning(self):
        """Test morning hours."""
        from src.helpers.misc import get_part_of_day

        assert get_part_of_day(7) == "morning"
        assert get_part_of_day(9) == "morning"
        assert get_part_of_day(11) == "morning"

    def test_get_part_of_day_afternoon(self):
        """Test afternoon hours."""
        from src.helpers.misc import get_part_of_day

        assert get_part_of_day(12) == "afternoon"
        assert get_part_of_day(14) == "afternoon"
        assert get_part_of_day(16) == "afternoon"

    def test_get_part_of_day_evening(self):
        """Test evening hours."""
        from src.helpers.misc import get_part_of_day

        assert get_part_of_day(17) == "evening"
        assert get_part_of_day(19) == "evening"
        assert get_part_of_day(20) == "evening"

    def test_get_part_of_day_night(self):
        """Test night hours."""
        from src.helpers.misc import get_part_of_day

        assert get_part_of_day(21) == "night"
        assert get_part_of_day(22) == "night"
        assert get_part_of_day(23) == "night"

    def test_get_part_of_day_late_night(self):
        """Test late night hours."""
        from src.helpers.misc import get_part_of_day

        assert get_part_of_day(0) == "late_night"
        assert get_part_of_day(3) == "late_night"
        assert get_part_of_day(6) == "late_night"

    def test_debug_print(self, capsys):
        """Test debug_print outputs correctly."""
        from src.helpers.misc import debug_print

        debug_print("Test Title", "Test Message")

        captured = capsys.readouterr()
        assert "Test Title" in captured.out
        assert "Test Message" in captured.out
        assert "---" in captured.out


class TestFhirHelpers:
    """Test suite for src.helpers.fhir module."""

    def test_create_fhir_json_from_reading(self):
        """Test FHIR JSON creation."""
        from src.helpers.fhir import create_fhir_json_from_reading

        reading = {
            "timestamp": 1638840420.0,
            "time": "2021-12-07T01:27:00+00:00",
            "value": 104.0,
            "patient": "559",
        }

        result = create_fhir_json_from_reading(reading)

        assert isinstance(result, str)

        # Parse and verify structure
        fhir = json.loads(result)
        assert fhir["status"] == "final"
        assert fhir["valueQuantity"]["value"] == 104.0
        assert fhir["valueQuantity"]["unit"] == "mg/dL"
        assert fhir["subject"]["identifier"] == "559"
        assert fhir["effectiveDateTime"] == "2021-12-07T01:27:00+00:00"

    def test_create_fhir_json_custom_unit(self):
        """Test FHIR JSON with custom unit."""
        from src.helpers.fhir import create_fhir_json_from_reading

        reading = {
            "time": "2021-12-07T01:27:00+00:00",
            "value": 5.8,
            "patient": "559",
        }

        result = create_fhir_json_from_reading(reading, value_unit="mmol/L")
        fhir = json.loads(result)

        assert fhir["valueQuantity"]["unit"] == "mmol/L"


class TestDataframeHelpers:
    """Test suite for src.helpers.dataframe module."""

    def test_save_and_read_df(self, tmp_path):
        """Test DataFrame save and read roundtrip."""
        import pandas as pd

        from src.helpers.dataframe import read_df, save_df

        # Create test DataFrame
        df = pd.DataFrame(
            {"time": [1, 2, 3], "bg_value": [100, 110, 105], "id": ["a", "a", "a"]}
        )

        # Save and read back
        filepath = str(tmp_path / "test.pkl")
        save_df(df, filepath)
        loaded_df = read_df(filepath)

        # Verify
        assert len(loaded_df) == 3
        assert list(loaded_df.columns) == ["time", "bg_value", "id"]
        assert loaded_df["bg_value"].tolist() == [100, 110, 105]


class TestMadexHelpers:
    """Test suite for src.helpers.diabetes.madex module."""

    def test_madex_perfect_prediction(self):
        """Test MADEX with perfect predictions."""
        from src.helpers.diabetes.madex import madex

        y = [100, 120, 140]
        y_pred = [100, 120, 140]

        error = madex(y, y_pred)

        assert error == 0.0

    def test_madex_with_error(self):
        """Test MADEX with prediction errors."""
        from src.helpers.diabetes.madex import madex

        y = [100, 120, 140]
        y_pred = [110, 130, 150]

        error = madex(y, y_pred)

        assert error > 0

    def test_rmadex(self):
        """Test root MADEX."""
        import math

        from src.helpers.diabetes.madex import madex, rmadex

        y = [100, 120, 140]
        y_pred = [110, 130, 150]

        madex_val = madex(y, y_pred)
        rmadex_val = rmadex(y, y_pred)

        assert rmadex_val == pytest.approx(math.sqrt(madex_val), rel=1e-6)

    def test_mean_adjusted_exponent_error(self):
        """Test mean_adjusted_exponent_error function."""
        from src.helpers.diabetes.madex import mean_adjusted_exponent_error

        y = [100.0, 120.0]
        y_pred = [100.0, 120.0]

        error = mean_adjusted_exponent_error(y, y_pred)

        assert error == 0.0
