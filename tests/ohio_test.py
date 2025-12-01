"""Tests for Ohio BGC provider."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.bgc_providers.ohio_bgc_provider import OhioBgcProvider


class TestOhioBgcProvider:
    """Tests for OhioBgcProvider class."""

    @pytest.fixture
    def provider(self):
        """Create a provider instance."""
        return OhioBgcProvider(ohio_no=559)

    def test_provider_initialization(self, provider):
        """Test provider initializes correctly."""
        assert provider.patient == 559
        assert "559" in provider.source_file
        assert provider.root is not None

    def test_get_glycose_levels(self, provider):
        """Test getting glucose levels returns list."""
        data = provider.get_glycose_levels()
        assert isinstance(data, list)
        assert data[3].attrib["value"] == "112"

    def test_get_glycose_levels_with_start(self, provider):
        """Test getting glucose levels with start offset."""
        all_data = provider.get_glycose_levels(start=0)
        offset_data = provider.get_glycose_levels(start=5)
        assert len(offset_data) == len(all_data) - 5

    def test_stream(self, provider):
        """Test glucose stream generates correct values."""
        stream = provider.simulate_glucose_stream()
        assert next(stream)["value"] == 101.0
        assert next(stream)["value"] == 98.0

    def test_simulate_glucose_stream_with_shift(self, provider):
        """Test glucose stream with shift offset."""
        stream_no_shift = list(provider.simulate_glucose_stream(shift=0))
        stream_with_shift = list(provider.simulate_glucose_stream(shift=5))
        assert len(stream_with_shift) == len(stream_no_shift) - 5

    def test_simulate_glucose_stream_contains_required_keys(self, provider):
        """Test stream items contain all required keys."""
        stream = provider.simulate_glucose_stream()
        item = next(stream)
        assert "timestamp" in item
        assert "time" in item
        assert "value" in item
        assert "patient" in item

    def test_ts_to_datetime(self, provider):
        """Test timestamp string to datetime conversion."""
        ts = "15-06-2018 10:30:00"
        result = provider.ts_to_datetime(ts)
        assert isinstance(result, datetime)
        assert result.day == 15
        assert result.month == 6
        assert result.year == 2018
        assert result.hour == 10
        assert result.minute == 30

    def test_ts_to_timestamp(self, provider):
        """Test timestamp string to Unix timestamp conversion."""
        ts = "15-06-2018 10:30:00"
        result = provider.ts_to_timestamp(ts)
        assert isinstance(result, float)
        expected = datetime(2018, 6, 15, 10, 30, 0, tzinfo=timezone.utc).timestamp()
        assert result == expected

    def test_ts_to_iso(self, provider):
        """Test timestamp string to ISO format conversion."""
        ts = "15-06-2018 10:30:00"
        result = provider.ts_to_iso(ts)
        assert "2018-06-15" in result
        assert "10:30:00" in result

    def test_tsfresh_dataframe(self, provider):
        """Test tsfresh dataframe creation."""
        df = provider.tsfresh_dataframe(truncate=10)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert "bg_value" in df.columns
        assert "time" in df.columns
        assert "id" in df.columns
        assert "part_of_day" in df.columns
        assert "mock_date" in df.columns
        assert "time_of_day" in df.columns

    def test_tsfresh_dataframe_no_truncate(self, provider):
        """Test tsfresh dataframe without truncation."""
        df = provider.tsfresh_dataframe(truncate=0)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 10  # Should have more than 10 rows

    def test_tsfresh_dataframe_id_column(self, provider):
        """Test tsfresh dataframe has correct id column."""
        df = provider.tsfresh_dataframe(truncate=5)
        assert all(df["id"] == "a")

    def test_provider_with_different_scope(self):
        """Test provider with test scope."""
        provider = OhioBgcProvider(ohio_no=559, scope="test")
        assert "test" in provider.source_file
        data = provider.get_glycose_levels()
        assert isinstance(data, list)
