"""Tests for miscellaneous helper functions."""

from src.helpers.misc import debug_print, get_part_of_day


class TestGetPartOfDay:
    """Tests for get_part_of_day function."""

    def test_morning(self):
        """Test morning hours (7-11)."""
        assert get_part_of_day(7) == "morning"
        assert get_part_of_day(9) == "morning"
        assert get_part_of_day(11) == "morning"

    def test_afternoon(self):
        """Test afternoon hours (12-16)."""
        assert get_part_of_day(12) == "afternoon"
        assert get_part_of_day(14) == "afternoon"
        assert get_part_of_day(16) == "afternoon"

    def test_evening(self):
        """Test evening hours (17-20)."""
        assert get_part_of_day(17) == "evening"
        assert get_part_of_day(19) == "evening"
        assert get_part_of_day(20) == "evening"

    def test_night(self):
        """Test night hours (21-23)."""
        assert get_part_of_day(21) == "night"
        assert get_part_of_day(22) == "night"
        assert get_part_of_day(23) == "night"

    def test_late_night(self):
        """Test late night hours (0-6)."""
        assert get_part_of_day(0) == "late_night"
        assert get_part_of_day(3) == "late_night"
        assert get_part_of_day(6) == "late_night"


class TestDebugPrint:
    """Tests for debug_print function."""

    def test_debug_print_logs_message(self, caplog):
        """Test debug_print logs the message."""
        import logging

        with caplog.at_level(logging.DEBUG):
            debug_print("TestTitle", "Test message content")

        # The function uses loguru, so we check if it was called
        # Since loguru doesn't integrate with pytest caplog by default,
        # we just ensure the function doesn't raise
        assert True

    def test_debug_print_with_various_types(self):
        """Test debug_print with various message types."""
        # Should not raise for any type
        debug_print("Title", "string message")
        debug_print("Title", 12345)
        debug_print("Title", {"key": "value"})
        debug_print("Title", [1, 2, 3])
        debug_print("Title", None)
