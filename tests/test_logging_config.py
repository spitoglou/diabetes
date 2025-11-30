"""Tests for logging configuration."""

import sys
from unittest.mock import MagicMock, patch

from loguru import logger

from src.logging_config import get_experiment_logger, get_logger, setup_logging


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_default(self):
        """Test setup_logging with default parameters."""
        with patch.object(logger, "remove") as mock_remove:
            with patch.object(logger, "add") as mock_add:
                with patch.object(logger, "configure") as mock_configure:
                    setup_logging()

                    mock_remove.assert_called_once()
                    mock_add.assert_called_once()
                    mock_configure.assert_called_once_with(extra={"context": ""})

    def test_setup_logging_custom_level(self):
        """Test setup_logging with custom log level."""
        with patch.object(logger, "remove"):
            with patch.object(logger, "add") as mock_add:
                with patch.object(logger, "configure"):
                    setup_logging(level="DEBUG")

                    call_kwargs = mock_add.call_args[1]
                    assert call_kwargs["level"] == "DEBUG"

    def test_setup_logging_with_file(self):
        """Test setup_logging with file output."""
        with patch.object(logger, "remove"):
            with patch.object(logger, "add") as mock_add:
                with patch.object(logger, "configure"):
                    setup_logging(log_file="/tmp/test.log")

                    # Should be called twice: once for stdout, once for file
                    assert mock_add.call_count == 2

    def test_setup_logging_stdout_handler(self):
        """Test that stdout handler is added."""
        with patch.object(logger, "remove"):
            with patch.object(logger, "add") as mock_add:
                with patch.object(logger, "configure"):
                    setup_logging()

                    call_args = mock_add.call_args[0]
                    assert call_args[0] == sys.stdout


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a bound logger."""
        with patch.object(logger, "bind") as mock_bind:
            mock_bind.return_value = MagicMock()
            result = get_logger()

            mock_bind.assert_called_once_with(context="")
            assert result is not None

    def test_get_logger_with_name(self):
        """Test get_logger with custom name."""
        with patch.object(logger, "bind") as mock_bind:
            mock_bind.return_value = MagicMock()
            get_logger("custom_name")

            # Name is accepted but not used (loguru doesn't use it the same way)
            mock_bind.assert_called_once_with(context="")


class TestGetExperimentLogger:
    """Tests for get_experiment_logger function."""

    def test_get_experiment_logger_context(self):
        """Test that experiment logger has correct context."""
        with patch.object(logger, "bind") as mock_bind:
            mock_bind.return_value = MagicMock()
            get_experiment_logger(patient=559, window=12, horizon=6)

            mock_bind.assert_called_once_with(context="[P559/W12/H6]")

    def test_get_experiment_logger_different_params(self):
        """Test experiment logger with different parameters."""
        with patch.object(logger, "bind") as mock_bind:
            mock_bind.return_value = MagicMock()
            get_experiment_logger(patient=570, window=6, horizon=1)

            mock_bind.assert_called_once_with(context="[P570/W6/H1]")
