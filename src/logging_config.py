"""
Centralized logging configuration using loguru.
"""

import sys

from loguru import logger

from src.config import get_config


def setup_logging(
    level: str | None = None,
    log_file: str | None = None,
    format_string: str | None = None,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to config value.
        log_file: Optional file path to write logs to.
        format_string: Custom log format string.
    """
    config = get_config()

    # Use provided level or fall back to config
    log_level = level or config.log_level

    # Default format with context support
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "{extra[context]} "
            "<level>{message}</level>"
        )

    # Remove default handler
    logger.remove()

    # Add stderr handler
    logger.add(
        sys.stderr,
        format=format_string,
        level=log_level,
        colorize=True,
        filter=lambda record: record["extra"].setdefault("context", ""),
    )

    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            format=format_string,
            level=log_level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            filter=lambda record: record["extra"].setdefault("context", ""),
        )


def get_logger(name: str = __name__):
    """
    Get a logger instance with optional context binding.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Logger instance.
    """
    return logger.bind(context="")


def get_experiment_logger(patient: int, window: int, horizon: int):
    """
    Get a logger with experiment context bound.

    Args:
        patient: Patient ID.
        window: Window size.
        horizon: Prediction horizon.

    Returns:
        Logger instance with context.
    """
    context = f"[P{patient}/W{window}/H{horizon}]"
    return logger.bind(context=context)
