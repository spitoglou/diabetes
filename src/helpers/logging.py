"""
Centralized logging configuration for the diabetes prediction system.

Usage:
    from src.helpers.logging import setup_logging, get_logger

    # At application startup
    setup_logging()

    # In modules
    logger = get_logger(__name__)
    logger.info("Message")
"""

from __future__ import annotations

import sys
from typing import Optional

from loguru import logger

from config.settings import settings


def setup_logging(
    level: Optional[str] = None,
    json_format: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """Configure application-wide logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to INFO or DEBUG if settings.DEBUG.
        json_format: If True, output logs in JSON format for structured logging.
        log_file: Optional path to log file. If provided, logs will also be written to file.
    """
    # Remove default handler
    logger.remove()

    # Determine log level
    if level is None:
        level = "DEBUG" if settings.DEBUG else "INFO"

    # Define format
    if json_format:
        log_format = (
            '{{"timestamp": "{time:YYYY-MM-DDTHH:mm:ss.SSSZ}", '
            '"level": "{level}", '
            '"module": "{name}", '
            '"function": "{function}", '
            '"line": {line}, '
            '"message": "{message}"}}'
        )
    else:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Add stderr handler
    logger.add(
        sys.stderr,
        format=log_format,
        level=level,
        colorize=not json_format,
    )

    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            format=log_format,
            level=level,
            rotation="10 MB",
            retention="7 days",
            compression="gz",
        )

    logger.info(
        f"Logging configured: level={level}, json={json_format}, file={log_file}"
    )


def get_logger(name: str) -> "logger":
    """Get a logger instance for a module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logger.bind(name=name)


# Convenience re-export
__all__ = ["setup_logging", "get_logger", "logger"]
