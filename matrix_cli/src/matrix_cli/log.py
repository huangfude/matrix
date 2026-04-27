"""Matrix CLI logging configuration.

This module provides centralized logging configuration for all matrix_cli components.

Usage:
    from matrix_cli.log import logger
    logger.info("Your message here")
"""

import logging
import sys

LOG_FORMAT = "[matrix-cli] %(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_LEVEL = logging.INFO


def configure_logging(
    level: int | str = DEFAULT_LOG_LEVEL,
    fmt: str = LOG_FORMAT,
    datefmt: str = DATE_FORMAT,
    stream: str | None = None,
) -> None:
    """Configure logging for the matrix_cli module."""
    global logger  # 引用全局logger对象

    if isinstance(level, str):
        level = getattr(logging, level.upper(), DEFAULT_LOG_LEVEL)

    # Check if root logger has already been configured to avoid issues with tests
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        # Only use basicConfig if no handlers exist yet
        logging.basicConfig(
            level=level,
            format=fmt,
            datefmt=datefmt,
            stream=stream or sys.stderr,
        )
    else:
        # If root logger already has handlers, just set the level
        root_logger.setLevel(level)

    # Ensure our specific logger has the correct level
    logger.setLevel(level)


def get_logger(name: str = "matrix-cli") -> logging.Logger:
    """Get a logger instance for matrix_cli."""
    return logging.getLogger(name)


# Initialize logger instance
logger = get_logger()

# Check if any logging is already configured, and if not, apply default configuration
root_logger = logging.getLogger()
if not root_logger.handlers and not logger.handlers:
    configure_logging()
else:
    # If logging is already configured, ensure logger has correct level
    logger.setLevel(DEFAULT_LOG_LEVEL)

