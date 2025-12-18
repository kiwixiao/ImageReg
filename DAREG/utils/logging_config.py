"""
DAREG Logging Configuration

Professional logging setup with:
- Console output with color formatting
- Optional file logging
- Module-level loggers
- Performance timing utilities
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColorFormatter(logging.Formatter):
    """Custom formatter with color support for console output"""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")

        # Create formatted message
        level_short = record.levelname[0]  # D, I, W, E, C
        message = f"{color}[{timestamp}] {level_short} | {record.name}: {record.getMessage()}{reset}"

        return message


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    module_name: str = "DAREG"
) -> logging.Logger:
    """
    Setup professional logging for DAREG

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        module_name: Root module name for logger hierarchy

    Returns:
        Configured root logger
    """
    # Get root logger for DAREG
    logger = logging.getLogger(module_name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(ColorFormatter())
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a module-level logger

    Args:
        name: Module name (will be prefixed with DAREG)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"DAREG.{name}")


class Timer:
    """Context manager for timing code sections"""

    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or get_logger("timer")
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, *args):
        self.elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(f"{self.name}: {self.elapsed:.2f}s")
