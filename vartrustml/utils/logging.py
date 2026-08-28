"""
Logging configuration utilities.
"""

import logging
import sys
from typing import Optional


def setup_logging(verbose: int = 1, log_file: Optional[str] = None):
    """Set up logging configuration based on verbosity level.

    Parameters
    ----------
    verbose : int
        Verbosity level (1=tqdm only, 2=INFO, 3=DEBUG).
    log_file : str, optional
        File path to write logs to.

    Returns
    -------
    logging.Logger
        Configured root logger.
    """
    # Map verbose levels to logging levels
    level_map = {
        0: logging.WARNING,  # No output
        1: logging.WARNING,  # Only tqdm progress bars (suppress INFO)
        2: logging.INFO,  # INFO level logging
        3: logging.DEBUG,  # DEBUG level (all verbosity)
    }

    log_level = level_map.get(verbose, logging.INFO)

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    simple_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler with simple format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)

    # File handler with detailed format (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)

    # Set levels for third-party loggers to avoid spam
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Only show sklearn warnings and errors
    logging.getLogger("sklearn").setLevel(logging.WARNING)

    # Be more permissive with our own logger
    logging.getLogger("vartrustml").setLevel(log_level)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module."""
    return logging.getLogger(f"vartrustml.{name}")
