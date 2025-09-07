"""
Responsible for setting up loggers.
"""

import logging
import sys

from .config import LOGGING_LEVEL


def setup_logger(name: str) -> logging.Logger:
    """
    Create and return a logger that outputs to stderr.

    Args:
        name (str): Name of the logger (usually __name__ of the module).

    Returns:
        logging.Logger: Configured logger instance.
    """

    log_level = getattr(logging, LOGGING_LEVEL.upper(), logging.DEBUG)
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(funcName)s | [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
