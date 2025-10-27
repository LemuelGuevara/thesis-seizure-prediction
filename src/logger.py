"""
Responsible for setting up loggers.
"""

import logging
import sys

from src.config import LoggingConfig


def setup_logger(name: str) -> logging.Logger:
    """
    Create and return a logger that outputs to stderr.

    Args:
        name (str): Name of the logger (usually __name__ of the module).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logging_config = LoggingConfig()

    log_level = getattr(logging, logging_config.logging_level.upper(), logging.DEBUG)
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(funcName)s | [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def get_all_active_loggers():
    """Get all loggers that have been created and have handlers"""

    loggers = []
    for name in logging.getLogger().manager.loggerDict:
        logger = logging.getLogger(name)
        if logger.handlers:  # Only loggers that actually output something
            loggers.append(logger)
    return loggers
