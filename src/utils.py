import logging
import os
import sys
from datetime import datetime, timedelta

import toml


def load_toml_config(path: str) -> dict:
    """
    Loads a toml config file.

    Args:
        path(str): Path of the toml file.

    Returns:
        dict: Loaded toml file.
    """

    with open(path, "r") as f:
        config = toml.load(f)
    return config


def parse_time_str(time_str: str) -> datetime:
    """
    Parse HH:MM:SS, handling '24:..' as midnight next day.

    Args:
        time_str(str): HH:MM:SS in a string format

    Returns:
        datetime: Parsed time string.
    """
    parts = time_str.split(":")
    hour = int(parts[0])
    minute = int(parts[1])
    second = int(parts[2])
    if hour == 24:
        return datetime.min.replace(hour=0, minute=minute, second=second) + timedelta(
            days=1
        )
    else:
        return datetime.min.replace(hour=hour, minute=minute, second=second)


def load_patient_summary(patient_id: str, dataset_path: str):
    """
    Load the summary file for a given patient ID.

    Args:
        patient_id(str): Zero padded patient id (e.g. 01)
        dataset_path(str): Path of the EEG recordings
    """

    return open(
        os.path.join(dataset_path, f"chb{patient_id}", f"chb{patient_id}-summary.txt"),
        "r",
    )


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Reusable logger setup.

    Args:
        name(str): Name of the logger.
        level(int): Logging level (e.g. logging.DEBUG)
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    return logger
