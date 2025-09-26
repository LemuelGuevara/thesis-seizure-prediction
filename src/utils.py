"""
Functions that can be resued as utility to other modules.
"""

import os
import platform
import random
from dataclasses import fields
from datetime import datetime, timedelta

import numpy as np
import toml
import torch


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


def is_precomputed_data_exists(data_path: str) -> bool:
    return os.path.exists(data_path) and len(os.listdir(data_path)) > 0


def get_torch_device() -> torch.device:
    if platform.system() in ("Windows", "Linux") and torch.cuda.is_available():
        return torch.device("cuda")
    elif platform.system() == "Darwin" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def classvars_to_dict(cls):
    """Extract ClassVar attributes from a dataclass class."""
    return {f.name: getattr(cls, f.name) for f in fields(cls) if hasattr(cls, f.name)}
