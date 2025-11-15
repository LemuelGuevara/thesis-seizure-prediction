"""
Functions that can be resued as utility to other modules.
"""

import csv
import json
import os
import platform
import random
from datetime import datetime, timedelta
from re import sub
from typing import Any, Literal, Optional

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

    if hour >= 24:
        # Calculate how many days to add
        days_to_add = hour // 24
        normalized_hour = hour % 24
        return datetime.min.replace(
            hour=normalized_hour, minute=minute, second=second
        ) + timedelta(days=days_to_add)
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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def export_to_csv(
    path: str,
    fieldnames: list[str],
    data: list[dict[str, Any]],
    mode: Literal["w", "a"] = "w",
    json_metadata: Optional[tuple[str, Any]] = None,
) -> None:
    file_exists = os.path.isfile(path)

    # Add metadata column name to fieldnames if provided
    if json_metadata is not None:
        meta_column_name, meta_value = json_metadata
        if meta_column_name not in fieldnames:
            fieldnames = fieldnames + [meta_column_name]

    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if mode == "w" or not file_exists:
            writer.writeheader()

        for row in data:
            row_to_write = row.copy()

            if json_metadata is not None:
                meta_column_name, meta_value = json_metadata
                row_to_write[meta_column_name] = json.dumps(meta_value)

            writer.writerow(row_to_write)


def snake_case(s: str) -> str:
    return "_".join(
        sub(
            "([A-Z][a-z]+)", r" \1", sub("([A-Z]+)", r" \1", s.replace("-", " "))
        ).split()
    ).lower()
