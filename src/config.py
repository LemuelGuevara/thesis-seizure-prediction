"""
Derived config values from config.toml
"""

import os
from dataclasses import dataclass
from typing import ClassVar, Literal

from src.utils import load_toml_config

config_path = os.path.join(os.path.dirname(__file__), "..", "config.toml")
config = load_toml_config(config_path)


@dataclass
class DataConfig:
    dataset_path: ClassVar[str] = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", config["data"]["dataset_path"])
    )
    precomputed_data_path: ClassVar[str] = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", config["data"]["precomputed_data_path"]
        )
    )
    number_of_patients: ClassVar[int] = config["data"]["number_of_patients"]


@dataclass
class PreprocessingConfig:
    sample_rate: ClassVar[int] = config["preprocessing"]["sample_rate"]
    preictal_minutes: ClassVar[int] = config["preprocessing"]["preictal_minutes"]
    epoch_length: ClassVar[int] = config["preprocessing"]["epoch_length"]
    low_cutoff_filter: ClassVar[float] = config["preprocessing"]["low_cutoff_filter"]
    high_cutoff_filter: ClassVar[float] = config["preprocessing"]["high_cutoff_filter"]
    notch_filter: ClassVar[float] = config["preprocessing"]["notch_filter"]

    # now these are also class-level constants
    selected_channels: ClassVar[list[str]] = config["preprocessing"][
        "selected_channels"
    ]
    normalization_method: ClassVar[Literal["minmax", "zscore"]] = config[
        "preprocessing"
    ]["normalization_method"]
    band_defs: ClassVar[dict[str, list[float]]] = config["preprocessing"]["band_defs"]


@dataclass
class LoggingConfig:
    logging_level: ClassVar[Literal["DEBUG", "VERBOSE", "INFO", "WARNING", "ERROR"]] = (
        config["logging"]["logging_level"]
    )
