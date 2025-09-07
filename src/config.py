"""
Derived config values from config.toml
"""

import os

from src.utils import load_toml_config

config_path = os.path.join(os.path.dirname(__file__), "..", "config.toml")
config = load_toml_config(config_path)

# Data paths
DATASET_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", config["data"]["dataset_path"])
)
PRECOMPUTED_DATA_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", config["data"]["precomputed_data_path"]
    )
)

# Dataset settings
NUMBER_OF_PATIENTS: int = config["dataset"]["number_of_patients"]

# Preprocessing parameters
SAMPLE_RATE: int = config["preprocessing"]["sample_rate"]
PREICTAL_MINUTES: int = config["preprocessing"]["preictal_minutes"]
EPOCH_WINDOW_DURATION_SECONDS: int = config["preprocessing"][
    "epoch_window_duration_seconds"
]
LOW_CUTOFF_FILTER: float = config["preprocessing"]["low_cutoff_filter"]
HIGH_CUTOFF_FILTER: float = config["preprocessing"]["high_cutoff_filter"]
NOTCH_FILTER: float = config["preprocessing"]["notch_filter"]
SELECTED_CHANNELS: list[str] = config["preprocessing"]["selected_channels"]
NORMALIZATION_METHOD: str = config["preprocessing"]["normalization_method"]
BAND_DEFS: dict[str, list[float]] = config["preprocessing"]["band_defs"]

# Logging
LOGGING_LEVEL = config["logging"]["logging_level"]
