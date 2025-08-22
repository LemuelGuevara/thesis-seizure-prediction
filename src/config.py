import os

from utils import load_toml_config

config_path = os.path.join(os.path.dirname(__file__), "..", "config.toml")
config = load_toml_config(config_path)

# Data paths
DATASET_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", config["data"]["dataset_path"])
)
SPECS_OUTPUT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", config["data"]["specs_output_path"])
)

# Dataset settings
NUMBER_OF_PATIENTS = config["dataset"]["number_of_patients"]

# Preprocessing parameters
SAMPLE_RATE = config["preprocessing"]["sample_rate"]
STFT_WINDOW_SIZE = int(SAMPLE_RATE * 30)
WINDOW_TIME_STEP = STFT_WINDOW_SIZE // 2
PREICTAL_MINUTES = config["preprocessing"]["preictal_minutes"]
EPOCH_WINDOW_DURATION_SECONDS = config["preprocessing"]["epoch_window_duration_seconds"]
LOW_CUTOFF_FILTER = config["preprocessing"]["low_cutoff_filter"]
HIGH_CUTOFF_FILTER = config["preprocessing"]["high_cutoff_filter"]
NOTCH_FILTER = config["preprocessing"]["notch_filter"]
SELECTED_CHANNELS = config["preprocessing"]["selected_channels"]
