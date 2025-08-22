import os

from utils import load_toml_config

config_path = os.path.join(os.path.dirname(__file__), "..", "config.toml")
config = load_toml_config(config_path)["preprocessing"]

DATASET_PATH = config["dataset_path"]
DATASET_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", DATASET_PATH)
)
SPECS_OUTPUT_PATH = config["specs_output_path"]
SPECS_OUTPUT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", SPECS_OUTPUT_PATH)
)

SAMPLE_RATE = config["sample_rate"]
STFT_WINDOW_SIZE = int(SAMPLE_RATE * 30)
WINDOW_TIME_STEP = STFT_WINDOW_SIZE // 2
PREICTAL_MINUTES = config["preictal_minutes"]
EPOCH_WINDOW_DURATION_SECONDS = config["epoch_window_duration_seconds"]
LOW_CUTOFF_FILTER = config["low_cutoff_filter"]
HIGH_CUTOFF_FILTER = config["high_cutoff_filter"]
NOTCH_FILTER = config["notch_filter"]
SELECTED_CHANNELS = config["selected_channels"]
NUMBER_OF_PATIENTS = config["number_of_patients"]
