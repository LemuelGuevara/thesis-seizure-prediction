"""
Loaders module

This module is a utility for creating loader functions that can be reused across other
modules.
"""

import os

import mne
import numpy as np
from mne.io import BaseRaw
from tqdm import tqdm

from src.config import DataConfig, PreprocessingConfig
from src.datatypes import StftData
from src.logger import setup_logger

logger = setup_logger(name="loaders")


def load_raw_recordings(patient_id: str, file_names: list[str]) -> list[BaseRaw]:
    """
    Loads all raw EDF recordings of a patient without filtering.

    Args:
        patient_id (str): Zero-padded patient ID (e.g. "01").

    Returns:
        list[BaseRaw]: List of unprocessed raw recordings.
    """

    logger.info(f"Loading EDF files for patient {patient_id}")
    patient_folder = os.path.join(DataConfig.dataset_path, f"chb{patient_id}")
    raw_edf_list: list[BaseRaw] = []

    # Load EDF files
    for file_name in tqdm(
        file_names, desc=f"Reading EDF files for patient {patient_id}"
    ):
        recording_path = os.path.join(patient_folder, file_name)
        logger.info(f"Reading file: {file_name}")

        raw_edf = mne.io.read_raw_edf(recording_path, preload=False, verbose="error")
        raw_channels = set(raw_edf.ch_names)
        selected_channels = set(PreprocessingConfig.selected_channels)

        # Only append if all selected channels are present
        if selected_channels.issubset(raw_channels):
            raw_edf.pick(PreprocessingConfig.selected_channels)
            raw_edf_list.append(raw_edf)
        else:
            logger.warning(
                f"Skipping {file_name}: missing channels {selected_channels - raw_channels}"
            )

    if not raw_edf_list:
        raise FileNotFoundError(f"No EDF files found in {patient_folder}")

    return raw_edf_list


def load_precomputed_stfts(patient_stfts_dir: str) -> list[StftData]:
    """
    Load precomputed STFT (.h5) files for a single patient when STFTs are stored per epoch.

    Args:
        patient_stfts_dir (str): Path containing all precomputed STFT .h5 files.

    Returns:
        list[StftStore]: List of STFTs for all epochs.
    """

    # List all .h5 files sorted
    epoch_files = sorted(
        f for f in os.listdir(patient_stfts_dir) if f.lower().endswith(".npz")
    )

    stft_store_list: list[StftData] = []

    for epoch_file in epoch_files:
        full_path = os.path.join(patient_stfts_dir, epoch_file)
        data = np.load(full_path)

        stft_store = StftData(
            phase=data["phase"],
            start=int(data["start"]),
            end=int(data["end"]),
            stft_db=data["stft_db"],
            power=data["power"]
            if "power" in data
            else np.empty((0,), dtype=np.float32),
            Zxx=data["Zxx"],
            mag=data["mag"],
            freqs=data["freqs"],
            times=data["times"],
            seizure_id=int(data["seizure_id"]) if "seizure_id" in data else -1,
            file_name=str(data["file_name"]) if "file_name" in data else epoch_file,
        )

        stft_store_list.append(stft_store)
    return stft_store_list
