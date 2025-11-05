"""
Loaders module

This module is a utility for creating loader functions that can be reused across other
modules.
"""

import os

import numpy as np

from src.datatypes import StftData


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
