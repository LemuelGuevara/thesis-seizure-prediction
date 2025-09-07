"""
Loaders module

This module is util module used for creating loader functions that can be reused across other
modules.
"""

import os

import numpy as np

from src.datatypes import StftStore


def load_precomputed_stfts(patient_stfts_dir: str) -> dict[str, list[StftStore]]:
    """
    Load the precomputed STFT (.npz) for a single patient.

    Args:
        patient_stfts_dir (str): Path of the patient sfts.

    Returns:
        dict[str, list[StftStore]]: A dictionary containing each channel with each epochs
            - {ch_name: [stft_store_list]}
    """

    stfts_by_channel: dict[str, list[StftStore]] = {}

    for ch_name in os.listdir(patient_stfts_dir):
        ch_path = os.path.join(patient_stfts_dir, ch_name)
        if not os.path.isdir(ch_path):
            continue  # skip files if any

        # List .npz files in channel folder
        epoch_files = sorted(
            f for f in os.listdir(ch_path) if f.lower().endswith(".npz")
        )

        stft_store_list: list[StftStore] = []
        for epoch_file in epoch_files:
            full_path = os.path.join(ch_path, epoch_file)
            data = np.load(full_path)
            stft_store_list.append(
                StftStore(
                    phase=data["phase"],
                    start=data["start"],
                    end=data["end"],
                    stft_db=data["stft_db"],
                    power=data["power"],
                    Zxx=data["Zxx"],
                    freqs=data["freqs"],
                    times=data["times"],
                )
            )

        if stft_store_list:
            stfts_by_channel[ch_name] = stft_store_list

    return stfts_by_channel
