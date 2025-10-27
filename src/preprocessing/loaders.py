"""
Loaders module

This module is a utility for creating loader functions that can be reused across other
modules.
"""

import os

import h5py
import numpy as np

from src.datatypes import StftStore


def load_precomputed_stfts(patient_stfts_dir: str) -> list[StftStore]:
    """
    Load precomputed STFT (.h5) files for a single patient when STFTs are stored per epoch.

    Args:
        patient_stfts_dir (str): Path containing all precomputed STFT .h5 files.

    Returns:
        list[StftStore]: List of STFTs for all epochs.
    """

    # List all .h5 files sorted
    epoch_files = sorted(
        f for f in os.listdir(patient_stfts_dir) if f.lower().endswith(".h5")
    )

    stft_store_list: list[StftStore] = []

    for epoch_file in epoch_files:
        full_path = os.path.join(patient_stfts_dir, epoch_file)

        with h5py.File(full_path, "r") as f:
            stft_store_list.append(
                StftStore(
                    phase=f.attrs["phase"],
                    start=f.attrs["start"],
                    end=f.attrs["end"],
                    stft_db=f["stft_db"][:],
                    power=f["power"][:]
                    if "power" in f
                    else np.empty((0,), dtype=np.float32),
                    Zxx=f["Zxx"][:],
                    mag=f["mag"][:],
                    freqs=f["freqs"][:],
                    times=f["times"][:],
                    seizure_id=int(f.attrs.get("seizure_id", -1)),
                    file_name=str(f.attrs["file_name"]),
                )
            )

    return stft_store_list
