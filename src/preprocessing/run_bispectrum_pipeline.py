"""
Bispectrum Pipeline Module

This module is the main module for piecing togother all the functions that are needed to process
the bispectrum features and extract them into mosaic tensor.
"""

import os
from typing import cast

import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.config import DataConfig
from src.datatypes import StftData
from src.logger import get_all_active_loggers, setup_logger
from src.preprocessing.bispectrum_branch import bispectrum_estimation
from src.preprocessing.data_transformation import (
    create_efficientnet_img,
    group_freqs_into_bands,
)
from src.preprocessing.loaders import load_precomputed_stfts
from src.utils import is_precomputed_data_exists

logger = setup_logger(name="bispectrum_pipeline")
active_loggers = get_all_active_loggers()


def main():
    """
    Data Preprocessing
    - Data transformation for bispectrum branch
        - PAC at band-level
        - Bispectrum estimation at band-level (per channel)
        - Global normalization
        - Resize to 224x224x3
        - Save to npz
    """
    with logging_redirect_tqdm(loggers=active_loggers):
        for patient in tqdm(DataConfig.patients_to_process, desc="Patients"):
            patient_id = f"{patient:02d}"
            logger.info(f"Processing patient {patient_id}")

            patient_stfts_dir = os.path.join(
                DataConfig.precomputed_data_path, f"patient_{patient_id}", "stfts"
            )
            patient_bispectrum_dir = os.path.join(
                DataConfig.precomputed_data_path, f"patient_{patient_id}", "bispectrum"
            )
            os.makedirs(patient_bispectrum_dir, exist_ok=True)

            if is_precomputed_data_exists(data_path=patient_bispectrum_dir):
                logger.info(
                    f"Precomputed bispectrum mosaics found for patient {patient_id} — skipping."
                )
                continue

            # 1. Load precomputed STFTs
            loaded_stft_epochs = load_precomputed_stfts(patient_stfts_dir)

            for idx, stft_epoch in enumerate(
                tqdm(loaded_stft_epochs, desc="Processing epochs", leave=False)
            ):
                stft_epoch = cast(StftData, stft_epoch)
                Zxx = stft_epoch.Zxx  # (C, F, T)
                freqs = stft_epoch.freqs

                # 2–3. Compute bispectrum per channel*

                bispectrum_mag, _ = bispectrum_estimation(Zxx=Zxx, freqs=freqs)
                bis_grouped_by_band = group_freqs_into_bands(bispectrum_mag, freqs)

                # 4. Resize to 224x224x3
                bispectra = create_efficientnet_img(bis_grouped_by_band)
                logger.info(
                    f"Bispectrum shape: {bis_grouped_by_band.shape} | "
                    f"Epoch index: {idx} | "
                    f"Start: {stft_epoch.start} | "
                    f"End: {stft_epoch.end} | "
                    f"Phase: {stft_epoch.phase}"
                )

                # 7. Save to npz
                out_name = f"{stft_epoch.file_name}_bis"
                filename = os.path.join(patient_bispectrum_dir, f"{out_name}.npz")

                if not os.path.exists(filename):
                    np.savez_compressed(
                        filename,
                        tensor=bispectra,
                        start=stft_epoch.start,
                        end=stft_epoch.end,
                        phase=stft_epoch.phase,
                        freqs=stft_epoch.freqs,
                        seizure_id=stft_epoch.seizure_id,
                        n_channels=stft_epoch.Zxx.shape[0],
                        file_name=out_name,
                    )

            logger.info(
                f"Finished building bispectrum npz files for patient {patient_id}"
            )
            logger.info(f"Results saved in: {patient_bispectrum_dir}")


if __name__ == "__main__":
    main()
