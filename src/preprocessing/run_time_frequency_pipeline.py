"""
Time-Frequency Pipeline Module

This module is the main module for piecing together all the functions that are needed to process
the time-frequency features and extract them into mosaic tensor.
"""

import os
from typing import cast

import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.config import DataConfig
from src.datatypes import StftStore
from src.logger import get_all_active_loggers, setup_logger
from src.preprocessing.data_transformation import (
    create_efficientnet_img,
)
from src.utils import is_precomputed_data_exists

from .loaders import load_precomputed_stfts

logger = setup_logger(name="time_frequency_pipeline")
active_loggers = get_all_active_loggers()


def main():
    """
    Data Preprocessing
    - Data transformation for time-frequency branch
        - Reduce channels: R=mean, G=max, B=std
        - Global normalization
        - Resize to 224x224x3
        - Save to npz
    """
    logger.info("Starting time-frequency pipeline")

    with logging_redirect_tqdm(loggers=active_loggers):
        for patient in tqdm(DataConfig.patients_to_process, desc="Patients"):
            patient_id = f"{patient:02d}"
            logger.info(f"Processing patient {patient_id}")

            patient_stfts_dir = os.path.join(
                DataConfig.precomputed_data_path, f"patient_{patient_id}", "stfts"
            )
            patient_time_frequency_dir = os.path.join(
                DataConfig.precomputed_data_path,
                f"patient_{patient_id}",
                "time-frequency",
            )
            os.makedirs(patient_time_frequency_dir, exist_ok=True)

            if is_precomputed_data_exists(data_path=patient_time_frequency_dir):
                logger.info(
                    f"Precomputed time-frequency mosaics found for patient {patient_id} — skipping."
                )
                continue

            # 1. Load precomputed stfts (all channels)
            loaded_stft_epochs = load_precomputed_stfts(patient_stfts_dir)

            # 3. Process each window
            for idx, stft_epoch in enumerate(
                tqdm(loaded_stft_epochs, desc="Processing epochs", leave=False)
            ):
                stft_epoch = cast(StftStore, stft_epoch)

                # stft_band_averaged = band_average_stft(stft)
                stft_mag = stft_epoch.mag
                stft_mag_avg = np.mean(stft_mag, axis=0)

                # 6. Resize to 224x224x3
                spectrogram_mosaic = create_efficientnet_img(stft_mag_avg)
                logger.info(
                    f"Spectrogram shape: {spectrogram_mosaic.shape} | "
                    f"Epoch index: {idx} | "
                    f"Start: {stft_epoch.start} | "
                    f"End: {stft_epoch.end} | "
                    f"Phase: {stft_epoch.phase}"
                )

                # 7. Save to npz
                out_name = f"{stft_epoch.file_name}_tf"
                filename = os.path.join(patient_time_frequency_dir, f"{out_name}.npz")

                if not os.path.exists(filename):
                    np.savez_compressed(
                        filename,
                        tensor=spectrogram_mosaic,
                        start=stft_epoch.start,
                        end=stft_epoch.end,
                        phase=stft_epoch.phase,
                        freqs=stft_epoch.freqs,
                        seizure_id=stft_epoch.seizure_id,
                        n_channels=stft_epoch.stft_db.shape[0],
                        file_name=out_name,
                    )

            logger.info(
                f"Finished building time-frequency mosaics for patient {patient_id}"
            )
            logger.info(f"Results saved in: {patient_time_frequency_dir}")

            os.makedirs(DataConfig.runs_dir, exist_ok=True)

            precomputed_tf_summary_path = os.path.join(
                DataConfig.runs_dir, "precomputed_tf.csv"
            )
            fieldnames = ["patient_id", "mosaics"]


if __name__ == "__main__":
    main()
