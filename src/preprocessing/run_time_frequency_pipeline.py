"""
Time-Frequency Pipeline Module

This module is the main module for piecing together all the functions that are needed to process
the time-frequency features and extract them into mosaic tensor.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def process_epoch(stft_epoch: StftStore, patient_time_frequency_dir: str, idx: int):
    """
    Process one STFT epoch → create mosaic → save .npz
    """
    stft_mag = stft_epoch.mag
    stft_mag_avg = np.mean(stft_mag, axis=0)
    spectrogram_mosaic = create_efficientnet_img(stft_mag_avg)

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


def main():
    """
    Data Preprocessing
    - Data transformation for time-frequency branch
      (parallelized with ThreadPoolExecutor)
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

            # 2. Process each epoch in parallel
            with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
                futures = [
                    executor.submit(
                        process_epoch,
                        cast(StftStore, stft),
                        patient_time_frequency_dir,
                        idx,
                    )
                    for idx, stft in enumerate(loaded_stft_epochs)
                ]

                # tqdm progress bar while futures complete
                for _ in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Processing epochs",
                    leave=False,
                ):
                    pass

            logger.info(
                f"Finished building time-frequency mosaics for patient {patient_id}"
            )
            logger.info(f"Results saved in: {patient_time_frequency_dir}")


if __name__ == "__main__":
    main()
