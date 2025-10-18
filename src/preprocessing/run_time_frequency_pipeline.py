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
    resize_to_224,
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
            loaded_stfts = load_precomputed_stfts(patient_stfts_dir)

            # stft_spec = [stft.stft_db for stft in loaded_stfts]
            # stft_spec_global_min = np.min([np.abs(z).min() for z in stft_spec])
            # stft_spec_global_max = np.max([np.abs(z).max() for z in stft_spec])

            # 3. Process each window
            for idx, stft in enumerate(
                tqdm(loaded_stfts, desc="Processing epochs", leave=False)
            ):
                stft = cast(StftStore, stft)
                stft_feat = stft.stft_db

                if stft_feat.shape[0] > 3:
                    logger.info(f"Averaging across all {stft_feat.shape[0]}")
                    stft_feat_mean = np.mean(stft_feat, axis=0)
                else:
                    stft_feat_mean = stft_feat

                freqs = stft.freqs

                logger.info(
                    f"STFT freqs range: {freqs.min():.2f}-{freqs.max():.2f} Hz, len={len(freqs)}"
                )

                # stft_feat_mean_norm = stft_feat_mean / (stft_spec_global_max + 1e-10)

                # 6. Resize to 224x224x3
                resized = resize_to_224(stft_feat_mean)

                logger.info(f"After resize: resized shape = {resized.shape}")

                # 7. Save to npz
                filename = os.path.join(
                    patient_time_frequency_dir,
                    f"{stft.phase}_tf_{idx:06d}.npz",
                )

                if not os.path.exists(filename):
                    np.savez_compressed(
                        filename,
                        tensor=resized,
                        start=stft.start,
                        end=stft.end,
                        phase=stft.phase,
                        freqs=stft.freqs,
                        seizure_id=stft.seizure_id,
                        n_channels=stft.stft_db.shape[0],
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
