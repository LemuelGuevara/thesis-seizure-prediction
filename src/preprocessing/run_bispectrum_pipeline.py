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
from src.datatypes import StftStore
from src.logger import get_all_active_loggers, setup_logger
from src.preprocessing.bispectrum_branch import compute_bispectrum_estimation
from src.preprocessing.data_transformation import resize_to_224
from src.utils import is_precomputed_data_exists

from .loaders import load_precomputed_stfts

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
            loaded_stfts = load_precomputed_stfts(patient_stfts_dir)

            all_zxx = [stft.Zxx for stft in loaded_stfts]
            zxx_global_min = np.min([np.abs(z).min() for z in all_zxx])
            zxx_global_max = np.max([np.abs(z).max() for z in all_zxx])

            for idx, stft in enumerate(
                tqdm(loaded_stfts, desc="Processing epochs", leave=False)
            ):
                stft = cast(StftStore, stft)
                Zxx = stft.Zxx  # (C, F, T)
                freqs = stft.freqs
                # Zxx_norm = stft.Zxx / (zxx_global_max + 1e-10)

                # 2–3. Compute bispectrum per channel*
                bis_list = []
                for ch in range(Zxx.shape[0]):
                    _, bis_db = compute_bispectrum_estimation(Zxx=Zxx[ch], freqs=freqs)
                    bis_list.append(bis_db)

                bispectrum_db_avg = np.mean(bis_list, axis=0)

                # 4. Resize to 224x224x3
                resized = resize_to_224(bispectrum_db_avg)

                # 7. Save to npz
                filename = os.path.join(
                    patient_bispectrum_dir,
                    f"{stft.phase}_bispectrum_{idx:06d}.npz",
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
                        n_channels=stft.Zxx.shape[0],
                    )

            logger.info(
                f"Finished building time-frequency mosaics for patient {patient_id}"
            )
            logger.info(f"Results saved in: {patient_bispectrum_dir}")

            os.makedirs(DataConfig.runs_dir, exist_ok=True)

            precomputed_tf_summary_path = os.path.join(
                DataConfig.runs_dir, "precomputed_bis.csv"
            )
            fieldnames = ["patient_id", "mosaics"]


if __name__ == "__main__":
    main()
