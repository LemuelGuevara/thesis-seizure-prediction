from src.utils import is_precomputed_data_exists

"""
Bispectrum Pipeline Module

This module is the main module for piecing togother all the functions that are needed to process
the bispectrum features and extract them into mosaic tensor.
"""

import os

import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.config import DataConfig
from src.datatypes import StftStore
from src.logger import get_all_active_loggers, setup_logger

from .bispectrum_branch import (
    compute_bispectrum_estimation,
)
from .data_transformation import build_epoch_mosaic, normalize_globally, resize_to_224
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
        for idx in tqdm(range(1, DataConfig.number_of_patients + 1), desc="Patients"):
            patient_id = f"{idx:02d}"
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
            stfts_by_channel: dict[str, list[StftStore]] = load_precomputed_stfts(
                patient_stfts_dir
            )

            epochs_first_channel = next(iter(stfts_by_channel.values()))
            n_epochs = len(epochs_first_channel)

            processed_files = 0

            for epoch_idx in tqdm(
                range(n_epochs), desc=f"Processing epochs for patient {patient_id}"
            ):
                # 2–3. Compute bispectrum per channel*
                bispectra_db: list[np.ndarray] = []
                for ch_name, epochs in stfts_by_channel.items():
                    epoch_meta: StftStore = epochs[epoch_idx]

                    _, bispectrum_db = compute_bispectrum_estimation(
                        Zxx=epoch_meta.Zxx,
                        freqs=epoch_meta.freqs,
                    )

                    bispectra_db.append(bispectrum_db)

                # stack into (n_channels, n_bands, n_bands)
                bispectra_db = np.stack(bispectra_db, axis=0)

                # 4. Build per-epoch mosaic
                mosaic, mode = build_epoch_mosaic(
                    epoch_idx=epoch_idx,
                    mode="bispectrum",
                    stfts_by_channel=stfts_by_channel,
                    bispectrum_arr_list=bispectra_db,
                )

                # 5. Global normalization
                normalized = normalize_globally(data=mosaic)

                # 6. Resize to 224x224x3
                resized = resize_to_224(normalized)

                # 7. Save to npz
                epoch_meta = epochs_first_channel[epoch_idx]
                filename = os.path.join(
                    patient_bispectrum_dir,
                    f"{epoch_meta.phase}_bispectrum_{epoch_meta.start}_{epoch_meta.end}.npz",
                )

                if not os.path.exists(filename):
                    np.savez_compressed(
                        filename,
                        tensor=resized,
                        start=epoch_meta.start,
                        end=epoch_meta.end,
                        phase=epoch_meta.phase,
                    )

                    processed_files += 1

            logger.info(
                f"Finished building time-frequency mosaics for patient {patient_id}"
            )
            logger.info(f"Total mosaics files created: {processed_files}")
            logger.info(f"Results saved in: {patient_bispectrum_dir}")


if __name__ == "__main__":
    main()
