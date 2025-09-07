"""
Time-Frquency Pipeline Module

This module is the main module for piecing togother all the functions that are needed to process
the time-frequency features and extract them into mosaic tensor.
"""

import os

import numpy as np
from tqdm import tqdm

from src.config import NUMBER_OF_PATIENTS, PRECOMPUTED_DATA_PATH
from src.datatypes import StftStore
from src.logger import setup_logger
from src.utils import is_precomputed_data_exists

from .data_transformation import build_epoch_mosaic, normalize_globally, resize_to_224
from .loaders import load_precomputed_stfts
from .time_frequency_branch import create_band_groupings_per_channel

logger = setup_logger(name="time_frequency_pipeline")


def main():
    """
    Data Preprocessing
    - Data transformation for time-frequency branch
        - Band groupings per channel
        - Mosaic for STFT matrix
        - Global normalization
        - Resize to 224x224x3
        - Save to npz
    """
    logger.info("Starting time-frequency pipeline")

    for idx in tqdm(range(1, NUMBER_OF_PATIENTS + 1), desc="Patients"):
        patient_id = f"{idx:02d}"
        logger.info(f"Processing patient {patient_id}")

        patient_stfts_dir = os.path.join(
            PRECOMPUTED_DATA_PATH, f"patient_{patient_id}", "stfts"
        )
        patient_time_frequency_dir = os.path.join(
            PRECOMPUTED_DATA_PATH, f"patient_{patient_id}", "time-frequency"
        )
        os.makedirs(patient_time_frequency_dir, exist_ok=True)

        if is_precomputed_data_exists(data_path=patient_time_frequency_dir):
            logger.info(
                f"Precomputed mosaics found for patient {patient_id} — skipping."
            )
            continue

        # 1. Load precomputed stfts
        stfts_by_channel: dict[str, list[StftStore]] = load_precomputed_stfts(
            patient_stfts_dir
        )

        # 2. Band groupings per channel
        band_maps = create_band_groupings_per_channel(stfts_by_channel)

        # Get epochs from first channel
        first_channel_epochs = next(iter(stfts_by_channel.values()))
        n_epochs = len(first_channel_epochs)

        # 3. Iterate over epochs and build mosaics
        processed_files = 0
        for epoch_idx in tqdm(
            range(n_epochs), desc=f"Processing epochs for patient {patient_id}"
        ):
            epoch_meta: StftStore = first_channel_epochs[epoch_idx]

            mosaic, mode = build_epoch_mosaic(
                band_maps=band_maps,
                stfts_by_channel=stfts_by_channel,
                epoch_idx=epoch_idx,
                type="time_frequency_band",
            )

            # 4. Global normalization
            normalized = normalize_globally(mosaic)

            # 5. Resize to 224x224x3
            resized = resize_to_224(normalized)

            # 6. Save to npz
            filename = os.path.join(
                patient_time_frequency_dir,
                f"{mode}_epoch_{epoch_meta.start}_{epoch_meta.end}.npz",
            )

            if not os.path.exists(filename):
                np.savez_compressed(
                    filename,
                    type=str(type),
                    tensor=resized,
                    start=epoch_meta.start,
                    end=epoch_meta.end,
                    phase=epoch_meta.phase,
                )
                processed_files += 1

        logger.info(f"Finished building mosaics for patient {patient_id}")
        logger.info(f"Total mosaics files created: {processed_files}")
        logger.info(f"Results saved in: {patient_time_frequency_dir}")


if __name__ == "__main__":
    main()
