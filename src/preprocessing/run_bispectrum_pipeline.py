"""
Bispectrum Pipeline Module

This module is the main module for piecing togother all the functions that are needed to process
the bispectrum features and extract them into mosaic tensor.
"""

import os

import numpy as np
from tqdm import tqdm

from src.config import NUMBER_OF_PATIENTS, PRECOMPUTED_DATA_PATH
from src.datatypes import StftStore
from src.logger import setup_logger

from .bispectrum_branch import (
    compute_band_average_stft_coeffs,
    compute_band_level_bispectrum,
)
from .data_transformation import build_epoch_mosaic, normalize_globally, resize_to_224
from .loaders import load_precomputed_stfts

logger = setup_logger(name="bispectrum_pipeline")


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

    for idx in tqdm(range(1, NUMBER_OF_PATIENTS + 1), desc="Patients"):
        patient_id = f"{idx:02d}"
        logger.info(f"Processing patient {patient_id}")

        patient_stfts_dir = os.path.join(
            PRECOMPUTED_DATA_PATH, f"patient_{patient_id}", "stfts"
        )
        patient_bispectrum_dir = os.path.join(
            PRECOMPUTED_DATA_PATH, f"patient_{patient_id}", "bispectrum"
        )
        os.makedirs(patient_bispectrum_dir, exist_ok=True)

        # 1. Load precomputed STFTs
        stfts_by_channel: dict[str, list[StftStore]] = load_precomputed_stfts(
            patient_stfts_dir
        )

        epochs_first_channel = next(iter(stfts_by_channel.values()))
        n_epochs = len(epochs_first_channel)

        for epoch_idx in tqdm(
            range(n_epochs), desc=f"Processing epochs for patient {patient_id}"
        ):
            # 2–3. Compute bispectrum *per channel*
            bispectra_db: list[np.ndarray] = []
            for ch_name, epochs in stfts_by_channel.items():
                epoch_meta: StftStore = epochs[epoch_idx]

                band_complex, band_centers = compute_band_average_stft_coeffs(
                    Zxx=epoch_meta.Zxx, freqs=epoch_meta.freqs
                )

                _, bispectrum_db = compute_band_level_bispectrum(
                    Zxx=epoch_meta.Zxx,
                    freqs=epoch_meta.freqs,
                    band_complex=band_complex,
                    band_centers=band_centers,
                )

                bispectra_db.append(bispectrum_db)

            # stack into (n_channels, n_bands, n_bands)
            bispectra_db = np.stack(bispectra_db, axis=0)

            # 4. Build per-epoch mosaic
            mosaic, mode = build_epoch_mosaic(
                stfts_by_channel=stfts_by_channel,
                epoch_idx=epoch_idx,
                type="bispectrum",
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
                f"bispectrum_mosaic_epoch_{epoch_meta.start}_{epoch_meta.end}.npz",
            )

            processed_files = 0
            if not os.path.exists(filename):
                np.savez_compressed(
                    filename,
                    tensor=resized,
                    start=epoch_meta.start,
                    end=epoch_meta.end,
                    phase=epoch_meta.phase,
                    freqs=band_centers,
                )

                processed_files += 1

        logger.info(
            f"Finished building time-frequency mosaics for patient {patient_id}"
        )
        logger.info(f"Total mosaics files created: {processed_files}")
        logger.info(f"Results saved in: {patient_bispectrum_dir}")


if __name__ == "__main__":
    main()
