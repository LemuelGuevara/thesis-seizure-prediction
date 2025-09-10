"""
Precompute STFTS Pipeline Module

This module is the main module for piecing togother all the functions that are needed to precompute
STFTS from the EEG recordings.
"""

import os

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.config import DataConfig
from src.logger import get_all_active_loggers, setup_logger
from src.utils import is_precomputed_data_exists, load_patient_summary

from .data_cleaning import (
    apply_filters,
    extract_seizure_intervals,
    load_patient_recording,
    segment_intervals,
)
from .data_transformation import precompute_stfts

logger = setup_logger(name="run_precompute_stfts")
active_loggers = get_all_active_loggers()


def main():

    logger.info("Starting precomputation of STFTs for all patients")

    with logging_redirect_tqdm(loggers=active_loggers):
        for idx in tqdm(range(1, DataConfig.number_of_patients + 1), desc="Patients"):
            patient_id = f"{idx:02d}"

            logger.info(f"Processing patient {patient_id}")
            patient_stfts_dir = os.path.join(
                DataConfig.precomputed_data_path, f"patient_{patient_id}", "stfts"
            )
            os.makedirs(patient_stfts_dir, exist_ok=True)

            if is_precomputed_data_exists(data_path=patient_stfts_dir):
                logger.info(
                    f"Precomputed STFTs found for patient {patient_id} — skipping reading/precompute."
                )
                continue

            """
            Data Preparation
            - Extraction of seizure intervals
            - Normalizing time format 
            - Loading patient recordings
            - Concatenating all recordings of each patient into 1 continuous recording
            """

            # 1. Extraction of seizure intervals and normalization of time format
            with load_patient_summary(
                patient_id, DataConfig.dataset_path
            ) as patient_summary:
                (
                    preictal_intervals,
                    interictal_intervals,
                    ictal_intervals,
                ) = extract_seizure_intervals(patient_summary)
                logger.info(f"Number of seizures: {len(ictal_intervals)}")

            # 2. Loading patient recordings and concatenating all recordings into 1 continuous recording
            recording = load_patient_recording(patient_id)

            """
            Data Preprocessing
            - Filtering
            - 30-second epoch segmentation
            - Computation of STFT
            """

            # 1. Filtering
            filtered_recording = apply_filters(recording)

            # 2. 30-second epoch segmentation
            segmented_intervals = segment_intervals(
                preictal_intervals
                + interictal_intervals
                # Combining all extracted intervals
            )

            # 3. Computation of STFT
            # NOTE: STFTS will be precomputed and saved on the disk

            # Void function for precomputing the stfts, this will also instantly save these stfts
            # on the disk
            precompute_stfts(
                recording=filtered_recording,
                patient_stfts_dir=patient_stfts_dir,
                segmented_intervals=segmented_intervals,
            )


if __name__ == "__main__":
    main()
