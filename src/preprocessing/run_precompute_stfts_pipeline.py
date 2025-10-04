"""
Precompute STFTS Pipeline Module

This module is the main module for piecing togother all the functions that are needed to precompute
STFTS from the EEG recordings.
"""

import gc
import os
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.config import DataConfig
from src.datatypes import EpochInterval
from src.logger import get_all_active_loggers, setup_logger
from src.utils import export_to_csv, is_precomputed_data_exists, load_patient_summary

from .data_cleaning import (
    apply_filters,
    extract_seizure_intervals,
    load_raw_recordings,
    segment_intervals,
)
from .data_transformation import precompute_stfts

logger = setup_logger(name="run_precompute_stfts")
active_loggers = get_all_active_loggers()


def main():
    logger.info("Starting precomputation of STFTs for all patients")

    precomputed_stfts_summary: list[dict[str, int]] = []

    with logging_redirect_tqdm(loggers=active_loggers):
        for patient in tqdm(DataConfig.patients_to_process, desc="Patients"):
            patient_id = f"{patient:02d}"

            logger.info(f"Processing patient {patient_id}")
            precomputed_stfts_summary: list[dict[str, int]] = []

            """
                Data Preparation
                - Extraction of seizure intervals
                - Normalizing time format 
                - Loading patient recordings
                - Concatenating all recordings of each patient into 1 continuous recording
                """

            # 1. Extraction of seizure intervals and normalization of time format
            # --- Always extract seizure intervals for summary data ---
            with load_patient_summary(
                patient_id, DataConfig.dataset_path
            ) as patient_summary:
                (
                    preictal_intervals,
                    interictal_intervals,
                    ictal_intervals,
                ) = extract_seizure_intervals(patient_summary)
                logger.info(f"Number of seizures: {len(ictal_intervals)}")

            patient_stfts_dir = os.path.join(
                DataConfig.precomputed_data_path, f"patient_{patient_id}", "stfts"
            )
            os.makedirs(patient_stfts_dir, exist_ok=True)

            all_intervals = preictal_intervals + interictal_intervals
            file_intervals: dict[str, list[EpochInterval]] = {}
            for interval in all_intervals:
                if interval.file_name is None:
                    continue
                file_intervals.setdefault(interval.file_name, []).append(interval)

            # Initialize phase counts once per patient
            phase_counts: dict[str, int] = {}

            if is_precomputed_data_exists(data_path=patient_stfts_dir):
                logger.info(
                    f"Precomputed STFTs found for patient {patient_id} — skipping reading/precompute."
                )
                # Still count the intervals even though we skip processing
                segmented_intervals = segment_intervals(
                    preictal_intervals + interictal_intervals
                )
                for epoch in segmented_intervals:
                    phase_counts[epoch.phase] = phase_counts.get(epoch.phase, 0) + 1
            else:
                # 2. Loading patient recordings and concatenating all recordings into 1 continuous recording
                raw_recordings = load_raw_recordings(patient_id)

                """
                    Data Preprocessing
                    - Filtering
                    - 30-second epoch segmentation
                    - Computation of STFT
                    """

                for raw_recording in raw_recordings:
                    file_name = Path(raw_recording.filenames[0]).name
                    intervals = file_intervals.get(file_name, [])
                    if not intervals:
                        continue

                    loaded_raw = raw_recording.load_data()

                    logger.info(f"Processing file: {file_name}")
                    logger.info(
                        f"Loading data into memory: {raw_recording.filenames[0]}"
                    )
                    logger.info(
                        f"Loaded Raw Memory size: {loaded_raw.get_data().nbytes / (1024**2):.2f} MB"
                    )

                    # 1. Filtering
                    filtered_recording = apply_filters(loaded_raw)

                    # Removed the loaded raw after filtering to save ram.
                    # The filtered recording will then be loaded again in the precompute stfts epoch
                    del loaded_raw
                    gc.collect()

                    # 2. 30-second epoch segmentation
                    segmented_intervals = segment_intervals(intervals)

                    # 3. Computation of STFT
                    # NOTE: STFTS will be precomputed and saved on the disk
                    file_phase_counts = precompute_stfts(
                        recording=filtered_recording,
                        patient_stfts_dir=patient_stfts_dir,
                        segmented_intervals=segmented_intervals,
                    )

                    # Aggregate counts across files
                    for phase, count in file_phase_counts.items():
                        phase_counts[phase] = phase_counts.get(phase, 0) + count

                    del filtered_recording
                    gc.collect()

            # Append **once per patient**
            precomputed_stfts_summary.append(
                {
                    "patient_index": patient,
                    "number_of_seizures": len(ictal_intervals),
                    "preictal_intervals": phase_counts.get("preictal", 0),
                    "interictal_intervals": phase_counts.get("interictal", 0),
                }
            )

    # Write to csv the summarized patient precomputed stfts
    precomputed_stfts_summary_path = os.path.join(
        DataConfig.precomputed_data_path, "precomputed_stfts_summary.csv"
    )
    fieldnames = [
        "patient_index",
        "number_of_seizures",
        "preictal_intervals",
        "interictal_intervals",
    ]
    export_to_csv(
        path=precomputed_stfts_summary_path,
        fieldnames=fieldnames,
        data=precomputed_stfts_summary,
    )


if __name__ == "__main__":
    main()
