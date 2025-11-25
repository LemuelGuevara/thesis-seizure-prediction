"""
Precompute STFTS Pipeline Module

This module is the main module for piecing togother all the functions that are needed to precompute
STFTS from the EEG recordings.
"""

import os

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.config import DataConfig, PreprocessingConfig
from src.logger import get_all_active_loggers, setup_logger
from src.preprocessing.data_transformation import precompute_stft
from src.preprocessing.loaders import load_raw_recordings
from src.utils import (
    export_to_csv,
    is_precomputed_data_exists,
    load_patient_summary,
    set_seed,
)

from .data_cleaning import (
    apply_filters,
    apply_ica,
    parse_patient_summary_intervals,
    segment_recordings,
)

logger = setup_logger(name="run_precompute_stfts")
active_loggers = get_all_active_loggers()


def main():
    set_seed()
    logger.info("Starting precomputation of STFTs for all patients")

    with logging_redirect_tqdm(loggers=active_loggers):
        for patient in tqdm(DataConfig.patients_to_process, desc="Patients"):
            patient_id = f"{patient:02d}"

            logger.info(f"Processing patient {patient_id}")

            """
                Data Preparation
                - Extraction of seizure recordings
                - Normalizing time format 
                - Loading patient recordings
                - Concatenating all recordings of each patient into 1 continuous recording
            """

            patient_stfts_dir = os.path.join(
                DataConfig.precomputed_data_path, f"patient_{patient_id}", "stfts"
            )
            os.makedirs(patient_stfts_dir, exist_ok=True)

            # 1. Extraction of seizure intervals and normalization of time format
            with load_patient_summary(
                patient_id, DataConfig.dataset_path
            ) as patient_summary:
                (
                    preictal_intervals,
                    interictal_intervals,
                    ictal_intervals,
                    seizure_files_data,
                    no_seizure_files_data,
                ) = parse_patient_summary_intervals(patient_summary)
                logger.info(f"Number of seizures: {len(ictal_intervals)}")

            combined_intervals = preictal_intervals + interictal_intervals
            total_phase_counts = {"preictal": 0, "interictal": 0}

            # 2. 30-second epoch segmentation
            segmented_recording_epochs = segment_recordings(
                combined_intervals, undersampling=True
            )

            logger.info(
                f"Total epochs to process for patient: {len(segmented_recording_epochs)}"
            )

            if is_precomputed_data_exists(data_path=patient_stfts_dir):
                logger.info(
                    f"Precomputed STFTs found for patient {patient_id} — skipping reading/precompute."
                )
            else:
                # 2. Loading patient recordings and concatenating all recordings into 1 continuous recording
                valid_files = sorted(
                    {
                        os.path.basename(str(iv.file_name))
                        for iv in segmented_recording_epochs
                    }
                )

                raw_recordings = load_raw_recordings(patient_id, valid_files)

                """
                    Data Preprocessing
                    - Filtering
                    - 30-second epoch segmentation
                    - Computation of STFT
                """

                for raw in raw_recordings:
                    # NOTE: we should only process recordings that are from the intervals extracted
                    # meaning if the the intervals only have 3 files/filenames, we should in use load
                    # those recordings.
                    raw_recording_filename = os.path.basename(str(raw.filenames[0]))

                    logger.info(f"Processing file: {raw.filenames[0]}")
                    logger.info("Loading data into memory")
                    loaded_raw = raw.load_data()
                    logger.info(
                        f"Memory size: {loaded_raw.get_data().nbytes / (1024**2):.2f} MB"
                    )

                    # 1. Filtering
                    filtered_recording = apply_filters(loaded_raw)
                    cleaned_recording = (
                        apply_ica(filtered_recording)
                        if PreprocessingConfig.apply_ica
                        else filtered_recording
                    )

                    epochs_this_file = [
                        iv
                        for iv in segmented_recording_epochs
                        if os.path.basename(str(iv.file_name)) == raw_recording_filename
                    ]
                    if not epochs_this_file:
                        logger.info(
                            f"No intervals for {raw_recording_filename}; skipping."
                        )
                        continue

                    # 3. Computation of STFT
                    # NOTE: STFTS will be precomputed and saved on the disk
                    # Void function for precomputing the stfts, this will also instantly save these stfts
                    # on the disk
                    current_phase_counts = precompute_stft(
                        recording=cleaned_recording,
                        patient_stfts_dir=patient_stfts_dir,
                        segmented_epochs=epochs_this_file,
                    )

                    total_phase_counts["preictal"] += current_phase_counts.get(
                        "preictal", 0
                    )
                    total_phase_counts["interictal"] += current_phase_counts.get(
                        "interictal", 0
                    )

                patient_stfts_summary = {
                    "patient_id": patient_id,
                    "number_of_seizures": len(ictal_intervals),
                    "preictal_epochs": total_phase_counts.get("preictal", 0),
                    "interictal_epochs": total_phase_counts.get("interictal", 0),
                }

                # Write to csv the summarized patient precomputed stfts
                os.makedirs(DataConfig.runs_dir, exist_ok=True)

                precomputed_stfts_summary_path = os.path.join(
                    DataConfig.runs_dir, "precomputed_stfts.csv"
                )
                fieldnames = [
                    "patient_id",
                    "number_of_seizures",
                    "preictal_epochs",
                    "interictal_epochs",
                ]
                export_to_csv(
                    path=precomputed_stfts_summary_path,
                    fieldnames=fieldnames,
                    data=[patient_stfts_summary],
                    mode="a",
                )


if __name__ == "__main__":
    main()
