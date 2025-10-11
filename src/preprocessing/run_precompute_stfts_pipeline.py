"""
Precompute STFTS Pipeline Module

This module is the main module for piecing togother all the functions that are needed to precompute
STFTS from the EEG recordings.
"""

import gc
import os
import random
from typing import cast

import numpy as np
from mne.io.base import BaseRaw, concatenate_raws
from natsort import natsorted
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.config import DataConfig
from src.logger import get_all_active_loggers, setup_logger
from src.preprocessing.data_transformation import precompute_stfts
from src.utils import (
    export_to_csv,
    is_precomputed_data_exists,
    load_patient_summary,
    set_seed,
)

from .data_cleaning import (
    apply_filters,
    extract_seizure_intervals,
    load_raw_recordings,
    segment_intervals,
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
                    seizure_files_data,
                    no_seizure_files_data,
                ) = extract_seizure_intervals(patient_summary)
                logger.info(f"Number of seizures: {len(ictal_intervals)}")

            # Initial values
            seizure_filenames = [file.file_name for file in seizure_files_data]
            no_seizure_filenames = [file.file_name for file in no_seizure_files_data]

            cropped_no_seizure_files = no_seizure_files_data
            cropped_no_seizure_filenames = {
                file.file_name for file in cropped_no_seizure_files
            }
            filtered_interictal_intervals = interictal_intervals
            combined_file_data = seizure_files_data + no_seizure_files_data

            if combined_file_data:
                durations = [interval.duration for interval in combined_file_data]
                average_recording_duration = round(np.mean(durations))  # in hours

            logger.info(f"Average recording duration (s): {average_recording_duration}")

            # We only do non-seizure file reduction when the average recording time
            # of the patient is above 3600s (or 2hr)
            if average_recording_duration > 7200:
                # Filter interictal intervals to only include the files we want to keep
                num_to_keep = len(ictal_intervals)
                cropped_no_seizure_files = random.sample(
                    no_seizure_files_data, min(num_to_keep, len(no_seizure_files_data))
                )
                cropped_no_seizure_filenames = {
                    file.file_name for file in cropped_no_seizure_files
                }
                # Only keep interictal intervals from the cropped file list
                filtered_interictal_intervals = [
                    interval
                    for interval in interictal_intervals
                    if interval.file_name in cropped_no_seizure_filenames
                ]

                no_seizure_filenames = [
                    file.file_name for file in cropped_no_seizure_files
                ]

            combined_intervals = preictal_intervals + filtered_interictal_intervals

            logger.info(
                f"Keeping {len(cropped_no_seizure_files)} of {len(no_seizure_files_data)} non-seizure files "
                f"with {len(filtered_interictal_intervals)} interictal intervals"
            )
            logger.info(
                f"Selected non-seizure files for interictal intervals: {cropped_no_seizure_filenames}"
            )

            file_names = seizure_filenames + no_seizure_filenames
            # We need to sort the filenames beacuse when then non-seizure file reduction
            # clause in the run_precompute_stfts_pipeline has been activated, picking of non
            # nonn-seiuze files is random therefore not maintaning the order of it in the list.
            # And so once the filenames list has been passed into this function, the load order
            # is no longer chronological. Which then once concatenated, the recording not in
            # its natural order anymore. Natsorted is fuction from a module called natsort, it allows
            # us to naturally sort objects e.g. chb01_01, cbh01_02, and so on.

            sorted_file_names = natsorted(file_names)

            patient_stfts_dir = os.path.join(
                DataConfig.precomputed_data_path, f"patient_{patient_id}", "stfts"
            )
            os.makedirs(patient_stfts_dir, exist_ok=True)

            if is_precomputed_data_exists(data_path=patient_stfts_dir):
                logger.info(
                    f"Precomputed STFTs found for patient {patient_id} — skipping reading/precompute."
                )
                # Still count the intervals even though we skip processing
                segmented_intervals = segment_intervals(combined_intervals)
                phase_counts: dict[str, int] = {}
                for epoch in segmented_intervals:
                    phase_counts[epoch.phase] = phase_counts.get(epoch.phase, 0) + 1
            else:
                # 2. Loading patient recordings and concatenating all recordings into 1 continuous recording
                raw_recordings = load_raw_recordings(patient_id, sorted_file_names)
                raw_concatenated = cast(
                    BaseRaw, concatenate_raws(raw_recordings, preload=False)
                )

                """
                    Data Preprocessing
                    - Filtering
                    - 30-second epoch segmentation
                    - Computation of STFT
                """

                logger.info("Loading data into memory")
                loaded_raw = raw_concatenated.load_data()
                logger.info(
                    f"Memory size: {loaded_raw.get_data().nbytes / (1024**2):.2f} MB"
                )

                # 1. Filtering
                filtered_recording = apply_filters(loaded_raw)

                # 2. 30-second epoch segmentation
                segmented_intervals = segment_intervals(combined_intervals)

                # Removed the loadded raw recording after filtering to save ram.
                # The filtered recording will then be loaded again in the precompute stfts epoch
                del loaded_raw
                gc.collect()

                # 3. Computation of STFT
                # NOTE: STFTS will be precomputed and saved on the disk

                # Void function for precomputing the stfts, this will also instantly save these stfts
                # on the disk
                phase_counts = precompute_stfts(
                    recording=filtered_recording,
                    patient_stfts_dir=patient_stfts_dir,
                    segmented_intervals=segmented_intervals,
                )

                patient_stfts_summary = {
                    "patient_id": patient_id,
                    "number_of_seizures": len(ictal_intervals),
                    "preictal_intervals": phase_counts.get("preictal", 0),
                    "interictal_intervals": phase_counts.get("interictal", 0),
                }

                # Write to csv the summarized patient precomputed stfts
                os.makedirs(DataConfig.runs_dir, exist_ok=True)

                precomputed_stfts_summary_path = os.path.join(
                    DataConfig.runs_dir, "precomputed_stfts.csv"
                )
                fieldnames = [
                    "patient_id",
                    "number_of_seizures",
                    "preictal_intervals",
                    "interictal_intervals",
                ]
                export_to_csv(
                    path=precomputed_stfts_summary_path,
                    fieldnames=fieldnames,
                    data=[patient_stfts_summary],
                    mode="a",
                )


if __name__ == "__main__":
    main()
