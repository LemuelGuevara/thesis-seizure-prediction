"""
Data Cleaning Module

This module provides functions for cleaning the EEG recordings before
passing them to time-frequency and bispectrum processing.

"""

import os
import random
import re
import sys
import typing
from datetime import timedelta
from typing import cast

import mne
from mne.io import BaseRaw
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from src.config import DataConfig, PreprocessingConfig
from src.datatypes import IntervalMeta, RecordingFileInfo
from src.logger import setup_logger
from src.utils import parse_time_str

mne.set_log_level("ERROR")

logger = setup_logger(name="data_cleaning")

# Counter for Spectrogram windows
preictal_windows_count: int = 0
interictal_windows_count: int = 0


def parse_patient_summary_intervals(patient_summary: typing.TextIO):
    """
    Parses a patient's summary file and extract all interval information.
    Interval in the context of seizure is the seizure timeline and interval.
    On the other hand, interval in the context of interictals are the full timeline
    of non-seizure files limited to 1hr.
    """

    preictal_intervals: list[IntervalMeta] = []
    interictal_intervals: list[IntervalMeta] = []
    ictal_intervals: list[IntervalMeta] = []

    seizure_files_data: list[RecordingFileInfo] = []
    no_seizure_files_data: list[RecordingFileInfo] = []

    file_name = ""
    pending_seizure_start = None
    file_start = None
    file_end = None
    seizure_counter = 0
    interictal_counter = 0
    last_ictal_end_by_file: dict[str, int] = {}

    logger.info("Only taking preictals that are more than 15mins in duration")

    for line in patient_summary:
        line = line.strip()

        if line.startswith("File Name:"):
            file_name = line.split(":", 1)[1].strip()
        elif line.startswith("File Start Time:"):
            file_start = parse_time_str(line.split(":", 1)[1].strip())
        elif line.startswith("File End Time:"):
            file_end = parse_time_str(line.split(":", 1)[1].strip())
        elif line.startswith("Number of Seizures in File:"):
            no_of_seizures = int(line.split(":", 1)[1].strip().split()[0])
            if no_of_seizures == 0 and file_start and file_end:
                duration_seconds = int((file_end - file_start).total_seconds())
                recording = IntervalMeta(
                    phase="interictal",
                    start=0,
                    end=duration_seconds,
                    file_name=file_name,
                    seizure_id=interictal_counter,
                )
                interictal_intervals.append(recording)
                no_seizure_files_data.append(
                    RecordingFileInfo(file_name, recording, duration_seconds)
                )
                interictal_counter += 1
        elif re.match(r"Seizure(\s+\d+)?\s+Start Time:", line):
            pending_seizure_start = int(line.split(":")[-1].strip().split()[0])
        elif (
            re.match(r"Seizure(\s+\d+)?\s+End Time:", line)
            and pending_seizure_start is not None
        ):
            end_sec = int(line.split(":")[-1].strip().split()[0])
            duration_seconds = 0
            if file_start and file_end:
                duration_seconds = int((file_end - file_start).total_seconds())

            ictal_recording = IntervalMeta(
                phase="ictal",
                start=pending_seizure_start,
                end=end_sec,
                file_name=file_name,
                seizure_id=seizure_counter,
            )
            seizure_files_data.append(
                RecordingFileInfo(file_name, ictal_recording, duration_seconds)
            )
            ictal_intervals.append(ictal_recording)

            # Preictal intervals
            preictal_minutes = PreprocessingConfig.preictal_minutes
            seizure_start_sec = pending_seizure_start

            preictal_start_sec = max(0, seizure_start_sec - preictal_minutes * 60)

            prev_end = last_ictal_end_by_file.get(file_name, 0)
            preictal_start_sec = max(preictal_start_sec, prev_end)

            if file_start:
                preictal_start_dt = file_start + timedelta(seconds=preictal_start_sec)
                seizure_start_dt = file_start + timedelta(seconds=seizure_start_sec)
                duration_min = (
                    seizure_start_dt - preictal_start_dt
                ).total_seconds() / 60.0

                logger.info(
                    f"[{file_name}] Preictal {seizure_counter}: "
                    f"start={preictal_start_dt.time()} end={seizure_start_dt.time()} "
                    f"duration={duration_min:.1f} min"
                )

            if (seizure_start_sec - preictal_start_sec) >= 900:
                preictal_intervals.append(
                    IntervalMeta(
                        phase="preictal",
                        start=preictal_start_sec,
                        end=seizure_start_sec,
                        file_name=file_name,
                        seizure_id=seizure_counter,
                    )
                )

            last_ictal_end_by_file[file_name] = end_sec
            pending_seizure_start = None
            seizure_counter += 1

    return (
        preictal_intervals,
        interictal_intervals,
        ictal_intervals,
        seizure_files_data,
        no_seizure_files_data,
    )


# 1. Loads all EDF recordings of a patient (since we have multiple recordings per patient)
def load_raw_recordings(patient_id: str, file_names: list[str]) -> list[BaseRaw]:
    """
    Loads all raw EDF recordings of a patient without filtering.

    Args:
        patient_id (str): Zero-padded patient ID (e.g. "01").

    Returns:
        list[BaseRaw]: List of unprocessed raw recordings.
    """

    logger.info(f"Loading EDF files for patient {patient_id}")
    patient_folder = os.path.join(DataConfig.dataset_path, f"chb{patient_id}")
    raw_edf_list: list[BaseRaw] = []

    # Load EDF files
    for file_name in tqdm(
        file_names, desc=f"Reading EDF files for patient {patient_id}"
    ):
        recording_path = os.path.join(patient_folder, file_name)
        logger.info(f"Reading file: {file_name}")

        raw_edf = mne.io.read_raw_edf(recording_path, preload=False, verbose="error")
        raw_channels = set(raw_edf.ch_names)
        selected_channels = set(PreprocessingConfig.selected_channels)

        # Only append if all selected channels are present
        if selected_channels.issubset(raw_channels):
            raw_edf.pick(PreprocessingConfig.selected_channels)
            raw_edf_list.append(raw_edf)
        else:
            logger.warning(
                f"Skipping {file_name}: missing channels {selected_channels - raw_channels}"
            )

    if not raw_edf_list:
        raise FileNotFoundError(f"No EDF files found in {patient_folder}")

    return raw_edf_list


# 2. Preprocessing: Filtering
def apply_filters(
    raw: BaseRaw,
) -> BaseRaw:
    """
    Applies bandpass and notch filtering to EEG recording.

    Args:
        raw (BaseRaw): Raw EEG recording.

    Returns:
        BaseRaw: Filtered EEG recording.
    """

    # notch filtering
    raw.notch_filter(PreprocessingConfig.notch_filter)
    logger.info(f"Notch filtered: {PreprocessingConfig.notch_filter} Hz")

    # bandpass filtering

    logger.info("Applying bandpass and notch filters")
    raw.filter(
        l_freq=PreprocessingConfig.low_cutoff_filter,
        h_freq=PreprocessingConfig.high_cutoff_filter,
    )
    logger.info(
        f"Bandpass filtered: {PreprocessingConfig.low_cutoff_filter} - {PreprocessingConfig.high_cutoff_filter} Hz"
    )

    return raw


def balance_epochs(epochs: list[IntervalMeta]):
    logger.info("Balancing epochs (matching total interictal to total preictal)...")

    # Separate by phase
    preictal = [ep for ep in epochs if ep.phase.lower() == "preictal"]
    interictal = [ep for ep in epochs if ep.phase.lower() == "interictal"]

    n_pre = len(preictal)
    n_inter = len(interictal)

    logger.info(f"Before balancing: preictal={n_pre}, interictal={n_inter}")

    # If we have more interictals, randomly downsample them
    if n_inter > n_pre:
        interictal = random.sample(interictal, n_pre)
        logger.info(f"Downsampled interictal from {n_inter} → {len(interictal)}")

    # If we have fewer interictals, oversample (with replacement)
    elif n_inter < n_pre and n_inter > 0:
        extra = random.choices(interictal, k=n_pre - n_inter)
        interictal += extra
        logger.info(f"Oversampled interictal from {n_inter} → {len(interictal)}")

    # Combine balanced sets
    balanced = preictal + interictal

    logger.info(
        f"After balancing: preictal={len(preictal)}, interictal={len(interictal)}"
    )
    logger.info(f"Total balanced epochs: {len(balanced)}")

    return balanced


# 3. Preprocessing: Segmentation into epochs
def segment_recordings(
    intervals: list[IntervalMeta],
    undersampling: bool,
    overlap: float = 0.5,  # 50% overlap
) -> list[IntervalMeta]:
    """
    Segments a seizure interval into overlapping, labeled epochs.

    Args:
        interval (EpochInterval): Interval to segment; uses interval.phase.
        epoch_length (int): Window length in seconds.
        overlap (float): Fractional overlap.

    Returns:
        List[EpochInterval]: List of EpochInterval instances.
    """

    logger.info("Segmenting recordings into 30-second epochs")
    epoch_length = PreprocessingConfig.epoch_length

    epochs: list[IntervalMeta] = []
    step = max(1, int(epoch_length * (1 - overlap)))

    # Match number of interictals with the number preictals recordings
    preictal_intervals = [
        interval for interval in intervals if interval.phase == "preictal"
    ]
    interictal_intervals = [
        interval for interval in intervals if interval.phase == "interictal"
    ]

    if len(preictal_intervals) < len(interictal_intervals):
        sampled_interictals = random.sample(
            interictal_intervals, len(preictal_intervals)
        )
    else:
        sampled_interictals = interictal_intervals

    # Reinintialize precital and interictal seizure ids to start at 0 again
    for new_id, preictal in enumerate(preictal_intervals):
        preictal.seizure_id = new_id

    for new_id, interictal in enumerate(sampled_interictals):
        interictal.seizure_id = new_id

    balanced_recordings = preictal_intervals + (
        sampled_interictals
        if len(preictal_intervals) < len(interictal_intervals)
        else interictal_intervals
    )

    for idx in tqdm(range(len(balanced_recordings))):
        interval = cast(IntervalMeta, balanced_recordings[idx])
        start_sec, end_sec = int(interval.start), int(interval.end)

        current_start = start_sec
        interval_epochs = 0

        while current_start + epoch_length <= end_sec:
            epoch = min(current_start + epoch_length, end_sec)

            epochs.append(
                IntervalMeta(
                    start=current_start,
                    end=epoch,
                    phase=interval.phase,
                    seizure_id=interval.seizure_id,
                    file_name=interval.file_name,
                )
            )
            interval_epochs += 1
            current_start += step

        logger.info(
            f"Interval {idx + 1} [{interval.file_name}] ({interval.phase}): created {interval_epochs} epochs"
        )

    logger.info(f"Total epochs created: {len(epochs)}")

    if undersampling:
        epochs = balance_epochs(epochs)

    return epochs
