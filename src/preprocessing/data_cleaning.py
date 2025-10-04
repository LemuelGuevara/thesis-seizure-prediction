"""
Data Cleaning Module

This module provides functions for cleaning the EEG recordings before
passing them to time-frequency and bispectrum processing.

"""

import os
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
from src.datatypes import EpochInterval
from src.logger import setup_logger
from src.utils import parse_time_str

mne.set_log_level("ERROR")

logger = setup_logger(name="data_cleaning")

# Counter for Spectrogram windows
preictal_windows_count: int = 0
interictal_windows_count: int = 0


def extract_seizure_intervals(
    patient_summary: typing.TextIO,
) -> tuple[list[EpochInterval], list[EpochInterval], list[EpochInterval]]:
    """
    Extracts seizure intervals from a patient summart text file.

    Necessary conversions since there are two time formats (seconds and datetime).

    Args:
        patient_summary (TextIO): Patient summary.txt file.

    Returns:
        tuple[list[SeizureInterval], list[SeizureInterval], list[SeizureInterval]]:
            - preictal_intervals: Intervals before seizures.
            - interictal_intervals: Intervals between seizures.
            - ictal_intervals: Actual seizure intervals.
    """

    preictal_intervals: list[EpochInterval] = []
    interictal_intervals: list[EpochInterval] = []
    ictal_intervals: list[EpochInterval] = []

    pending_seizure_start = None
    file_start = None
    file_end = None
    file_name = ""

    for line in patient_summary:
        # In order to get the preictal and interictal intervals, we need to loop through the
        # patient summary file that contains the file start time, file end time, seizure start time,
        # and seizure end time that would determine the file start, file end, and the
        # seizures themselves. Then a seizure array will be used to store when the seizures started
        # which will be used to find the preictals and interictals.

        line = line.strip()

        if line.startswith("File Name:"):
            file_name = line.split(":", 1)[1].strip()
            file_name = os.path.basename(file_name)
        elif line.startswith("File Start Time:"):
            file_start = parse_time_str(line.split(":", 1)[1].strip())
        elif line.startswith("File End Time:"):
            file_end = parse_time_str(line.split(":", 1)[1].strip())
        elif line.startswith("Number of Seizures in File:"):
            no_of_seizures = int(line.split(":", 1)[1].strip().split()[0])
            if no_of_seizures == 0:
                interictal_intervals.append(
                    EpochInterval(
                        phase="interictal", start=0, end=3600, file_name=file_name
                    )
                )
        elif re.match(r"Seizure(\s+\d+)?\s+Start Time:", line):
            pending_seizure_start = int(line.split(":")[-1].strip().split()[0])
        elif (
            re.match(r"Seizure(\s+\d+)?\s+End Time:", line)
            and pending_seizure_start is not None
        ):
            end_sec = int(line.split(":")[-1].strip().split()[0])
            ictal_intervals.append(
                EpochInterval(phase="ictal", start=pending_seizure_start, end=end_sec)
            )
            pending_seizure_start = None

    assert file_start is not None, "file_start cannot be None"
    assert file_end is not None, "file_end cannot be None"

    current_start = file_start
    for seizure_idx, seizure in enumerate(ictal_intervals, 1):
        # Since the file start and times are in datatime format, in order to get the
        # absolute datetimes of the seizure start/end from seconds, conversion has by using
        # timedelta and adding it to the file_start datetime.

        seizure_start_time = file_start + timedelta(seconds=seizure.start)
        seizure_end_time = file_start + timedelta(seconds=seizure.end)
        preictal_start_time = seizure_start_time - timedelta(
            minutes=PreprocessingConfig.preictal_minutes
        )

        logger.debug(
            "seizure_start: %s, seizure_end: %s",
            seizure.start,
            seizure.end,
        )

        # We need to make sure that our times will never be negative
        if preictal_start_time < current_start:
            preictal_start_time = current_start
        if preictal_start_time >= seizure_start_time:
            preictal_start_time = file_start

        # Interictal before preictal
        if preictal_start_time > current_start:
            interictal_intervals.append(
                EpochInterval(
                    phase="interictal",
                    start=int((current_start - file_start).total_seconds()),
                    end=int((preictal_start_time - file_start).total_seconds()),
                    file_name=file_name,
                )
            )

        preictal_intervals.append(
            EpochInterval(
                phase="preictal",
                start=int((preictal_start_time - file_start).total_seconds()),
                end=int((seizure_start_time - file_start).total_seconds()),
                file_name=file_name,
            )
        )

        current_start = seizure_end_time

    # Interictal after last seizure
    if file_end is not None and current_start < file_end:
        interictal_intervals.append(
            EpochInterval(
                phase="interictal",
                start=int((current_start - file_start).total_seconds()),
                end=int((file_end - file_start).total_seconds()),
                file_name=file_name,
            )
        )

    return (
        preictal_intervals,
        interictal_intervals,
        ictal_intervals,
    )


# 1. Loads all EDF recordings of a patient (since we have multiple recordings per patient)
def load_raw_recordings(patient_id: str) -> list[BaseRaw]:
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

    for root, dirs, files in os.walk(patient_folder):
        files = [f for f in files if f.lower().endswith(".edf")]
        files.sort()

        for file in tqdm(files, desc=f"Reading EDF files in {os.path.basename(root)}"):
            recording_path = os.path.join(root, file)
            logger.debug(f"Reading file: {file}")

            raw_edf = mne.io.read_raw_edf(
                recording_path, preload=False, verbose="error"
            )
            raw_edf.pick(
                PreprocessingConfig.selected_channels
            )  # only keep channels defined in config
            raw_edf_list.append(raw_edf)

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

    logger.info("Applying bandpass and notch filters")

    # bandpass filtering
    raw.filter(
        l_freq=PreprocessingConfig.low_cutoff_filter,
        h_freq=PreprocessingConfig.high_cutoff_filter,
    )
    logger.info(
        f"Bandpass filtered: {PreprocessingConfig.low_cutoff_filter} - {PreprocessingConfig.high_cutoff_filter} Hz"
    )

    # notch filtering
    raw.notch_filter(PreprocessingConfig.notch_filter)
    logger.info(f"Notch filtered: {PreprocessingConfig.notch_filter} Hz")
    return raw


def load_patient_recording(patient_id: str) -> BaseRaw:
    """
    Loads and preprocesses a patient's EEG recordings
    - Reads EDFs
    - Concatenates them

    Args:
        patient_id (str): Patient ID (e.g. "01")

    Returns:
        BaseRaw: Concatenated and filtered EEG recording.
    """

    raw_edf_list: list[BaseRaw] = load_raw_recordings(patient_id)

    # Concatenate all files into one continuous recording
    logger.info("Concatenating EDF files")
    raw_concatenated: BaseRaw = cast(
        BaseRaw, mne.concatenate_raws(raw_edf_list, preload=False)
    )

    return raw_concatenated


# 3. Preprocessing: Segmentation into epochs
def segment_intervals(
    intervals: list[EpochInterval],
    overlap: float = 0.5,  # 50% overlap
) -> list[EpochInterval]:
    """
    Segments a seizure interval into overlapping, labeled epochs.

    Args:
        interval (EpochInterval): Interval to segment; uses interval.phase.
        epoch_length (int): Window length in seconds.
        overlap (float): Fractional overlap.

    Returns:
        List[EpochInterval]: List of EpochInterval instances.
    """

    logger.info("Segmenting intervals into 30-second epochs")
    epoch_length = PreprocessingConfig.epoch_length

    windows: list[EpochInterval] = []
    step = max(1, int(epoch_length * (1 - overlap)))

    for idx in tqdm(range(len(intervals))):
        interval = intervals[idx]
        start_sec, end_sec = int(interval.start), int(interval.end)

        current_start = start_sec
        interval_windows = 0

        while current_start + epoch_length <= end_sec:
            window_end = min(current_start + epoch_length, end_sec)

            windows.append(
                EpochInterval(start=current_start, end=window_end, phase=interval.phase)
            )
            interval_windows += 1
            current_start += step

        logger.info(
            f"Interval {idx + 1} ({interval.phase}): created {interval_windows} windows"
        )

    logger.info(f"Total windows created: {len(windows)}")

    return windows
