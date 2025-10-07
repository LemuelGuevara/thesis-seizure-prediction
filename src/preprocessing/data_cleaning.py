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

import mne
from mne.io import BaseRaw
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from src.config import DataConfig, PreprocessingConfig
from src.datatypes import EpochInterval, IntervalFileInfo
from src.logger import setup_logger
from src.utils import parse_time_str

mne.set_log_level("ERROR")

logger = setup_logger(name="data_cleaning")

# Counter for Spectrogram windows
preictal_windows_count: int = 0
interictal_windows_count: int = 0


def extract_seizure_intervals(
    patient_summary: typing.TextIO,
) -> tuple[
    list[EpochInterval],
    list[EpochInterval],
    list[EpochInterval],
    list[IntervalFileInfo],
    list[IntervalFileInfo],
]:
    """
    Extracts seizure intervals from a patient summart text file.

    Necessary conversions since there are two time formats (seconds and datetime).

    Args:
        patient_summary (TextIO): Patient summary.txt file.

    Returns:
        tuple[list[SeizureInterval], list[SeizureInterval], list[SeizureInterval]
            list[IntervalFileInfo], list[IntervalFileInfo], int]:
            - preictal_intervals: Intervals before seizures.
            - interictal_intervals: Intervals between seizures.
            - ictal_intervals: Actual seizure intervals.
            - seizure_files_data: Files (recordings) with seizures.
            - no_seizure_files_data: Files

    """

    preictal_intervals: list[EpochInterval] = []
    interictal_intervals: list[EpochInterval] = []
    ictal_intervals: list[EpochInterval] = []

    seizure_files_data: list[IntervalFileInfo] = []
    no_seizure_files_data: list[IntervalFileInfo] = []

    file_name = ""
    pending_seizure_start = None
    file_start = None
    file_end = None

    for line in patient_summary:
        # In order to get the preictal and interictal intervals, we need to loop through the
        # patient summary file that contains the file start time, file end time, seizure start time,
        # and seizure end time that would determine the file start, file end, and the
        # seizures themselves. Then a seizure array will be used to store when the seizures started
        # which will be used to find the preictals and interictals.

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
                interval = EpochInterval(
                    phase="interictal",
                    start=0,
                    end=duration_seconds,
                    file_name=file_name,
                )
                interictal_intervals.append(interval)
                no_seizure_files_data.append(
                    IntervalFileInfo(file_name, interval, duration_seconds)
                )
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

            ictal_interval = EpochInterval(
                phase="ictal",
                start=pending_seizure_start,
                end=end_sec,
                file_name=file_name,
            )
            seizure_files_data.append(
                IntervalFileInfo(file_name, ictal_interval, duration_seconds)
            )
            ictal_intervals.append(ictal_interval)
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
        seizure_files_data,
        no_seizure_files_data,
    )


# 1. Loads all EDF recordings of a patient (since we have multiple recordings per patient)
def load_raw_recordings(patient_id: str, file_names: list[str]) -> list[BaseRaw]:
    """
    Loads all raw EDF recordings of a patient without filtering.

    Args:
        patient_id (str): Zero-padded patient ID (e.g. "01").
        combined_intervals (list[EpochInterval]): List of the preictal, ictal,
            and interictal intervals combined

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
        logger.debug(f"Reading file: {file_name}")

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
                EpochInterval(
                    start=current_start,
                    end=window_end,
                    phase=interval.phase,
                )
            )
            interval_windows += 1
            current_start += step

        logger.info(
            f"Interval {idx + 1} ({interval.phase}): created {interval_windows} windows"
        )

    logger.info(f"Total windows created: {len(windows)}")

    return windows
