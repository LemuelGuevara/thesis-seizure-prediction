import logging
import os
import sys
import typing
from dataclasses import dataclass
from datetime import timedelta
from itertools import chain
from typing import Literal, Optional

import mne
import numpy as np
from mne.io import BaseRaw
from numpy._typing._array_like import NDArray
from PIL import Image
from prettytable import PrettyTable
from scipy.signal import stft
from sklearn import preprocessing as p
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from config import (
    DATASET_PATH,
    EPOCH_WINDOW_DURATION_SECONDS,
    PREICTAL_MINUTES,
    SELECTED_CHANNELS,
    LOW_CUTOFF_FILTER,
    HIGH_CUTOFF_FILTER,
    NOTCH_FILTER,
)
from utils import load_patient_summary, parse_time_str, setup_logger

mne.set_log_level("ERROR")

# Counter for Spectrogram windows
preictal_windows_count: int = 0
interictal_windows_count: int = 0

@dataclass
class SeizureInterval:
    phase: Literal["preictal", "interictal", "ictal"]
    start: int
    end: int
    duration: Optional[int] = None
    windows_created: Optional[int] = None


@dataclass
class CombinedIntervals:
    preictal_intervals: list[SeizureInterval]
    interictal_intervals: list[SeizureInterval]
    ictal_intervals: list[SeizureInterval]


# Extracts seizure intervals from a patient summary text file.
# Necessary conversions since there are two time formats (seconds and datetime)
def extract_seizure_intervals(
    patient_summary: typing.TextIO,
) -> tuple[list[SeizureInterval], list[SeizureInterval], list[SeizureInterval]]:
    """
    Extracts seizure intervals from a patient summart text file.

    Args:
        patient_summary (TextIO): Patient summary.txt file.

    Returns:
        tuple[list[SeizureInterval], list[SeizureInterval], list[SeizureInterval]]:
            - preictal_intervals: Intervals before seizures.
            - interictal_intervals: Intervals between seizures.
            - ictal_intervals: Actual seizure intervals.
    """

    logger = setup_logger(name="extract_seizure_intervals", level=logging.DEBUG)

    preictal_intervals: list[SeizureInterval] = []
    interictal_intervals: list[SeizureInterval] = []
    ictal_intervals: list[SeizureInterval] = []

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
        if line.startswith("File Start Time:"):
            file_start = parse_time_str(line.split(":", 1)[1].strip())
        elif line.startswith("File End Time:"):
            file_end = parse_time_str(line.split(":", 1)[1].strip())
        elif line.startswith("Seizure Start Time:"):
            pending_seizure_start = int(line.split(":")[1].strip().split()[0])
        elif line.startswith("Seizure End Time:"):
            end_sec = int(float(line.split(":", 1)[1].strip().split()[0]))
            if pending_seizure_start is not None:
                ictal_intervals.append(
                    SeizureInterval(
                        phase="ictal", start=pending_seizure_start, end=end_sec
                    )
                )
            pending_seizure_start = None

    assert file_start is not None, "file_start cannot be None"

    current_start = file_start
    for seizure_idx, seizure in enumerate(ictal_intervals, 1):
        # Since the file start and times are in datatime format, in order to get the
        # absolute datetimes of the seizure start/end from seconds, conversion has by using
        # timedelta and adding it to the file_start datetime.

        seizure_start_time = file_start + timedelta(seconds=seizure.start)
        seizure_end_time = file_start + timedelta(seconds=seizure.end)
        preictal_start_time = seizure_start_time - timedelta(minutes=PREICTAL_MINUTES)

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
                SeizureInterval(
                    phase="interictal",
                    start=int((current_start - file_start).total_seconds()),
                    end=int((preictal_start_time - file_start).total_seconds()),
                )
            )

        preictal_intervals.append(
            SeizureInterval(
                phase="preictal",
                start=int((preictal_start_time - file_start).total_seconds()),
                end=int((seizure_start_time - file_start).total_seconds()),
            )
        )

        current_start = seizure_end_time

    # Interictal after last seizure
    if file_end is not None and current_start < file_end:
        interictal_intervals.append(
            SeizureInterval(
                phase="interictal",
                start=int((current_start - file_start).total_seconds()),
                end=int((file_end - file_start).total_seconds()),
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

    logger = setup_logger(name="load_raw_recordings", level=logging.DEBUG)
    logger.debug(f"Loading EDF files for patient {patient_id}")

    patient_folder = os.path.join(DATASET_PATH, f"chb{patient_id}")
    raw_edf_list: list[BaseRaw] = []

    for root, dirs, files in os.walk(patient_folder):
        files.sort()
        for file in files:
            if file.lower().endswith(".edf"):
                recording_path = os.path.join(root, file)
                logger.debug(f"Reading file: {file}")
                raw_edf = mne.io.read_raw_edf(recording_path, preload=True, verbose="error")
                raw_edf.pick(SELECTED_CHANNELS)  # only keep channels defined in config
                raw_edf_list.append(raw_edf)

    if not raw_edf_list:
        raise FileNotFoundError(f"No EDF files found in {patient_folder}")

    return raw_edf_list


# 2. Preprocessing: Filtering
def apply_filters(
    raw: BaseRaw,
    l_freq=LOW_CUTOFF_FILTER,
    h_freq=HIGH_CUTOFF_FILTER,
    notch_freq=NOTCH_FILTER
) -> BaseRaw:
    """
    Applies bandpass and notch filtering to EEG recording.

    Args:
        raw (BaseRaw): Raw EEG recording.

    Returns:
        BaseRaw: Filtered EEG recording.
    """
    logger = setup_logger(name="apply_filters", level=logging.DEBUG)
    logger.debug("Applying bandpass and notch filters")

    #bandpass filtering
    raw.filter(l_freq, h_freq)
    logger.debug(f"Bandpass filtered: {l_freq} - {h_freq} Hz")

    #notch filtering
    raw.notch_filter(notch_freq)
    logger.debug(f"Notch filtered: {notch_freq} Hz")

    return raw


# Loads, concatenates, and applies filters to a patient's EEG recordings
def load_patient_recording(patient_id: str) -> BaseRaw:
    """
    Loads and preprocesses a patient's EEG recordings:
    - Reads EDFs
    - Concatenates them
    - Applies filters

    Args:
        patient_id (str): Patient ID (e.g. "01")

    Returns:
        BaseRaw: Concatenated and filtered EEG recording.
    """
    logger = setup_logger(name="load_patient_recording", level=logging.DEBUG)
    raw_edf_list = load_raw_recordings(patient_id)

    # Concatenate all files into one continuous recording
    logger.debug("Concatenating EDF files")
    raw_concatenated = mne.concatenate_raws(raw_edf_list)

    # Apply filtering
    raw_filtered = apply_filters(raw_concatenated)
    return raw_filtered


# 3. Preprocessing: Segmentation into epochs
def segment_intervals(
    interval: SeizureInterval,
    epoch_length: int = EPOCH_WINDOW_DURATION_SECONDS,  # 30s
    overlap: float = 0.5,  # 50% overlap
) -> list[dict]:
    """
    Segments a seizure interval into overlapping, labeled epochs.

    Args:
        interval (SeizureInterval): Interval to segment; uses interval.phase.
        epoch_length (int): Window length in seconds.
        overlap (float): Fractional overlap.

    Returns:
        List[dict]: each dict has {"start": int, "end": int, "phase": str}
    """
    start_sec, end_sec = int(interval.start), int(interval.end)
    step = max(1, int(epoch_length * (1 - overlap)))
    windows: list[dict] = []

    current_start = start_sec
    while current_start + epoch_length <= end_sec:
        window_end = current_start + epoch_length
        windows.append({"start": current_start, "end": window_end, "phase": interval.phase})
        current_start += step

    return windows

