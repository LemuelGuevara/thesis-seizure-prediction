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
    NUMBER_OF_PATIENTS,
    PREICTAL_MINUTES,
    SAMPLE_RATE,
    SELECTED_CHANNELS,
    SPECS_OUTPUT_PATH,
)
from utils import load_patient_summary, parse_time_str, setup_logger

mne.set_log_level("ERROR")

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
        # patient summary file that contains the file start time, file end time, seizure start
        # time, seizure end time that would determine the file start, file end, and the
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


def load_patient_recording(patient_id: str) -> BaseRaw:
    """
    Loads all eeg recordings of a patient and concatenates all of
    them into one continous recording.

    Args:
        patient_id (str): Index of the patient.

    Returns:
       BaseRaw: The concatenated raw recordings.
    """

    logger = setup_logger(name="load_patient_recording", level=logging.DEBUG)
    logger.debug(f"loading_patient_{patient_id}_recordings")

    patient_folder = os.path.join(DATASET_PATH, f"chb{patient_id}")
    raw_edf_list: list[BaseRaw] = []

    for root, dirs, files in os.walk(patient_folder):
        for idx in tqdm(range(len(files))):
            files.sort()
            file = files[idx]

            if file.lower().endswith(".edf"):
                recording = os.path.join(root, file)
                logger.debug(f"reading_current_file: {file}")
                raw_edf: BaseRaw = mne.io.read_raw_edf(
                    recording, preload=True, verbose="error"
                )
                raw_edf.pick(SELECTED_CHANNELS)  # specified in the config.toml file
                # Band pass filtering along with a nothc filter
                raw_edf.filter(l_freq=0.5, h_freq=40.0)
                raw_edf.notch_filter(freqs=60.0)
                raw_edf_list.append(raw_edf)

    if not raw_edf_list:
        raise FileNotFoundError(f"No EDF files found in {patient_folder}")

    logger.debug("concatenated_raw_edf_list")
    return mne.concatenate_raws(raw_edf_list)  # type: ignore


def compute_stft_from_data(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the stft of the EEG recording.

    Args:
        data(np.ndarray): Converted numpy array of the BasRaw recording data.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - frequency_bins: Array of frequency values (in Hz) corresponding to the rows of the STFT.
            - times: Array of time points (in seconds) corresponding to the columns of the STFT.
            - stft_matrix: 2D complex-valued STFT result, where each element represents the magnitude
            and phase of a specific frequency at a specific time.
    """

    if data.ndim > 1:
        data = np.squeeze(data)

    frequency_bins, times, stft_matrix = stft(data, fs=SAMPLE_RATE, nperseg=SAMPLE_RATE)
    stft_magnitude = np.abs(stft_matrix)

    # Conversion of magnitude to decibles
    stft_magnitude_db = 20 * np.log10(stft_magnitude + 1e-8)

    # Limit to <= 40 Hz
    freq_mask = frequency_bins <= 40
    f = frequency_bins[freq_mask]
    stft_magnitude_db = stft_magnitude_db[freq_mask, :]

    return stft_magnitude_db, f, times


def get_best_channel(
    recording: BaseRaw, combined_intervals: list[SeizureInterval]
) -> int:
    """
    Gets the best channel from the preictal and interictal windows

    Args:
        recording (mne.io.BaseRaw): The raw recording returned by MNE.
        combined_intervals (list[SeizureInterval]): A combined list of SeizureInterval objects.

    Returns:
        int: Index of  the best channel found in intervals.
    """

    preictal_windows: list[np.ndarray] = []
    interictal_windows: list[np.ndarray] = []

    # Get the preictal and interictal windows from the combined intervals
    # and append them into separate lists to compute the variances
    for interval in combined_intervals:
        start_sample = int(interval.start * recording.info["sfreq"])
        end_sample = int(interval.end * recording.info["sfreq"])
        data_2d = recording.get_data(start=start_sample, stop=end_sample)
        data_2d_arr = np.asarray(data_2d)

        if interval.phase == "preictal":
            preictal_windows.append(data_2d_arr)
        elif interval.phase == "interictal":
            interictal_windows.append(data_2d_arr)

    # Concatenate windows for a long continous signal
    preictal_concat: NDArray[np.float64] = np.concatenate(preictal_windows, axis=1)
    interictal_concat: NDArray[np.float64] = np.concatenate(interictal_windows, axis=1)

    preictal_variance = float(np.var(preictal_concat))  # type: ignore
    interictal_variance = float(np.var(interictal_concat))  # type: ignore

    variance_diff = preictal_variance - interictal_variance
    return int(np.argmax(variance_diff))


def normalize_stft(stft_matrix: np.ndarray) -> np.ndarray:
    """
    Normalizes the stft to be compatible with EfficientNetB0.

    Args:
        stft_matrix(np.ndarray): Matrix of the stft.

    Returns:
        np.ndarray: (224, 224, 3) matrix that is compatible with EfficientNetB0
    """

    # Make sure that we are only getting a 2D matrix (frequency_bins, time_bins)
    if stft_matrix.ndim > 2:
        stft_matrix = np.squeeze(stft_matrix)

    # Normalize to 0–1
    min_max_scaler = p.MinMaxScaler()
    stft_norm_matrix: np.ndarray = min_max_scaler.fit_transform(stft_matrix)

    # Convert to PIL Image and resize
    img = Image.fromarray(stft_norm_matrix, mode="RGB")
    img = img.resize((224, 224), resample=Image.Resampling.BICUBIC)

    return np.array(img)  # shape (224,224,3)


def create_spectrograms(
    intervals: CombinedIntervals,
    patient_specs_dir: str,
    recording: BaseRaw,
) -> None:
    """
    Creates the spectrograms of the seizure intervals (preictal and interictal).

    Args:
        intervals(CombinedIntervals): Class containing the combined intervals (preictal, interictal, ictal).
        patient_specs_dir(str): Directory of where the spectrograms of the patient will be stored.
        recording(BaseRaw): Concatenated EEG recording of the patient (see load_patient_recording).
    """

    logger = setup_logger(name="create_spectrograms", level=logging.INFO)

    global preictal_windows_count, interictal_windows_count
    preictal_windows_count = 0
    interictal_windows_count = 0

    # In order to create those, the preictal_intervals and interictal_intervals must be combined first into
    # a single array and loop through those, keep in mind that this is array contains the SeizureInterval objects
    # that was appended in the extract_seizure_intervals function.

    combined_intervals: list[SeizureInterval] = list(
        chain(
            intervals.preictal_intervals,
            intervals.interictal_intervals,
            intervals.ictal_intervals,
        )
    )

    logger.info(
        f"Generating {len(intervals.preictal_intervals)} preictal and {len(intervals.interictal_intervals)} interictal intervals"
    )

    best_channel_idx = get_best_channel(
        recording=recording, combined_intervals=combined_intervals
    )
    logger.info(f"Best channel: {recording.ch_names[best_channel_idx]}")

    # This part of the function segments the combined_intervals that contains both preictal_intervals and
    # interictal_intervals. The segmentation is done by creating 30 second windows with a 50% overlap.
    for idx in tqdm(range(len(combined_intervals))):
        interval = combined_intervals[idx]

        start_sec: int = int(interval.start)
        end_sec: int = int(interval.end)
        start_sample: int = int(start_sec * recording.info["sfreq"])
        current_start_sec: int = start_sec

        interval.duration = interval.end - interval.start

        # NOTE: currently the condition interval.phase != "ictal" exists for debugging
        # purposes.
        if interval.phase != "ictal":
            interval.windows_created = (
                interval.duration - EPOCH_WINDOW_DURATION_SECONDS
            ) // (EPOCH_WINDOW_DURATION_SECONDS // 2) + 1

        if interval.phase != "ictal":
            while current_start_sec + EPOCH_WINDOW_DURATION_SECONDS <= end_sec:
                # The next window starting point, for example our first window is starting from 0 seconds,
                # then its ending point is at 0 seconds + EPOCH_WINDOW_DURATION_SECONDS(30s) which is 30 seconds.
                window_end_sec: int = current_start_sec + EPOCH_WINDOW_DURATION_SECONDS

                if window_end_sec > end_sec:
                    window_end_sec = end_sec
                window_end_sample: int = int(window_end_sec * recording.info["sfreq"])

                if window_end_sample > start_sample:
                    filename = f"spec_{interval.phase[:2]}_{current_start_sec}_{window_end_sec}.npz"
                    file_path: str = os.path.join(
                        f"{patient_specs_dir}/{interval.phase}", filename
                    )
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)

                    if not os.path.exists(file_path):
                        data_interval = recording.get_data(
                            picks=[best_channel_idx],
                            start=int(current_start_sec * recording.info["sfreq"]),
                            stop=window_end_sample,
                        )
                        data_interval_2d = np.asarray(data_interval)

                        if data_interval_2d.size > 0:
                            stft_matrix, freqs, times = compute_stft_from_data(
                                data_interval_2d
                            )

                            np.savez_compressed(
                                file_path,
                                spectrogram=normalize_stft(stft_matrix),
                                stft=stft_matrix,
                                freqs=freqs,
                                times=times,
                                channel=recording.ch_names[best_channel_idx],
                            )
                            logger.info(f"Generated file: {filename}")

                            if interval.phase == "preictal":
                                preictal_windows_count += 1
                            elif interval.phase == "interictal":
                                interictal_windows_count += 1

                # We apply 50% overlap in the next window so that we can capture
                # better features.
                current_start_sec += EPOCH_WINDOW_DURATION_SECONDS // 2


def log_patient_preprocessing(intervals: list[SeizureInterval]):
    """
    Logs the summary of patient preprocessing.

    Args:
        intervals(list[SeizureInterval]): List of seizure intervals (preictal, interictal, ictal).
    """

    logger = setup_logger(name="log_patient_preprocessing", level=logging.INFO)

    table = PrettyTable()
    table.field_names = [
        "Phase",
        "Start (s)",
        "End (s)",
        "Duration (s)",
        "Windows Created",
    ]
    table.title = "Patient Preprocessing Summary"

    intervals.sort(key=lambda interval: interval.end)

    for interval in intervals:
        table.add_row(
            [
                interval.phase,
                interval.start,
                interval.end,
                interval.duration,
                interval.windows_created,
            ]
        )
    print(table)
    logger.info(
        f"Preictal windows: {preictal_windows_count}, Interictal windows: {interictal_windows_count}"
    )


def main():
    logger = setup_logger(name="main", level=logging.INFO)
    print("am i working")

    for idx in range(1, NUMBER_OF_PATIENTS + 1):
        patient_id = f"{idx:02d}"

        logger.info(f"{'=' * 10} Working on patient {patient_id} {'=' * 10}")
        with load_patient_summary(patient_id, DATASET_PATH) as patient_summary:
            (
                preictal_intervals,
                interictal_intervals,
                ictal_intervals,
            ) = extract_seizure_intervals(patient_summary)
            logger.info(f"Number of seizures: {len(ictal_intervals)}")

        recording = load_patient_recording(patient_id)
        patient_specs_folder = os.path.join(SPECS_OUTPUT_PATH, f"patient_{patient_id}")
        os.makedirs(patient_specs_folder, exist_ok=True)
        combined_intervals = CombinedIntervals(
            preictal_intervals, interictal_intervals, ictal_intervals
        )

        create_spectrograms(
            combined_intervals,
            patient_specs_folder,
            recording,
        )
        log_patient_preprocessing(
            preictal_intervals + interictal_intervals + ictal_intervals
        )


if __name__ == "__main__":
    main()
