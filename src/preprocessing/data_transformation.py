"""
Data Transformation Module

This module provides functions for transforming the data. Key important functions under
this module are: compute_stft_epoch, compute_stft_epoch, precompute_stfts. These functions
are the backbone for creating the precomputed STFTS as a whole.
"""

import os
import random
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import h5py
import numpy as np
from mne.io import BaseRaw
from mne.io.base import BaseRaw
from PIL import Image
from scipy.signal import stft
from tqdm import tqdm

from src.config import DataConfig, PreprocessingConfig
from src.datatypes import EpochInterval, StftStore
from src.logger import setup_logger

logger = setup_logger(name="data_transformation")

# small epsilon for log computations
EPS = 1e-8


def stft_db_to_power(stft_db: np.ndarray) -> np.ndarray:
    """
    Converts stft db to linear power.

    The scaling factor is 10 dB per decade because the stft_db represents a power ratio.

    Args:
        stft_db(np.ndarray): The stft array containing db values.

    Returns:
        np.ndarray: STFT power array
    """

    return np.power(10.0, np.divide(stft_db, 10.0))


def normalize_power_matrix(power_matrix: np.ndarray) -> np.ndarray:
    """
    Normalize a linear power matrix using specified normalization method.

    Args:
        power_matrix (np.ndarray): Input power matrix with shape (F, T) where F is the
            number of frequency bins and T is the number of time frames.
        method (str, optional): Normalization method to apply. Defaults to "minmax".

    Returns:
        np.ndarray: Normalized power matrix with the same shape as input (F, T).
            Values are scaled according to the specified method.

    Raises:
        ValueError: If an unknown normalization method is specified.
    """

    if power_matrix is None or power_matrix.size == 0:
        return power_matrix

    # Supported methods:
    # - "minmax": Min-max normalization, (x - min) / (max - min)
    # - "zscore": Z-score normalization, (x - mean) / std

    normalization_method = PreprocessingConfig.normalization_method
    if normalization_method == "minmax":
        mn = np.min(power_matrix)
        mx = np.max(power_matrix)
        if mx - mn == 0:
            return np.zeros_like(power_matrix)
        return np.divide((power_matrix - mn), (mx - mn))
    elif normalization_method == "zscore":
        mu = np.mean(power_matrix)
        sd = np.std(power_matrix)
        if sd == 0:
            return np.zeros_like(power_matrix)
        return np.divide((power_matrix - mu), sd)
    else:
        raise ValueError(f"Unknown normalization method: {normalization_method}")


def resize_to_224(data: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    """
    Resize a 2D or 3D spectrogram to target size (224x224) while keeping float values in 0-255.
    Handles both (H, W) and (C, H, W) shapes.
    """

    logger.info(f"Data shape (before): {data.shape}")
    data = np.asarray(data, dtype=float)

    # Min-max scaling to 0-255
    scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    scaled_data = (scaled_data * 255).astype(np.uint8)

    if scaled_data.ndim == 2:
        rgb_data = np.stack([scaled_data] * 3, axis=-1)  # (H, W, 3)

    elif scaled_data.ndim == 3:
        if scaled_data.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
            rgb_data = np.transpose(scaled_data, (1, 2, 0))
        else:
            raise ValueError(f"Expected 3 channels, got shape {scaled_data.shape}")

    else:
        raise ValueError(f"Unsupported data shape: {scaled_data.shape}")

    logger.info(f"Data shape (after channel fix): {rgb_data.shape}")

    img = Image.fromarray(rgb_data, "RGB")
    img = img.resize(target_size, resample=Image.Resampling.BICUBIC)
    return np.asarray(img)


def normalize_to_imagenet(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    img = img / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (img - mean) / std


def compute_stft_epoch(
    epoch_signal: np.ndarray,
    max_freq: float = 40.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Short-Time Fourier Transform (STFT) for a 1-D epoch signal.

    Args:
        epoch_signal (np.ndarray): Input 1-D time-domain signal for one epoch.
        max_freq (float, optional): Maximum frequency (Hz) to keep in the output.
            Frequencies above this threshold are discarded. Defaults to 40.0.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - stft_db: Magnitude of the STFT in decibels (20 * log10(|Zxx|)),
              shape (F_trim, T).
            - freqs_trim: Frequency bins (Hz) corresponding to the trimmed STFT,
              shape (F_trim,).
            - times: Time bins (s) corresponding to the STFT columns,
              shape (T,).
            - Zxx_trimmed: Complex STFT values before magnitude/log scaling,
              shape (F_trim, T).
    """

    nperseg = PreprocessingConfig.sample_rate
    noverlap = PreprocessingConfig.sample_rate // 2

    x = np.asarray(epoch_signal)
    if x.ndim > 1:
        x = np.squeeze(x)

    freqs, times, Zxx_full = stft(
        x, fs=PreprocessingConfig.sample_rate, nperseg=nperseg, noverlap=noverlap
    )
    mask = freqs <= max_freq

    Zxx = Zxx_full[mask, :]  # complex (F_trim, T)
    mag = np.abs(Zxx)  # magnitude
    stft_db = 20.0 * np.log10(mag + EPS)  # dB

    freqs_trim = freqs[mask]

    return stft_db, freqs_trim, times, Zxx


def compute_stft_for_epoch(
    data: np.ndarray, interval: EpochInterval, normalize_power: bool, epoch_idx: int
) -> StftStore:
    """
    Compute STFT for one EEG segment that includes multiple channels.

    Args:
        data (np.ndarray): Input array with shape (n_channels, n_samples)
        interval (EpochInterval): Current epoch interval to process.
        normalize_power (bool): Whether to normalize STFT power to dB.

    Returns:
        StftStore: stores multi-channel STFT results (each key stores per-channel arrays)
    """
    if data.size == 0:
        return StftStore(
            start=interval.start,
            end=interval.end,
            phase=interval.phase,
            stft_db=np.empty((0, 0, 0)),
            power=np.empty((0, 0, 0)),
            Zxx=np.empty((0, 0, 0), dtype=np.complex64),
            freqs=np.empty(0),
            times=np.empty(0),
        )

    # pca = PCA(n_components=3)
    # data = pca.fit_transform(data.T).T

    stft_db_list, power_list, Zxx_list = [], [], []
    logger.info(f"Number of channels: {data.shape[0]}, Epoch index: {epoch_idx}")

    for ch in range(data.shape[0]):
        sig = np.squeeze(data[ch])
        stft_db, freqs, times, Zxx = compute_stft_epoch(epoch_signal=sig)
        power = stft_db_to_power(stft_db)

        if normalize_power and power is not None:
            power = normalize_power_matrix(power)

        stft_db_list.append(stft_db)
        power_list.append(power)
        Zxx_list.append(Zxx)

    # Stack into one 3D array (channels, freq, time)
    stft_db_all = np.stack(stft_db_list, axis=0)
    power_all = np.stack(power_list, axis=0)
    Zxx_all = np.stack(Zxx_list, axis=0)

    return StftStore(
        start=interval.start,
        end=interval.end,
        phase=interval.phase,
        stft_db=stft_db_all,
        power=power_all,
        Zxx=Zxx_all,
        freqs=freqs,
        times=times,
        seizure_id=interval.seizure_id,
    )


def undersample_inteictal(
    segmented_intervals: list[EpochInterval],
) -> list[EpochInterval]:
    preictal_epochs, interictal_epochs = [], []

    for interval in segmented_intervals:
        if interval.phase == "preictal":
            preictal_epochs.append(interval)
        else:
            interictal_epochs.append(interval)

    balanced_segmented = preictal_epochs + random.sample(
        interictal_epochs, len(preictal_epochs)
    )
    return balanced_segmented


def precompute_stfts(
    recording: BaseRaw,
    patient_stfts_dir: str,
    segmented_intervals: list[EpochInterval],
    normalize_power: bool,
    normalization_method: str = "minmax",
):
    """
    Precompute STFTs for all epochs and save each epoch as an HDF5 file.
    All channels are saved in a single file per epoch.
    """
    # TODO: perform undersampling here wherein we match the number of epochs of intericals
    # with the number of epochs of preictals

    # The logic:
    # 1. loop through the segmented intervals then make 2 lists for preictals and interictals separately
    # 2. add the epoch intervals accordingly to their phase e.g. interictal -> interictals_list
    # 3. apply random.sample to interictals list wherein N is the number of preictal epochs

    if DataConfig.interictal_undersampling:
        segmented = undersample_inteictal(segmented_intervals)
    else:
        segmented = segmented_intervals

    phase_counts = {}
    for epoch in segmented:
        phase_counts[epoch.phase] = phase_counts.get(epoch.phase, 0) + 1
    logger.info(f"Normalize power: {normalize_power}")
    logger.info(f"Windows by phase: {phase_counts}")

    sfreq = float(recording.info["sfreq"])
    os.makedirs(patient_stfts_dir, exist_ok=True)

    h5_lock = Lock()

    tasks = list(enumerate(segmented))

    def _save_stft_epoch(task: tuple[int, EpochInterval]) -> bool:
        idx, epoch = task
        start_sample = int(epoch.start * sfreq)
        end_sample = int(epoch.end * sfreq)
        filename = os.path.join(patient_stfts_dir, f"stft_epoch_{idx:06d}.h5")

        # Quick check to skip existing files
        if os.path.exists(filename):
            return False

        # Extract EEG data for selected channels
        data = recording.get_data(
            picks=PreprocessingConfig.selected_channels,
            start=start_sample,
            stop=end_sample,
        ).astype(np.float32)

        # Compute STFT for this epoch
        computed_stft = compute_stft_for_epoch(
            np.asarray(data), epoch, normalize_power, idx
        )

        try:
            # Thread-safe write
            with h5_lock:
                with h5py.File(filename, "w") as f:
                    f.create_dataset("Zxx", data=computed_stft.Zxx, compression="gzip")
                    f.create_dataset(
                        "stft_db", data=computed_stft.stft_db, compression="gzip"
                    )
                    f.create_dataset(
                        "freqs", data=computed_stft.freqs, compression="gzip"
                    )
                    f.create_dataset(
                        "times", data=computed_stft.times, compression="gzip"
                    )
                    f.create_dataset(
                        "epochs", data=computed_stft.times, compression="gzip"
                    )
                    if getattr(computed_stft, "power", None) is not None:
                        f.create_dataset(
                            "power", data=computed_stft.power, compression="gzip"
                        )

                    # Save metadata as attributes
                    f.attrs["phase"] = computed_stft.phase
                    f.attrs["start"] = computed_stft.start
                    f.attrs["end"] = computed_stft.end
                    f.attrs["seizure_id"] = computed_stft.seizure_id

            return True
        except Exception as e:
            logger.error(f"Failed to save {filename}: {e}")
            return False

    # Parallel execution for STFT computation
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        results = list(
            tqdm(
                pool.map(_save_stft_epoch, tasks),
                total=len(tasks),
                desc="Computing STFTs",
            )
        )

    logger.info("STFT precomputation completed successfully!")
    logger.info(f"Processed {len(results)} STFT epochs")
    logger.info(f"Total STFT files created: {sum(results)}")
    logger.info(f"Results saved in: {patient_stfts_dir}")

    return phase_counts
