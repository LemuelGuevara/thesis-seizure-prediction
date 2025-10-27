"""
Data Transformation Module

This module provides functions for transforming the data. Key important functions under
this module are: compute_stft_epoch, compute_stft_epoch, precompute_stfts. These functions
are the backbone for creating the precomputed STFTS as a whole.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Tuple

import h5py
import numpy as np
from mne.io.base import BaseRaw
from PIL import Image
from scipy.signal import stft
from tqdm import tqdm

from src.config import PreprocessingConfig
from src.datatypes import IntervalMeta, StftStore
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


def normalize_array(
    arr: np.ndarray, method: str = "minmax", eps: float = 1e-8
) -> np.ndarray:
    """
    Generic normalization function for spectrograms or tensors.

    Args:
        arr (np.ndarray): Input array to normalize.
        method (str): Normalization method — one of:
            - "minmax": scales to [0, 1]
            - "zscore": zero mean, unit variance
        eps (float): Small constant to avoid division by zero.

    Returns:
        np.ndarray: Normalized array as float32.
    """
    arr = arr.astype(np.float32)

    if method == "minmax":
        normed = (arr - arr.min()) / (arr.max() - arr.min() + eps)
    elif method == "zscore":
        normed = (arr - arr.mean()) / (arr.std() + eps)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normed.astype(np.float32)


def compute_stft_epoch(
    epoch_signal: np.ndarray,
    max_freq: float = 40.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    stft_log_mag = np.log1p(mag)

    freqs_trim = freqs[mask]

    return stft_db, freqs_trim, times, Zxx, stft_log_mag


def compute_stft_for_epoch(
    data: np.ndarray, interval: IntervalMeta, normalize_power: bool, epoch_idx: int
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
            mag=np.empty(0, 0, 0),
            freqs=np.empty(0),
            times=np.empty(0),
        )

    stft_db_list, power_list, Zxx_list, stft_mag_list = [], [], [], []
    logger.info(
        f"Number of channels: {data.shape[0]} | "
        f"Epoch index: {epoch_idx} | "
        f"Start: {interval.start} | "
        f"End: {interval.end} | "
        f"Phase: {interval.phase} | "
        f"Recording: {interval.file_name}"
    )

    for ch in range(data.shape[0]):
        sig = np.squeeze(data[ch])
        stft_db, freqs, times, Zxx, stft_log_mag = compute_stft_epoch(epoch_signal=sig)
        power = stft_db_to_power(stft_db)

        if normalize_power and power is not None:
            power = normalize_power_matrix(power)

        stft_db_list.append(normalize_array(stft_db))
        power_list.append(power)
        Zxx_list.append(Zxx)
        stft_mag_list.append(normalize_array(stft_log_mag))

    # Stack into one 3D array (channels, freq, time)
    stft_db_all = np.stack(stft_db_list, axis=0)
    power_all = np.stack(power_list, axis=0)
    Zxx_all = np.stack(Zxx_list, axis=0)
    stft_mag_all = np.stack(stft_mag_list, axis=0)

    return StftStore(
        start=interval.start,
        end=interval.end,
        phase=interval.phase,
        stft_db=stft_db_all,
        power=power_all,
        Zxx=Zxx_all,
        mag=stft_mag_all,
        freqs=freqs,
        times=times,
        seizure_id=interval.seizure_id,
    )


def precompute_stft(
    recording: BaseRaw,
    patient_stfts_dir: str,
    segmented_epochs: list[IntervalMeta],
    normalize_power: bool = False,
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

    phase_counts = {}
    for epoch in segmented_epochs:
        phase_counts[epoch.phase] = phase_counts.get(epoch.phase, 0) + 1
    logger.info(f"Epochs by phase: {phase_counts}")

    logger.info(f"Normalize power: {normalize_power}")

    sfreq = float(recording.info["sfreq"])
    os.makedirs(patient_stfts_dir, exist_ok=True)

    h5_lock = Lock()

    tasks = list(enumerate(segmented_epochs))

    def _save_stft_epoch(task: tuple[int, IntervalMeta]) -> bool:
        idx, epoch = task
        start_sample = int(epoch.start * sfreq)
        end_sample = int(epoch.end * sfreq)

        stem = os.path.splitext(os.path.basename(str(epoch.file_name)))[0]
        base_name = f"{epoch.phase}_{stem}_{int(epoch.start):06d}_{int(epoch.end):06d}"
        out_name = f"{base_name}.h5"
        filename = os.path.join(patient_stfts_dir, out_name)

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
                    f.create_dataset("mag", data=computed_stft.mag, compression="gzip")
                    f.create_dataset(
                        "freqs", data=computed_stft.freqs, compression="gzip"
                    )
                    f.create_dataset(
                        "times", data=computed_stft.times, compression="gzip"
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
                    f.attrs["file_name"] = base_name

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


def normalize_to_imagenet(img: np.ndarray) -> np.ndarray:
    """
    Normalize an image to ImageNet statistics in channel-first format (C, H, W).
    Converts grayscale to 3-channel if needed.
    """
    img = np.asarray(img, dtype=np.float32)

    # Ensure channel-first (C, H, W)
    if img.ndim == 2:  # (H, W)
        img = np.stack([img] * 3, axis=0)
    elif img.ndim == 3 and img.shape[0] not in (1, 3):  # (H, W, C) -> (C, H, W)
        img = np.transpose(img, (2, 0, 1))
    elif img.ndim == 3 and img.shape[0] == 1:  # (1, H, W)
        img = np.repeat(img, 3, axis=0)

    # Scale to [0, 1]
    img = img / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]  # (3,1,1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]  # (3,1,1)
    return (img - mean) / std


def create_efficientnet_img(data: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    """
    Resize a 2D or 3D spectrogram to target size (224x224) while keeping float values in 0–1.
    Returns channel-first (C, H, W) array for PyTorch.

    Accepts:
        - (H, W)
        - (H, W, 3)
        - (3, H, W)
    """
    logger.info(f"Data shape (before): {data.shape}")
    data = np.asarray(data, dtype=float)

    scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    scaled_data = (scaled_data * 255).astype(np.uint8)

    if scaled_data.ndim == 2:
        rgb_data = np.stack([scaled_data] * 3, axis=-1)  # (H, W, 3)
    elif scaled_data.ndim == 3:
        if scaled_data.shape[0] == 3:
            rgb_data = np.transpose(scaled_data, (1, 2, 0))
        elif scaled_data.shape[2] == 3:
            rgb_data = scaled_data
        else:
            raise ValueError(f"Expected 3 channels, got shape {scaled_data.shape}")
    else:
        raise ValueError(f"Unsupported data shape: {scaled_data.shape}")

    img = Image.fromarray(rgb_data, "RGB")
    img = img.resize(target_size, resample=Image.Resampling.BICUBIC)

    img_arr = np.array(img).astype(np.float32)
    img_arr = np.transpose(img_arr, (2, 0, 1))  # (H, W, C) -> (C, H, W)

    logger.info(f"Data shape (after): {img_arr.shape}")
    return img_arr
