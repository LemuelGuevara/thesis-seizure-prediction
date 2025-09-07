"""
Data Transformation Module

This module provides functions for transforming the data. Key important functions under
this module are: compute_stft_epoch, compute_stft_epoch, precompute_stfts. These functions
are the backbone for creating the precomputed STFTS as a whole.
"""

import os
from typing import List, Literal, Optional, Tuple

import numpy as np
from mne.io.base import BaseRaw
from PIL import Image
from scipy.signal import stft
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

from src.config import NORMALIZATION_METHOD, SAMPLE_RATE, SELECTED_CHANNELS
from src.datatypes import BandTimeStore, EegConfig, EpochInterval, StftStore
from src.logger import setup_logger

logger = setup_logger(name="data_transformation")

# small epsilon for log computations
EPS = 1e-8


def compute_stft_epoch(
    epoch_signal: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    nperseg: int = SAMPLE_RATE,
    noverlap: int = SAMPLE_RATE // 2,
    max_freq: float = 40.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Short-Time Fourier Transform (STFT) for a 1-D epoch signal.

    Args:
        epoch_signal (np.ndarray): Input 1-D time-domain signal for one epoch.
        sample_rate (int, optional): Sampling rate of the signal in Hz. Defaults to SAMPLE_RATE.
        nperseg (int, optional): Length of each STFT segment in samples. Defaults to SAMPLE_RATE.
        noverlap (int, optional): Number of samples to overlap between segments. Defaults to SAMPLE_RATE // 2.
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

    x = np.asarray(epoch_signal)
    if x.ndim > 1:
        x = np.squeeze(x)

    freqs, times, Zxx_full = stft(x, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    mask = freqs <= max_freq

    Zxx = Zxx_full[mask, :]  # complex (F_trim, T)
    mag = np.abs(Zxx)  # magnitude
    stft_db = 20.0 * np.log10(mag + EPS)  # dB

    freqs_trim = freqs[mask]

    return stft_db, freqs_trim, times, Zxx


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

    if NORMALIZATION_METHOD == "minmax":
        mn = np.min(power_matrix)
        mx = np.max(power_matrix)
        if mx - mn == 0:
            return np.zeros_like(power_matrix)
        return np.divide((power_matrix - mn), (mx - mn))
    elif NORMALIZATION_METHOD == "zscore":
        mu = np.mean(power_matrix)
        sd = np.std(power_matrix)
        if sd == 0:
            return np.zeros_like(power_matrix)
        return np.divide((power_matrix - mu), sd)
    else:
        raise ValueError(f"Unknown normalization method: {NORMALIZATION_METHOD}")


def normalize_to_unint8_rgb(arr: np.ndarray) -> np.ndarray:
    """
    Nomrmalizes ndarray float to ndarray uint8 in RGB to be compatible with EfficientNetB0.

    Args:
        arr (np.ndarray): Array to be converted.

    Returns:
        np.ndarray: Normalized uint8 RGB np.ndarray.
    """

    data = np.asarray(arr, dtype=float)
    # minmax scaling
    scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    scaled_data = (scaled_data * 255).astype(np.uint8)

    # Stack 2D data into 3 channels to create RGB
    rgb_data = np.stack([scaled_data, scaled_data, scaled_data], axis=-1)

    img = Image.fromarray(rgb_data, "RGB")
    img = img.resize((224, 224), resample=Image.Resampling.BICUBIC)

    return np.asarray(img)


def normalize_globally(data: np.ndarray) -> np.ndarray:
    """
    Normalize an array to the [0, 1] range using its global minimum and maximum.

    Args:
        data (np.ndarray): Input array to normalize.

    Returns:
        np.ndarray: Normalized array with values scaled between 0 and 1.
                    Returns an array of zeros if input is empty or all values are equal.
    """

    data = np.asarray(data, dtype=np.float64)
    if data.size == 0:
        return np.zeros_like(data)

    mn = np.nanmin(data)
    mx = np.nanmax(data)

    if mx - mn == 0:
        return np.zeros_like(data)

    normalized = (data - mn) / (mx - mn)
    return normalized


def resize_to_224(data: np.ndarray, is_global_normalization: bool = False):
    """
    Normalize and convert an array to uint8 RGB format suitable for 224x224 input.

    Args:
        data (np.ndarray): Input array to process.
        is_global_normalization (bool, optional): Whether to apply global [0,1] normalization
            before conversion. Defaults to False.

    Returns:
        np.ndarray: Array normalized (if specified) and converted to uint8 RGB.
    """

    if is_global_normalization:
        data = normalize_globally(data)

    return normalize_to_unint8_rgb(data)


def get_top_channels(
    data: np.ndarray,
    n_components: int = 3,
    top_k: int = 3,
    method: Literal["ica", "pca"] = "pca",
) -> list[int]:
    """
    Returns the indices of the top channels contributing to the first n_components
    ICA or PCA components.

    Args:
        data (np.ndarray): Input array to process.
        n_components (int): Number of components to compute.
        top_k (int): How many top channels to return.
        method (str): "ica" or "pca"

    Returns:
        best_channel_indices (list[int]): List of top channel indices.
    """

    if method == "ica":
        decomposer = FastICA(n_components=n_components)
    elif method == "pca":
        decomposer = PCA(n_components=n_components)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ica' or 'pca'.")

    decomposer.fit(data.T)  # transpose: (n_samples, n_features)

    # Compute absolute contribution of each channel
    channel_scores = np.abs(decomposer.components_).sum(axis=0)

    # Get top_k channels
    best_channel_indices = np.argsort(channel_scores)[::-1][:top_k]
    return best_channel_indices.tolist()


def compute_stft_for_segment(
    data: np.ndarray,
    interval: EpochInterval,
    eeg_config: EegConfig,
    normalize_power: bool,
) -> StftStore:
    """
    Compute STFT for one signal segment (single channel, single epoch).

    Args:
        data (np.ndarray): Input array to process.
        interval (EpochInterval): Current epoch interval to process.
        eeg_config (EegConfig): Instance of the EEG config made.
        normalize_power (bool): Whether to normalize STFT power to db.
    """
    if data.size == 0:
        return StftStore(
            start=interval.start,
            end=interval.end,
            phase=interval.phase,
            stft_db=np.empty((0, 0)),
            power=np.empty((0, 0)),
            Zxx=np.empty((0, 0)),
            freqs=np.empty(0),
            times=np.empty(0),
        )

    sig = np.squeeze(data)
    sample_rate, nperseg, noverlap, max_freq = (
        eeg_config.sample_rate,
        eeg_config.nperseg,
        eeg_config.noverlap,
        eeg_config.max_freq,
    )
    stft_db, freqs, times, Zxx = compute_stft_epoch(
        sig, sample_rate, nperseg, noverlap, max_freq
    )

    power = stft_db_to_power(stft_db)
    if normalize_power and power is not None:
        power = normalize_power_matrix(power)

    return StftStore(
        start=interval.start,
        end=interval.end,
        phase=interval.phase,
        stft_db=stft_db,
        power=power,
        Zxx=Zxx,
        freqs=freqs,
        times=times,
    )


def precompute_stfts(
    recording: BaseRaw,
    patient_stfts_dir: str,
    segmented_intervals: List[EpochInterval],
    normalize_power: bool = False,
    normalization_method: str = "minmax",
):
    """
    Precomputes Short-Time Fourier Transform (STFT) representations for all labeled epochs
    and all selected channels in a single patient recording.
    The STFTs will also be saved in the disk by the use of npz.
    This is intended to be used by both TF and bispectrum branches.
    Args:
        recording (BaseRaw): The continuous EEG recording instance.
        patient_stft_dir (str): Path where the patient's STFTs will be saved.
            Results are organized by channel, e.g.:
                <patient_stft_dir>/<channel_name>/epoch_<start>_<end>.npz
        segmented_intervals (List[EpochInterval]): List of intervals that define the
            start, end, and phase of each epoch to be analyzed.
        normalize_power (bool, optional): Whether to normalize the power matrix.
            Defaults to False.
        normalization_method (str, optional): Normalization method to apply if
            normalize_power is True. Defaults to "minmax".

    Saved NPZ file contents (per epoch, per channel):
        - start (float): Start time of the epoch (in seconds).
        - end (float): End time of the epoch (in seconds).
        - phase (str): Phase label associated with the epoch.
        - stft_db (ndarray): STFT in decibel scale, shape (F, T).
        - power (ndarray): Linear power matrix (or empty if disabled), shape (F, T).
        - Zxx (ndarray): Complex STFT coefficients, shape (F, T).
        - freqs (ndarray): Frequency bins, shape (F,).
        - times (ndarray): Time bins, shape (T,).
    """

    logger.info("Starting STFT precomputation")
    logger.info(f"Total intervals to process: {len(segmented_intervals)}")
    logger.info(f"Normalize power: {normalize_power}")

    if normalize_power:
        logger.info(f"Normalization method: {normalization_method}")

    eeg_config = EegConfig(channel_names=SELECTED_CHANNELS)
    channel_names = eeg_config.channel_names
    sfreq = float(recording.info["sfreq"])

    logger.info(f"Selected channels: {channel_names}")
    logger.info(f"Sampling frequency: {sfreq} Hz")

    ch_index_map = {name: recording.ch_names.index(name) for name in channel_names}

    # Count intervals by phase for summary
    phase_counts = {}
    for epoch in segmented_intervals:
        phase_counts[epoch.phase] = phase_counts.get(epoch.phase, 0) + 1
    logger.info(f"Intervals by phase: {phase_counts}")

    processed_files = 0

    for epoch_idx in tqdm(range(len(segmented_intervals)), desc="Processing epochs"):
        epoch = segmented_intervals[epoch_idx]
        start_sec, end_sec, phase = epoch.start, epoch.end, epoch.phase
        start_sample, end_sample = int(start_sec * sfreq), int(end_sec * sfreq)

        logger.debug(
            f"Epoch: "
            f"{phase} [{start_sec}-{end_sec}s] samples [{start_sample}-{end_sample}]"
        )

        for ch in SELECTED_CHANNELS:
            idx_ch = ch_index_map[ch]

            # Save inside <patient_dir>/<channel>/
            channel_dir = os.path.join(patient_stfts_dir, ch)
            os.makedirs(channel_dir, exist_ok=True)
            filename = os.path.join(channel_dir, f"epoch_{epoch.start}_{epoch.end}.npz")

            if not os.path.exists(filename):
                data = recording.get_data(
                    picks=[idx_ch], start=start_sample, stop=end_sample
                )
                data_2d = np.asarray(data)

                computed_stft = compute_stft_for_segment(
                    data_2d,
                    epoch,
                    eeg_config,
                    normalize_power,
                )

                try:
                    np.savez_compressed(
                        filename,
                        raw_data=data_2d,
                        phase=computed_stft.phase,
                        start=computed_stft.start,
                        end=computed_stft.end,
                        stft_db=computed_stft.stft_db,
                        power=computed_stft.power,
                        Zxx=computed_stft.Zxx,
                        freqs=computed_stft.freqs,
                        times=computed_stft.times,
                    )
                    processed_files += 1
                except Exception as e:
                    logger.error(f"Failed to save {filename}: {e}")

    logger.info("STFT precomputation completed successfully!")
    logger.info(
        f"Processed {len(segmented_intervals)} epochs across {len(SELECTED_CHANNELS)} channels"
    )
    logger.info(f"Total STFT files created: {processed_files}")
    logger.info(f"Results saved in: {patient_stfts_dir}")


def build_epoch_mosaic(
    epoch_idx: int,
    band_maps: Optional[dict[str, list[BandTimeStore]]] = None,
    stfts_by_channel: dict[str, list[StftStore]] = {},
    type: Literal[
        "time_frequency_band", "time_frequency_detailed", "bispectrum"
    ] = "time_frequency_band",
    per_tile_normalization: bool = True,
    grid: tuple[int, int] = (4, 4),
    bispectrum_arr_list: Optional[list[np.ndarray]] = None,
) -> tuple[np.ndarray, str]:
    """
    Build a 2D mosaic representation of an EEG epoch from multiple channels.

    Args:
        epoch_idx (int): Index of the epoch to process.
        band_maps (dict[str, list[BandTimeStore]], optional): Precomputed band-level data per channel.
        stfts_by_channel (dict[str, list[StftStore]]): STFT results for each channel.
        type (str, optional): Type of representation to build:
            "time_frequency_band", "time_frequency_detailed", or "bispectrum".
            Defaults to "time_frequency_band".
        per_tile_normalization (bool, optional): Whether to apply min-max normalization to each tile. Defaults to True.
        grid (tuple[int, int], optional): Grid layout (rows, cols) for the mosaic. Defaults to (4, 4).
        bispectrum_arr_list (list[np.ndarray], optional): Optional bispectrum arrays to use instead of STFT/power.

    Returns:
        tuple[np.ndarray, str]:
            - Mosaic array of shape (H, W) after concatenating all tiles.
            - The type of representation used ("time_frequency_band", "time_frequency_detailed", or "bispectrum").
    """

    rows, cols = grid
    logger.info(
        f"Building epoch mosaic - epoch_idx: {epoch_idx}, mode: '{type}', "
        f"per_tile_normalization: {per_tile_normalization}, grid: {rows}x{cols}"
    )

    tiles: list[np.ndarray] = []
    channel_names = list(stfts_by_channel.keys())
    logger.debug(f"Processing {len(channel_names)} channels: {channel_names}")

    for ch_idx, (ch_name, epochs) in enumerate(stfts_by_channel.items()):
        if bispectrum_arr_list is not None:
            tile = bispectrum_arr_list[ch_idx]
        else:
            if type == "time_frequency_band":
                if band_maps is None:
                    raise ValueError(
                        "band_maps is required when type == 'time_frequency_band'"
                    )
                tile = band_maps[ch_name][epoch_idx].band_time
            else:
                tile = epochs[epoch_idx].power

        logger.debug(
            f"Channel '{ch_name}' ({ch_idx + 1}/{len(channel_names)}): "
            f"extracted {type} data, shape: {tile.shape}"
        )

        if per_tile_normalization:
            normalized_tile = minmax_scale(tile.flatten()).reshape(tile.shape)
            logger.debug(f"Channel '{ch_name}': applied minmax normalization")
        else:
            normalized_tile = tile

        tiles.append(np.asarray(normalized_tile, dtype=np.float64))

    # Calculate grid requirements
    n_tiles = len(tiles)
    required_tiles = rows * cols
    logger.debug(f"Grid requires {required_tiles} tiles, have {n_tiles} channels")

    # Pad if needed
    if n_tiles < required_tiles:
        padding_needed = required_tiles - n_tiles
        logger.debug(f"Padding with {padding_needed} zero tiles")
        for _ in range(padding_needed):
            tiles.append(np.zeros_like(tiles[0]))

    # Build grid
    logger.debug(f"Building {rows}x{cols} grid from {len(tiles)} tiles")
    grid_rows: list[np.ndarray] = []
    for r in range(rows):
        row_tiles = tiles[r * cols : (r + 1) * cols]
        grid_row = np.concatenate(row_tiles, axis=1)
        grid_rows.append(grid_row)
        logger.debug(
            f"Row {r}: concatenated {len(row_tiles)} tiles, shape: {grid_row.shape}"
        )

    mosaic = np.concatenate(grid_rows, axis=0)
    logger.info(
        f"Mosaic construction complete - final shape: {mosaic.shape}, "
        f"used {n_tiles} channels in {rows}x{cols} grid"
    )

    return mosaic, type
