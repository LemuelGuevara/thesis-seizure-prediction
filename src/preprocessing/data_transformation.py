# data_transformation.py
from __future__ import annotations
import os
import csv
import logging
from typing import Dict, Tuple, List, Any, Optional
import numpy as np
from scipy.signal import stft
from PIL import Image

from config import (
    SAMPLE_RATE,
    SELECTED_CHANNELS,
    BAND_DEFS,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# small epsilon for log computations
EPS = 1e-8

# HELPER FUNCTIONS
# Normalizes a numpy array to uint8
# uint8 images are the standard EfficientNet format
def _minmax_float_to_uint8(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    mn, mx = np.nanmin(a), np.nanmax(a)
    if mx - mn < eps:
        return np.zeros_like(a, dtype=np.uint8)
    scaled = (a - mn) / (mx - mn)
    return (scaled * 255.0).astype(np.uint8)


# duplicates a single-channel image to three equal channels for EfficientNet
def replicate_grayscale_to_rgb(single: np.ndarray) -> np.ndarray:
    """Replicate a 2D uint8 array into 3 identical channels (H, W) -> (H, W, 3)."""
    if single.dtype != np.uint8:
        single = _minmax_float_to_uint8(single)
    return np.stack([single, single, single], axis=-1)


# for debugging - computes for the expected STFT frame count (T) for an epoch
def expected_num_frames(epoch_length_sec: int, sample_rate: int, nperseg: int, noverlap: int) -> int:
    """
    Formula: 1 + floor((N - nperseg) / hop) where hop = nperseg - noverlap.
    """
    N = int(epoch_length_sec * sample_rate)
    hop = nperseg - noverlap
    if N < nperseg or hop <= 0:
        return 0
    return 1 + (N - nperseg) // hop



def compute_stft_epoch(
    epoch_signal: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    nperseg: int = SAMPLE_RATE,
    noverlap: int = SAMPLE_RATE // 2,
    max_freq: float = 40.0,
    return_complex: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Compute STFT for one 1-D epoch signal.

    Backward-compatible behavior:
      - By default (return_complex=False) returns a 3-tuple:
          (stft_db_trimmed, freqs_trim, times)
        where stft_db_trimmed is 20*log10(|Zxx|) (shape F_trim x T).
      - If return_complex=True, returns a 4-tuple:
          (stft_db_trimmed, freqs_trim, times, Zxx_trimmed)
        where Zxx_trimmed is the complex STFT (F_trim x T).

    Notes:
      - Keep nperseg/noverlap the same as your previous code so results match.
      - Use max_freq to trim frequency rows (e.g., <= 40 Hz).
      - Use small EPS to avoid log(0).
    """
    x = np.asarray(epoch_signal)
    if x.ndim > 1:
        x = np.squeeze(x)

    freqs, times, Zxx_full = stft(x, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, boundary=None)
    mask = freqs <= max_freq

    Zxx = Zxx_full[mask, :]  # complex (F_trim, T)
    mag = np.abs(Zxx)  # magnitude
    stft_db = 20.0 * np.log10(mag + EPS)  # dB

    freqs_trim = freqs[mask]

    if return_complex:
        # Return complex STFT as well (4-tuple)
        return stft_db, freqs_trim, times, Zxx
    else:
        # Backwards compatible (3-tuple)
        return stft_db, freqs_trim, times


# Convert db to linear power for band averaging
def stft_db_to_power(stft_db: np.ndarray) -> np.ndarray:
    """
    Convert stft dB (20*log10(mag)) to linear power (mag^2).
    Because 20*log10(mag) => mag = 10^(dB/20); power ∝ mag^2 => 10^(dB/10).
    """
    return 10.0 ** (np.asarray(stft_db) / 10.0)


# Normalize the linear power immediately after computing power
def normalize_power_matrix(power: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize a linear power matrix (F x T).
    Supported methods:
      - "minmax": (x - min) / (max - min)
      - "zscore": (x - mean) / std
    Returns normalized matrix (same shape).
    """
    if power is None or power.size == 0:
        return power
    if method == "minmax":
        mn = np.nanmin(power)
        mx = np.nanmax(power)
        if mx - mn == 0:
            return np.zeros_like(power)
        return (power - mn) / (mx - mn)
    elif method == "zscore":
        mu = np.nanmean(power)
        sd = np.nanstd(power)
        if sd == 0:
            return np.zeros_like(power)
        return (power - mu) / sd
    else:
        raise ValueError(f"Unknown normalization method: {method}")



def compute_stft_all_epochs_channels(
    recording,
    labeled_epochs: List[Dict[str, Any]],
    channel_names: Optional[List[str]] = None,
    sample_rate: int = SAMPLE_RATE,
    nperseg: int = SAMPLE_RATE,
    noverlap: int = SAMPLE_RATE // 2,
    max_freq: float = 40.0,
    compute_power: bool = True,
    normalize_power: bool = False,
    normalization_method: str = "minmax",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compute complex STFT and derived representations (dB & linear power) once for every
    channel and epoch.

    Returns a comprehensive store:
      stft_store[channel] = [ { 'start','end','phase',
                                'stft_db' : (F x T) dB,
                                'power'   : (F x T) linear power (or None if compute_power False),
                                'Zxx'     : (F x T) complex coefficients,
                                'freqs'   : (F,), 'times' : (T,) }, ... ]
    This is intended to be used by both TF and bispectrum branches.
    """
    if channel_names is None:
        channel_names = SELECTED_CHANNELS

    sfreq = float(recording.info["sfreq"])
    ch_index_map = {name: recording.ch_names.index(name) for name in channel_names}

    stft_store: Dict[str, List[Dict[str, Any]]] = {ch: [] for ch in channel_names}

    for ep in labeled_epochs:
        start_sec = int(ep["start"])
        end_sec = int(ep["end"])
        phase = ep["phase"]
        start_sample = int(start_sec * sfreq)
        end_sample = int(end_sec * sfreq)

        for ch in channel_names:
            idx = ch_index_map[ch]
            data = recording.get_data(picks=[idx], start=start_sample, stop=end_sample)
            if data.size == 0:
                stft_store[ch].append({"start": start_sec, "end": end_sec, "phase": phase, "stft_db": None, "power": None, "Zxx": None, "freqs": None, "times": None})
                continue

            sig = np.squeeze(data)
            # Request complex so we can derive everything from Zxx (no double STFT)
            stft_db, freqs, times, Zxx = compute_stft_epoch(sig, sample_rate=sample_rate, nperseg=nperseg, noverlap=noverlap, max_freq=max_freq, return_complex=True)

            power = None
            if compute_power:
                power = stft_db_to_power(stft_db)  # linear power
                if normalize_power and power is not None:
                    power = normalize_power_matrix(power, method=normalization_method)

            stft_store[ch].append({
                "start": start_sec,
                "end": end_sec,
                "phase": phase,
                "stft_db": stft_db,   # dB matrix
                "power": power,       # linear power (or None)
                "Zxx": Zxx,           # complex coefficients
                "freqs": freqs,
                "times": times
            })

    return stft_store


def precompute_stfts_for_all_patients(
    number_of_patients: int,
    dataset_path: str,
    out_root: str,
    selected_channels: List[str],
    labeled_epochs_builder,  # callable(patient_id) -> list of epoch dicts
    sample_rate: int = SAMPLE_RATE,
    nperseg: int = SAMPLE_RATE,
    noverlap: int = SAMPLE_RATE // 2,
    max_freq: float = 40.0,
    compute_power: bool = True,
    normalize_power_flag: bool = False,
    normalization_method: str = "minmax",
    skip_existing: bool = True,
):
    """
    Precompute STFT (Zxx), stft_db, and power for all patients and save per-channel-per-epoch npz files.

    - labeled_epochs_builder: function that given patient_id returns labeled_epochs (list of {'start','end','phase'}).
    - out_root: directory to store precomputed STFT files (eg SPECS_OUTPUT_PATH/precomputed_stft)
    """
    os.makedirs(out_root, exist_ok=True)
    log = logger.getChild("precompute_stfts")
    log.info("Starting STFT precomputation for %d patients", number_of_patients)

    for idx in range(1, number_of_patients + 1):
        pid = f"{idx:02d}"
        log.info("Patient %s: building labeled epochs", pid)
        labeled_epochs = labeled_epochs_builder(pid)
        if not labeled_epochs:
            log.info("Patient %s: no labeled epochs found, skipping", pid)
            continue

        # load recording (caller must provide compatible loader)
        # Here we expect the caller to pass a recording or implement a loader inside labeled_epochs_builder
        # For convenience we allow labeled_epochs_builder to also return a tuple (epochs, recording) if desired.
        # If labeled_epochs is a tuple: (epochs, recording)
        recording = None
        if isinstance(labeled_epochs, tuple) and len(labeled_epochs) == 2:
            labeled_epochs, recording = labeled_epochs
        if recording is None:
            # if no recording passed, assume caller will call compute_stft_all_epochs_channels themselves
            raise RuntimeError("precompute_stfts_for_all_patients requires labeled_epochs_builder to return (epochs, recording) or provide a recording separately")

        patient_root = os.path.join(out_root, f"patient_{pid}", "stft")
        os.makedirs(patient_root, exist_ok=True)

        sfreq = float(recording.info["sfreq"])
        ch_index_map = {name: recording.ch_names.index(name) for name in selected_channels}

        for ep in labeled_epochs:
            start = int(ep["start"]); end = int(ep["end"]); phase = ep["phase"]
            for ch in selected_channels:
                idx_ch = ch_index_map[ch]
                data = recording.get_data(picks=[idx_ch], start=int(start * sfreq), stop=int(end * sfreq))
                ch_folder = os.path.join(patient_root, f"channel_{ch}")
                os.makedirs(ch_folder, exist_ok=True)
                out_fn = os.path.join(ch_folder, f"epoch_{start}_{end}.npz")
                if skip_existing and os.path.exists(out_fn):
                    continue
                if data.size == 0:
                    np.savez_compressed(out_fn,
                                        Zxx=np.array([]),
                                        stft_db=np.array([]),
                                        power=np.array([]),
                                        freqs=np.array([]),
                                        times=np.array([]),
                                        phase=str(phase))
                    continue
                sig = np.squeeze(data)
                stft_db, freqs, times, Zxx = compute_stft_epoch(sig,
                                                                sample_rate=sample_rate,
                                                                nperseg=nperseg,
                                                                noverlap=noverlap,
                                                                max_freq=max_freq,
                                                                return_complex=True)
                power = None
                if compute_power:
                    power = stft_db_to_power(stft_db)
                    if normalize_power_flag and power is not None:
                        power = normalize_power_matrix(power, method=normalization_method)

                # save arrays (Zxx complex allowed)
                np.savez_compressed(out_fn,
                                    Zxx=Zxx,
                                    stft_db=stft_db.astype(np.float32),
                                    power=(power.astype(np.float32) if power is not None else np.array([])),
                                    freqs=freqs.astype(np.float32),
                                    times=times.astype(np.float32),
                                    phase=str(phase))
        log.info("Patient %s: finished", pid)

    log.info("All patients STFT precomputation completed.")


def load_precomputed_stft(patient_id: str, channel: str, start: int, end: int, precomputed_root: str) -> Dict[str, Any]:
    """
    Load the precomputed .npz for a single patient/channel/epoch.
    Returns dict with keys: Zxx, stft_db, power, freqs, times, phase
    If file missing returns dict with None values.
    """
    fn = os.path.join(precomputed_root, f"patient_{patient_id}", "stft", f"channel_{channel}", f"epoch_{start}_{end}.npz")
    if not os.path.exists(fn):
        return {"Zxx": None, "stft_db": None, "power": None, "freqs": None, "times": None, "phase": None}
    d = np.load(fn, allow_pickle=True)
    def _get(k):
        if k in d.files and d[k].size:
            return d[k]
        return None
    return {
        "Zxx": _get("Zxx"),
        "stft_db": _get("stft_db"),
        "power": _get("power"),
        "freqs": _get("freqs"),
        "times": _get("times"),
        "phase": str(d["phase"]) if "phase" in d.files else None,
    }


def replicate_resize_and_save(
    mosaic_2d: np.ndarray,
    out_path: str,
    metadata: Dict[str, Any],
    resize_to: Tuple[int, int] = (224, 224),
    manifest_path: Optional[str] = None,
    manifest_row: Optional[List[Any]] = None,
    global_normalize_before_uint8: bool = False,
):
    """
    Convert mosaic_2d (float) -> uint8 RGB -> resize -> save .npz and optionally append manifest row.
    - If global_normalize_before_uint8 = True, min-max normalize mosaic_2d to [0,1] before conversion.
    """
    # Global mosaic min-max normalization option
    if global_normalize_before_uint8:
        mn = float(np.nanmin(mosaic_2d)) if mosaic_2d.size else 0.0
        mx = float(np.nanmax(mosaic_2d)) if mosaic_2d.size else 0.0
        if mx - mn == 0:
            mosaic_2d = np.zeros_like(mosaic_2d)
        else:
            mosaic_2d = (mosaic_2d - mn) / (mx - mn)

    # Convert to uint8 RGB
    u8 = _minmax_float_to_uint8(mosaic_2d)
    rgb = replicate_grayscale_to_rgb(u8)
    img = Image.fromarray(rgb)
    img = img.resize(resize_to, resample=Image.Resampling.BICUBIC)
    final_arr = np.asarray(img)

    # Save as compressed npz with metadata
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np_meta = {}
    for k, v in metadata.items():
        try:
            np_meta[k] = np.array(v)
        except Exception:
            np_meta[k] = np.array([str(v)])
    np.savez_compressed(out_path, image=final_arr, **np_meta)

    # to load labels and for file locations
    if manifest_path and manifest_row is not None:
        write_header = not os.path.exists(manifest_path)
        with open(manifest_path, "a", newline="") as mf:
            writer = csv.writer(mf)
            if write_header:
                writer.writerow([
                    "filepath", "patient_id", "start_sec", "end_sec", "phase",
                    "channels", "band_names_or_freqs", "mosaic_pre_H", "mosaic_pre_W",
                    "tile_T", "nperseg", "noverlap", "mode",
                ])
            writer.writerow(manifest_row)