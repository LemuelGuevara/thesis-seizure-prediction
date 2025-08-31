"""
Bispectrum branch:
 - If precomputed_stft_root provided, load per-epoch/channel .npz
 - Else, accepts stft_store (in-memory) produced by compute_stft_all_epochs_channels
 - Produces per-epoch mosaics of bispectrum band matrices and writes manifest rows
"""
from typing import List, Dict, Any, Optional, Tuple
import os
import csv
import numpy as np

# Import helpers from your data_transformation module
from data_transformation import (
    load_precomputed_stft,
    replicate_resize_and_save,
    BAND_DEFS,
    SELECTED_CHANNELS,
)

# small numeric eps
_EPS = 1e-12


def group_complex_into_bands(
    Zxx: np.ndarray,
    freqs: np.ndarray,
    band_defs: Dict[str, Tuple[float, float]] = BAND_DEFS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Average complex STFT coefficients across frequency bins for each canonical band.

    Returns:
      - band_complex: (n_bands, T) complex array
      - band_centers: (n_bands,) float center frequency of each band
    """
    bands = list(band_defs.keys())
    n_bands = len(bands)

    if Zxx is None or Zxx.size == 0 or freqs is None:
        return np.zeros((n_bands, 0), dtype=np.complex128), np.array([ (band_defs[b][0] + band_defs[b][1]) / 2.0 for b in bands ])

    T = Zxx.shape[1]
    band_complex = np.zeros((n_bands, T), dtype=np.complex128)
    band_centers = np.zeros(n_bands, dtype=float)

    for i, b in enumerate(bands):
        low, high = band_defs[b]
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            band_complex[i, :] = 0.0 + 0.0j
            band_centers[i] = (low + high) / 2.0
        else:
            band_complex[i, :] = np.mean(Zxx[mask, :], axis=0)
            band_centers[i] = float(np.mean(freqs[mask]))
    return band_complex, band_centers


def compute_bispectrum_band_matrix(
    Zxx: np.ndarray,
    freqs: np.ndarray,
    band_complex: np.ndarray,
    band_centers: np.ndarray,
    eps: float = _EPS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute band-level bispectrum using triple product:
      B[i,j] = mean_t( X_i(t) * X_j(t) * conj(X_sum(t)) ),
    where X_sum(t) is the complex coefficient at freq closest to center_i + center_j.

    Returns:
      - B_complex: complex matrix (n_bands x n_bands)
      - B_db: real matrix in dB (20*log10(|B_complex| + eps))
    """
    n_bands = band_complex.shape[0]
    B = np.zeros((n_bands, n_bands), dtype=np.complex128)

    if freqs is None or freqs.size == 0:
        return B, 20.0 * np.log10(np.abs(B) + eps)

    max_freq = freqs[-1]
    # precompute nearest freq index for each pair
    sum_idx = np.full((n_bands, n_bands), -1, dtype=int)
    for i in range(n_bands):
        for j in range(n_bands):
            target = band_centers[i] + band_centers[j]
            if target <= max_freq:
                sum_idx[i, j] = int(np.argmin(np.abs(freqs - target)))
            else:
                sum_idx[i, j] = -1

    # compute triple product
    for i in range(n_bands):
        Xi = band_complex[i, :]
        for j in range(n_bands):
            Xj = band_complex[j, :]
            idx = sum_idx[i, j]
            if idx == -1:
                B[i, j] = 0.0 + 0.0j
            else:
                Xsum = Zxx[idx, :]
                B[i, j] = np.mean(Xi * Xj * np.conjugate(Xsum))

    B_db = 20.0 * np.log10(np.abs(B) + eps)
    return B, B_db


def process_patient_bispectrum(
    *,
    patient_id: str,
    labeled_epochs: List[Dict[str, Any]],
    out_dir: str,
    channel_names: Optional[List[str]] = None,
    precomputed_stft_root: Optional[str] = None,
    stft_store: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    grid: Tuple[int, int] = (4, 4),
    per_tile_normalization: bool = True,
    tile_norm_method: str = "minmax",
    global_normalize_mosaic: bool = True,
    resize_to: Tuple[int, int] = (224, 224),
    manifest: bool = True,
    manifest_filename: str = "manifest_bispec.csv",
    skip_existing: bool = True,
    nperseg: int = 256,
    noverlap: int = 128,
) -> Dict[str, Any]:
    """
    Orchestrator for bispectrum processing for one patient.
    Returns dict with saved files list and counts.
    """
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, manifest_filename) if manifest else None
    channel_names = channel_names or SELECTED_CHANNELS

    # If stft_store not provided, and no precomputed root, compute stft_store on-the-fly
    if stft_store is None and precomputed_stft_root is None:
        raise RuntimeError("Provide either stft_store (in-memory) or precomputed_stft_root (on-disk).")

    # Build bispec tiles per channel per epoch
    bispec_tiles: Dict[str, List[np.ndarray]] = {ch: [] for ch in channel_names}

    for ep in labeled_epochs:
        start = int(ep["start"]); end = int(ep["end"]); phase = ep["phase"]
        for ch in channel_names:
            if precomputed_stft_root:
                d = load_precomputed_stft(patient_id, ch, start, end, precomputed_stft_root)
                Zxx = d["Zxx"]
                freqs = d["freqs"]
            else:
                # find matching rec in stft_store
                recs = stft_store.get(ch, [])
                idx = next((i for i, rr in enumerate(recs) if rr["start"] == start and rr["end"] == end), None)
                if idx is None:
                    Zxx = None; freqs = None
                else:
                    Zxx = recs[idx]["Zxx"]
                    freqs = recs[idx]["freqs"]

            if Zxx is None or Zxx.size == 0:
                # placeholder zero tile (n_bands x n_bands)
                n_bands = len(BAND_DEFS)
                bispec_tiles[ch].append(np.zeros((n_bands, n_bands), dtype=float))
                continue

            # 1) band-complex averaging
            band_complex, band_centers = group_complex_into_bands(Zxx, freqs, BAND_DEFS)

            # 2) bispectrum computation (band-level)
            B_complex, B_db = compute_bispectrum_band_matrix(Zxx, freqs, band_complex, band_centers)

            # 3) per-tile normalization if desired
            tile = B_db.copy()
            if per_tile_normalization:
                if tile_norm_method == "minmax":
                    mn = np.nanmin(tile); mx = np.nanmax(tile)
                    tile = np.zeros_like(tile) if mx - mn == 0 else (tile - mn) / (mx - mn)
                elif tile_norm_method == "zscore":
                    mu = np.nanmean(tile); sd = np.nanstd(tile)
                    tile = np.zeros_like(tile) if sd == 0 else (tile - mu) / sd

            bispec_tiles[ch].append(tile)

    # Build mosaics epoch-by-epoch and save
    rows, cols = grid
    cells = rows * cols
    selected = channel_names[:cells]
    saved_files = []
    n_bands = len(BAND_DEFS)
    for ep_idx, ep in enumerate(labeled_epochs):
        start = int(ep["start"]); end = int(ep["end"]); phase = ep["phase"]
        fname = f"bispec_epoch_{start}_{end}.npz"
        out_path = os.path.join(out_dir, fname)

        if skip_existing and os.path.exists(out_path):
            saved_files.append(out_path)
            continue

        # assemble mosaic (tiles are square n_bands x n_bands)
        tile_H = n_bands; tile_W = n_bands
        mosaic_H = rows * tile_H; mosaic_W = cols * tile_W
        mosaic = np.zeros((mosaic_H, mosaic_W), dtype=float)
        idx = 0
        for r in range(rows):
            for c in range(cols):
                ch = selected[idx]
                tile = bispec_tiles[ch][ep_idx] if ep_idx < len(bispec_tiles[ch]) else np.zeros((n_bands, n_bands))
                r0 = r * tile_H; c0 = c * tile_W
                mosaic[r0:r0+tile_H, c0:c0+tile_W] = tile
                idx += 1

        # Optional global normalize
        if global_normalize_mosaic:
            mn = float(np.nanmin(mosaic)) if mosaic.size else 0.0
            mx = float(np.nanmax(mosaic)) if mosaic.size else 0.0
            if mx - mn == 0:
                mosaic = np.zeros_like(mosaic)
            else:
                mosaic = (mosaic - mn) / (mx - mn)

        metadata = {
            "start_sec": start,
            "end_sec": end,
            "channels": selected,
            "band_names": list(BAND_DEFS.keys()),
            "phase": ep["phase"],
        }

        manifest_row = None
        if manifest and manifest_path:
            manifest_row = [out_path, patient_id, start, end, phase, ";".join(selected), ";".join(list(BAND_DEFS.keys())), mosaic_H, mosaic_W, tile_W, nperseg, noverlap, "bispec"]

        replicate_resize_and_save(
            mosaic_2d=mosaic,
            out_path=out_path,
            metadata=metadata,
            resize_to=resize_to,
            manifest_path=manifest_path,
            manifest_row=manifest_row,
            global_normalize_before_uint8=False,
        )
        saved_files.append(out_path)

    return {"saved_files": saved_files}
