"""
Time-frequency branch (band / detailed).
Uses precomputed STFT files (preferred) or computes STFTs once via compute_stft_all_epochs_channels.
Produces mosaics (band-mode or detailed-mode) and writes manifest CSV.
"""
from typing import List, Dict, Any, Optional, Tuple
import os
import csv
import numpy as np

from data_transformation import (
    load_precomputed_stft,
    compute_stft_all_epochs_channels,
    replicate_resize_and_save,
    BAND_DEFS,
    SELECTED_CHANNELS,
    SAMPLE_RATE,
)

# build a view compatible with existing stft_results consumers
def build_tf_view(
    stft_store: Dict[str, List[Dict[str, Any]]],
    tf_type: str = "power"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build the 'stft_results' view expected by downstream functions (where each rec has 'tf','freqs','times','start','end','phase').

    tf_type:
      - "power"   -> rec['tf'] = rec['power']   (used for band maps)
      - "stft_db" -> rec['tf'] = rec['stft_db'] (used for detailed spectrogram images)
      - "Zxx"     -> rec['tf'] = rec['Zxx']     (if you want direct complex tiles)
    """
    if tf_type not in ("power", "stft_db", "Zxx"):
        raise ValueError("tf_type must be 'power', 'stft_db', or 'Zxx'")

    view: Dict[str, List[Dict[str, Any]]] = {}
    for ch, recs in stft_store.items():
        out = []
        for r in recs:
            tf = None
            if tf_type == "power":
                tf = r.get("power", None)
            elif tf_type == "stft_db":
                tf = r.get("stft_db", None)
            else:
                tf = r.get("Zxx", None)
            out.append({"start": r["start"], "end": r["end"], "phase": r["phase"], "tf": tf, "freqs": r.get("freqs", None), "times": r.get("times", None)})
        view[ch] = out
    return view


# Backwards-compatible wrapper: compute and return old stft_results view
def apply_stft_all_epochs_channels(
    recording,
    labeled_epochs: List[Dict[str, Any]],
    channel_names: Optional[List[str]] = None,
    sample_rate: int = SAMPLE_RATE,
    nperseg: int = SAMPLE_RATE,
    noverlap: int = SAMPLE_RATE // 2,
    max_freq: float = 40.0,
    convert_to_power: bool = False,
    normalize_power: bool = False,
    normalization_method: str = "minmax",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Backwards-compatible wrapper: computes STFT store once and returns the old-style
    stft_results where rec['tf'] contains either linear power (if convert_to_power True)
    or stft_db (if convert_to_power False). Internally it calls compute_stft_all_epochs_channels.
    """
    stft_store = compute_stft_all_epochs_channels(
        recording=recording,
        labeled_epochs=labeled_epochs,
        channel_names=channel_names,
        sample_rate=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        max_freq=max_freq,
        compute_power=convert_to_power,
        normalize_power=normalize_power,
        normalization_method=normalization_method,
    )

    tf_type = "power" if convert_to_power else "stft_db"
    return build_tf_view(stft_store, tf_type=tf_type)


def group_power_into_bands(
    power: np.ndarray,
    freqs: np.ndarray,
    band_defs: Dict[str, Tuple[float, float]] = BAND_DEFS,
) -> np.ndarray:
    """
    Convert linear power (F x T) to band-time map (n_bands x T) by averaging power across frequency bins in each band.
    Returns band_time (n_bands, T).
    """
    if power is None or freqs is None or power.size == 0:
        return np.zeros((len(band_defs), 0), dtype=float)
    bands = list(band_defs.keys())
    band_time = np.zeros((len(bands), power.shape[1]), dtype=float)
    for i, b in enumerate(bands):
        low, high = band_defs[b]
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            band_time[i, :] = 0.0
        else:
            band_power = power[mask, :]  # (Fb, T)
            band_time[i, :] = np.mean(band_power, axis=0)
    return band_time


def create_band_time_maps_per_channel(
    stft_results: Dict[str, List[Dict[str, Any]]],
    band_defs: Dict[str, Tuple[float, float]] = BAND_DEFS,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convert stft_results (which contain linear power in 'tf') into band-time maps per channel.
    Output: channel -> list of epoch dicts: {"start","end","phase","band_time","times"}
    """
    band_maps: Dict[str, List[Dict[str, Any]]] = {}
    for ch, recs in stft_results.items():
        out_list = []
        for rec in recs:
            tf = rec["tf"]
            freqs = rec["freqs"]
            times = rec["times"]
            phase = rec["phase"]
            if tf is None:
                band_time = np.zeros((len(band_defs), 0), dtype=float)
            else:
                # We assume tf is already linear power (or normalized linear power).
                band_time = group_power_into_bands(tf, freqs, band_defs)
            out_list.append({"start": rec["start"], "end": rec["end"], "phase": phase, "band_time": band_time, "times": times})
        band_maps[ch] = out_list
    return band_maps



def build_mosaic_for_epoch(
    *,
    mode: str,
    stft_results: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    band_maps: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    channel_order: List[str],
    epoch_idx: int,
    grid: Tuple[int, int] = (4, 4),
    per_tile_normalization: bool = False,
    tile_norm_method: str = "minmax",
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Build a 2D mosaic for a single epoch index.
    mode: "band" or "detailed".
      - band: uses band_maps; each tile is (n_bands x T)
      - detailed: uses stft_results; each tile is (F_trim x T) (full spectrogram)
    per_tile_normalization: if True, normalize each tile individually (minmax or zscore).
    Returns (mosaic_2d_float, (H, W)) before resizing.
    """
    rows, cols = grid
    cells = rows * cols
    selected_channels = channel_order[:cells]

    # create mosaic from band maps (dimensions: rows x 5)
    if mode == "band":
        assert band_maps is not None
        n_bands = len(BAND_DEFS)

        # Determine tile_T (max frames among channels)
        tile_T = 0

        for ch in selected_channels:
            recs = band_maps.get(ch, [])
            if epoch_idx < len(recs):
                bt = recs[epoch_idx]["band_time"]
                if bt is not None and bt.size:
                    tile_T = max(tile_T, bt.shape[1])
        if tile_T == 0:
            tile_T = 1

        tiles = []
        for ch in selected_channels:
            recs = band_maps.get(ch, [])
            if epoch_idx >= len(recs) or recs[epoch_idx]["band_time"] is None or recs[epoch_idx]["band_time"].size == 0:
                tile = np.zeros((n_bands, tile_T), dtype=float)
            else:
                tile = recs[epoch_idx]["band_time"]  # (n_bands, Tch)
                if tile.shape[1] < tile_T:
                    tile = np.pad(tile, ((0, 0), (0, tile_T - tile.shape[1])), mode="constant", constant_values=0.0)
                elif tile.shape[1] > tile_T:
                    tile = tile[:, :tile_T]

                # Per-tile normalization option
                if per_tile_normalization:
                    if tile_norm_method == "minmax":
                        mn = np.nanmin(tile)
                        mx = np.nanmax(tile)
                        tile = np.zeros_like(tile) if mx - mn == 0 else (tile - mn) / (mx - mn)
                    elif tile_norm_method == "zscore":
                        mu = np.nanmean(tile)
                        sd = np.nanstd(tile)
                        tile = np.zeros_like(tile) if sd == 0 else (tile - mu) / sd
            tiles.append(tile)

        mosaic_h = rows * n_bands
        mosaic_w = cols * tile_T
        mosaic = np.zeros((mosaic_h, mosaic_w), dtype=float)
        idx = 0
        for r in range(rows):
            for c in range(cols):
                tile = tiles[idx]
                r0 = r * n_bands
                c0 = c * tile_T
                mosaic[r0:r0 + n_bands, c0:c0 + tile_T] = tile
                idx += 1
        return mosaic, (mosaic_h, mosaic_w)

    # create mosaic from full stft (dimensions: rows x freqs bins)
    elif mode == "detailed":
        assert stft_results is not None

        # Determine tile_T and n_freqs (max across channels)
        tile_T = 0
        n_freqs = 0

        for ch in selected_channels:
            recs = stft_results.get(ch, [])
            if epoch_idx < len(recs):
                tf = recs[epoch_idx]["tf"]
                if tf is not None and tf.size:
                    n_freqs = max(n_freqs, tf.shape[0])
                    tile_T = max(tile_T, tf.shape[1])
        if tile_T == 0:
            tile_T = 1
        if n_freqs == 0:
            n_freqs = 1

        tiles = []
        for ch in selected_channels:
            recs = stft_results.get(ch, [])
            if epoch_idx >= len(recs) or recs[epoch_idx]["tf"] is None:
                tile = np.zeros((n_freqs, tile_T), dtype=float)
            else:
                tile = recs[epoch_idx]["tf"]  # (Fch, Tch)
                # pad/truncate freq-axis
                if tile.shape[0] < n_freqs:
                    tile = np.pad(tile, ((0, n_freqs - tile.shape[0]), (0, 0)), mode="constant", constant_values=0.0)
                elif tile.shape[0] > n_freqs:
                    tile = tile[:n_freqs, :]
                # pad/truncate time-axis
                if tile.shape[1] < tile_T:
                    tile = np.pad(tile, ((0, 0), (0, tile_T - tile.shape[1])), mode="constant", constant_values=0.0)
                elif tile.shape[1] > tile_T:
                    tile = tile[:, :tile_T]

                # Per-tile normalization option
                if per_tile_normalization:
                    if tile_norm_method == "minmax":
                        mn = np.nanmin(tile)
                        mx = np.nanmax(tile)
                        tile = np.zeros_like(tile) if mx - mn == 0 else (tile - mn) / (mx - mn)
                    elif tile_norm_method == "zscore":
                        mu = np.nanmean(tile)
                        sd = np.nanstd(tile)
                        tile = np.zeros_like(tile) if sd == 0 else (tile - mu) / sd
            tiles.append(tile)

        mosaic_h = rows * n_freqs
        mosaic_w = cols * tile_T
        mosaic = np.zeros((mosaic_h, mosaic_w), dtype=float)
        idx = 0
        for r in range(rows):
            for c in range(cols):
                tile = tiles[idx]
                r0 = r * n_freqs
                c0 = c * tile_T
                mosaic[r0:r0 + n_freqs, c0:c0 + tile_T] = tile
                idx += 1
        return mosaic, (mosaic_h, mosaic_w)

    else:
        raise ValueError("mode must be 'band' or 'detailed'")



def process_patient_timefrequency(
    *,
    patient_id: str,
    labeled_epochs: List[Dict[str, Any]],
    out_dir: str,
    mode: str = "band",  # 'band' or 'detailed'
    channel_names: Optional[List[str]] = None,
    precomputed_stft_root: Optional[str] = None,
    stft_store: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    nperseg: int = SAMPLE_RATE,
    noverlap: int = SAMPLE_RATE // 2,
    max_freq: float = 40.0,
    convert_to_power: bool = True,    # for band mode, should be True
    normalize_power: bool = True,
    normalization_method: str = "minmax",
    per_tile_normalization: bool = False,
    tile_norm_method: str = "minmax",
    global_normalize_mosaic: bool = False,
    grid: Tuple[int,int] = (4,4),
    resize_to: Tuple[int,int] = (224,224),
    manifest: bool = True,
    manifest_filename: str = "manifest_tf.csv",
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """
    Orchestrator for the TF branch for one patient.
    Returns dict with saved_files and counts.
    """
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, manifest_filename) if manifest else None
    channel_names = channel_names or SELECTED_CHANNELS

    # 1) Acquire stft_results view (either from precomputed files, from stft_store, or compute on-the-fly)
    if precomputed_stft_root:
        stft_results: Dict[str, List[Dict[str, Any]]] = {ch: [] for ch in channel_names}
        for ep in labeled_epochs:
            start = int(ep["start"]); end = int(ep["end"]); phase = ep["phase"]
            for ch in channel_names:
                d = load_precomputed_stft(patient_id, ch, start, end, precomputed_stft_root)
                # choose tf depending on mode
                tf = d["power"] if mode == "band" else d["stft_db"]
                stft_results[ch].append({"start": start, "end": end, "phase": phase, "tf": tf, "freqs": d["freqs"], "times": d["times"]})
    else:
        # if caller did not provide stft_store, compute STFTs (once) using compute_stft_all_epochs_channels
        if stft_store is None:
            # compute power when mode == 'band'
            stft_store = compute_stft_all_epochs_channels(
                recording=None,  # caller may have passed recording into higher-level runner; here we expect stft_store or precomputed root
                labeled_epochs=labeled_epochs,
                channel_names=channel_names,
                sample_rate=SAMPLE_RATE,
                nperseg=nperseg,
                noverlap=noverlap,
                max_freq=max_freq,
                compute_power=(mode == "band"),
                normalize_power=normalize_power,
                normalization_method=normalization_method,
            )
            # NOTE: above compute_stft_all_epochs_channels requires a recording object; if you plan to call this path
            # pass in the `recording` or call compute_stft_all_epochs_channels at caller level and pass stft_store here.
            raise RuntimeError("compute_stft_all_epochs_channels requires a recording and cannot be called here without it. Provide stft_store or precomputed_stft_root.")
        # Build view
        stft_results = build_tf_view(stft_store, tf_type="power" if mode == "band" else "stft_db")

    # 2) If band mode, convert to band maps
    band_maps = None
    if mode == "band":
        band_maps = create_band_time_maps_per_channel(stft_results, band_defs=BAND_DEFS)

    # 3) Build mosaics per epoch and save
    saved_files = []
    num_epochs = len(labeled_epochs)
    for epoch_idx in range(num_epochs):
        ep = labeled_epochs[epoch_idx]
        start = ep["start"]; end = ep["end"]; phase = ep["phase"]
        fname = f"tf_{mode}_epoch_{start}_{end}.npz"
        out_path = os.path.join(out_dir, fname)

        if skip_existing and os.path.exists(out_path):
            saved_files.append(out_path)
            continue

        mosaic_2d, (H, W) = build_mosaic_for_epoch(
            mode=mode,
            stft_results=stft_results,
            band_maps=band_maps,
            channel_order=channel_names,
            epoch_idx=epoch_idx,
            grid=grid,
            per_tile_normalization=per_tile_normalization,
            tile_norm_method=tile_norm_method,
        )

        # Optionally global normalize mosaic
        if global_normalize_mosaic:
            mn = float(np.nanmin(mosaic_2d)) if mosaic_2d.size else 0.0
            mx = float(np.nanmax(mosaic_2d)) if mosaic_2d.size else 0.0
            if mx - mn == 0:
                mosaic_2d = np.zeros_like(mosaic_2d)
            else:
                mosaic_2d = (mosaic_2d - mn) / (mx - mn)

        metadata = {
            "start_sec": start,
            "end_sec": end,
            "channels": channel_names[: grid[0] * grid[1]],
            "band_names": list(BAND_DEFS.keys()) if mode == "band" else [],
            "phase": phase,
        }

        manifest_row = None
        if manifest and manifest_path:
            band_or_freq = ";".join(list(BAND_DEFS.keys())) if mode == "band" else "freqs"
            # tile_T heuristic: W / cols
            tile_T = W // grid[1] if grid[1] else W
            manifest_row = [out_path, patient_id, start, end, phase, ";".join(channel_names[:grid[0]*grid[1]]), band_or_freq, H, W, tile_T, nperseg, noverlap, mode]

        replicate_resize_and_save(
            mosaic_2d=mosaic_2d,
            out_path=out_path,
            metadata=metadata,
            resize_to=resize_to,
            manifest_path=manifest_path,
            manifest_row=manifest_row,
            global_normalize_before_uint8=False,
        )
        saved_files.append(out_path)

    return {"saved_files": saved_files}
