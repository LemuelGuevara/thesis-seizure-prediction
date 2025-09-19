"""
Bispectrum Processing Module

This module provides functions for procesing STFT complex coefficients into bispectrum
estimations.

NOTE: Same precomputed STFT will be used in here as inputs for the functions.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.config import PreprocessingConfig
from src.logger import setup_logger

logger = setup_logger(name="bispectrum_branch")

# small numeric eps
_EPS = 1e-12


def compute_band_average_stft_coeffs(
    Zxx: np.ndarray, freqs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute band-averaged complex STFT coefficients for canonical frequency bands.

    Args:
        Zxx (np.ndarray): STFT complex coefficients, shape (n_freqs, T)
        freqs (np.ndarray): STFT frequency bins, shape (n_freqs,)

    Returns:
        band_complex (np.ndarray): Averaged complex STFT per band, shape (n_bands, T)
        band_centers (np.ndarray): Center frequency of each band, shape (n_bands,)
    """
    band_defs = PreprocessingConfig.band_defs
    bands = list(band_defs.keys())
    n_bands = len(bands)

    logger.info(f"Computing band-averaged STFTs for {n_bands} bands.")

    if Zxx is None or Zxx.size == 0 or freqs is None:
        band_centers = np.array(
            [(band_defs[b][0] + band_defs[b][1]) / 2.0 for b in bands], dtype=float
        )
        band_complex = np.zeros((n_bands, 0), dtype=np.complex128)
        return band_complex, band_centers

    T = Zxx.shape[1]
    band_complex = np.zeros((n_bands, T), dtype=np.complex128)
    band_centers = np.zeros(n_bands, dtype=float)

    for i, b in enumerate(bands):
        low, high = band_defs[b]
        mask = (freqs >= low) & (freqs <= high)

        logger.debug(f"Band {b}: range=({low}, {high}), bins={np.count_nonzero(mask)}")

        if not np.any(mask):
            band_complex[i, :] = np.zeros(T, dtype=np.complex128)
            band_centers[i] = (low + high) / 2.0
            logger.debug(
                f"No frequencies in range, using center {band_centers[i]:.2f} Hz."
            )
        else:
            band_complex[i, :] = np.mean(Zxx[mask, :], axis=0).astype(np.complex128)
            band_centers[i] = float(np.mean(freqs[mask]))
            logger.debug(f"Center frequency set to {band_centers[i]:.2f} Hz.")

    return band_complex, band_centers


def compute_bispectrum_estimation(
    Zxx: np.ndarray,
    freqs: np.ndarray,
    eps: float = _EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the  bispectrum estimation using the triple product:
        B[i, j] = mean_t( X_i(t) * X_j(t) * conj(X_sum(t)) )
    where X_sum(t) is the STFT coefficient at the frequency bin
    nearest to (center_i + center_j).

    Additionally, this function also handles the flexibility of either computing
    at band-level or at full stft level.

    Args:
        Zxx (np.ndarray): Complex STFT array of shape (n_freqs, T).
        freqs (np.ndarray): Array of frequency bin centers, shape (n_freqs,).
        eps (float): Small constant added for numerical stability in log scaling.

    Returns:
        B_complex: Complex bispectrum matrix of shape (n_bands, n_bands).
        B_db: Real bispectrum magnitude in dB, same shape as B_complex.
    """

    # Flexibility for either computing at band level or full stft
    if PreprocessingConfig.band_level_bispectrum:
        band_complex, band_centers = compute_band_average_stft_coeffs(
            Zxx=Zxx, freqs=freqs
        )
        coeffs, centers = band_complex, band_centers
    else:
        coeffs, centers = Zxx, freqs

    n_bins = coeffs.shape[0]
    logger.info(
        f"Computing {'band-level' if PreprocessingConfig.band_level_bispectrum else 'full'} bispectrum for {n_bins} bins."
    )

    B_complex = np.zeros((n_bins, n_bins), dtype=np.complex128)
    if freqs.size == 0 or Zxx.size == 0:
        return B_complex, 20.0 * np.log10(np.abs(B_complex) + eps)

    # Pairwise target freqs
    targets = centers[:, None] + centers[None, :]
    logger.debug("Computed pairwise target frequencies.")

    # Nearest index lookup (vectorized)
    pos = np.searchsorted(freqs, targets)
    left = np.clip(pos - 1, 0, len(freqs) - 1)
    right = np.clip(pos, 0, len(freqs) - 1)
    nearest = np.where(
        np.abs(freqs[left] - targets) <= np.abs(freqs[right] - targets), left, right
    )
    sum_idx = np.where(targets <= freqs[-1], nearest, -1).astype(int)
    logger.debug("Computed nearest frequency indices for all band pairs.")

    conj_Zxx = np.conjugate(Zxx)

    def _bispectrum_estimate_task(args: tuple[int, int, np.ndarray]) -> complex:
        # Bispectrum estimation through triple product
        i, j, Xi = args
        idx = sum_idx[i, j]
        if idx == -1:
            logger.debug(f"B[{i},{j}] skipped (target frequency out of range).")
            return 0j
        value = np.mean(Xi * coeffs[j] * conj_Zxx[idx])
        logger.debug(
            f"B[{i},{j}] computed using freq idx {idx} "
            f"(center={centers[i] + centers[j]:.2f} Hz, "
            f"{'band-level' if PreprocessingConfig.band_level_bispectrum else 'full'})."
        )
        return complex(value)

    bispec_tasks: list[tuple[int, int, np.ndarray]] = [
        (i, j, coeffs[i]) for i, j in np.ndindex(n_bins, n_bins)
    ]

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        results: list[complex] = list(pool.map(_bispectrum_estimate_task, bispec_tasks))

    B_complex = np.array(results).reshape(n_bins, n_bins)

    B_db = 20.0 * np.log10(np.abs(B_complex) + eps)
    logger.info("Finished computing bispectrum matrices.")
    return B_complex, B_db
