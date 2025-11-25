"""
Bispectrum Processing Module

This module provides functions for procesing STFT complex coefficients into bispectrum
estimations.

NOTE: Same precomputed STFT will be used in here as inputs for the functions.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.logger import setup_logger

logger = setup_logger(name="bispectrum_branch")


def bispectrum_estimation(
    Zxx: np.ndarray, freqs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Multithreaded bispectrum estimation.

    Args:
        Zxx (np.ndarray): STFT complex coefficients (C, F, T)
        freqs (np.ndarray): Frequency bins (F,)

    Returns:
        tuple: (bispectrum_mag, bispectrum_db)
            - bispectrum_mag: (C, F, F) in log scale for better visualization
            - bispectrum_db: (C, F, F) in decibels per channel
    """
    n_channels, n_bins, _ = Zxx.shape
    bispectrum = np.zeros((n_channels, n_bins, n_bins), dtype=np.complex128)

    Zxx_norm = Zxx / (np.max(np.abs(Zxx), axis=(1, 2), keepdims=True) + 1e-20)

    def compute_row(freq1: int):
        """Compute one row of the bispectrum (fixed freq1)."""
        row = np.zeros((n_channels, n_bins), dtype=np.complex128)
        for freq2 in range(n_bins - freq1):
            freq3 = freq1 + freq2
            if freq3 < n_bins:
                # Mean over time (axis=-1)
                row[:, freq2] = np.mean(
                    Zxx_norm[:, freq1, :]
                    * Zxx_norm[:, freq2, :]
                    * np.conj(Zxx_norm[:, freq3, :]),
                    axis=-1,
                )
        return freq1, row

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
        for freq1, row in executor.map(compute_row, range(n_bins)):
            bispectrum[:, freq1, :] = row

    bispectrum_mag = np.abs(bispectrum)  # (C, F, F)
    bispectrum_db = 20.0 * bispectrum_mag

    return bispectrum_mag, bispectrum_db
