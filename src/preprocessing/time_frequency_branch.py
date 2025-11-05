"""
Time-Frequency Band Processing Module

This module provides functions to convert STFT power representations of EEG signals
into band-averaged time-frequency maps.

NOTE: STFT related that will be passed in the functions will be the precomputed STFTS.
"""

import numpy as np

from src.config import PreprocessingConfig
from src.datatypes import BandTimeStore, StftData
from src.logger import setup_logger

logger = setup_logger(name="time_frequency_branch")


def group_power_into_bands(
    power: np.ndarray,
    freqs: np.ndarray,
) -> np.ndarray:
    """
    Convert linear power (F x T) to band-time map (n_bands x T) by averaging power across frequency bins in each band.

    Args:
        power (np.ndarray): STFT power array.
        freqs (np.ndarray): STFT freqs array.

    Returns:
        np.ndarray: band_time matrix (n_bands, T)
    """

    logger.info("Starting power grouping")

    power = np.asarray(power)
    freqs = np.asarray(freqs).flatten()
    band_defs = PreprocessingConfig.band_defs

    if power is None or freqs is None or power.size == 0:
        return np.zeros((len(band_defs), 0), dtype=float)

    bands = list(band_defs.keys())
    band_time = np.zeros((len(bands), power.shape[1]), dtype=float)

    logger.info(f"Processing {len(bands)} frequency bands: {bands}")

    for idx, band in enumerate(bands):
        low = band_defs[band][0]
        high = band_defs[band][1]

        mask = np.logical_and(np.greater_equal(freqs, low), np.less_equal(freqs, high))
        if np.any(mask):
            band_time[idx, :] = np.mean(power[mask, :], axis=0)  # type: ignore
            logger.debug(
                f"Band '{band}' ({low}-{high} Hz): {np.sum(mask)} frequency bins averaged"
            )

    logger.info(f"Band grouping complete - output shape: {band_time.shape}")
    return band_time


def create_band_groupings_per_channel(
    stfts_by_channel: dict[str, list[StftData]],
) -> dict[str, list[BandTimeStore]]:
    """
    Convert stfts_by_channel (which contain linear power) into band-time maps per channel.

    Args:
        stfts_by_channel (dict[str, list[StftStore]]): Mapping from channel names to lists of StftStore objects.

    Returns:
        dict[str, list[BandTimeStore]]: Dict mapping channel names to lists of BandTimeStore objects
        containing band-averaged power per epoch.
            - {ch_name: [band_time_store_list]}
    """

    logger.info(f"Creating band groupings for {len(stfts_by_channel)} channels")

    band_maps: dict[str, list[BandTimeStore]] = {}
    total_epochs = 0

    for ch_name, epochs in stfts_by_channel.items():
        logger.debug(f"Processing channel '{ch_name}' with {len(epochs)} epochs")
        band_maps[ch_name] = []

        for epoch_idx, epoch in enumerate(epochs):
            logger.debug(
                f"Channel '{ch_name}', epoch {epoch_idx}: processing epoch {epoch.start}-{epoch.end}s, phase '{epoch.phase}'"
            )
            band_time = group_power_into_bands(power=epoch.power, freqs=epoch.freqs)

            band_store = BandTimeStore(
                start=epoch.start,
                end=epoch.end,
                phase=epoch.phase,
                band_time=band_time,
                times=epoch.times,
            )

            band_maps[ch_name].append(band_store)
            total_epochs += 1

    logger.info(
        f"Band grouping complete - processed {total_epochs} total epochs across {len(stfts_by_channel)} channels"
    )
    return band_maps
