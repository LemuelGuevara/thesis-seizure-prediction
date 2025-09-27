"""
Datatypes that will be used to other modules.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


@dataclass
class EpochInterval:
    phase: Literal["preictal", "interictal", "ictal"]
    start: int
    end: int
    duration: Optional[int] = None
    windows_created: Optional[int] = None


@dataclass
class CombinedIntervals:
    preictal_intervals: list[EpochInterval]
    interictal_intervals: list[EpochInterval]
    ictal_intervals: list[EpochInterval]


@dataclass(kw_only=True)
class StftStore(EpochInterval):
    stft_db: np.ndarray
    power: np.ndarray
    Zxx: np.ndarray
    freqs: np.ndarray
    times: np.ndarray


@dataclass(kw_only=True)
class BandTimeStore(EpochInterval):
    band_time: np.ndarray  # shape: (n_bands, n_times)
    times: np.ndarray


@dataclass
class PrecomputedStftSummary:
    patient_index: int
    number_of_seizures: int
    preictal_intervals: int
    interictal_intervals: int
