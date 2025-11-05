"""
Datatypes that will be used to other modules.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


@dataclass
class IntervalMeta:
    phase: Literal["preictal", "interictal", "ictal"]
    start: int
    end: int
    duration: Optional[int] = None
    windows_created: Optional[int] = None
    file_name: Optional[str] = None
    seizure_id: Optional[int] = None


@dataclass(kw_only=True)
class StftData(IntervalMeta):
    stft_db: np.ndarray
    power: np.ndarray
    Zxx: np.ndarray
    freqs: np.ndarray
    times: np.ndarray
    mag: np.ndarray


@dataclass(kw_only=True)
class BandTimeStore(IntervalMeta):
    band_time: np.ndarray  # shape: (n_bands, n_times)
    times: np.ndarray


@dataclass
class PrecomputedStftSummary:
    patient_index: int
    number_of_seizures: int
    preictal_intervals: int
    interictal_intervals: int


@dataclass
class RecordingFileInfo:
    file_name: str
    recording: IntervalMeta
    duration: int
