import os

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from src.config import DataConfig, DataLoaderConfig
from src.logger import setup_logger

logger = setup_logger(name="data")


def get_filename_suffix(filename) -> str:
    return "_".join(filename.replace(".npz", "").split("_")[-2:])


def get_paired_dataset(patient_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads paired (tf_tensor, bispec_tensor, label) for a specific patient.

    Args:
        patient_id: e.g., "patient_01"

    Returns:
        X_tf: array of time-frequency tensors
        X_bis: array of bispectrum tensors
        y: array of labels (0 = interictal, 1 = preictal)
    """

    patient_path = f"patient_{patient_id}"
    tf_dir = os.path.join(
        DataConfig.precomputed_data_path, patient_path, "time-frequency"
    )
    bis_dir = os.path.join(DataConfig.precomputed_data_path, patient_path, "bispectrum")

    tf_files = [f for f in os.listdir(tf_dir) if f.endswith(".npz")]
    bispec_files = [f for f in os.listdir(bis_dir) if f.endswith(".npz")]

    tf_map = {get_filename_suffix(f): os.path.join(tf_dir, f) for f in tf_files}
    bis_map = {get_filename_suffix(f): os.path.join(bis_dir, f) for f in bispec_files}

    common_keys = sorted(set(tf_map.keys()) & set(bis_map.keys()))

    tf_features, bis_features, labels = [], [], []

    for key in common_keys:
        tf_path = tf_map[key]
        bis_path = bis_map[key]

        tf_tensor = np.load(tf_path)["tensor"]
        bis_tensor = np.load(bis_path)["tensor"]

        label = 1 if "preictal" in os.path.basename(tf_path) else 0
        # Log for Verification (SORRY IF MADAMI LMAO)
        logger.info(
            f"Labelled {label} for {os.path.basename(tf_path)} and {os.path.basename(bis_path)}"
        )
        tf_features.append(tf_tensor)
        bis_features.append(bis_tensor)
        labels.append(label)

    return np.array(tf_features), np.array(bis_features), np.array(labels)


def get_loocv_fold(
    tf_features: np.ndarray,
    bis_features: np.ndarray,
    labels: np.ndarray,
    sample_idx: int,
) -> tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]:
    """
    Creates split train and test data through leave-one-out cross-validation (LOOCV)

    Args:
        tf_features (np.ndarray): Numpy array of time-frequency features
        bis_features (np.ndarray): Numpy array of bispectrum features
        labels (np.ndarray): Numpy array of feature labels (0 = interictal, 1 = preictal)

    Returns:
        tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]:
            The train and test tensors
    """

    logger.info(
        f"Fold {sample_idx + 1} / {len(tf_features)} (leaving out sample {sample_idx})"
    )

    tf_train = torch.tensor(
        np.delete(tf_features, sample_idx, axis=0), dtype=torch.float32
    ).permute(0, 3, 1, 2)
    bis_train = torch.tensor(
        np.delete(bis_features, sample_idx, axis=0), dtype=torch.float32
    ).permute(0, 3, 1, 2)
    labels_train = torch.tensor(np.delete(labels, sample_idx, axis=0), dtype=torch.long)

    tf_test = (
        torch.tensor(tf_features[sample_idx], dtype=torch.float32)
        .unsqueeze(0)
        .permute(0, 3, 1, 2)
    )
    bis_test = (
        torch.tensor(bis_features[sample_idx], dtype=torch.float32)
        .unsqueeze(0)
        .permute(0, 3, 1, 2)
    )
    labels_test = torch.tensor([labels[sample_idx]], dtype=torch.long)

    return (tf_train, bis_train, labels_train), (tf_test, bis_test, labels_test)


def create_data_loader(tensor_dataset: TensorDataset) -> DataLoader:
    """
    Creates a data loader for a dataset or tensor

    Args:
        dataset (Union[Dataset, Sequence[Tensor]]): Input to be used, can be either torch dataset
        or tensor

    Returns:
        DataLoader: The created dataloader
    """

    return DataLoader(
        dataset=tensor_dataset,
        batch_size=DataLoaderConfig.batch_size,
        shuffle=DataLoaderConfig.shuffle,
        num_workers=DataLoaderConfig.num_workers,
        pin_memory=True,
    )
