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
    tf_tensor: Tensor,
    bis_tensor: Tensor,
    labels_tensor: Tensor,
    sample_idx: int,
    undersample: bool,
) -> tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]:
    """
    Creates split train and test data through leave-one-out cross-validation (LOOCV)

    Args:
        tf_features (np.ndarray): Numpy array of time-frequency features
        bis_features (np.ndarray): Numpy array of bispectrum features
        labels (np.ndarray): Numpy array of feature labels (0 = interictal, 1 = preictal)
        undersample: if True, applies random undersampling

    Returns:
        tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]:
            The train and test tensors
    """

    logger.info(
        f"Fold {sample_idx + 1} / {len(tf_tensor)} (leaving out sample {sample_idx})"
    )

    mask = torch.arange(len(labels_tensor)) != sample_idx

    tf_train = tf_tensor[mask]
    bis_train = bis_tensor[mask]
    labels_train = labels_tensor[mask]

    tf_test = tf_tensor[~mask]
    bis_test = bis_tensor[~mask]
    labels_test = labels_tensor[~mask]

    # (optional) undersampling still supported
    if undersample:
        unique, counts = torch.unique(labels_train, return_counts=True)
        minority = torch.argmin(counts)
        minority_count = counts[minority].item()

        indices_min = torch.where(labels_train == minority)[0]
        indices_maj = torch.where(labels_train != minority)[0]
        sampled_maj = indices_maj[torch.randperm(len(indices_maj))[:minority_count]]
        idx = torch.cat((indices_min, sampled_maj))
        tf_train, bis_train, labels_train = (
            tf_train[idx],
            bis_train[idx],
            labels_train[idx],
        )

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
        num_workers=DataLoader.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
