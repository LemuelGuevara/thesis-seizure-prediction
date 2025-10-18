import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.config import DataConfig, DataLoaderConfig
from src.logger import setup_logger
from src.preprocessing.data_transformation import normalize_to_imagenet

logger = setup_logger(name="data")


def get_filename_suffix(filename: str) -> str:
    return filename.replace(".npz", "").split("_")[-1]


def get_paired_dataset(patient_id: str):
    """
    Loads paired (tf_tensor, bispec_tensor, label, seizure_id) for a specific patient.
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

    tf_features, bis_features, labels, seizure_ids = [], [], [], []

    for key in common_keys:
        tf_path = tf_map[key]
        bis_path = bis_map[key]

        tf_tensor = np.array(np.load(tf_path)["tensor"], dtype=np.float32)
        bis_tensor = np.array(np.load(bis_path)["tensor"], dtype=np.float32)

        tf_tensor = normalize_to_imagenet(tf_tensor)
        bis_tensor = normalize_to_imagenet(bis_tensor)

        label = 1 if "preictal" in os.path.basename(tf_path) else 0
        logger.info(
            f"Labelled {label} for {os.path.basename(tf_path)} and {os.path.basename(bis_path)}"
        )
        seizure_id = int(
            np.load(tf_path)["seizure_id"]
        )  # assuming seizure_id saved in npz

        tf_features.append(tf_tensor)
        bis_features.append(bis_tensor)
        labels.append(label)
        seizure_ids.append(seizure_id)

    return (
        np.array(tf_features),
        np.array(bis_features),
        np.array(labels),
        np.array(seizure_ids),
    )


def get_loso_fold(
    tf_features: torch.Tensor,
    bis_features: torch.Tensor,
    labels: torch.Tensor,
    seizure_ids: torch.Tensor,
    fold_idx: int,
    n_parts: int,
):
    # Separate preictal vs interictal
    preictal_mask = labels == 1
    interictal_mask = labels == 0

    # Gathering of features, ids, and labels for both preictal and interictal
    preictal_tf = tf_features[preictal_mask]
    preictal_bis = bis_features[preictal_mask]
    preictal_ids = seizure_ids[preictal_mask]
    preictal_labels = labels[preictal_mask]

    interictal_tf = tf_features[interictal_mask]
    interictal_bis = bis_features[interictal_mask]
    interictal_labels = labels[interictal_mask]

    # LOSO preictal split
    unique_preictal_ids = torch.unique(preictal_ids)
    logger.info(f"Unique preictal seizure IDs: {unique_preictal_ids.tolist()}")

    unique_preictal_ids = torch.unique(preictal_ids)
    test_id = unique_preictal_ids[fold_idx]
    preictal_test_mask = preictal_ids == test_id
    preictal_train_mask = ~preictal_test_mask

    preictal_tf_train = preictal_tf[preictal_train_mask]
    preictal_bis_train = preictal_bis[preictal_train_mask]
    preictal_labels_train = preictal_labels[preictal_train_mask]

    preictal_tf_test = preictal_tf[preictal_test_mask]
    preictal_bis_test = preictal_bis[preictal_test_mask]
    preictal_labels_test = preictal_labels[preictal_test_mask]

    # LOSO interictal split
    # Shuffle first
    g = torch.Generator().manual_seed(42 + fold_idx)
    perm = torch.randperm(len(interictal_tf), generator=g)

    interictal_tf = interictal_tf[perm]
    interictal_bis = interictal_bis[perm]
    interictal_labels = interictal_labels[perm]

    interictal_tf_splits = torch.tensor_split(interictal_tf, n_parts)
    interictal_bis_splits = torch.tensor_split(interictal_bis, n_parts)
    interictal_labels_splits = torch.tensor_split(interictal_labels, n_parts)

    interictal_tf_test = interictal_tf_splits[fold_idx]
    interictal_bis_test = interictal_bis_splits[fold_idx]
    interictal_labels_test = interictal_labels_splits[fold_idx]

    interictal_tf_train = torch.cat(
        [s for i, s in enumerate(interictal_tf_splits) if i != fold_idx]
    )
    interictal_bis_train = torch.cat(
        [s for i, s in enumerate(interictal_bis_splits) if i != fold_idx]
    )
    interictal_labels_train = torch.cat(
        [s for i, s in enumerate(interictal_labels_splits) if i != fold_idx]
    )

    # Concatenate features now
    logger.info(f"Preictal tf train shape: {preictal_tf_train.shape}")
    logger.info(f"Interictal tf train shape: {interictal_tf_train.shape}")

    tf_train = torch.cat([preictal_tf_train, interictal_tf_train], dim=0)
    bis_train = torch.cat([preictal_bis_train, interictal_bis_train], dim=0)
    labels_train = torch.cat([preictal_labels_train, interictal_labels_train], dim=0)

    tf_test = torch.cat([preictal_tf_test, interictal_tf_test], dim=0)
    bis_test = torch.cat([preictal_bis_test, interictal_bis_test], dim=0)
    labels_test = torch.cat([preictal_labels_test, interictal_labels_test], dim=0)

    return (tf_train, bis_train, labels_train), (tf_test, bis_test, labels_test)


def create_data_loader(
    tensor_dataset: TensorDataset, use_sampler: bool = False
) -> DataLoader:
    """
    Creates a data loader for a dataset or tensor, optionally using class balancing.

    Args:
        tensor_dataset: TensorDataset containing (tf, bis, labels)
        use_sampler: whether to balance classes with WeightedRandomSampler

    Returns:
        DataLoader
    """
    dataset_size = len(tensor_dataset)
    batch_size = 32 if dataset_size < 1000 else 256
    sampler = None

    if use_sampler:
        # Assume labels are the last tensor in TensorDataset
        labels = tensor_dataset.tensors[-1]
        class_counts = torch.bincount(labels)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )

    logger.info(f"TensorDataset size: {dataset_size}, using batch size: {batch_size}")
    if sampler:
        logger.info("Using WeightedRandomSampler for balanced classes")

    return DataLoader(
        dataset=tensor_dataset,
        batch_size=32,
        shuffle=not use_sampler,
        sampler=sampler,
        num_workers=DataLoaderConfig.num_workers,
        pin_memory=True,
    )
