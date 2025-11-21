import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.config import DataLoaderConfig, Trainconfig
from src.logger import setup_logger
from src.preprocessing.data_transformation import normalize_to_imagenet

logger = setup_logger(name="data")


def assert_modalities():
    assert any(m in ["tf", "bis"] for m in Trainconfig.modalities), (
        "modalities must include 'tf', 'bis', or both"
    )


def epoch_key(filename: str) -> str:
    """
    Creates a unique key for matching TF and BIS files.
    Keeps everything except the trailing suffix like '_tf' or '_bis'.
    Example:
        'preictal_chb02_16+_002282_002312_tf.npz' → 'preictal_chb02_16+_002282_002312'
        'interictal_chb02_02_003075_003105_tf.npz' → 'interictal_chb02_02_003075_003105'
    """
    stem = Path(filename).stem  # drops .npz
    parts = stem.split("_")

    if parts[-1].isalpha():
        parts = parts[:-1]

    return "_".join(parts)


class PairedEEGDataset(Dataset):
    def __init__(self, patient_id: str, root_dir: str):
        patient_path = f"patient_{patient_id}"
        tf_dir = os.path.join(root_dir, patient_path, "time-frequency")
        bis_dir = os.path.join(root_dir, patient_path, "bispectrum")

        tf_files = [f for f in os.listdir(tf_dir) if f.endswith(".npz")]
        bispec_files = [f for f in os.listdir(bis_dir) if f.endswith(".npz")]

        tf_map = {epoch_key(f): os.path.join(tf_dir, f) for f in tf_files}
        bis_map = {epoch_key(f): os.path.join(bis_dir, f) for f in bispec_files}

        common_keys = sorted(set(tf_map.keys()) & set(bis_map.keys()))

        tf_list: list[torch.Tensor] = []
        bis_list: list[torch.Tensor] = []
        labels_list: list[int] = []
        seizure_id_list: list[int] = []

        interictal_file_names: list[str] = []

        for key in common_keys:
            tf_path = tf_map[key]
            bis_path = bis_map[key]

            # TODO: Change 'tensor' to 'image'
            tf_npz = np.load(tf_path)
            bis_npz = np.load(bis_path)

            tf_image = np.array(tf_npz["tensor"], dtype=np.float32)
            bis_image = np.array(bis_npz["tensor"], dtype=np.float32)

            # Normalize to imagenet specs
            tf_image = normalize_to_imagenet(tf_image)

            bis_image = bis_image / 255.0
            bis_image = (bis_image - bis_image.mean()) / (bis_image.std() + 1e-8)

            tf_tensor = torch.from_numpy(tf_image).permute(2, 0, 1)
            bis_tensor = torch.from_numpy(bis_image).permute(2, 0, 1)

            seizure_id = int(np.load(tf_path)["seizure_id"])

            label = 1 if "preictal" in os.path.basename(tf_path) else 0
            logger.info(
                f"Labelled {label} for {os.path.basename(tf_path)} and {os.path.basename(bis_path)} | seizure id: {seizure_id}"
            )

            tf_list.append(tf_tensor)
            bis_list.append(bis_tensor)
            labels_list.append(label)
            seizure_id_list.append(seizure_id)

        # Convert features, labels, and ids into tensors
        self.tf_features: torch.Tensor = torch.stack(tf_list)
        self.bis_features: torch.Tensor = torch.stack(bis_list)
        self.labels: torch.Tensor = torch.tensor(labels_list, dtype=torch.long)
        self.seizure_ids: torch.Tensor = torch.tensor(seizure_id_list, dtype=torch.long)

    def __len__(self):
        return len(self.tf_features)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tf_tensor: torch.Tensor = self.tf_features[idx]
        bis_tensor: torch.Tensor = self.bis_features[idx]
        label: torch.Tensor = self.labels[idx]
        seizure_id: torch.Tensor = self.seizure_ids[idx]

        return tf_tensor, bis_tensor, label, seizure_id


def get_loocv_fold(
    dataset: PairedEEGDataset,
    n_chunks: int,
    n_folds: int,
    fold_idx: int,
    undersample: bool = True,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    logger.info(f"Leaving out fold {fold_idx}")

    tf_features = dataset.tf_features
    bis_features = dataset.bis_features
    labels = dataset.labels

    # Split features and labels into N
    # The splitting works by splitting the entire tensor into chunks.
    # Each tensor would have equal chunks based on n_splits
    #
    # Another thing to keep in mind is that N seizures is the
    # amount of samples we have per fold so N is 5 then each fold would have around 5
    # samples each but some may have more less depends if its its divisible, but torch.split
    # wil handle those cases.

    preictal_mask = labels == 1
    interictal_mask = labels == 0

    preictal_idx = torch.where(preictal_mask)[0]
    interictal_idx = torch.where(interictal_mask)[0]

    preictal_count = len(preictal_idx)
    interictal_count = len(interictal_idx)

    if interictal_count > preictal_count:
        logger.info(
            "Undersampling interictals as balancing epochs did not work in preprocessing "
        )
        logger.info(f"Before undersampling interictals: {interictal_count}")

        perm = torch.randperm(interictal_count)[:preictal_count]
        interictal_idx = interictal_idx[perm]

        logger.info(f"After undersampling interictals: {len(interictal_idx)}")

    elif preictal_count > interictal_count:
        logger.info(
            "Undersampling preictals as balancing epochs did not work in preprocessing "
        )
        logger.info(f"Before undersampling preictals: {preictal_count}")

        perm = torch.randperm(preictal_count)[:interictal_count]
        preictal_idx = preictal_idx[perm]

        logger.info(f"After undersampling preictals: {len(preictal_idx)}")

    else:
        logger.info("Classes already balanced")

    logger.info(
        f"Preictal indices count: {len(preictal_idx)}, interictal indices count: {len(interictal_idx)}"
    )

    preictal_splits = torch.split(preictal_idx, n_chunks)
    interictal_splits = torch.split(interictal_idx, n_chunks)

    assert len(preictal_splits) == len(interictal_splits), "Classes must split evenly"
    assert 0 <= fold_idx < n_folds, "fold_idx exceeds available folds"

    logger.info(f"Preictal splits size: {len(preictal_splits)}")
    logger.info(f"Interictal splits size: {len(interictal_splits)}")

    preictal_train = torch.cat(
        [preictal_splits[index] for index in range(n_folds) if index != fold_idx]
    )
    interictal_train = torch.cat(
        [interictal_splits[index] for index in range(n_folds) if index != fold_idx]
    )

    # We then combine back both preictals and interictals into a single tensor
    train_mask = torch.cat([preictal_train, interictal_train])
    validation_mask = torch.cat(
        [preictal_splits[fold_idx], interictal_splits[fold_idx]]
    )

    # After combining the preictals and interictals, we get the time-frequency features,
    # bispectrum features, and the labels
    tf_train = tf_features[train_mask]
    bis_train = bis_features[train_mask]
    labels_train = labels[train_mask]

    tf_validation = tf_features[validation_mask]
    bis_validation = bis_features[validation_mask]
    labels_validation = labels[validation_mask]

    logger.info(
        f"""Normalization stats:
    TF Train: min={tf_train.min():.4f}, max={tf_train.max():.4f}, mean={tf_train.mean():.4f}, std={tf_train.std():.4f}
    BIS Train: min={bis_train.min():.4f}, max={bis_train.max():.4f}, mean={bis_train.mean():.4f}, std={bis_train.std():.4f}
    TF Validation:  min={tf_validation.min():.4f},  max={tf_validation.max():.4f},  mean={tf_validation.mean():.4f},  std={tf_validation.std():.4f}
    BIS Validation: min={bis_validation.min():.4f}, max={bis_validation.max():.4f}, mean={bis_validation.mean():.4f}, std={bis_validation.std():.4f}
    """
    )

    return (tf_train, bis_train, labels_train), (
        tf_validation,
        bis_validation,
        labels_validation,
    )


def get_data_loaders(
    train_set: TensorDataset,
    test_set: TensorDataset,
    batch_size: int = 32,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=DataLoaderConfig.num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=DataLoaderConfig.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader


def get_model_outputs(model, batch_tf: torch.Tensor, batch_bis: torch.Tensor):
    modalities = Trainconfig.modalities

    if "tf" in modalities and "bis" not in modalities:
        return model(batch_tf)
    elif "bis" in modalities and "tf" not in modalities:
        return model(batch_bis)
    else:
        return model(batch_tf, batch_bis)
