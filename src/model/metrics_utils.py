import os
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from prettytable import PrettyTable

from src.config import DataConfig, Trainconfig, config
from src.logger import setup_logger
from src.utils import export_to_csv


@dataclass
class PatientTrainLoss:
    patient_id: str
    train_losses: list[float]
    val_losses: list[float]
    all_preds: list[int]
    all_labels: list[int]
    timestamp: str


@dataclass
class TrainingResults:
    patient_id: str
    run_timestamp: str
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    accuracy: float
    recall: float
    f1_score: float


logger = setup_logger(name="metrics_utils")

# Define runs directory
training_runs_dir = os.path.join(DataConfig.runs_dir, "training")
os.makedirs(training_runs_dir, exist_ok=True)

setup_dir = os.path.join(training_runs_dir, Trainconfig.setup_name)
os.makedirs(setup_dir, exist_ok=True)


def show_training_results(
    field_names: list[str], training_results: TrainingResults
) -> None:
    table = PrettyTable()
    table.field_names = field_names
    table.title = f"Patient {training_results.patient_id} Training Results"
    table.add_row(
        [
            training_results.patient_id,
            Trainconfig.setup_name,
            training_results.run_timestamp,
            training_results.true_positives,
            training_results.false_positives,
            training_results.true_negatives,
            training_results.false_negatives,
            training_results.accuracy,
            training_results.recall,
            training_results.f1_score,
        ]
    )
    print(f"\n{table}")


def plot_training_loss(patient_train_loss: PatientTrainLoss) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(patient_train_loss.train_losses) + 1),
        patient_train_loss.train_losses,
        label="Train Loss",
    )
    plt.plot(
        range(1, len(patient_train_loss.val_losses) + 1),
        patient_train_loss.val_losses,
        label="Validation Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"Loss Curve - Patient {patient_train_loss.patient_id} - {Trainconfig.setup_name}"
    )
    plt.legend()
    plt.grid(False)
    plt.xlim(0, 30)

    patient_dir = os.path.join(setup_dir, f"patient_{patient_train_loss.patient_id}")
    os.makedirs(patient_dir, exist_ok=True)

    # Save in runs/training with patient_id_timestamp.png
    filename = f"patient_{patient_train_loss.patient_id}_{patient_train_loss.timestamp}_loss_graph.png"
    train_loss_path = os.path.join(patient_dir, filename)

    plt.savefig(train_loss_path)
    plt.close()
    logger.info(f"Saved loss plot to {train_loss_path}")


def plot_confusion_matrix(
    patient_id: str, cf_matrix: np.ndarray, timestamp: str
) -> None:
    classnames = ["Interictal", "Preictal"]

    # Normalize by row
    cf_matrix_normalized = (
        cf_matrix.astype("float") / cf_matrix.sum(axis=1)[:, np.newaxis]
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cf_matrix_normalized,
        annot=True,
        fmt=".2f",
        xticklabels=classnames,
        yticklabels=classnames,
        cmap="Blues",
        cbar=True,
    )

    plt.title(f"Confusion Matrix - Patient {patient_id} - {Trainconfig.setup_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    patient_dir = os.path.join(setup_dir, f"patient_{patient_id}")
    os.makedirs(patient_dir, exist_ok=True)

    filename = f"patient_{patient_id}_{timestamp}_cf_matrix.png"
    cf_matrix_path = os.path.join(patient_dir, filename)

    plt.savefig(cf_matrix_path)
    plt.close()
    logger.info(f"Saved confusion matrix to {cf_matrix_path}")


def export_training_results(
    field_names: list[str], training_results: list[TrainingResults]
):
    # Save CSV in runs/training
    training_results_path = os.path.join(
        training_runs_dir, f"{Trainconfig.setup_name}_training_results.csv"
    )

    training_results_dicts = [asdict(tr) for tr in training_results]

    export_to_csv(
        path=training_results_path,
        fieldnames=field_names,
        data=training_results_dicts,
        mode="a",
        json_metadata=("config", config),
    )
    logger.info(f"Saved training results CSV to {training_results_path}")
