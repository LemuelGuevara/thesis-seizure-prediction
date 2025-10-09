import os
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from prettytable import PrettyTable

from src.config import DataConfig, Trainconfig
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
class PatientTrainAccuracy:
    patient_id: str
    train_accuracies: list[float]
    val_accuracies: list[float]
    timestamp: str


@dataclass
class TrainingResults:
    patient_id: str
    setup_name: str
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
    field_names: list[str], training_results: list["TrainingResults"]
) -> None:
    table = PrettyTable()
    table.field_names = field_names
    table.title = "Patient Training Results"

    for result in training_results:
        table.add_row(list(asdict(result).values()))

    print(f"\n{table}")


def plot_training_loss(patient_train_loss: PatientTrainLoss) -> None:
    def smooth_curve(values: list[float], window_size: int = 3):
        if len(values) < window_size:
            return values
        return np.convolve(values, np.ones(window_size) / window_size, mode="valid")

    plt.figure(figsize=(8, 5))

    train_losses_smooth = smooth_curve(patient_train_loss.train_losses, window_size=5)
    val_losses_smooth = smooth_curve(patient_train_loss.val_losses, window_size=5)

    plt.plot(
        range(1, len(train_losses_smooth) + 1),
        train_losses_smooth,
        label="Train Loss (smoothed)",
    )
    plt.plot(
        range(1, len(val_losses_smooth) + 1),
        val_losses_smooth,
        label="Validation Loss (smoothed)",
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


def plot_training_accuracy(patient_training_acc: PatientTrainAccuracy) -> None:
    plt.figure(figsize=(8, 5))

    # Plot raw training and validation accuracy
    plt.plot(
        range(1, len(patient_training_acc.train_accuracies) + 1),
        patient_training_acc.train_accuracies,
        label="Train Accuracy",
        color="tab:blue",
        linewidth=2,
    )
    plt.plot(
        range(1, len(patient_training_acc.val_accuracies) + 1),
        patient_training_acc.val_accuracies,
        label="Validation Accuracy",
        color="tab:orange",
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        f"Accuracy Curve - Patient {patient_training_acc.patient_id} - {Trainconfig.setup_name}"
    )
    plt.legend()
    plt.grid(False)
    plt.xlim(0, max(len(patient_training_acc.train_accuracies), 30))
    plt.ylim(0, 1)

    patient_dir = os.path.join(setup_dir, f"patient_{patient_training_acc.patient_id}")
    os.makedirs(patient_dir, exist_ok=True)

    filename = f"patient_{patient_training_acc.patient_id}_{patient_training_acc.timestamp}_accuracy_graph.png"
    acc_plot_path = os.path.join(patient_dir, filename)

    plt.savefig(acc_plot_path)
    plt.close()
    logger.info(f"Saved accuracy plot to {acc_plot_path}")


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
    )
    logger.info(f"Saved training results CSV to {training_results_path}")
