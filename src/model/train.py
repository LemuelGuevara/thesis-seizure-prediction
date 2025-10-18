import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from torch.amp import GradScaler, autocast
from torch.utils.data import TensorDataset
from tqdm import tqdm

from src.config import DataConfig, Trainconfig
from src.logger import setup_logger
from src.model.classification.concat_model import ConcatModel
from src.model.classification.multi_seizure_model import MultimodalSeizureModel
from src.model.data import (
    create_data_loader,
    get_loso_fold,
    get_paired_dataset,
)
from src.model.metrics_utils import (
    PatientTrainAccuracy,
    PatientTrainLoss,
    TrainingResults,
    compute_train_accuracy,
    export_training_results,
    plot_confusion_matrix,
    plot_training_accuracy,
    plot_training_loss,
    show_training_results,
)
from src.utils import get_torch_device, set_seed

logger = setup_logger(name="train")
device = get_torch_device()


def get_model(sample_size: int):
    if Trainconfig.gated:
        model = MultimodalSeizureModel(use_cbam=Trainconfig.use_cbam)
    else:
        model = ConcatModel(use_cbam=Trainconfig.use_cbam)

    if torch.cuda.device_count() > 1 and sample_size > 1000:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)
    return model


def main():
    set_seed()
    logger.info(f"Torch device: {device}")
    logger.info("Starting training for all patients")

    training_results_list: list[TrainingResults] = []

    for patient in tqdm(DataConfig.patients_to_process, desc="Patients"):
        patient_id = f"{patient:02d}"
        timestamp = datetime.now().strftime("%b%d_%H-%M-%S")

        saved_models_path = os.path.join(
            os.path.dirname(__file__), "saved_models", f"patient_{patient_id}.pt"
        )
        os.makedirs(os.path.dirname(saved_models_path), exist_ok=True)

        # Load paired dataset
        tf_features, bis_features, labels, seizure_ids = get_paired_dataset(patient_id)

        tf_features = torch.tensor(tf_features, dtype=torch.float32).permute(0, 3, 1, 2)
        bis_features = torch.tensor(bis_features, dtype=torch.float32).permute(
            0, 3, 1, 2
        )
        labels = torch.tensor(labels, dtype=torch.long)
        seizure_ids = torch.tensor(seizure_ids, dtype=torch.long)

        pre_ids = torch.unique(seizure_ids[labels == 1])
        N_parts = len(pre_ids)

        # Patient-level accumulators
        all_preds, all_labels = [], []

        patient_train_losses, patient_val_losses = [], []
        patient_train_accuracies, patient_val_accuracies = [], []

        for fold_idx in tqdm(range(N_parts), desc="Seizure folds"):
            train_losses, val_losses = [], []
            train_accuracies, val_accuracies = [], []

            (
                (train_tf, train_bis, train_labels),
                (test_tf, test_bis, test_labels),
            ) = get_loso_fold(
                tf_features, bis_features, labels, seizure_ids, fold_idx, N_parts
            )

            train_loader = create_data_loader(
                TensorDataset(train_tf, train_bis, train_labels)
            )
            test_loader = create_data_loader(
                TensorDataset(test_tf, test_bis, test_labels)
            )

            unique_train, counts_train = torch.unique(train_labels, return_counts=True)
            unique_test, counts_test = torch.unique(test_labels, return_counts=True)
            logger.info(
                f"[Patient {patient_id} | Fold {fold_idx}] "
                f"Train class dist: {dict(zip(unique_train.tolist(), counts_train.tolist()))} | "
                f"Test class dist: {dict(zip(unique_test.tolist(), counts_test.tolist()))}"
            )

            # Initialize model
            model = get_model(len(train_tf))

            if Trainconfig.class_weighting:
                unique_labels, counts = torch.unique(train_labels, return_counts=True)
                class_frequencies = counts / len(train_labels)
                class_weights = 1.0 / class_frequencies
                class_weights = (1.0 / class_frequencies).float().to(device)

                logger.info(f"Using class weights: {class_weights}")

                criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
                criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=Trainconfig.lr)
            scaler = GradScaler(device.type)

            for epoch in range(Trainconfig.num_epochs):
                model.train()
                train_loss_sum = 0.0
                train_acc_sum = 0.0

                for batch_tf, batch_bis, batch_labels in tqdm(
                    train_loader,
                    desc=f"Train Batches (Patient {patient_id}, Fold {fold_idx + 1})",
                    leave=False,
                ):
                    batch_tf, batch_bis, batch_labels = (
                        batch_tf.to(device),
                        batch_bis.to(device),
                        batch_labels.to(device),
                    )
                    optimizer.zero_grad()

                    with autocast(device.type):
                        outputs = model(batch_tf, batch_bis)
                        loss = criterion(outputs, batch_labels)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    train_loss_sum += loss.item()
                    train_acc_sum += compute_train_accuracy(outputs, batch_labels)

                average_train_loss = train_loss_sum / len(train_loader)
                average_train_accuracy = train_acc_sum / len(train_loader)

                model.eval()
                val_loss_sum = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch_tf, batch_bis, batch_labels in test_loader:
                        batch_tf, batch_bis, batch_labels = (
                            batch_tf.to(device),
                            batch_bis.to(device),
                            batch_labels.to(device),
                        )
                        outputs = model(batch_tf, batch_bis)
                        loss = criterion(outputs, batch_labels)
                        preds = torch.argmax(outputs, dim=1)

                        val_loss_sum += loss.item()
                        val_correct += (preds == batch_labels).sum().item()
                        val_total += batch_labels.size(0)

                average_validation_loss = val_loss_sum / len(test_loader)
                average_validation_accuracy = val_correct / max(1, val_total)

                train_losses.append(average_train_loss)
                train_accuracies.append(average_train_accuracy)
                val_losses.append(average_validation_loss)
                val_accuracies.append(average_validation_accuracy)

                logger.info(
                    f"[Patient {patient_id} | Fold {fold_idx + 1}] "
                    f"Epoch {epoch + 1}/{Trainconfig.num_epochs} - "
                    f"Train Loss: {average_train_loss:.4f} | "
                    f"Train Acc: {average_train_accuracy * 100:.2f}% | "
                    f"Val Loss: {average_validation_loss:.4f} | "
                    f"Val Acc: {average_validation_accuracy * 100:.2f}%"
                )

            # ONE final evaluation pass for fold-level preds/labels
            model.eval()
            fold_preds, fold_gt = [], []
            with torch.no_grad():
                for batch_tf, batch_bis, batch_labels in test_loader:
                    batch_tf, batch_bis, batch_labels = (
                        batch_tf.to(device),
                        batch_bis.to(device),
                        batch_labels.to(device),
                    )
                    outputs = model(batch_tf, batch_bis)
                    preds = torch.argmax(outputs, dim=1)
                    fold_preds.extend(preds.cpu().numpy())
                    fold_gt.extend(batch_labels.cpu().numpy())

            all_preds.extend(fold_preds)
            all_labels.extend(fold_gt)

            patient_train_losses = train_losses
            patient_val_losses = val_losses
            patient_train_accuracies = train_accuracies
            patient_val_accuracies = val_accuracies

        accuracy = round(accuracy_score(all_labels, all_preds), 4)
        recall = round(recall_score(all_labels, all_preds), 4)
        f1 = round(f1_score(all_labels, all_preds), 4)
        cf_matrix = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        TN, FP, FN, TP = cf_matrix.ravel()

        # Export per-patient summary row
        training_result_fieldnames = [
            "patient_id",
            "setup_name",
            "run_timestamp",
            "true_positives",
            "false_positives",
            "true_negatives",
            "false_negatives",
            "accuracy",
            "recall",
            "f1_score",
        ]

        training_results = TrainingResults(
            patient_id,
            Trainconfig.setup_name,
            timestamp,
            TP,
            FP,
            TN,
            FN,
            accuracy,
            recall,
            f1,
        )

        # Visualizations
        plot_training_loss(
            PatientTrainLoss(
                patient_id,
                patient_train_losses,
                patient_val_losses,
                all_preds,
                all_labels,
                timestamp,
            )
        )
        plot_training_accuracy(
            PatientTrainAccuracy(
                patient_id,
                patient_train_accuracies,
                patient_val_accuracies,
                timestamp,
            )
        )
        plot_confusion_matrix(patient_id, cf_matrix, timestamp)

        training_results_list.append(training_results)
        export_training_results(training_result_fieldnames, [training_results])

    show_training_results(
        [
            "patient_id",
            "setup_name",
            "run_timestamp",
            "true_positives",
            "false_positives",
            "true_negatives",
            "false_negatives",
            "accuracy",
            "recall",
            "f1_score",
        ],
        training_results_list,
    )


if __name__ == "__main__":
    main()
