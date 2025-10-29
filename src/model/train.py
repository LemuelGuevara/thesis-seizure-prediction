import math
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from torch.utils.data import TensorDataset
from tqdm import tqdm

from src.config import DataConfig, DataLoaderConfig, Trainconfig
from src.logger import setup_logger
from src.model.classification.concat_model import ConcatModel
from src.model.classification.multi_seizure_model import MultimodalSeizureModel
from src.model.classification.uni_seizure_model import UnimodalSeizureModel
from src.model.data import (
    PairedEEGDataset,
    get_data_loaders,
    get_loocv_fold,
    get_model_outputs,
)
from src.model.early_stopping import EarlyStopping
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


def get_model():
    modalities = Trainconfig.modalities

    if len(modalities) == 1:
        logger.info(f"Using unimodal model with modality: {modalities[0]}")
        model = UnimodalSeizureModel(use_cbam=Trainconfig.use_cbam)

    elif len(modalities) == 2:
        if Trainconfig.gated:
            logger.info("Using gated multimodal seizure model")
            model = MultimodalSeizureModel(use_cbam=Trainconfig.use_cbam)
        else:
            logger.info("Using concat multimodal seizure model")
            model = ConcatModel(use_cbam=Trainconfig.use_cbam)

    else:
        raise ValueError(
            f"Unsupported number of modalities: {len(modalities)}. "
            f"Expected 1 or 2 but got {modalities}."
        )

    return model.to(device)


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
        dataset = PairedEEGDataset(
            patient_id, root_dir=DataConfig.precomputed_data_path
        )

        unique_seizure_ids = torch.unique(dataset.seizure_ids)

        # Number of folds will be Total features / N seizures
        # So if our total features for lets say preictal is 200 and N is 5 then
        # 200 / 5 = 40 folds.
        #
        # NOTE: n_chnks and n_folds are different
        # n_folds = The number folds for the loocv
        # n_splits = The number of samples/chunks per fold. The number of seizures
        # will dictate this.

        n_chunks = len(unique_seizure_ids)

        preictal_count = int((dataset.labels == 1).sum())
        interictal_count = int((dataset.labels == 0).sum())

        preictal_folds = math.ceil(preictal_count / n_chunks)
        interictal_folds = math.ceil(interictal_count / n_chunks)

        n_folds = min(preictal_folds, interictal_folds)

        # Patient-level accumulators
        all_preds, all_labels = [], []

        patient_train_losses, patient_val_losses = [], []
        patient_train_accuracies, patient_val_accuracies = [], []

        for fold_idx in tqdm(range(n_folds), desc="Seizure folds"):
            train_losses, val_losses = [], []
            train_accuracies, val_accuracies = [], []

            # LOSO Fold features
            (tf_train, bis_train, labels_train), (tf_test, bis_test, labels_test) = (
                get_loocv_fold(dataset, n_chunks, n_folds, fold_idx)
            )

            train_set = TensorDataset(tf_train, bis_train, labels_train)
            test_set = TensorDataset(tf_test, bis_test, labels_test)

            # Data loaders
            train_loader, test_loader = get_data_loaders(
                train_set,
                test_set,
                batch_size=DataLoaderConfig.batch_size,
                pin_memory=False,
            )

            unique_train, counts_train = torch.unique(labels_train, return_counts=True)
            unique_test, counts_test = torch.unique(labels_test, return_counts=True)
            logger.info(
                f"[Patient {patient_id} | Fold {fold_idx}] "
                f"Train class dist: {dict(zip(unique_train.tolist(), counts_train.tolist()))} | "
                f"Test class dist: {dict(zip(unique_test.tolist(), counts_test.tolist()))}"
            )

            # Initialize model
            model = get_model()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=Trainconfig.lr)
            early_stopping = EarlyStopping(patience=5, delta=1e-4)

            for epoch in range(Trainconfig.num_epochs):
                model.train()
                train_loss_sum = 0.0
                train_acc_sum = 0.0

                for batch_tf, batch_bis, batch_labels in tqdm(
                    train_loader,
                    desc=f"Train Batches (Patient {patient_id}, Fold {fold_idx + 1}, Modalities: {Trainconfig.modalities})",
                    leave=False,
                ):
                    batch_tf, batch_bis, batch_labels = (
                        batch_tf.to(device),
                        batch_bis.to(device),
                        batch_labels.to(device),
                    )
                    optimizer.zero_grad()

                    outputs = get_model_outputs(model, batch_tf, batch_bis)
                    loss = criterion(outputs, batch_labels)

                    loss.backward()
                    optimizer.step()

                    train_loss_sum += loss.item()
                    train_acc_sum += compute_train_accuracy(outputs, batch_labels)

                average_train_loss = train_loss_sum / len(train_loader)
                average_train_accuracy = train_acc_sum / len(train_loader)

                val_loss_sum = 0.0
                val_correct = 0
                val_total = 0

                model.eval()
                with torch.no_grad():
                    for batch_tf, batch_bis, batch_labels in test_loader:
                        batch_tf, batch_bis, batch_labels = (
                            batch_tf.to(device),
                            batch_bis.to(device),
                            batch_labels.to(device),
                        )
                        outputs = get_model_outputs(model, batch_tf, batch_bis)
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

                early_stopping(average_validation_loss, model)
                if early_stopping.early_stop:
                    logger.info(
                        f"Early stopping triggered — best epoch: {epoch + 1 - early_stopping.counter}"
                    )
                    break

            if early_stopping.best_model_state is not None:
                model.load_state_dict(early_stopping.best_model_state)

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
                    outputs = get_model_outputs(model, batch_tf, batch_bis)
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
