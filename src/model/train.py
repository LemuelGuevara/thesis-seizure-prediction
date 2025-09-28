import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.amp import GradScaler, autocast
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import DataConfig, Trainconfig, config
from src.logger import setup_logger
from src.model.classification.multi_seizure_model import MultimodalSeizureModel
from src.model.data import create_data_loader, get_loocv_fold, get_paired_dataset
from src.model.early_stopping import EarlyStopping
from src.utils import export_to_csv, get_torch_device, set_seed

logger = setup_logger(name="train")
device = get_torch_device()
writer = SummaryWriter()


def main():
    set_seed()
    logger.info(f"Torch device: {device}")
    logger.info("Starting training for all patients")

    loocv_results: list[dict[str, float | str]] = []

    for patient in tqdm(DataConfig.patients_to_process, desc="Patients"):
        patient_id = f"{patient:02d}"

        checkpoint_path = os.path.join(
            os.path.dirname(__file__),
            "checkpoints",
            f"patient_{patient_id}.pt",
        )

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        # Get get paired dataset
        tf_features, bis_features, labels = get_paired_dataset(patient_id=patient_id)

        # # Normalize to float16 [0, 1]
        tf_features = tf_features.astype(np.float16) / 255.0
        bis_features = bis_features.astype(np.float16) / 255.0
        labels = labels.astype(np.int8)

        logger.info(
            f"Normalized features: {{tf: {tf_features.dtype}, {tf_features.min():.4f}-{tf_features.max():.4f}; "
            f"bis: {bis_features.dtype}, {bis_features.min():.4f}-{bis_features.max():.4f}}}"
        )

        all_preds, all_labels = [], []

        # Tracker for the best val loss for each patient
        patient_best_val_loss = float("inf")

        for sample_idx in tqdm(range(len(tf_features)), desc="Samples"):
            # Split train/test data through loocv
            (tf_train, bis_train, labels_train), (tf_test, bis_test, labels_test) = (
                get_loocv_fold(
                    tf_features=tf_features,
                    bis_features=bis_features,
                    labels=labels,
                    sample_idx=sample_idx,
                    undersample=Trainconfig.undersample,
                )
            )

            train_loader = create_data_loader(
                tensor_dataset=TensorDataset(tf_train, bis_train, labels_train)
            )
            test_loader = create_data_loader(
                tensor_dataset=TensorDataset(tf_test, bis_test, labels_test)
            )

            # Initialize model, criterion, optimizer, scaler
            model = MultimodalSeizureModel(use_cbam=Trainconfig.use_cbam).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=Trainconfig.lr)
            scaler = GradScaler(device.type)
            early_stopping = EarlyStopping()

            # Training
            for epoch in range(Trainconfig.num_epochs):
                model.train()
                epoch_loss = 0.0

                for batch_tf, batch_bis, batch_labels in tqdm(
                    train_loader, desc="Train Batches", leave=False
                ):
                    batch_tf = batch_tf.to(device, non_blocking=True)
                    batch_bis = batch_bis.to(device, non_blocking=True)
                    batch_labels = batch_labels.to(device, non_blocking=True)

                    optimizer.zero_grad()

                    # Mixed precision
                    with autocast(device.type):
                        outputs = model(batch_tf, batch_bis)
                        loss = criterion(outputs, batch_labels)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss += loss.item()

                epoch_loss /= len(train_loader)

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_tf, batch_bis, batch_labels in test_loader:
                        batch_tf = batch_tf.to(device, non_blocking=True)
                        batch_bis = batch_bis.to(device, non_blocking=True)
                        batch_labels = batch_labels.to(device, non_blocking=True)

                        outputs = model(batch_tf, batch_bis)
                        loss = criterion(outputs, batch_labels)
                        val_loss += loss.item()

                        preds = torch.argmax(outputs, dim=1).cpu().numpy()
                        all_preds.extend(preds)
                        all_labels.extend(batch_labels.cpu().numpy())

                    writer.add_scalar("Loss/train", epoch_loss, epoch)
                    writer.flush()
                val_loss /= len(test_loader)

                # Apply early stopping to stop overfitting the model
                early_stopping(val_loss, model)
                if early_stopping.best_score is not None:
                    fold_best_val = -float(early_stopping.best_score)
                    if fold_best_val < patient_best_val_loss:
                        patient_best_val = fold_best_val
                        torch.save(model.state_dict(), checkpoint_path)

                if early_stopping.early_stop:
                    break

                logger.info(
                    f"Patient {patient_id} Fold {sample_idx + 1} "
                    f"Epoch {epoch + 1}/{Trainconfig.num_epochs} "
                    f"- train_loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}"
                )
                writer.add_scalar(f"{patient_id}/Loss/train", epoch_loss, epoch)
                writer.add_scalar(f"{patient_id}/Loss/val", val_loss, epoch)
                writer.flush()

        acc = accuracy_score(all_labels, all_preds)
        rec = recall_score(all_labels, all_preds, average="binary")
        f1 = f1_score(all_labels, all_preds, average="binary")

        loocv_result_fieldnames = [
            "patient",
            "run_timestamp",
            "recall",
            "accuracy",
            "f1-score",
        ]

        timestamp = datetime.now().strftime("%b%d_%H-%M-%S")

        # Show results of loocv in table format
        table = PrettyTable()
        table.field_names = loocv_result_fieldnames
        table.title = f"Patient {patient_id} LOOCV Results"
        table.add_row([patient_id, timestamp, f"{acc:.4f}", f"{rec:.4f}", f"{f1:.4f}"])
        print(f"\n{table}")

        loocv_results.append(
            {
                "patient": patient_id,
                "run_timestamp": timestamp,
                "recall": round(rec, 4),
                "accuracy": round(acc, 4),
                "f1-score": round(f1, 4),
                "config": config,
            }
        )

        # Save all results of each patient in csv file
        all_patients_results_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "runs",
            "all_patients_results.csv",
        )

        export_to_csv(
            path=all_patients_results_path,
            fieldnames=loocv_result_fieldnames,
            data=loocv_results,
            mode="a",
            json_metadata=("config", config),
        )


if __name__ == "__main__":
    main()
