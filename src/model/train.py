import os
from datetime import datetime

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
from src.model.classification.concat_model import ConcatModel
from src.model.classification.multi_seizure_model import MultimodalSeizureModel
from src.model.data import create_data_loader, get_loocv_fold, get_paired_dataset
from src.model.early_stopping import EarlyStopping
from src.utils import export_to_csv, get_torch_device, set_seed

logger = setup_logger(name="train")
device = get_torch_device()


def main():
    set_seed()
    logger.info(f"Torch device: {device}")
    logger.info("Starting training for all patients")

    loocv_results: list[dict[str, float | str]] = []

    for patient in tqdm(DataConfig.patients_to_process, desc="Patients"):
        patient_id = f"{patient:02d}"
        timestamp = datetime.now().strftime("%b%d_%H-%M-%S")

        log_dir = os.path.join(
            os.path.dirname(__file__), "runs", f"patient_{patient_id}", timestamp
        )
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

        checkpoint_path = os.path.join(
            os.path.dirname(__file__),
            "checkpoints",
            f"patient_{patient_id}.pt",
        )

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        # Get get paired dataset
        tf_features, bis_features, labels = get_paired_dataset(patient_id=patient_id)
        tf_tensor = torch.tensor(tf_features, dtype=torch.float32).permute(0, 3, 1, 2)
        bis_tensor = torch.tensor(bis_features, dtype=torch.float32).permute(0, 3, 1, 2)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

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
                    tf_tensor,
                    bis_tensor,
                    labels_tensor,
                    sample_idx=sample_idx,
                    undersample=Trainconfig.undersample
                    and not Trainconfig.class_weighting,
                )
            )

            train_loader = create_data_loader(
                tensor_dataset=TensorDataset(tf_train, bis_train, labels_train)
            )
            batch_tf, batch_bis, batch_labels = next(iter(train_loader))
            logger.info(
                f"Train batch types: tf={batch_tf.dtype}, bis={batch_bis.dtype}, labels={batch_labels.dtype}"
            )
            logger.info(
                f"Train batch shapes: tf={batch_tf.shape}, bis={batch_bis.shape}, labels={batch_labels.shape}"
            )

            test_loader = create_data_loader(
                tensor_dataset=TensorDataset(tf_test, bis_test, labels_test)
            )

            # Initialize model, criterion, optimizer, scaler
            if Trainconfig.gated:
                model = MultimodalSeizureModel(use_cbam=Trainconfig.use_cbam).to(device)
            else:
                model = ConcatModel(use_cbam=Trainconfig.use_cbam).to(device)

            if Trainconfig.class_weighting:
                unique_labels, counts = torch.unique(labels_train, return_counts=True)
                class_frequencies = counts / len(labels_train)
                class_weights = 1.0 / class_frequencies
                class_weights = (1.0 / class_frequencies).float().to(device)

                logger.info(f"Using class weights: {class_weights}")

                criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
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

                # Compute average training loss
                avg_train_loss = epoch_loss / len(train_loader)

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

                avg_val_loss = val_loss / len(test_loader)

                early_stopping(avg_val_loss, model)
                if early_stopping.best_score is not None:
                    fold_best_val = -float(early_stopping.best_score)
                    if fold_best_val < patient_best_val_loss:
                        patient_best_val = fold_best_val
                        torch.save(model.state_dict(), checkpoint_path)

                if early_stopping.early_stop:
                    break

                logger.info(
                    f"[Patient {patient_id} | Fold {sample_idx + 1}] "
                    f"Epoch {epoch + 1}/{Trainconfig.num_epochs} "
                    f"- avg_train_loss: {avg_train_loss:.4f} | avg_val_loss: {avg_val_loss:.4f}"
                )

                writer.add_scalar(
                    f"patient_{patient_id}/avg_train_loss", avg_train_loss, epoch
                )
                writer.add_scalar(
                    f"patient_{patient_id}/avg_val_loss", avg_val_loss, epoch
                )
                writer.flush()

        acc = accuracy_score(all_labels, all_preds)
        rec = recall_score(all_labels, all_preds, average="binary")
        f1 = f1_score(all_labels, all_preds, average="binary")

        loocv_result_fieldnames = [
            "patient",
            "setup_name",
            "run_timestamp",
            "recall",
            "accuracy",
            "f1-score",
        ]

        # Show results of loocv in table format
        table = PrettyTable()
        table.field_names = loocv_result_fieldnames
        table.title = f"Patient {patient_id} LOOCV Results"
        table.add_row(
            [
                patient_id,
                Trainconfig.setup_name,
                timestamp,
                f"{acc:.4f}",
                f"{rec:.4f}",
                f"{f1:.4f}",
            ]
        )
        print("\nREMINDER: BALIKTAD ANG RECALL AT ACCURACY COLUMN!!!")
        print(f"\n{table}")

        loocv_results.append(
            {
                "patient": patient_id,
                "setup_name": Trainconfig.setup_name,
                "run_timestamp": timestamp,
                "recall": round(rec, 4),
                "accuracy": round(acc, 4),
                "f1-score": round(f1, 4),
                "config": config,
            }
        )

        # Save all results of each patient in csv file
        all_patients_results_path = os.path.join(
            log_dir,
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
