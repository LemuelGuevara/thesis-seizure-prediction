import csv
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.amp import GradScaler, autocast
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import DataConfig, Trainconfig
from src.logger import setup_logger
from src.model.classification.multi_seizure_model import MultimodalSeizureModel
from src.model.data import create_data_loader, get_loocv_fold, get_paired_dataset
from src.model.early_stopping import EarlyStopping
from src.utils import get_torch_device

logger = setup_logger(name="train")
device = get_torch_device()
writer = SummaryWriter()


def main():
    logger.info(f"Torch device: {device}")
    logger.info("Starting training for all patients")

    loocv_results: list[dict[str, float | str]] = []

    for idx in tqdm(range(1, DataConfig.number_of_patients + 1), desc="Patients"):
        patient_id = f"{idx:02d}"

        timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
        patient_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "runs",
            "results",
            f"patient_{patient_id}",
        )
        os.makedirs(patient_dir, exist_ok=True)

        timestamped_dir = os.path.join(patient_dir, timestamp)
        os.makedirs(timestamped_dir, exist_ok=True)
        # Get get paired dataset
        tf_features, bis_features, labels = get_paired_dataset(patient_id=patient_id)
        all_preds, all_labels = [], []

        for sample_idx in tqdm(range(len(tf_features)), desc="Samples"):
            # Split train/test data through loocv
            (tf_train, bis_train, labels_train), (tf_test, bis_test, labels_test) = (
                get_loocv_fold(
                    tf_features=tf_features,
                    bis_features=bis_features,
                    labels=labels,
                    sample_idx=sample_idx,
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

            checkpoint_path = os.path.join(
                os.path.dirname(__file__),
                "checkpoints",
                f"patient{patient_id}_fold{sample_idx}.pt",
            )

            early_stopping = EarlyStopping(
                path=checkpoint_path,
                patience=10,
                verbose=True,
            )

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

                logger.info(
                    f"Patient {patient_id} Fold {sample_idx + 1} "
                    f"Epoch {epoch + 1}/{Trainconfig.num_epochs} "
                    f"- train_loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}"
                )
                writer.add_scalar(f"{patient_id}/Loss/train", epoch_loss, epoch)
                writer.add_scalar(f"{patient_id}/Loss/val", val_loss, epoch)
                writer.flush()

                # Apply early stopping to stop overfitting the model
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    logger.info("Early stopping triggered.")
                    break

        acc = accuracy_score(all_labels, all_preds)
        rec = recall_score(all_labels, all_preds, average="binary")
        f1 = f1_score(all_labels, all_preds, average="binary")

        loocv_results.append(
            {"patient": patient_id, "accuracy": round(acc, 4), "f1-score": round(f1, 4)}
        )

        print("\n===== Final LOOCV Results =====")
        print(f"Accuracy: {acc:.4f}")
        print(f"Recall:   {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")

    patient_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "runs",
        "results",
        f"patient_{patient_id}",
    )
    os.makedirs(patient_dir, exist_ok=True)
    patient_path = os.path.join(patient_dir, "results.csv")

    with open(patient_path, "w", newline="") as results_csv:
        fieldnames = ["patient", "accuracy", "f1-score"]
        csv_writer = csv.DictWriter(results_csv, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(loocv_results)


if __name__ == "__main__":
    main()
