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
from src.utils import get_torch_device

logger = setup_logger(name="train")
device = get_torch_device()
writer = SummaryWriter()


def main():
    logger.info(f"Torch device: {device}")
    logger.info("Starting training for all patients")

    for idx in tqdm(range(1, DataConfig.number_of_patients + 1), desc="Patients"):
        patient_id = f"{idx:02d}"

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
            model = MultimodalSeizureModel().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=Trainconfig.lr)
            scaler = GradScaler(device.type)

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

                logger.info(
                    f"Fold {sample_idx + 1} Epoch {epoch + 1}/{Trainconfig.num_epochs} "
                    f"- train_loss: {epoch_loss:.4f}"
                )

                model.eval()
                with torch.no_grad():
                    for batch_tf, batch_bis, batch_labels in test_loader:
                        batch_tf = batch_tf.to(device, non_blocking=True)
                        batch_bis = batch_bis.to(device, non_blocking=True)
                        batch_labels = batch_labels.to(device, non_blocking=True)

                        outputs = model(batch_tf, batch_bis)

                        preds = torch.argmax(outputs, dim=1).cpu().numpy()
                        all_preds.extend(preds)
                        all_labels.extend(batch_labels.cpu().numpy())

                    writer.add_scalar("Loss/train", epoch_loss, epoch)
                    writer.flush()

        acc = accuracy_score(all_labels, all_preds)
        rec = recall_score(all_labels, all_preds, average="binary")
        f1 = f1_score(all_labels, all_preds, average="binary")

        print("\n===== Final LOOCV Results =====")
        print(f"Accuracy: {acc:.4f}")
        print(f"Recall:   {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")


if __name__ == "__main__":
    main()
