import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, f1_score
from src.model.class_labels import get_paired_dataset
from src.model.multimodalseizuremodel import MultimodalSeizureModel

def leave_one_out_cv(patient_id: str, num_epochs=5, lr=1e-4):
    # Load dataset
    X_tf, X_bis, y = get_paired_dataset(patient_id)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_preds, all_labels = [], []

    for i in range(len(X_tf)):
        print(f"\n[LOOCV] Fold {i+1}/{len(X_tf)} (leaving out sample {i})")

        # Split train/test
        X_tf_train = torch.tensor([x for j, x in enumerate(X_tf) if j != i], dtype=torch.float32)
        X_bis_train = torch.tensor([x for j, x in enumerate(X_bis) if j != i], dtype=torch.float32)
        y_train = torch.tensor([l for j, l in enumerate(y) if j != i], dtype=torch.long)

        X_tf_test = torch.tensor(X_tf[i], dtype=torch.float32).unsqueeze(0)
        X_bis_test = torch.tensor(X_bis[i], dtype=torch.float32).unsqueeze(0)
        y_test = torch.tensor([y[i]], dtype=torch.long)

        # Model
        model = MultimodalSeizureModel().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(X_tf_train.to(device), X_bis_train.to(device))
            loss = criterion(outputs, y_train.to(device))
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X_tf_test.to(device), X_bis_test.to(device))
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_test.numpy())

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")

    print("\n===== Final LOOCV Results =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Recall:   {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return acc, rec, f1


if __name__ == "__main__":
    leave_one_out_cv("patient_01")
