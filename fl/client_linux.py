import os
import sys

# ganti path projectnya
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Masukkan root project ke sys.path paling depan
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import flwr as fl
import torch
from torch.utils.data import DataLoader
from config import *
from utils.param import get_model_params, set_model_params
from utils.data import load_and_preprocess, create_dataloaders, split_data
from models.base import ECGCNN  # Definisi model
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_DIR = "fl_client_results"
os.makedirs(RESULT_DIR, exist_ok=True)

# --- Load & preprocess data ---
X_np, y_np, encoder = load_and_preprocess(MITDB_PATH)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_np, y_np)
train_loader, val_loader, test_loader = create_dataloaders(
    X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32
)

# --- Buat model kosong ---
model = ECGCNN(
    input_channels=X_np.shape[2],
    num_classes=len(np.unique(y_np)),
    input_length=X_np.shape[1]
).to(DEVICE)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def evaluate_model(loader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    acc = correct / total
    return acc

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return get_model_params(model)

    def fit(self, parameters, config):
        # --- Set parameter global dari server ---
        set_model_params(model, parameters)

        # --- Evaluasi sebelum training ---
        acc_before = evaluate_model(val_loader)
        print(f"🏁 Accuracy sebelum training: {acc_before:.4f}")

        # --- Training ---
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # --- Evaluasi setelah training ---
        acc_after = evaluate_model(val_loader)
        print(f"✅ Accuracy setelah training: {acc_after:.4f}")

        # --- Simpan model lokal client ---
        torch.save(model.state_dict(), os.path.join(RESULT_DIR, "model_local.pth"))

        return get_model_params(model), len(train_loader.dataset), {
            "accuracy_before": acc_before,
            "accuracy_after": acc_after
        }

    def evaluate(self, parameters, config):
        set_model_params(model, parameters)
        acc = evaluate_model(val_loader)
        loss = 1 - acc
        return float(loss), len(val_loader.dataset), {"accuracy": acc}

if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address=LINUX_HOST,
        client=FlowerClient()
    )
