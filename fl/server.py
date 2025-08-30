import os
import sys

# ganti path projectnya
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Masukkan root project ke sys.path paling depan
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import csv
import flwr as fl
import torch
from utils.eval import load_model
from utils.param import get_model_params, set_model_params
from config import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(RESULT_DIR, exist_ok=True)
HISTORY_CSV = os.path.join(RESULT_DIR, "training_history.csv")

# Load pretrained model
global_model = load_model(
    path=ECGCNN_MODEL_PATH,
    input_channels=1,   # sesuaikan data
    num_classes=5,
    input_length=29,
    device=DEVICE
)

# --- Helper untuk simpan history ---
def append_history(round_num, metrics, filename=HISTORY_CSV):
    fieldnames = ["round", "num_samples", "accuracy", "loss"]
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {"round": round_num}
        if metrics:
            row.update(metrics)
        writer.writerow(row)

class MyStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model):
        self.model = model
        super().__init__()

    def initialize_parameters(self, client_manager):
        print("🔹 Broadcasting pretrained model to clients")
        return fl.common.ndarrays_to_parameters(get_model_params(self.model))

    def aggregate_fit(self, rnd, results, failures):
        aggregated = super().aggregate_fit(rnd, results, failures)

        if aggregated is not None:
            # --- Ambil Parameters jika tuple (parameters, metrics) ---
            if isinstance(aggregated, tuple):
                parameters, metrics_list = aggregated
                # rata-rata metrics dari semua client
                metrics = {}
                if metrics_list:
                    num_samples = sum([m.get("num_examples", 0) for m in metrics_list])
                    accuracy = sum([m.get("accuracy_after", 0) * m.get("num_examples", 0) for m in metrics_list])
                    accuracy = accuracy / num_samples if num_samples else 0.0
                    metrics = {"num_samples": num_samples, "accuracy": accuracy, "loss": 1-accuracy}
            else:
                parameters = aggregated
                metrics = None

            # --- Set parameter ke model global ---
            params_ndarrays = fl.common.parameters_to_ndarrays(parameters)
            set_model_params(self.model, params_ndarrays)

            # --- Simpan model global tiap round ---
            torch.save(self.model.state_dict(), os.path.join(RESULT_DIR, f"model_global_round{rnd}.pth"))
            print(f"💾 Global model updated and saved after round {rnd}")

            # --- Simpan history ke CSV ---
            append_history(rnd, metrics)

            # --- Return Parameters object + metrics jika ada ---
            return aggregated

        return None

if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=MyStrategy(global_model),
        config=fl.server.ServerConfig(num_rounds=3),
    )
