import torch
from datetime import datetime
import os

# Paths
DATA_PATH = '../../dataset creation/dataset code/feature_store/combined_features_aami_clean_scaled.csv_balanced.csv'
MITDB_PATH = "D:\S2\code\dataset creation\dataset code\\feature_store\combined_features_aami_clean_scaled_cleaned.csv"
SVDB_PATH = ""
ECGCNN_MODEL_PATH = "D:\S2\code\model\Thesis\\results\\2025-08-22_20-31-42\\best_model.pth"
RESULT_DIR = os.path.join("results", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
LOG_PATH = os.path.join(RESULT_DIR, "log.csv")

# Training parameters
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 100
PATIENCE = 10

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU = torch.device("cpu")
# Create result dir if needed
os.makedirs(RESULT_DIR, exist_ok=True)
