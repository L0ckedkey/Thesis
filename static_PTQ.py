import os
import torch.quantization as tq
import torch
from config import *
from utils.data import load_and_preprocess, split_data, create_dataloaders, save_label_distribution
from utils.eval import save_confusion_matrix
from models.base import ECGCNN
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

DEVICE = torch.device("cpu") 

# ---------------- Data ----------------
X, y, encoder = load_and_preprocess(DATA_PATH)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
train_loader, val_loader, test_loader = create_dataloaders(
    X_train, X_val, X_test, y_train, y_val, y_test, BATCH_SIZE
)

# ---------------- Model ----------------
model = ECGCNN(input_channels=1, num_classes=len(np.unique(y)), input_length=X_train.shape[1]).to(DEVICE)

# Load model yang sudah disimpan
# model_path = os.path.join("D:\S2\code\model\Thesis\\results\\2025-08-22_20-31-42", "best_model.pth")
model_path = os.path.join("/mnt/d/S2/code/model/Thesis/results/2025-08-22_20-31-42", "best_model.pth")
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()
print("✅ Model loaded from", model_path)

# ---------------- Static Quantization ----------------
# 1. Set quantization config
model.qconfig = tq.QConfig(
    activation=tq.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine),
    weight=tq.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine)
)

# 2. Prepare
model_prepared = tq.prepare(model)

# 3. Calibrate dengan beberapa batch
model_prepared.eval()
with torch.no_grad():
    for i, (inputs, _) in enumerate(train_loader):
        inputs = inputs.to(DEVICE)
        _ = model_prepared(inputs)
        if i >= 10:
            break

# 4. Convert
quantized_model = tq.convert(model_prepared)
print("✅ Static PTQ completed (per-tensor affine)")

# ---------------- Evaluasi ----------------
save_confusion_matrix(
    quantized_model, test_loader, encoder,
    "Quantized Test Confusion", "quant_test_confusion.png", DEVICE, RESULT_DIR
)
print("✅ Confusion matrix saved.")
