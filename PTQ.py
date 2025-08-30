import torch
import torch.quantization as quant
import os
from models.base import ECGCNN
from utils.data import load_and_preprocess, split_data, create_dataloaders, save_label_distribution
from utils.eval import save_confusion_matrix
from utils.train import get_model_size, evaluate_model
from config import *
import numpy as np

MODEL_PATH = 'results\\2025-08-14_18-12-11\\'
DEVICE = torch.device("cpu")

X, y, encoder = load_and_preprocess(DATA_PATH)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
train_loader, val_loader, test_loader = create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, BATCH_SIZE)
save_label_distribution(y_val, "Validation", "label_val.png", encoder, RESULT_DIR)

# 1️⃣ Load model float terlatih
model = ECGCNN(input_channels=1, num_classes=len(np.unique(y)), input_length=X_train.shape[1]).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "best_model.pth"), map_location="cpu"))
model.eval()

# 2️⃣ Dynamic quantization (hanya Linear layer)
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)


# 3️⃣ Simpan model quantized
torch.save(model.state_dict(), os.path.join(RESULT_DIR, "best_model_quantized.pth"))

criterion = torch.nn.CrossEntropyLoss()
val_loss, val_acc, inf_time_batch, inf_time_sample = evaluate_model(model, val_loader, criterion, DEVICE)

# ===== 4. Ukuran model =====
model_size = get_model_size(model)

save_confusion_matrix(model, val_loader, encoder, "Validation Confusion", "val_confusion.png", DEVICE, RESULT_DIR)

with open(os.path.join(RESULT_DIR, "quantized_eval_log.txt"), "w") as f:
    f.write(f"Quantized Model Evaluation\n")
    f.write(f"Validation Loss: {val_loss:.4f}\n")
    f.write(f"Validation Accuracy: {val_acc:.4f}\n")
    f.write(f"Model Size: {model_size:.2f} MB\n")
    f.write(f"Inference Time / Batch: {inf_time_batch*1000:.3f} ms\n")
    f.write(f"Inference Time / Sample: {inf_time_sample*1000:.3f} ms\n")

print("✅ Quantized model saved & metrics logged.")
