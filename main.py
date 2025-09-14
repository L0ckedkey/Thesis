from config import *
from utils.data import load_and_preprocess, split_data, create_dataloaders, save_label_distribution
from models.base_linux import ECGCNN
from utils.train import train_model
from utils.eval import save_confusion_matrix
import torch
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

import argparse
import importlib

# --------------------
# 0. Config
# --------------------

parser = argparse.ArgumentParser()
parser.add_argument("-l","--linux", action="store_true", help="Enable linux treatment")
args = parser.parse_args()

if args.linux:
    model_module = importlib.import_module("models.base_linux")
else:
    model_module = importlib.import_module("models.base")

ECGCNN = model_module.ECGCNN

print("✅ CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("❌ GPU not available.")
    exit()

X, y, encoder = load_and_preprocess(DATA_PATH)
print("Shape X: ", X.shape)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
print("X_train shape: ", X_train.shape)

# overlap = 0
# duplicates = []
# for i, x_val in enumerate(X_val):
#     for j, x_train in enumerate(X_train):
#         if np.array_equal(x_val, x_train):
#             overlap += 1
#             duplicates.append((i, j))  # simpan index val & train

# print(f"Jumlah overlap data train-val: {overlap}")

# if overlap > 0:
#     print("Contoh pasangan duplikat (val_idx, train_idx):")
#     print(duplicates[:10])  # print maksimal 10
# print("yeay")

save_label_distribution(y, "Label Distribution (All)", "label_all.png", encoder, RESULT_DIR)
save_label_distribution(y_train, "Train", "label_train.png", encoder, RESULT_DIR)
save_label_distribution(y_val, "Validation", "label_val.png", encoder, RESULT_DIR)
save_label_distribution(y_test, "Test", "label_test.png", encoder, RESULT_DIR)

train_loader, val_loader, test_loader = create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, BATCH_SIZE)
# train_loader2, val_loader2, test_loader2 = create_dataloaders(X2_train, X2_val, X2_test, y2_train, y2_val, y2_test, BATCH_SIZE)

model = ECGCNN(input_channels=1, num_classes=5, input_length=X_train.shape[1]).to(DEVICE)

# class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
# class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
# criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

model = train_model(model, criterion, optimizer, train_loader, val_loader, DEVICE, EPOCHS, PATIENCE, LOG_PATH, RESULT_DIR)

save_confusion_matrix(model, train_loader, encoder, "Train Confusion", "train_confusion.png", DEVICE, RESULT_DIR)
save_confusion_matrix(model, val_loader, encoder, "Validation Confusion", "val_confusion.png", DEVICE, RESULT_DIR)
save_confusion_matrix(model, test_loader, encoder, "Test Confusion", "test_confusion.png", DEVICE, RESULT_DIR)
