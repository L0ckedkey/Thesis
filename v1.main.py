import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime


def save_label_distribution(y_data, title, filename):
    label_counts = pd.Series(y_data).value_counts().sort_index()
    plt.figure(figsize=(6,4))
    sns.barplot(x=encoder.inverse_transform(label_counts.index), y=label_counts.values, palette="viridis")
    plt.title(title)
    plt.xlabel("Label")
    plt.ylabel("Jumlah Sampel")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, filename))
    plt.close()

# Cek GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("❌ GPU not available. Using CPU.")
    sys.exit()
    
    
# ===== 0. Init folder for log result =====
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
result_dir = os.path.join("results", timestamp)
os.makedirs(result_dir, exist_ok=True)
log_path = os.path.join(result_dir, "log.csv")
with open(log_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])


# ===== 1. Load data =====
# path = "D:/S2/code/dataset creation/dataset code/feature_store/20250716_features_aami.csv"
path = '../../dataset creation/dataset code/feature_store/combined_features_aami_clean_scaled.csv_balanced.csv'
df = pd.read_csv(path,low_memory=False)

# ===== 2. Drop kolom yang tidak dibutuhkan =====
# Baru drop kolom-kolom non-fitur

# 1. Hapus kolom-kolom yang tidak dibutuhkan terlebih dahulu
# df = df.dropna(subset=['lf_hf_ratio', 'lfnu'])
# X = df.drop(columns=['aami_label','entropy','record_id'])
X = df.drop(columns=['aami_label'])

# 2. Ubah kolom 'sdsd' ke numerik, kalau gagal jadikan NaN
df['sdsd'] = pd.to_numeric(df['sdsd'], errors='coerce')  # konversi
df = df[df['sdsd'].notna()]  # hanya filter baris yang valid

df['sdsd'] = pd.to_numeric(df['sdsd'], errors='coerce')
df = df.dropna(subset=['sdsd'])

# 5. Ambil labelnya
y = df['aami_label']

# Note: cek isNa
# print(df.isna().sum())

# ===== 3. Encode AAMI labels =====
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ===== 4. Normalisasi =====
X = X.astype(float)
X = (X - X.mean()) / X.std()

# ===== 5. Bentuk data ke (samples, timesteps, channels) =====
X_np = np.expand_dims(X.values, axis=2).astype(np.float32)
y_np = y_encoded.astype(np.int64)

# ===== 6. Split data =====
X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# ===== 6.5. Save the dataset info =====
save_label_distribution(y_np, "Distribusi Label (Keseluruhan)", "label_dist_all.png")
save_label_distribution(y_train, "Distribusi Label (Train)", "label_dist_train.png")
save_label_distribution(y_val, "Distribusi Label (Validation)", "label_dist_val.png")
save_label_distribution(y_test, "Distribusi Label (Test)", "label_dist_test.png")

# ===== 7. DataLoader =====
# ===== 7. DataLoader =====
batch_size = 32
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# ===== 8. CNN Model PyTorch =====
class ECGCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(ECGCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv5 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(2)
        self.bn3 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self._get_flattened_size(X_train.shape[1], input_channels), 64)
        self.fc2 = nn.Linear(64, num_classes)


    def _get_flattened_size(self, input_length, input_channels):
        x = torch.zeros(1, input_channels, input_length)  # udah benar urutannya
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.pool3(F.relu(self.conv6(F.relu(self.conv5(x)))))
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, F, 1) -> (B, 1, F)
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.bn1(self.dropout(x))

        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.bn2(self.dropout(x))

        x = self.pool3(F.relu(self.conv6(F.relu(self.conv5(x)))))
        x = self.bn3(self.dropout(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ===== 9. Training =====
input_channels = 1
num_classes = len(np.unique(y))

model = ECGCNN(input_channels=1, num_classes=len(np.unique(y))).to(device)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===== Training with Early Stopping + Logging =====
patience = 10
best_val_loss = float("inf")
epochs_no_improve = 0
best_model_state = None

train_losses, val_losses = [], []
train_accs, val_accs = [], []

num_epochs = 100
for epoch in range(num_epochs):
    # === Train ===
    model.train()
    running_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    # === Validation ===
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += y_batch.size(0)
            val_correct += (predicted == y_batch).sum().item()

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    # Simpan history untuk plot
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    # Tulis ke CSV log
    with open(log_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc])

    print(f"Epoch {epoch+1}/{num_epochs} "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        torch.save(best_model_state, os.path.join(result_dir, "best_model.pth"))
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"⏹ Early stopping at epoch {epoch+1}")
            break

# ===== Load Best Model =====
model.load_state_dict(torch.load(os.path.join(result_dir, "best_model.pth")))

# ===== Plot Loss & Accuracy =====
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy per Epoch")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(result_dir, "loss_acc_plot.png"))
plt.close()

# ===== Confusion Matrix Function =====
def save_confusion_matrix(loader, title, filename):
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(result_dir, filename))
    plt.close()

# Simpan confusion matrix untuk train/val/test
save_confusion_matrix(train_loader, "Train Confusion Matrix", "train_confusion.png")
save_confusion_matrix(val_loader, "Validation Confusion Matrix", "val_confusion.png")
save_confusion_matrix(test_loader, "Test Confusion Matrix", "test_confusion.png")