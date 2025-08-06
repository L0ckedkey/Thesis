import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import sys

# Cek GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("❌ GPU not available. Using CPU.")
    sys.exit()

# ===== 1. Load data =====
# path = "D:/S2/code/dataset creation/dataset code/feature_store/20250716_features_aami.csv"
path = '../dataset creation/dataset code/feature_store/20250716_features_aami.csv'
df = pd.read_csv(path,low_memory=False)

# ===== 2. Drop kolom yang tidak dibutuhkan =====
# Baru drop kolom-kolom non-fitur

# 1. Hapus kolom-kolom yang tidak dibutuhkan terlebih dahulu
df = df.dropna(subset=['lf_hf_ratio', 'lfnu', 'hfnu'])
X = df.drop(columns=['label', 'aami_label', 'tinn', 'channel','entropy','record_id'])

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
print(X_train.shape)



# ===== 7. DataLoader =====
batch_size = 32
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

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
model = ECGCNN(input_channels, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    acc = correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")

# Evaluation on test data
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

test_acc = correct / total
print(f"🧪 Test Accuracy: {test_acc:.4f}")
