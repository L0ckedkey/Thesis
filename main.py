from config import *
from utils.data import load_and_preprocess, split_data, create_dataloaders, save_label_distribution
from models.base import ECGCNN
from utils.train import train_model
from utils.eval import save_confusion_matrix
import torch
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

print("✅ CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("❌ GPU not available.")
    exit()

X, y, encoder = load_and_preprocess(DATA_PATH)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
print(X_train.shape)

save_label_distribution(y, "Label Distribution (All)", "label_all.png", encoder, RESULT_DIR)
save_label_distribution(y_train, "Train", "label_train.png", encoder, RESULT_DIR)
save_label_distribution(y_val, "Validation", "label_val.png", encoder, RESULT_DIR)
save_label_distribution(y_test, "Test", "label_test.png", encoder, RESULT_DIR)

train_loader, val_loader, test_loader = create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, BATCH_SIZE)

model = ECGCNN(input_channels=1, num_classes=len(np.unique(y)), input_length=X_train.shape[1]).to(DEVICE)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

model = train_model(model, criterion, optimizer, train_loader, val_loader, DEVICE, EPOCHS, PATIENCE, LOG_PATH, RESULT_DIR)

save_confusion_matrix(model, train_loader, encoder, "Train Confusion", "train_confusion.png", DEVICE, RESULT_DIR)
save_confusion_matrix(model, val_loader, encoder, "Validation Confusion", "val_confusion.png", DEVICE, RESULT_DIR)
save_confusion_matrix(model, test_loader, encoder, "Test Confusion", "test_confusion.png", DEVICE, RESULT_DIR)
