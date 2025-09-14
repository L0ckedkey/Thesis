import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os
import importlib

def save_confusion_matrix(model, loader, encoder, title, filename, device, result_dir):
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Tentukan label names
    if isinstance(encoder, dict):  # pakai AAMI_MAP
        idx2label = {v: k for k, v in encoder.items()}
        classes = [f"{i} ({idx2label[i]})" for i in sorted(idx2label.keys())]
    else:  # pakai LabelEncoder
        classes = [f"{i} ({lab})" for i, lab in enumerate(encoder.classes_)]

    # Hitung confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(result_dir, filename))
    plt.close()

    # Hitung metrik
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)

    # Simpan ke file txt
    with open(os.path.join(result_dir, filename.replace(".png", "_metrics.txt")), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    
def load_model(path, input_channels, num_classes, input_length, device, linux=False):
    if not linux:
        model_module = importlib.import_module("models.base")
    else: 
        model_module = importlib.import_module("models.base_linux")
        
    ECGCNN = model_module.ECGCNN
    model = ECGCNN(input_channels, num_classes, input_length)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model
