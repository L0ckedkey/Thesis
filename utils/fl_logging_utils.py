import os
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def create_folder(base_dir, round_num):
    """Buat folder sesuai tanggal dan round"""
    date_str = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(base_dir, date_str, f"round_{round_num}")
    os.makedirs(path, exist_ok=True)
    return path

def save_metrics(metrics, base_dir, round_num, filename="metrics.json"):
    folder = create_folder(base_dir, round_num)
    filepath = os.path.join(folder, filename)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"📄 Metrics saved to {filepath}")

def save_confusion_matrix(y_true, y_pred, base_dir, round_num, class_names=None, filename="confusion_matrix.png"):
    folder = create_folder(base_dir, round_num)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    filepath = os.path.join(folder, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"📊 Confusion matrix saved to {filepath}")

def compute_classwise_metrics(y_true, y_pred):
    """Hitung akurasi, precision, recall, F1 per kelas"""
    classes = np.unique(y_true)
    class_metrics = {}
    for c in classes:
        idx = y_true == c
        y_true_c = y_true[idx]
        y_pred_c = y_pred[idx]
        acc = (y_true_c == y_pred_c).mean() if len(y_true_c) > 0 else None
        prec = precision_score(y_true_c, y_pred_c, average="binary", zero_division=0)
        rec = recall_score(y_true_c, y_pred_c, average="binary", zero_division=0)
        f1 = f1_score(y_true_c, y_pred_c, average="binary", zero_division=0)
        class_metrics[int(c)] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    return class_metrics

def log_all_metrics(y_true, y_pred, base_dir, round_num, class_names=None):
    metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "classwise": compute_classwise_metrics(y_true, y_pred)
    }
    save_metrics(metrics, base_dir, round_num)
    save_confusion_matrix(y_true, y_pred, base_dir, round_num, class_names)
