import os
import sys

# ganti path projectnya
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Masukkan root project ke sys.path paling depan
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import time
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, matthews_corrcoef,
                             cohen_kappa_score)
import onnxruntime as ort
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torchinfo import summary

from config import *
from utils.data import load_and_preprocess, split_data, create_dataloaders
from models.base import ECGCNN
from utils.train import get_general_model_size

# -------------------- 0. Config --------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ CUDA available:", torch.cuda.is_available())

# Load dataset
X, y, encoder = load_and_preprocess(DATA_PATH)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
train_loader, val_loader, test_loader = create_dataloaders(
    X_train, X_val, X_test, y_train, y_val, y_test, BATCH_SIZE
)

num_classes = len(np.unique(y))
input_length = X_train.shape[1]

# -------------------- 1. Load all models --------------------
def load_all_models(input_length, num_classes):
    # PyTorch
    model_path = os.path.join("results\\2025-08-22_20-31-42/best_model.pth")
    pytorch_model = ECGCNN(input_channels=1, num_classes=num_classes, input_length=input_length).to(DEVICE)
    summary(pytorch_model, input_size=(input_length, 1), batch_dim=0)  
    pytorch_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    pytorch_model.eval()
    print("✅ PyTorch model loaded.")

    # ONNX
    onnx_path = os.path.join("results\\2025-08-25_16-56-58\model.onnx")
    onnx_session = ort.InferenceSession(onnx_path)
    print("✅ ONNX model loaded.")

    # TensorFlow SavedModel
    tf_folder = os.path.join("results\\2025-08-25_16-56-58\\ecgcnn_tf")
    tf_model = tf.saved_model.load(tf_folder)
    print("✅ TensorFlow SavedModel loaded.")

    # TFLite
    tflite_path = os.path.join("results\\2025-08-25_16-56-58\\ecgcnn_quant.tflite")
    tflite_interpreter = tf.lite.Interpreter(model_path=tflite_path)
    tflite_interpreter.allocate_tensors()
    print("✅ TFLite model loaded.")

    return pytorch_model, onnx_session, tf_model, tflite_interpreter

pytorch_model, onnx_session, tf_model, tflite_interpreter = load_all_models(input_length, num_classes)

# -------------------- 2. Evaluation helpers --------------------
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred)
    }

def plot_confusion_matrix(y_true, y_pred, encoder, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(RESULT_DIR, filename))
    plt.close()

def evaluate_pytorch(model, loader, encoder, device):
    y_true, y_pred = [], []
    start_time = time.time()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds.cpu().numpy())
    end_time = time.time()
    latency = (end_time - start_time)/len(y_true)
    throughput = len(y_true)/(end_time - start_time)
    metrics = compute_metrics(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, encoder, "PyTorch Confusion", "confusion_pytorch.png")
    return metrics, latency, throughput

def evaluate_onnx(onnx_session, loader, encoder):
    y_true, y_pred = [], []
    start_time = time.time()
    for X_batch, y_batch in loader:
        X_batch_np = X_batch.numpy()
        outputs = onnx_session.run(None, {"input": X_batch_np})[0]
        y_true.extend(y_batch.numpy())
        y_pred.extend(np.argmax(outputs, axis=1))
    end_time = time.time()
    latency = (end_time - start_time)/len(y_true)
    throughput = len(y_true)/(end_time - start_time)
    metrics = compute_metrics(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, encoder, "ONNX Confusion", "confusion_onnx.png")
    return metrics, latency, throughput

def evaluate_tf(tf_model, loader, encoder):
    infer = tf_model.signatures["serving_default"]
    y_true, y_pred = [], []
    start_time = time.time()
    for X_batch, y_batch in loader:
        for i in range(X_batch.shape[0]):
            x_input = tf.convert_to_tensor(X_batch[i:i+1], dtype=tf.float32)
            out = infer(input=x_input)
            y_pred.append(np.argmax(out["output"].numpy()))
        y_true.extend(y_batch.numpy())
    end_time = time.time()
    latency = (end_time - start_time)/len(y_true)
    throughput = len(y_true)/(end_time - start_time)
    metrics = compute_metrics(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, encoder, "TensorFlow Confusion", "confusion_tf.png")
    return metrics, latency, throughput

def evaluate_tflite(interpreter, loader, encoder):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    y_true, y_pred = [], []
    start_time = time.time()
    for X_batch, y_batch in loader:
        for i in range(X_batch.shape[0]):
            interpreter.set_tensor(input_details[0]['index'], X_batch[i:i+1])
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            y_pred.append(np.argmax(output_data))
        y_true.extend(y_batch.numpy())
    end_time = time.time()
    latency = (end_time - start_time)/len(y_true)
    throughput = len(y_true)/(end_time - start_time)
    metrics = compute_metrics(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, encoder, "TFLite Confusion", "confusion_tflite.png")
    return metrics, latency, throughput

# -------------------- 3. Run evaluation & save metrics --------------------
formats = {
    "PyTorch": (evaluate_pytorch, pytorch_model),
    "ONNX": (evaluate_onnx, onnx_session),
    "TensorFlow": (evaluate_tf, tf_model),
    "TFLite": (evaluate_tflite, tflite_interpreter)
}

metrics_list = []

for name, (func, model_obj) in formats.items():
    if name == "PyTorch":
        metrics, latency, throughput = func(model_obj, test_loader, encoder, DEVICE)
    else:
        metrics, latency, throughput = func(model_obj, test_loader, encoder)
    print(f"\n{name} Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"Latency per sample: {latency*1000:.3f} ms, Throughput: {throughput:.2f} samples/s")
    
    # Simpan ke list untuk CSV
    row = {"format": name, "latency_ms": latency*1000, "throughput_sps": throughput}
    row.update(metrics)
    metrics_list.append(row)

# Simpan semua metrics ke CSV
metrics_df = pd.DataFrame(metrics_list)
csv_path = os.path.join(RESULT_DIR, "model_metrics.csv")
metrics_df.to_csv(csv_path, index=False)
print(f"\n✅ Semua metrics disimpan ke {csv_path}")
