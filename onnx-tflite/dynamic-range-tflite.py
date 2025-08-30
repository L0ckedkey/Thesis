import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import onnxruntime as ort
import matplotlib.pyplot as plt
import seaborn as sns

from config import *
from utils.data import load_and_preprocess, split_data, create_dataloaders
from models.base import ECGCNN
from utils.eval import save_confusion_matrix
from utils.train import get_general_model_size

# --------------------
# 0. Config
# --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("❌ GPU not available, using CPU.")

# Load dataset
X, y, encoder = load_and_preprocess(DATA_PATH)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# Dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    X_train, X_val, X_test, y_train, y_val, y_test, BATCH_SIZE
)

num_classes = len(np.unique(y))
input_length = X_train.shape[1]

# --------------------
# 1. Load PyTorch model
# --------------------
model_path = os.path.join("results\\2025-08-22_20-31-42\\best_model.pth")
model = ECGCNN(input_channels=1, num_classes=num_classes, input_length=input_length).to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ PyTorch model loaded. Total params: {total_params}, Trainable: {trainable_params}")
get_general_model_size(model_path)

# Confusion matrix PyTorch
save_confusion_matrix(model, test_loader, encoder, "Test Confusion (PyTorch)", "test_confusion_pytorch.png", DEVICE, RESULT_DIR)

# --------------------
# 2. Export ke ONNX jika belum ada
# --------------------
onnx_path = os.path.join(RESULT_DIR, "model.onnx")
if not os.path.exists(onnx_path):
    dummy_input = torch.randn(1, input_length, 1).to(DEVICE)
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11
    )
    print(f"✅ Model exported to ONNX: {onnx_path}")

get_general_model_size(onnx_path)

# Verifikasi ONNX
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("✅ ONNX model checked successfully.")

# --------------------
# 3. Convert ONNX → TensorFlow SavedModel jika belum ada
# --------------------
tf_folder = os.path.join(RESULT_DIR, "ecgcnn_tf")
if not os.path.exists(tf_folder):
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_folder)
    print(f"✅ Converted ONNX → TensorFlow SavedModel at {tf_folder}")
get_general_model_size(tf_folder)

# --------------------
# 4. Convert ke TFLite dengan quantization jika belum ada
# --------------------
tflite_path = os.path.join(RESULT_DIR, "ecgcnn_quant.tflite")
if not os.path.exists(tflite_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_folder)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"✅ TFLite quantized model saved as {tflite_path}")
get_general_model_size(tflite_path)

# --------------------
# 5. Evaluasi semua format
# --------------------

def evaluate_onnx(test_loader, encoder, onnx_path):
    ort_session = ort.InferenceSession(onnx_path)
    y_true, y_pred = [], []
    for X_batch, y_batch in test_loader:
        X_batch_np = X_batch.numpy()
        outputs = ort_session.run(None, {"input": X_batch_np})[0]
        y_true.extend(y_batch.numpy())
        y_pred.extend(np.argmax(outputs, axis=1))
    print("ONNX Accuracy:", accuracy_score(y_true, y_pred))
    cm_file = "test_confusion_onnx.png"
    plt.figure(figsize=(6,5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title("Test Confusion (ONNX)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(RESULT_DIR, cm_file))
    plt.close()

def evaluate_tf(test_loader, encoder, tf_folder):
    tf_model = tf.saved_model.load(tf_folder)
    infer = tf_model.signatures["serving_default"]
    y_true, y_pred = [], []
    for X_batch, y_batch in test_loader:
        for i in range(X_batch.shape[0]):
            x_input = tf.convert_to_tensor(X_batch[i:i+1], dtype=tf.float32)
            out = infer(input=x_input)
            y_pred.append(np.argmax(out["output"].numpy()))
        y_true.extend(y_batch.numpy())
    print("TensorFlow Accuracy:", accuracy_score(y_true, y_pred))
    cm_file = "test_confusion_tf.png"
    plt.figure(figsize=(6,5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title("Test Confusion (TensorFlow)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(RESULT_DIR, cm_file))
    plt.close()

def evaluate_tflite(test_loader, encoder, tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    y_true, y_pred = [], []
    for X_batch, y_batch in test_loader:
        for i in range(X_batch.shape[0]):
            interpreter.set_tensor(input_details[0]['index'], X_batch[i:i+1])
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            y_pred.append(np.argmax(output_data))
        y_true.extend(y_batch.numpy())
    print("TFLite Accuracy:", accuracy_score(y_true, y_pred))
    cm_file = "test_confusion_tflite.png"
    plt.figure(figsize=(6,5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title("Test Confusion (TFLite)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(RESULT_DIR, cm_file))
    plt.close()

# Run evaluations
evaluate_onnx(test_loader, encoder, onnx_path)
evaluate_tf(test_loader, encoder, tf_folder)
evaluate_tflite(test_loader, encoder, tflite_path)

print("✅ Evaluasi semua format selesai.")
