import torch
import torch.nn as nn
import numpy as np
import onnx
import tensorflow as tf
import os
from models.base import ECGCNN
from config import *
from onnx_tf.backend import prepare
from utils.train import get_general_model_size

# --------------------
# 1. Load PyTorch model
# --------------------
DEVICE = torch.device("cpu")

# Ganti sesuai datasetmu
input_length = 29   # contoh panjang sinyal ECG
num_classes = 5     # contoh jumlah kelas

model = ECGCNN(input_channels=1, num_classes=num_classes, input_length=input_length).to(DEVICE)
model.load_state_dict(torch.load("results/2025-08-22_20-31-42/best_model.pth", map_location=DEVICE))
model.eval()

print("✅ PyTorch model loaded.")

# --------------------
# 2. Export ke ONNX
# --------------------
dummy_input = torch.randn(1, input_length, 1).to(DEVICE) 

onnx_path = os.path.join(RESULT_DIR, "model.onnx")

torch.onnx.export(
    model, dummy_input, onnx_path,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)
print(f"✅ Model exported to {onnx_path}")

get_general_model_size(onnx_path)

# Verifikasi ONNX
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("✅ ONNX model checked successfully.")

# --------------------
# 3. Convert ONNX → TensorFlow
# --------------------
tf_rep = prepare(onnx_model)
tf_folder = os.path.join(RESULT_DIR, "ecgcnn_tf")
tf_rep.export_graph(tf_folder)  # save as SavedModel
print(f"✅ Converted ONNX → TensorFlow SavedModel at {tf_folder}")

# --------------------
# 4. Convert ke TFLite dengan quantization
# --------------------
converter = tf.lite.TFLiteConverter.from_saved_model(tf_folder)

# Dynamic range quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

tflite_path = os.path.join(RESULT_DIR, "ecgcnn_quant.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ TFLite quantized model saved as {tflite_path}")
get_general_model_size(tflite_path)
