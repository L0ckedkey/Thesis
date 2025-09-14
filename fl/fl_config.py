import torch
from datetime import datetime
import os

RESULT_DIR = os.path.join("fl","results", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
GLOBAL_MODEL_DIR = os.path.join(RESULT_DIR,"global_model")
CLIENT_MODEL_DIR = os.path.join(RESULT_DIR,"clients")
LOG_PATH = os.path.join(RESULT_DIR, "log.csv")


# model config
EPOCH=100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.001
MODEL_CLASS = 5

# FL config
ROUNDS = 200



DATA_PATH = './datasets/gan.csv'
MITDB_PATH = "./datasets/mitdb.csv"
SVDB_PATH = "./datasets/svdb.csv"

RAW_MITDB = "D:/S2/code/dataset creation/dataset code/feature_store/20250717_features_aami.csv"
RAW_SVDB = "D:/S2/code/dataset creation/dataset code/feature_store/20250716_features_aami.csv"

LINUX_DATA_PATH = '/home/hans/Documents/combined_features_aami_clean_scaled.csv_balanced.csv'
LINUX_MITDB_PATH = '/home/hans/Documents/combined_features_aami_clean_scaled_cleaned.csv'
LINUX_SVDB_PATH = ""
LINUX_RAW_SVDB = ""
LINUX_RAW_MITDB = ""

ECGCNN_MODEL_PATH = ".\\result_models\\ECGCNN_base_mapped.pth"
# ECGCNN_MODEL_PATH = ".\\result_models\\ECGCNN.pth"
ECGCNN_ADAPTIVE_POOL_MODEL_PATH = ".\\result_models\\ECGCNN-AdaptivePool.pth"
ECGCNN_DYNAMIC_RANGE_QUANTIZATION_MODEL_PATH = ".\\result_models\\ECGCNN-DynamicRangeQuantization.pth"

LINUX_ECGCNN_MODEL_PATH = "./result_models/ECGCNN.pth"
LINUX_ECGCNN_ADAPTIVE_POOL_MODEL_PATH = "./result_models/ECGCNN-AdaptivePool.pth"
LINUX_ECGCNN_DYNAMIC_RANGE_QUANTIZATION_MODEL_PATH = "./result_models/ECGCNN-DynamicRangeQuantization.pth"
