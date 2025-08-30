import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_and_preprocess(path):
    df = pd.read_csv(path, low_memory=False)
    df['sdsd'] = pd.to_numeric(df['sdsd'], errors='coerce')
    df = df.dropna(subset=['sdsd'])
    X = df.drop(columns=['aami_label']).astype(float)
    y = df['aami_label']

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X = (X - X.mean()) / X.std()
    X_np = np.expand_dims(X.values, axis=2).astype(np.float32)
    y_np = y_encoded.astype(np.int64)
    return X_np, y_np, encoder

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size):
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size),
        DataLoader(test_dataset, batch_size=batch_size)
    )

def save_label_distribution(y_data, title, filename, encoder, result_dir):
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
