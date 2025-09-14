import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import TensorDataset, DataLoader

AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
AAMI_MAP = {cls: idx for idx, cls in enumerate(AAMI_CLASSES)}

# def load_and_preprocess(path):
#     df = pd.read_csv(path, low_memory=False)
#     df['sdsd'] = pd.to_numeric(df['sdsd'], errors='coerce')
#     df = df.dropna(subset=['sdsd'])
#     df = df.drop(columns=['channel','label'])
#     X = df.drop(columns=['aami_label']).astype(float)
#     y = df['aami_label']

#     encoder = LabelEncoder()
#     y_encoded = encoder.fit_transform(y)

#     X = (X - X.mean()) / X.std()
#     X_np = np.expand_dims(X.values, axis=2).astype(np.float32)
#     y_np = y_encoded.astype(np.int64)
#     return X_np, y_np, encoder

def load_and_preprocess(path):
    df = pd.read_csv(path, low_memory=False)

    # case 1 → kalau nama file/path mengandung "features_aami"
    if "features_aami" in path:
        print("✅ Detected features_aami dataset")

        # drop baris yang tidak punya fitur penting
        df = df.dropna(subset=['lf_hf_ratio', 'lfnu', 'hfnu', 'sd2','ratio_sd2_sd1','csi','cvi','Modified_csi'])

        # ubah kolom 'sdsd' ke numerik
        df['sdsd'] = pd.to_numeric(df['sdsd'], errors='coerce')
        df = df[df['sdsd'].notna()]

        # drop kolom yang tidak dipakai
        X = df.drop(columns=['label', 'aami_label', 'tinn', 'channel','entropy','record_id', 'min_hr','mean_hr','sample',]).astype(float)

    # case 2 → dataset normal
    else:
        print("✅ Detected normal dataset")

        df['sdsd'] = pd.to_numeric(df['sdsd'], errors='coerce')
        df = df.dropna(subset=['sdsd'])
        # df = df.drop(columns=['channel','label'])
        X = df.drop(columns=['aami_label']).astype(float)

    # target encoding
    y = df['aami_label'].map(AAMI_MAP)
    y = y.dropna().astype(int)  # kalau ada label di luar N,S,V,F,Q (buang aja)
    X = X.loc[y.index]  # sync index

    # normalisasi
    X = (X - X.mean()) / (X.std() + 1e-8)

    # numpy format
    X_np = np.expand_dims(X.values, axis=2).astype(np.float32)
    y_np = y.astype(np.int64).values

    return X_np, y_np, AAMI_MAP

def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42, delDuplicate = True):
    """
    Split data jadi train, val, test tanpa overlap.
    Duplikat dihapus sebelum split.
    """

    if delDuplicate:
        # --- Buang duplikat ---
        Xy = np.hstack([X.reshape(len(X), -1), y.reshape(-1, 1)])  # gabungkan X dan y
        Xy_unique = np.unique(Xy, axis=0)  # hapus duplikat
        X_unique, y_unique = Xy_unique[:, :-1], Xy_unique[:, -1]

        # reshape balik X ke bentuk asli
        X_unique = X_unique.reshape(-1, *X.shape[1:])

        print(f"Dataset awal: {len(X)} | Setelah buang duplikat: {len(X_unique)}")

        # --- Split train-test ---
        X_train, X_test, y_train, y_test = train_test_split(
            X_unique, y_unique, test_size=test_size, random_state=random_state, shuffle=True
        )

        # --- Split train-val ---
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=random_state, shuffle=True
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_unique, y_unique, test_size=test_size, random_state=random_state, shuffle=True
        )

        # --- Split train-val ---
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=random_state, shuffle=True
        )

    return X_train, X_val, X_test, y_train, y_val, y_test

def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size):
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size),
        DataLoader(test_dataset, batch_size=batch_size)
    )

def save_label_distribution(y_data, title, filename, encoder, result_dir):
    # Hitung distribusi
    label_counts = pd.Series(y_data).value_counts().sort_index()
    label_counts.index = label_counts.index.astype(int)  # pastiin int

    # Mapping index ke nama label
    if isinstance(encoder, dict):  # AAMI_MAP case
        idx2label = {v: k for k, v in encoder.items()}
        labels = [f"{i} ({idx2label[i]})" for i in label_counts.index]
    else:  # LabelEncoder case
        labels = [
            f"{i} ({lab})"
            for i, lab in zip(label_counts.index, encoder.inverse_transform(label_counts.index))
        ]

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(
        x=labels,
        y=label_counts.values,
        hue=labels,
        palette="viridis",
        legend=False
    )
    plt.title(title)
    plt.xlabel("Label (Encoded → Asli)")
    plt.ylabel("Jumlah Sampel")
    plt.xticks(rotation=45)

    # Tambahin angka di atas bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', label_type='edge')

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, filename))
    plt.close()

