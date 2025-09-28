import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
from pathlib import Path
import random


# --------- data loading (pickle with data['train']) ---------
def load_pickle_train(path):
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    X = np.asarray(data["train"]["images"], dtype=np.float32)  # (N,3,60,60)
    y = np.asarray(data["train"]["labels"], dtype=np.int64)

    # Normalize to [0,1] if needed
    if X.max() > 1.5:
        X /= 255.0

    # Ensure labels are contiguous [0..C-1]
    uniq = np.unique(y)
    if not np.array_equal(uniq, np.arange(uniq.size)):
        remap = {v: i for i, v in enumerate(uniq)}
        y = np.vectorize(remap.get)(y).astype(np.int64)
    return X, y


# --------- required function ---------
def tuso_model(train_data, train_label, val_data, device, batch_size=1024):
    """
    Train on (train_data, train_label), then return predicted labels for val_data.
    Expects:
      train_data: np.ndarray (N,3,60,60) float32 in [0,1]
      train_label: np.ndarray (N,) int64
      val_data: np.ndarray (M,3,60,60) float32 in [0,1]
    """

    return val_preds


# --------- helper: 80/20 split ---------
def split_train_val(X, y, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    idx = np.arange(N)
    rng.shuffle(idx)
    cut = int(N * (1 - val_ratio))
    tr_idx, va_idx = idx[:cut], idx[cut:]
    return X[tr_idx], y[tr_idx], X[va_idx], y[va_idx]


# --- Main script ---
if __name__ == "__main__":
    # Path to pickle with data['train']
    pickle_path = Path("nasbench360/s2_cifar100")  # change if needed

    # Seeds & device
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load & split
    X, y = load_pickle_train(pickle_path)
    train_X, train_y, val_X, val_y = split_train_val(X, y, val_ratio=0.2, seed=42)

    # Run model (must ONLY return val preds)
    print("tuso_model_start")
    val_preds = tuso_model(train_X, train_y, val_X, device)
    print("tuso_model_end")

    # Metric
    val_metric = f1_score(val_y, val_preds, average="macro")
    print(f"tuso_evaluate: {val_metric:.4f}")
