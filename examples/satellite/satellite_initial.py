import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
import random

def tuso_model(train_data, train_label, val_data, device, batch_size=1032):

    return y_pred_labels

# --- Main script ---
if __name__ == "__main__":
    # Load .npy
    train_data = np.load("nasbench360/aide_satellite/train_data.npy", allow_pickle=True)
    train_label = np.load("nasbench360/aide_satellite/train_label.npy", allow_pickle=True)
    val_data   = np.load("nasbench360/aide_satellite/val_data.npy",   allow_pickle=True)
    val_label  = np.load("nasbench360/aide_satellite/val_label.npy",  allow_pickle=True)

    # Unwrap object scalars
    if train_data.shape == (): train_data = train_data.item()
    if train_label.shape == (): train_label = train_label.item()
    if val_data.shape   == (): val_data   = val_data.item()
    if val_label.shape  == (): val_label  = val_label.item()

    # Normalize types/shapes early
    train_data = np.asarray(train_data, dtype=np.float32).reshape(-1, 46)
    val_data   = np.asarray(val_data,   dtype=np.float32).reshape(-1, 46)
    train_label = np.asarray(train_label, dtype=np.int64).reshape(-1)
    val_label   = np.asarray(val_label,   dtype=np.int64).reshape(-1)

    # Basic sanity checks
    assert train_data.ndim == 2 and train_data.shape[1] == 46
    assert val_data.ndim   == 2 and val_data.shape[1]   == 46
    assert train_label.ndim == 1 and val_label.ndim == 1
    assert train_label.min() >= 1 and train_label.max() <= 24, "Labels must be 1..24"

    # Convert training labels to zero-based (0..23) so CrossEntropyLoss is happy.
    train_label_zero = train_label - 1

    # Repro + cudnn stability
    np.random.seed(42); random.seed(42)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("tuso_model_start")
    # Pass zero-based labels into the function
    raw_preds = tuso_model(train_data, train_label_zero, val_data, device)
    print("tuso_model_end")

    # Normalize outputs to 1..24 (tolerate either 0- or 1-based from the LLM)
    raw_preds = np.asarray(raw_preds)
    if raw_preds.ndim != 1 or len(raw_preds) != len(val_data):
        raise ValueError("tuso_model must return a 1D list/array of length len(val_data).")

    if raw_preds.min() >= 1 and raw_preds.max() <= 24:
        val_preds = raw_preds.astype(int)  # already 1..24
    elif raw_preds.min() >= 0 and raw_preds.max() <= 23:
        val_preds = (raw_preds + 1).astype(int)  # convert 0..23 -> 1..24
    else:
        raise ValueError("Predictions must be class ids in 0..23 or 1..24.")

    # --- Assess accuracy ---
    val_metric = accuracy_score(val_label, val_preds)
    print(f"tuso_evaluate: {val_metric}")
