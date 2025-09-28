import scanpy as sc
import pandas as pd
import numpy as np
import scipy as sp
from anndata import read_h5ad
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import issparse
import random
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

def evaluate(adata):
    """
    Compute the mean of [AUC_macro, AUC_micro, F1_macro, F1_micro]
    for binary 0/1 labels stored in:
        - adata.var['true_spatial_var_score']
        - adata.var['pred_spatial_var_score']

    Notes
    -----
    - AUC is computed by treating hard labels as scores (0/1), which
      is mathematically valid but less informative than using probabilities.
    - If AUC is undefined (e.g., only one class present in y_true),
      it's set to NaN and ignored in the mean.
    """
    if 'true_spatial_var_score' not in adata.var or 'pred_spatial_var_score' not in adata.var:
        raise KeyError("adata.var must contain 'true_spatial_var_score' and 'pred_spatial_var_score'.")

    y_true = adata.var['true_spatial_var_score']
    y_pred = adata.var['pred_spatial_var_score']

    # Drop rows with NaN in either column
    mask = y_true.notna() & y_pred.notna()
    y_true = y_true[mask].astype(int).to_numpy()
    y_pred = y_pred[mask].astype(int).to_numpy()

    if y_true.size == 0:
        raise ValueError("No valid rows after dropping NaNs.")

    # Ensure binary {0,1}
    if not set(np.unique(y_true)).issubset({0, 1}) or not set(np.unique(y_pred)).issubset({0, 1}):
        raise ValueError("Both columns must be binary (0/1).")

    # F1 scores
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_micro = f1_score(y_true, y_pred, average="micro")

    # AUC (macro/micro) via one-vs-rest on a 2-column one-hot representation
    # Using hard labels as 'scores' (0/1); valid but coarse.
    auc_macro = np.nan
    auc_micro = np.nan
    try:
        y_true_oh = np.column_stack((1 - y_true, y_true))
        y_pred_oh = np.column_stack((1 - y_pred, y_pred))
        # Requires at least two classes in y_true; otherwise ValueError
        auc_macro = roc_auc_score(y_true_oh, y_pred_oh, average="macro")
        auc_micro = roc_auc_score(y_true_oh, y_pred_oh, average="micro")
    except ValueError:
        # e.g., only one class present in y_true
        pass

    # Mean across available metrics (ignores NaNs)
    metrics = [auc_macro, auc_micro, f1_macro, f1_micro]
    return float(np.nanmean(metrics))

def tuso_model(adata):

    adata.var['pred_spatial_var_score'] = ...
    return adata


def main():
    # Load data
    adata = read_h5ad("openproblems_datasets/stereo_drosophila_e5_6_sim.h5ad")
    adata.X = adata.layers["counts"]

    # Set seeds
    random.seed(42)
    np.random.seed(42)

    # Run model
    print("tuso_model_start")
    adata = tuso_model(adata)  # assumes tuso_model modifies/returns AnnData
    print("tuso_model_end")

    # Evaluate
    val_metric = evaluate(adata)  # assumes this function exists
    print(f"tuso_evaluate: {val_metric}")

if __name__ == "__main__":
    main()
