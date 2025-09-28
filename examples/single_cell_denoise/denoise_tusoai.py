import scanpy as sc
import pandas as pd
import numpy as np
import scipy as sp
import magic
from anndata import read_h5ad
import scprep
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from anndata import AnnData
import random

def mse(adata):
    import anndata
    import scanpy as sc
    import scprep
    import sklearn.metrics

    test_data = anndata.AnnData(X=adata.obsm["test"], obs=adata.obs, var=adata.var)
    denoised_data = anndata.AnnData(
        X=adata.obsm["denoised"], obs=adata.obs, var=adata.var
    )

    # scaling and transformation
    target_sum = 10000

    sc.pp.normalize_total(test_data, target_sum=target_sum)
    sc.pp.log1p(test_data)

    sc.pp.normalize_total(denoised_data, target_sum=target_sum)
    sc.pp.log1p(denoised_data)

    error = sklearn.metrics.mean_squared_error(
        scprep.utils.toarray(test_data.X), denoised_data.X
    )
    return error
import numpy as np
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix
from anndata import AnnData

def tuso_model(adata):
    a = AnnData(
        X=adata.obsm["train"].copy(),
        obs=adata.obs.copy(),
        var=adata.var.copy()
    )
    
    data = a.X
    if isinstance(data, csr_matrix):
        data = data.toarray()
    
    noise_model = np.random.poisson(data)  
    dropout_rate = np.clip(np.mean(noise_model == 0, axis=0), 0.01, 0.99)
    
    rank = min(np.linalg.matrix_rank(data), 20)
    model = NMF(n_components=rank, init='nndsvd', random_state=0)
    
    for _ in range(5):
        W = model.fit_transform(noise_model)
        H = model.components_
        denoised_data = np.dot(W, H)
        denoised_data *= (1 - dropout_rate)
        noise_model = np.random.poisson(denoised_data)

    adata.obsm["denoised"] = denoised_data
    
    return adata
def main():
    np.random.seed(42)
    random.seed(42)
    adata = read_h5ad('openproblems_datasets/1k_pbmc_processed.h5ad')
    print("tuso_model_start")
    adata = tuso_model(adata)
    print("tuso_model_end")

    val_metric = 1-mse(adata)
    print(f"tuso_evaluate: {val_metric}")

main()
