import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import StratifiedKFold
import networkx as nx

def tuso_model(adata):
    spatial_coords = adata.obsm["spatial"]
    gene_expression = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    num_genes = gene_expression.shape[1]

    spatial_var_score = np.zeros(num_genes)

    quantile_transformer = QuantileTransformer()
    scaled_gene_expression = StandardScaler(with_mean=False).fit_transform(quantile_transformer.fit_transform(gene_expression))

    G = nx.Graph()
    G.add_nodes_from(range(spatial_coords.shape[0]))
    for i in range(spatial_coords.shape[0]):
        distances = np.linalg.norm(spatial_coords[i] - spatial_coords, axis=1)
        nearest_indices = np.argsort(distances)[1:11]
        G.add_edges_from((i, j, {'weight': distances[j]}) for j in nearest_indices)

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for gene_idx in range(num_genes):
        scores = []
        for train_index, test_index in skf.split(spatial_coords, adata.obs["annotation"]):
            ridge = RidgeCV(alphas=np.logspace(-3, 3, 7))
            ridge.fit(spatial_coords[train_index], scaled_gene_expression[train_index, gene_idx])
            neighbor_exp = np.array([np.mean(scaled_gene_expression[list(G.neighbors(i)), gene_idx]) for i in test_index])
            scores.append(np.std(neighbor_exp))

        spatial_var_score[gene_idx] = 1 if np.mean(scores) > 0.5 else 0

    adata.var['pred_spatial_var_score'] = spatial_var_score
    return adata
