import numpy as np

def tuso_model(adata, gene_list, gene_weight=None, log=True):
    gene_list = list(gene_list)
    gene_weight = list(gene_weight) if gene_weight is not None else None

    df_gene = adata.uns["SCDRS_PARAM"]["GENE_STATS"]
    v_score_weight = 1 / np.sqrt(df_gene.loc[gene_list, "var_tech"].values + 1e-2)

    if gene_weight is not None:
        v_score_weight = v_score_weight * np.array(gene_weight)

    v_score_weight = v_score_weight / v_score_weight.sum()

    if log:
        log_expression = np.log1p(adata[:, gene_list].X)
        v_raw_score = log_expression.dot(v_score_weight).reshape([-1])
    else:
        v_raw_score = adata[:, gene_list].X.dot(v_score_weight).reshape([-1])

    return v_raw_score, v_score_weight
