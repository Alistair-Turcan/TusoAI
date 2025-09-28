import scanpy as sc
from anndata import read_h5ad
import pandas as pd
import numpy as np
import scipy as sp
import os
import time
import argparse
from statsmodels.stats.multitest import multipletests
import scdrs
from tqdm import tqdm
import anndata
from typing import List, Dict, Tuple
from sklearn.metrics import average_precision_score

def convert_species_name(species):
    if species in ["Mouse", "mouse", "Mus_musculus", "mus_musculus", "mmusculus"]:
        return "mmusculus"
    if species in ["Human", "human", "Homo_sapiens", "homo_sapiens", "hsapiens"]:
        return "hsapiens"
    raise ValueError("# compute_score: species name %s not supported" % species)

def tuso_model(adata, gene_list, gene_weight):

    gene_list = list(gene_list)
    gene_weight = list(gene_weight)

    df_gene = adata.uns["SCDRS_PARAM"]["GENE_STATS"]
    v_score_weight = 1 / np.sqrt(df_gene.loc[gene_list, "var_tech"].values + 1e-2)

    if gene_weight is not None:
        v_score_weight = v_score_weight * np.array(gene_weight)
    v_score_weight = v_score_weight / v_score_weight.sum()
    v_raw_score = adata[:, gene_list].X.dot(v_score_weight).reshape([-1])

    return v_raw_score, v_score_weight

from scdrs_method import _select_ctrl_geneset, _correct_background, _get_p_from_empi_null

def score_cell(
    data,
    gene_list,
    gene_weight=None,
    ctrl_match_key="mean_var",
    n_ctrl=1000,
    n_genebin=200,
    weight_opt="vs",
    copy=False,
    return_ctrl_raw_score=False,
    return_ctrl_norm_score=False,
    random_seed=0,
    verbose=False,
    save_intermediate=None,
):


    np.random.seed(random_seed)
    adata = data.copy() if copy else data
    n_cell, n_gene = adata.shape

    gene_stats_set_expect = {"mean", "var", "var_tech"}
    gene_stats_set = set(adata.uns["SCDRS_PARAM"]["GENE_STATS"])

    # Load parameters
    flag_sparse = adata.uns["SCDRS_PARAM"]["FLAG_SPARSE"]
    flag_cov = adata.uns["SCDRS_PARAM"]["FLAG_COV"]

    df_gene = adata.uns["SCDRS_PARAM"]["GENE_STATS"].loc[adata.var_names].copy()
    df_gene["gene"] = df_gene.index
    df_gene.drop_duplicates(subset="gene", inplace=True)

    gene_list = list(gene_list)
    if gene_weight is not None:
        gene_weight = list(gene_weight)
    else:
        gene_weight = [1] * len(gene_list)

    # Overlap gene_list with df_gene["gene"]
    dic_gene_weight = {x: y for x, y in zip(gene_list, gene_weight)}
    gene_list = sorted(set(gene_list) & set(df_gene["gene"]))
    gene_weight = [dic_gene_weight[x] for x in gene_list]

    # Select control gene sets
    dic_ctrl_list, dic_ctrl_weight = _select_ctrl_geneset(
        df_gene, gene_list, gene_weight, ctrl_match_key, n_ctrl, n_genebin, random_seed
    )

    # Compute raw scores
    print("tuso_model_start")
    v_raw_score, v_score_weight = tuso_model(
        adata, gene_list, gene_weight
    )
    print("tuso_model_end")

    mat_ctrl_raw_score = np.zeros([n_cell, n_ctrl])
    mat_ctrl_weight = np.zeros([len(gene_list), n_ctrl])
    for i_ctrl in range(n_ctrl):
        v_ctrl_raw_score, v_ctrl_weight = tuso_model(
            adata, dic_ctrl_list[i_ctrl], dic_ctrl_weight[i_ctrl]
        )
        mat_ctrl_raw_score[:, i_ctrl] = v_ctrl_raw_score
        mat_ctrl_weight[:, i_ctrl] = v_ctrl_weight

    # Compute normalized scores
    v_var_ratio_c2t = np.ones(n_ctrl)
    if (ctrl_match_key == "mean_var") & (weight_opt in ["uniform", "vs", "inv_std"]):
        # For mean_var matched control genes and raw scores computed as weighted average,
        # estimate variance ratio assuming independence.
        for i_ctrl in range(n_ctrl):
            v_var_ratio_c2t[i_ctrl] = (
                df_gene.loc[dic_ctrl_list[i_ctrl], "var"]
                * mat_ctrl_weight[:, i_ctrl] ** 2
            ).sum()
        v_var_ratio_c2t /= (df_gene.loc[gene_list, "var"] * v_score_weight ** 2).sum()

    v_norm_score, mat_ctrl_norm_score = _correct_background(
        v_raw_score,
        mat_ctrl_raw_score,
        v_var_ratio_c2t,
        save_intermediate=save_intermediate,
    )

    # Get p-values
    mc_p = (1 + (mat_ctrl_norm_score.T >= v_norm_score).sum(axis=0)) / (1 + n_ctrl)
    pooled_p = _get_p_from_empi_null(v_norm_score, mat_ctrl_norm_score.flatten())
    nlog10_pooled_p = -np.log10(pooled_p)
    pooled_z = -sp.stats.norm.ppf(pooled_p).clip(min=-10, max=10)

    # Return result
    dic_res = {
        "raw_score": v_raw_score,
        "norm_score": v_norm_score,
        "mc_pval": mc_p,
        "pval": pooled_p,
        "nlog10_pval": nlog10_pooled_p,
        "zscore": pooled_z,
    }
    if return_ctrl_raw_score:
        for i in range(n_ctrl):
            dic_res["ctrl_raw_score_%d" % i] = mat_ctrl_raw_score[:, i]
    if return_ctrl_norm_score:
        for i in range(n_ctrl):
            dic_res["ctrl_norm_score_%d" % i] = mat_ctrl_norm_score[:, i]
    df_res = pd.DataFrame(index=adata.obs.index, data=dic_res, dtype=np.float32)
    return df_res

def eval_cell_level(df_res, adata, alpha=0.1):
    # BH adjust
    q = multipletests(df_res["pval"].values, method="fdr_bh")[1]
    df = df_res.copy()
    df["qval"] = q
    df["pred_pos"] = df["qval"] <= alpha

    # Ground truth aligned
    gt = adata.obs.loc[df.index, "ground_truth"].astype(bool).values
    pred = df["pred_pos"].values

    TP = int(np.sum(pred &  gt))
    FP = int(np.sum(pred & ~gt))
    TN = int(np.sum(~pred & ~gt))
    FN = int(np.sum(~pred &  gt))
    N  = TP + FP + TN + FN

    # Metrics
    power = TP / (TP + FN) if (TP + FN) > 0 else np.nan          # recall
    fdr   = FP / (TP + FP) if (TP + FP) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    accuracy  = (TP + TN) / N if N > 0 else np.nan

    # F1
    F1 = 0.0 if (precision + power) == 0 else 2 * precision * power / (precision + power)

    # AUPRC using p-values as scores (lower p = stronger ? higher score via -log10)
    eps = 1e-300
    scores = -np.log10(df["pval"].values + eps)
    try:
        auprc = average_precision_score(gt.astype(int), scores)
    except ValueError:
        # Degenerate case (e.g., all-True or all-False); fall back to 0.0
        auprc = 0.0

    combined = 0.5 * (auprc + F1)

    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN, "N": N,
        "power": power, "fdr": fdr, "precision": precision, "accuracy": accuracy,
        "F1": F1, "AUPRC": auprc, "combined": combined
    }, df

H5AD_SPECIES = 'mmusculus'
GS_SPECIES = 'mmusculus'

atlas_scores = []

for es in [0.15]:
    for overlap in [0.25]:
        H5AD_FILE = f"scdrs_sims/adata/TMS_FACS_10k_{overlap}_{es}.h5ad"
        print(H5AD_FILE)
        adata = read_h5ad(H5AD_FILE)
        
        sc.pp.filter_cells(adata, min_genes=250)
        sc.pp.filter_genes(adata, min_cells=50)
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
        sc.pp.log1p(adata)
        
        scdrs.preprocess(adata, cov=None, n_mean_bin=20, n_var_bin=20, copy=False)
        

        for i in range(3):
            trait = f'sim_{i}_{overlap}_{es}'
            GS_FILE = f'scdrs_sims/gs/{trait}.gs'
            print(trait)
            
            dict_gs = scdrs.util.load_gs(
                GS_FILE,
                src_species=GS_SPECIES,
                dst_species=H5AD_SPECIES,
                to_intersect=adata.var_names,
            )
            gene_list, gene_weights = dict_gs[trait]
            
            
            df_res = score_cell(
                adata,
                gene_list,
                gene_weight=gene_weights,
                ctrl_match_key='mean_var',
                n_ctrl=1000,
                weight_opt='vs',
                return_ctrl_raw_score=False,
                return_ctrl_norm_score=False,
                verbose=False,
            )
    
            metrics, df_with_q = eval_cell_level(df_res, adata, alpha=0.1)
            print(f"[es={es}, overlap={overlap}, trait={trait}] "
                  f"power={metrics['power']:.3f}, FDR={metrics['fdr']:.3f}, "
                  f"F1={metrics['F1']:.3f}, AUPRC={metrics['AUPRC']:.3f}, "
                  f"combined={(metrics['combined']):.3f}")
            atlas_scores.append(metrics['combined'])

print(f"tuso_evaluate:", np.mean(atlas_scores))
