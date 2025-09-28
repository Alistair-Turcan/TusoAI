import numpy as np
import pandas as pd
import xgboost as xgb
from typing import List, Union

def tuso_model(all_data: pd.DataFrame, chromosomes: np.ndarray, seed: int, LOO_colname: str, predictors: List[str], index_cols: List[str]) -> pd.DataFrame:
    if 'snp_gene_distance_inverse' not in all_data.columns:
        all_data['snp_gene_distance_inverse'] = 1 / (all_data['snp_gene_distance'] + 1e-6)
    if 'snp_gene_distance_normalized' not in all_data.columns:
        all_data['snp_gene_distance_normalized'] = (all_data['snp_gene_distance'] - all_data['snp_gene_distance'].min()) / (all_data['snp_gene_distance'].max() - all_data['snp_gene_distance'].min())

    predictors += ['snp_gene_distance_inverse', 'snp_gene_distance_normalized']

    if 'snp_position' in all_data.columns and 'tss' in all_data.columns:
        all_data['snp_tss_distance'] = np.abs(all_data['snp_position'] - all_data['tss'])
        predictors += ['snp_tss_distance']

    if 'closest_tss' in all_data.columns:
        all_data['snp_closest_tss_distance'] = np.abs(all_data['snp_position'] - all_data['tss'] * all_data['closest_tss'])
        predictors += ['snp_closest_tss_distance']

    all_data['snp_gene_distance_squared'] = all_data['snp_gene_distance'] ** 2
    predictors += ['snp_gene_distance_squared']

    all_data['snp_gene_tss_interaction'] = all_data['snp_tss_distance'] * all_data['snp_gene_distance']
    predictors += ['snp_gene_tss_interaction']

    all_data['biological_relevance'] = (all_data['snp_gene_distance'] < 50000).astype(int)
    predictors += ['biological_relevance']

    predictions_dfs = []

    for chrom in chromosomes:
        train_mask = (all_data['train_include'] == 1) & (all_data[LOO_colname] != chrom)
        X_train = all_data.loc[train_mask, predictors].values
        y_train = all_data.loc[train_mask, 'positive'].values

        params = {
            "max_depth": 10,
            "eta": 0.01,
            "gamma": 10,
            "min_child_weight": 6,
            "subsample": 0.6,
            "scale_pos_weight": 1,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "nthread": 24,
            "seed": seed,
        }

        dtrain = xgb.DMatrix(X_train, label=y_train)
        bst = xgb.train(params, dtrain, num_boost_round=1000, early_stopping_rounds=50, evals=[(dtrain, 'train')], verbose_eval=False)

        pred_mask = (all_data[LOO_colname] == chrom)
        X_pred = all_data.loc[pred_mask, predictors].values
        pred_prob = bst.predict(xgb.DMatrix(X_pred))

        chrom_df = all_data.loc[pred_mask, index_cols].copy()
        chrom_df["pgBoost"] = pred_prob
        predictions_dfs.append(chrom_df)

    predictions = pd.concat(predictions_dfs, ignore_index=True)
    return predictions
