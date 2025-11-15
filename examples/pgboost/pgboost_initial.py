import numpy as np
import pandas as pd
import xgboost as xgb
from typing import List, Union

def tuso_model(all_data: pd.DataFrame, chromosomes: np.ndarray, seed: int, LOO_colname: str, predictors: List[str], index_cols: List[str]) -> pd.DataFrame:
    predictions_dfs = []

    for chrom in chromosomes:
        train_mask = (all_data['train_include'] == 1) & (all_data[LOO_colname] != chrom)
        X_train = all_data.loc[train_mask, predictors].values
        y_train = all_data.loc[train_mask, 'positive'].values

        params = {
            "max_depth": 10,
            "eta": 0.05,
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
        bst = xgb.train(params, dtrain, num_boost_round=1000, verbose_eval=False)

        pred_mask = (all_data[LOO_colname] == chrom)
        X_pred = all_data.loc[pred_mask, predictors].values
        pred_prob = bst.predict(xgb.DMatrix(X_pred))

        chrom_df = all_data.loc[pred_mask, index_cols].copy()
        chrom_df["pgBoost"] = pred_prob
        predictions_dfs.append(chrom_df)

    predictions = pd.concat(predictions_dfs, ignore_index=True)
    return predictions
def enrichment_recall(df, column, 
                      min_recall=0.0, max_recall=1.0, extrapolate=False):
    tmp = df[[column, 'gold']].copy()
    tmp = tmp.sort_values(by=column, ascending=False).reset_index(drop=True)
    tmp['linked'] = 1 * (tmp[column] > 0)
    tmp['linked_cum'] = tmp['linked'].cumsum()
    tmp['gold_cum'] = tmp['gold'].cumsum()
    tmp['recall'] = tmp['gold_cum'] / tmp['gold_cum'].max()
    enrich_denom = tmp['gold_cum'].max() / len(tmp)
    tmp['enrichment'] = (tmp['gold_cum'] / tmp['linked_cum']) / enrich_denom
    unscored = tmp[tmp[column] == -1].reset_index(drop=True)
    tmp = tmp[tmp[column] > 0][['recall', 'enrichment']]
    
    if extrapolate and len(unscored) > 0 and unscored['gold'].sum() > 0:
        last_point = tmp.iloc[-1]
        last_recall, last_enrichment = last_point['recall'], last_point['enrichment']
        num_new_points = int(unscored['gold'].sum())
        recall_increment = (1 - last_recall) / num_new_points
        enrichment_increment = (last_enrichment - 1) / num_new_points
        new_recall = [last_recall + recall_increment * (i + 1) for i in range(num_new_points)]
        new_enrichment = [last_enrichment - enrichment_increment * i for i in range(num_new_points)]
        extrapolated_er = pd.DataFrame({'recall': new_recall, 'enrichment': new_enrichment})
        tmp = pd.concat([tmp, extrapolated_er])
    
    tmp = tmp[(tmp['recall'] <= max_recall) & (tmp['recall'] >= min_recall)]
    return tmp[['recall', 'enrichment']].drop_duplicates(subset='recall', keep='first')
def auerc(df, column, 
          min_recall=0.01, max_recall=1.0, extrapolate=False, weight=False):
    er = enrichment_recall(df, column, min_recall, max_recall, extrapolate=extrapolate)
    if weight:
        return np.average(er['enrichment'], weights=1 - er['recall'])
    else:
        return np.average(er['enrichment'])
def evaluate(predictions):

    abc_orig = 74.71814009731844
    eqtl_orig = 19.20851701419809
    crispr_orig = 7.393605510898142
    gwas_orig = 6.57356782483718

    orig_vals = [eqtl_orig, abc_orig, gwas_orig, crispr_orig]
    evals = ['eqtl', 'abc']  # , 'gwas', 'crispr'

    new_vals = []
    for i, eval in enumerate(evals):
        gold_file = f'pgboost/evaluation_sets/{eval}_evaluation.tsv'
        gold = pd.read_csv(gold_file, sep='\t')

        if eval == 'crispr':
            merged = predictions.merge(gold, on=['snp', 'gene'], how='inner')
        else:
            merged = predictions.merge(gold, on=['snp', 'gene'], how='left')
            if eval == 'gwas':
                # Restrict universe to SNPs with at least one gold-standard link
                gold_snps = gold['snp'].unique()
                merged = merged[merged['snp'].isin(gold_snps)]

        merged = merged.fillna(0)  # Negative evaluation links are specified as 0
        merged['gold'] = (1 * merged['gold'])  # Ensure integer
        print('%s candidate links: %s positives, %s negatives' % (
            len(merged),
            merged.value_counts('gold')[1],
            merged.value_counts('gold')[0]
        ))
        result = auerc(merged, 'pgBoost', extrapolate=True, weight=True)
        ratio = result / orig_vals[i]
        new_vals.append(ratio)
        print(result)

    mean_val = float(np.mean(new_vals))

    return mean_val
    
predictor_file = 'pgboost/features_files/predictors.txt'
data_file = 'pgboost/all_data.csv'
LOO_colname = 'chr'
seed = 511

all_data = pd.read_csv(data_file, index_col=0)

index_cols = ['snp', 'gene']

predictors = pd.read_table(predictor_file, header=None)[0].tolist()

chromosomes = pd.unique(all_data[LOO_colname])

print("tuso_model_start")
predictions = tuso_model(
    all_data=all_data,
    chromosomes=chromosomes,
    seed=seed,
    LOO_colname=LOO_colname,
    predictors=predictors,
    index_cols=index_cols
)
print("tuso_model_end")
predictions = all_data.merge(predictions, on=index_cols, how='left').fillna(-1)
tuso_evaluate = evaluate(predictions)
print(f"tuso_evaluate: {tuso_evaluate}")
