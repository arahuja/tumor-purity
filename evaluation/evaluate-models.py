import argparse
import numpy as np
import pytcga
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score


def load_purity_data(path='data/tcga-purity-measures.csv'):

    purity = pd.read_csv(path)
    purity.rename(columns={'CPE': 'Consensus'}, inplace=1)
    tcga_sample = purity['Sample ID'].str.rsplit('-', n=1, expand=True)
    tcga_sample.columns = ['TCGA ID', 'SampleType']
    purity = purity.join(tcga_sample)
    return purity

def load_rnaseq(cancer_codes):
    dfs = []
    for code in cancer_codes:
        data = pytcga.load_rnaseq_data(code)
        dfs.append(data)
    return pd.concat(dfs, copy=False)

def build_matrix(data, purity, target_variable='Consensus'):
    tcga_sample_gene = pd.pivot_table(data=data,
                  index=['TCGA_ID'],
                  values=['normalized_count'],
                  columns=['gene_name'])
    # Rename columns
    tcga_sample_gene.columns = tcga_sample_gene.columns.get_level_values(1)

    # Join with purity data
    tcga_sample_gene_purity = tcga_sample_gene.join(purity.set_index(['TCGA ID']))

    # Drop rows without a target measurement
    tcga_sample_gene_purity_val = tcga_sample_gene_purity[~tcga_sample_gene_purity[target_variable].isnull()]

    purity_cols = [col for col in purity.columns if col != "TCGA ID"]

    # Build an X and y matrix
    X = tcga_sample_gene_purity_val.drop(purity_cols, axis=1)
    y = np.array(tcga_sample_gene_purity_val[target_variable]).astype(float)
    return X, y


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate machine learning methods for tumor purity prediction')

    parser.add_argument('--cancer-codes', nargs='+', default=['LUAD'])
    parser.add_argument('--target-variable', default='Consensus', help='Purity target variable to predict.')
    parser.add_argument('--num_splits', default=5, type=int)

    args = parser.parse_args()

    models = [RidgeCV(), RandomForestRegressor(n_estimators=50), ]
    num_splits = args.num_splits

    purity = load_purity_data()
    cancer_codes = args.cancer_codes

    target_variable = args.target_variable
    X, y = build_matrix(load_rnaseq(cancer_codes),
                        purity,
                        target_variable=target_variable)

    for model in models:
        scores = cross_val_score(model, X, y, scoring='mean_absolute_error', cv=num_splits)
        print("Model type: {}, CV:{} , MAE: {}".format(model.__class__, num_splits, scores.mean()))