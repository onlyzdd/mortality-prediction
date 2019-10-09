import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


import warnings
warnings.filterwarnings('ignore')

np.random.seed(43)


def get_Xy(df_demos0, df_vitals0, df_labels0, ids):
    df_labels = df_labels0[df_labels0.hadm_id.isin(ids)]
    df_vitals = df_vitals0[df_vitals0.hadm_id.isin(ids)]
    df_demos = df_demos0[df_demos0.hadm_id.isin(ids)]
    df_demos = pd.get_dummies(df_demos, drop_first=True)
    df_vitals = df_vitals.drop('charttime', axis=1).groupby(
        'hadm_id').agg(['mean', 'max', 'min'])
    df_vitals.columns = ['_'.join(col).strip()
                         for col in df_vitals.columns.values]
    df_vitals = df_vitals.reset_index()
    print(len(df_vitals))
    df = pd.merge(df_demos, df_vitals, on='hadm_id')
    df = df.merge(df_labels, on='hadm_id')
    df.fillna(0, inplace=True)
    X = df[df.columns[:-1]]
    y = df['mortality']
    # print(X)

    return X, y


if __name__ == "__main__":
    df_demos = pd.read_csv('./data/preprocessed/train/train_demos.csv')
    df_labels = pd.read_csv('./data/preprocessed/train/train_labels.csv')
    df_vitals = pd.read_csv('./data/preprocessed/train/train_vitals.csv')

    adm_ids = np.random.permutation(df_labels.hadm_id.tolist())
    train_size = int(len(adm_ids) * 0.8)
    train_ids, test_ids = adm_ids[:train_size], adm_ids[train_size:]

    X_train, y_train = get_Xy(df_demos, df_vitals, df_labels, train_ids)
    X_test, y_test = get_Xy(df_demos, df_vitals, df_labels, test_ids)
