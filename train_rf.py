import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from lightgbm import LGBMClassifier


import warnings
warnings.filterwarnings('ignore')

np.random.seed(43)


def get_Xy(df_demos, df_vitals, df_labels):
    df_demos = pd.get_dummies(df_demos, drop_first=True)
    df_vitals = df_vitals.drop('charttime', axis=1).groupby(
        'hadm_id').agg(['mean', 'max', 'min'])
    df_vitals.columns = ['_'.join(col).strip()
                         for col in df_vitals.columns.values]
    df_vitals = df_vitals.reset_index()
    df = pd.merge(df_demos, df_vitals, on='hadm_id')
    df = df.merge(df_labels, on='hadm_id')
    df.fillna(0, inplace=True)
    X = df[df.columns[:-1]]
    y = df[['hadm_id', 'mortality']]
    print(len(X.columns))
    return X, y


def train(X, y):
    classifier = LGBMClassifier()
    classifier.fit(X, y)
    return classifier


def test(X, y, classifier):
    preds = classifier.predict(X)
    acc = accuracy_score(y, preds)
    probs = classifier.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs)
    print('AUC: {}, Accuracy: {}'.format(auc, acc))
    print(classification_report(y, preds))


if __name__ == "__main__":
    df_demos = pd.read_csv('./data/preprocessed/all/demos.csv')
    df_labels = pd.read_csv('./data/preprocessed/all/labels.csv')
    df_vitals = pd.read_csv('./data/preprocessed/all/vitals.csv')

    adm_ids = np.random.permutation(df_labels.hadm_id.tolist())
    train_size = int(len(adm_ids) * 0.8)
    train_ids, test_ids = adm_ids[:train_size], adm_ids[train_size:]

    X, y = get_Xy(df_demos, df_vitals, df_labels)
    columns = X.columns
    X_train, y_train = X[X.hadm_id.isin(train_ids)][columns[1:]], y[y.hadm_id.isin(train_ids)].mortality
    X_test, y_test = X[X.hadm_id.isin(test_ids)][columns[1:]], y[y.hadm_id.isin(test_ids)].mortality
    classifier = train(X_train, y_train)
    test(X_test, y_test, classifier)
