import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier


import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


def get_Xy(df_demos, df_vitals, df_labels):
    df_vitals = df_vitals.drop('charttime', axis=1).groupby(
        'adm_id').agg(['mean'])
    df_vitals.columns = ['_'.join(col).strip()
                         for col in df_vitals.columns.values]
    df_vitals = df_vitals.reset_index()
    df = pd.merge(df_demos, df_vitals, on='adm_id')
    df = df.merge(df_labels, on='adm_id')
    df.fillna(0, inplace=True)
    X = df[df.columns[:-1]]
    y = df[['adm_id', 'mortality']]
    return X, y


def train(X, y):
    classifier = LogisticRegressionCV(cv=5)
    # parameters = {'n_estimators': [x for x in range(20, 60, 10)],
    #         'learning_rate': [0.05, 0.1, 0.125]}
    # classifier = GridSearchCV(model, parameters, cv=5)
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
    df_demos = pd.read_csv('./data/preprocessed/train/train_demos.csv')
    df_labels = pd.read_csv('./data/preprocessed/train/train_labels.csv')
    df_vitals = pd.read_csv('./data/preprocessed/train/train_features.csv')

    df_demos2 = pd.read_csv('./data/preprocessed/test/test_demos.csv')
    df_labels2 = pd.read_csv('./data/preprocessed/test/test_labels.csv')
    df_vitals2 = pd.read_csv('./data/preprocessed/test/test_features.csv')

    adm_ids = np.random.permutation(df_labels.adm_id.tolist())
    adm_ids2 = df_labels2.adm_id.tolist()
    train_size = int(len(adm_ids) * 0.8)
    train_ids, test_ids = adm_ids[:train_size], adm_ids[train_size:]

    df_demos = pd.concat([df_demos, df_demos2])
    df_demos = pd.get_dummies(df_demos, drop_first=True)

    df_demos2 = df_demos[df_demos.adm_id.isin(adm_ids2)]
    df_demos = df_demos[df_demos.adm_id.isin(adm_ids)]
    # df_demos2 = df_demos[df_demos.adm_id.isin(adm_ids)]

    X, y = get_Xy(df_demos, df_vitals, df_labels)
    columns = X.columns
    X_train, y_train = X[X.adm_id.isin(train_ids)][columns[1:]], y[y.adm_id.isin(train_ids)].mortality
    X_test, y_test = X[X.adm_id.isin(test_ids)][columns[1:]], y[y.adm_id.isin(test_ids)].mortality
    print(X_train.columns)
    classifier = train(X_train, y_train)
    test(X_test, y_test, classifier)

    X_test2, y_test2 = get_Xy(df_demos2, df_vitals2, df_labels2)
    X_test2 = X_test2[columns[1:]]
    y_test2 = y_test2.mortality
    test(X_test2, y_test2, classifier)
