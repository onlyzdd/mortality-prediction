import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import random


np.random.seed(43)
random.seed(43)


def deidentify():
    adm_ids = pd.read_csv('./data/preprocessed/all/labels.csv')['adm_id'].tolist()
    radm_ids = np.random.permutation(adm_ids)
    df = pd.DataFrame.from_dict({'adm_id': adm_ids, 'adm_id': radm_ids})
    df.to_csv('./data/preprocessed/all/id_mapping.csv', index=None)


def gen_data():
    map_dict = pd.read_csv('./data/preprocessed/all/id_mapping.csv', index_col=0, squeeze=True).to_dict()
    adm_ids = pd.read_csv('./data/preprocessed/all/labels.csv')['adm_id'].tolist()
    train_ids, test_ids = train_test_split(adm_ids, test_size=0.1)
    for item in ['demos', 'labels', 'features']:
        df = pd.read_csv('./data/preprocessed/all/{}.csv'.format(item))
        train = df[df.adm_id.isin(train_ids)]
        test = df[df.adm_id.isin(test_ids)]
        train['adm_id'].replace(map_dict, inplace=True)
        test['adm_id'].replace(map_dict, inplace=True)
        train.to_csv('./data/preprocessed/train/train_{}.csv'.format(item), index=None)
        test.to_csv('./data/preprocessed/test/test_{}.csv'.format(item), index=None)


def gen_sumbission():
    df_labels = pd.read_csv('./data/preprocessed/test/test_labels.csv')
    df_labels.columns = ['adm_id', 'probability']
    df_labels['probability'] = df_labels['probability'].apply(lambda x: random.random())
    df_labels.to_csv('./data/test_submission.csv', index=None)


if __name__ == "__main__":
    features0 = pd.read_csv('./data/preprocessed/all/features.csv')
    adm_ids = features0.adm_id.unique().tolist()
    for item in ['demos', 'labels']:
        df = pd.read_csv('./data/preprocessed/all/{}.csv'.format(item))
        df = df[df.adm_id.isin(adm_ids)]
        df.to_csv('./data/preprocessed/all/{}.csv'.format(item), index=None)
    deidentify()
    gen_data()
    gen_sumbission()

