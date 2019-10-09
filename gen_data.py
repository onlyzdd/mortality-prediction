import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import random


np.random.seed(43)
random.seed(43)


def deidentify():
    adm_ids = pd.read_csv('./data/preprocessed/all/labels.csv')['hadm_id'].tolist()
    radm_ids = np.random.permutation(adm_ids)
    df = pd.DataFrame.from_dict({'hadm_id': adm_ids, 'adm_id': radm_ids})
    df.to_csv('./data/preprocessed/all/id_mapping.csv', index=None)


def gen_data():
    map_dict = pd.read_csv('./data/preprocessed/all/id_mapping.csv', index_col=0, squeeze=True).to_dict()
    adm_ids = pd.read_csv('./data/preprocessed/all/labels.csv')['hadm_id'].tolist()
    train_ids, test_ids = train_test_split(adm_ids, test_size=0.2)
    for item in ['bgs', 'demos', 'labels', 'labs', 'vitals']:
        df = pd.read_csv('./data/preprocessed/all/{}.csv'.format(item))
        train = df[df.hadm_id.isin(train_ids)]
        test = df[df.hadm_id.isin(test_ids)]
        train['hadm_id'].replace(map_dict, inplace=True)
        test['hadm_id'].replace(map_dict, inplace=True)
        train.to_csv('./data/preprocessed/train/train_{}.csv'.format(item), index=None)
        test.to_csv('./data/preprocessed/test/test_{}.csv'.format(item), index=None)


if __name__ == "__main__":
    # deidentify()
    gen_data()

