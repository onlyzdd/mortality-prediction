import pandas as pd
import numpy as np
from datetime import timedelta


def get_labels():
    adm = pd.read_csv('./data/mimic/admissions.csv')
    icu_details = pd.read_csv('./data/mimic/icustay_details.csv', parse_dates=['admittime'])
    adm_ids = adm[adm['has_chartevents_data'] == 1].adm_id
    icu_details = icu_details[icu_details.adm_id.isin(adm_ids)]
    icu_details = icu_details[
        (icu_details.los_hospital >= 1) &
        (icu_details.age >= 18) &
        (icu_details.los_icu >= 1)
    ]
    df = icu_details.groupby('adm_id').first().reset_index()
    df = df.groupby('subject_id').last().reset_index()
    df = df[['adm_id', 'admittime', 'mortality']]
    df[['adm_id', 'mortality']].sort_values('adm_id').to_csv(
        './data/preprocessed/all/labels.csv', index=None)


def get_demographics():
    adm = pd.read_csv('./data/mimic/admissions.csv')
    icu_details = pd.read_csv('./data/mimic/icustay_details.csv')
    labels = pd.read_csv('./data/preprocessed/all/labels.csv')[['adm_id']]
    demos = pd.merge(labels, icu_details[['adm_id', 'gender', 'age']], on='adm_id', how='left')
    demos = pd.merge(demos, adm[['adm_id', 'admission_type', 'admission_location',
                              'insurance', 'marital_status', 'ethnicity']], on='adm_id', how='left')
    demos = demos.drop_duplicates()
    demos.age = pd.qcut(demos.age, 5, ['very-young', 'young', 'middle', 'old', 'very-old'])
    demos.marital_status = demos.marital_status.fillna(value='UNKNOWN')
    demos.to_csv('./data/preprocessed/all/demos.csv', index=None)


def get_features():
    adm = pd.read_csv('./data/mimic/admissions.csv', parse_dates=['admittime'])[['adm_id', 'admittime']]
    labels = pd.read_csv('./data/preprocessed/all/labels.csv')[['adm_id']]
    adm = adm[adm.adm_id.isin(labels.adm_id)]
    for feature in ['vital', 'lab', 'bg']:
        df = pd.read_csv('./data/mimic/pivoted_{}.csv'.format(feature), parse_dates=['charttime'])
        columns = df.columns
        df = df.merge(adm, on='adm_id')
        df.charttime = (df.charttime - df.admittime) / np.timedelta64(1,'h')
        df = df[df.charttime < 48]
        df = df[columns]
        print(len(df.adm_id.unique().tolist()))
        df.to_csv('./data/preprocessed/all/{}s.csv'.format(feature), index=None)
    

if __name__ == "__main__":
    get_labels()
    get_demographics()
    # get_features()

