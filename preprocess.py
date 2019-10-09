import pandas as pd
import numpy as np
from datetime import timedelta


def get_labels():
    adm = pd.read_csv('./data/mimic/admissions.csv')
    icu_details = pd.read_csv('./data/mimic/icustay_details.csv', parse_dates=['admittime'])
    adm_ids = adm[adm['has_chartevents_data'] == 1].hadm_id
    icu_details = icu_details[icu_details.hadm_id.isin(adm_ids)]
    icu_details = icu_details[
        (icu_details.los_hospital >= 1) &
        (icu_details.age >= 18) &
        (icu_details.los_icu >= 1)
    ]
    df = icu_details.groupby('hadm_id').first().reset_index()
    df = df.groupby('subject_id').last().reset_index()
    df = df[['hadm_id', 'admittime', 'mortality']]
    for item in ['vital']:
        df_item = pd.read_csv('./data/mimic/pivoted_{}.csv'.format(item), parse_dates=['charttime'])
        columns = df_item.columns
        df_item = df_item.merge(df, on='hadm_id')
        df_item.charttime = (df_item.charttime - df_item.admittime) / np.timedelta64(1, 'h')
        df_item = df_item[
            (df_item.charttime > 0) &
            (df_item.charttime < 48)
        ]
        df = df[df.hadm_id.isin(df_item.hadm_id)]
    df[['hadm_id', 'mortality']].sort_values('hadm_id').to_csv(
        './data/preprocessed/all/labels.csv', index=None)


def get_demographics():
    adm = pd.read_csv('./data/mimic/admissions.csv')
    icu_details = pd.read_csv('./data/mimic/icustay_details.csv')
    labels = pd.read_csv('./data/preprocessed/all/labels.csv')[['hadm_id']]
    demos = pd.merge(labels, icu_details[['hadm_id', 'gender', 'age']], on='hadm_id', how='left')
    demos = pd.merge(demos, adm[['hadm_id', 'admission_type', 'admission_location',
                              'insurance', 'marital_status', 'ethnicity']], on='hadm_id', how='left')
    demos = demos.drop_duplicates()
    demos.to_csv('./data/preprocessed/all/demos.csv', index=None)


def get_features():
    adm = pd.read_csv('./data/mimic/admissions.csv', parse_dates=['admittime'])[['hadm_id', 'admittime']]
    labels = pd.read_csv('./data/preprocessed/all/labels.csv')[['hadm_id']]
    adm = adm[adm.hadm_id.isin(labels.hadm_id)]
    for feature in ['vital', 'lab', 'bg']:
        df = pd.read_csv('./data/mimic/pivoted_{}.csv'.format(feature), parse_dates=['charttime'])
        columns = df.columns
        df = df.merge(adm, on='hadm_id')
        df.charttime = (df.charttime - df.admittime) / np.timedelta64(1,'h')
        df = df[df.charttime < 48]
        df = df[columns]
        print(len(df.hadm_id.unique().tolist()))
        df.to_csv('./data/preprocessed/all/{}s.csv'.format(feature), index=None)
    

if __name__ == "__main__":
    # get_labels()
    get_demographics()
    # get_features()

