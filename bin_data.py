import pandas as pd
import numpy as np

from datetime import timedelta


if __name__ == "__main__":
    labels = pd.read_csv('./data/preprocessed/all/labels.csv')
    adms = pd.read_csv('./data/mimic/admissions.csv', parse_dates=['admittime'])
    adm_ids = labels.adm_id.unique().tolist()
    df = labels[['adm_id']]
    for item in ['bg', 'lab', 'vital']:
        df_item = pd.read_csv('./data/mimic/pivoted_{}.csv'.format(item), parse_dates=['charttime'])
        df_item = df_item.merge(adms[['adm_id', 'admittime']], on='adm_id')
        df_item = df_item[df_item.adm_id.isin(adm_ids)]
        df_item['hr'] = (df_item.charttime - df_item.admittime) / np.timedelta64(1, 'h')
        df_item = df_item[(df_item.hr <= 48) & (df_item.hr > 0)]
        df_item = df_item.set_index('adm_id').groupby('adm_id').resample('H', on='charttime').mean().reset_index()
        df_item.to_csv('./data/preprocessed/all/{}.csv'.format(item), index=None)
    df = pd.read_csv('./data/preprocessed/all/vital.csv', parse_dates=['charttime'])[['adm_id', 'charttime', 'heartrate','sysbp','diasbp','meanbp','resprate','tempc','spo2']]
    df_lab = pd.read_csv('./data/preprocessed/all/lab.csv', parse_dates=['charttime'])
    df = df.merge(df_lab, on=['adm_id', 'charttime'], how='outer')
    df_bg = pd.read_csv('./data/preprocessed/all/bg.csv', parse_dates=['charttime'])[['adm_id', 'charttime', 'calcium', 'ph', 'pco2']]
    df = df.merge(df_bg, on=['adm_id', 'charttime'], how='outer')
    df = df.merge(adms[['adm_id', 'admittime']], on='adm_id')
    df['charttime'] = ((df.charttime - df.admittime) / np.timedelta64(1, 'h'))
    df['charttime'] = df['charttime'].apply(np.ceil) + 1
    df = df[(df.charttime <= 48) & (df.charttime >= 1)]
    df = df.sort_values(['adm_id', 'charttime'])
    df['charttime'] = df['charttime'].map(lambda x: int(x))
    df = df.drop(['admittime', 'hr'], axis=1)
    na_thres = len(df.columns) - 2
    df = df.dropna(thresh=3)
    df.to_csv('./data/preprocessed/all/features.csv', index=None)
    labels = labels[labels.adm_id.isin(df.adm_id)]
    labels.to_csv('./data/preprocessed/all/labels.csv', index=None)





