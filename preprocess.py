import pandas as pd
from datetime import timedelta


def get_labels():
    adm = pd.read_csv('./data/mimic/admissions.csv')
    icu_details = pd.read_csv('./data/mimic/icustay_details.csv')
    adm_ids = adm[adm['has_chartevents_data'] == 1].hadm_id
    icu_details = icu_details[icu_details.hadm_id.isin(adm_ids)]
    icu_details = icu_details[
        (icu_details.los_hospital > 2) &
        (icu_details.age >= 18) &
        (icu_details.los_icu > 2)
    ]
    df = icu_details.groupby('hadm_id').first().reset_index()
    df = df.groupby('subject_id').last().reset_index()
    df[['hadm_id', 'mortality']].sort_values('hadm_id').to_csv(
        './data/preprocessed/all/labels.csv', index=None)


def get_demographics():
    adm = pd.read_csv('./data/mimic/admissions.csv')
    icu_details = pd.read_csv('./data/mimic/icustay_details.csv')
    labels = pd.read_csv('./data/preprocessed/all/labels.csv')[['hadm_id']]
    demos = pd.merge(labels, icu_details[['hadm_id', 'gender', 'age']], on='hadm_id', how='left')
    demos = pd.merge(demos, adm[['hadm_id', 'admission_type', 'admission_location',
                              'insurance', 'marital_status', 'ethnicity']], on='hadm_id', how='left')
    demos.to_csv('./data/preprocessed/all/demos.csv', index=None)


def get_features():
    pass


if __name__ == "__main__":
    get_labels()
    get_demographics()
