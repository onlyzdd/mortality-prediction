# In-hospital Mortality Prediction on ICU Data

## Introduction

![](https://imgur.com/hjkOTJs.png)

- Goal: In-hospital mortality using demographics and first 48 hours ICU data after admission
- Evaluation: AUROC
  - Basline: Logistic Regression 0.82
  - Submission: CSV file with `adm_id,probability` as headers

## Data

### Data Description

- train (29953 patients, 972976 records)
  - train_labels.csv: training labels
  - train_demos.csv: training demographics
  - train_features.csv: training features, first 48 hours ICU data extracted, including vital signs and lab tests
- test (3329 patients, 108749 records)
  - test_labels.csv: test labels
  - test_demos.csv: test demographics
  - test_features.csv: test features, first 48 hours ICU data extracted, including vital signs and lab tests

### Data Remarks

- In each file, `adm_id` is the de-identified patient ID.
- In the label file, `1` on the `mortality` column means in-hospital mortality while `0` means discharge.
- In the feature file, `charttime` is the time that features are collected and is offset by $T_{adm}=0$.

### Submission

- AUROC is used as the evaluation metric.
- CSV file with `adm_id,probability` as headers, each line for patient ID and the probability of the patient's in-hospital mortality.
- Please make sure order of patients' IDs is the same as that in the label file.
- A sample submission file can be found at [test_submission.csv](./data/test_submission.csv).
