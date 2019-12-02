import pandas as pd
from sklearn.metrics import roc_auc_score


if __name__ == '__main__':
    y_true = pd.read_csv('./data/preprocessed/test/test_labels.csv')['mortality'].tolist()
    y_probs = pd.read_csv('./data/test_submission.csv')['probability'].tolist()
    print(roc_auc_score(y_true, y_probs))
