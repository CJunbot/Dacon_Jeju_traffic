import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


train = pd.read_parquet('../data/train_after.parquet')
test = pd.read_parquet('../data/test_after.parquet')

y = train['target']
x = np.array(train.drop(columns=['target']))

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits = folds.split(x, y)
y_pred = np.zeros(len(test))

for fold, (train_idx, valid_idx) in enumerate(splits):
    print(f"============ Fold {fold} ============\n")
    x_train, x_val = x[train_idx], x[valid_idx]
    y_train, y_val = y[train_idx], y[valid_idx]

    params = {
        "criterion": "gini",
        "n_estimators": 300,
        "min_samples_split": 10,
        "min_samples_leaf": 2,
        "max_features": "auto",
        "oob_score": True,
        "random_state": 42,
        "n_jobs": -1,
    }

    model = RandomForestRegressor(**params)
    model.fit(
        x_train,
        y_train,
    )

    y_pred += model.predict(x_val)

y_pred /= 5

sample_submission = pd.read_csv('../data/sample_submission.csv')
sample_submission['target'] = y_pred
sample_submission.to_csv("../data/submit_RF.csv", index=False)
