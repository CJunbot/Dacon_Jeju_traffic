import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

train = pd.read_parquet('../data/train_after.parquet')
test = pd.read_parquet('../data/test_after.parquet')

y = train['target']
x = np.array(train.drop(columns=['target']))

n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

y_pred = np.zeros(len(test))

# KFold 사용
for tr_idx, val_idx in kf.split(x):
    x_train, x_val = x[tr_idx], x[val_idx]
    y_train, y_val = y[tr_idx], y[val_idx]

    params = {
        "criterion": "mae",
        "n_estimators": 300,
        "min_samples_split": 10,  # 8~
        "max_depth": 10,  # 6~100
        "min_samples_leaf": 2,  # 8~
        "max_leaf_nodes": 3,
        "min_weight_fraction_leaf": 0.01,
        "min_impurity_decrease": 0.01,
        "max_features": "auto",  # sqrt
        "bootstrap": True,  # False
        "oob_score": True,  # False
        "random_state": 42,
        "n_jobs": -1,
    }

    model = RandomForestRegressor(**params)
    model.fit(x_train, y_train)

    y_pred += model.predict(test)

y_pred /= 5

sample_submission = pd.read_csv('../data/sample_submission.csv')
sample_submission['target'] = y_pred
sample_submission.to_csv("../data/submit_RF.csv", index=False)