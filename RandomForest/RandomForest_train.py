import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae

train = pd.read_parquet('../data/train_rf.parquet')
test = pd.read_parquet('../data/test_rf.parquet')

y = train['target']
x = np.array(train.drop(columns=['target']))

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=42, shuffle=True)

params = {
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

score = model.predict(x_val)
print(f'MAE: {mae(y_val, x_val)}')

y_pred = model.predict(test)

sample_submission = pd.read_csv('../data/sample_submission.csv')
sample_submission['target'] = y_pred
sample_submission.to_csv("../data/submit_RF.csv", index=False)