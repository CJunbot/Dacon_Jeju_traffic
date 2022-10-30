from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import StratifiedKFold

train = pd.read_parquet('../data/train_after.parquet')
test = pd.read_parquet('../data/test_after.parquet')

y = train['target']
x = np.array(train.drop(columns=['target']))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random.seed(42)
y_pred = np.zeros(len(test))

for train_idx, valid_idx in skf.split(x,y):
    x_train, x_val = x[train_idx], x[valid_idx]
    y_train, y_val = y[train_idx], y[valid_idx]

    params = {}
    params['objective'] = 'regression'
    params["verbose"] = -1
    params['metric'] = 'MAE'
    params['device_type'] = 'gpu'
    params['boosting_type'] = 'gbdt'
    params['learning_rate'] = 0.05136291374083868
    # 예측력 상승
    params['num_iterations'] = 5000  # = num round, num_boost_round
    params['min_child_samples'] = 110
    params['n_estimators'] = 5999  # 8500
    params['subsample'] = 0.8488291
    params['num_leaves'] = 5533
    params['max_depth'] = 35  # 26?
    # overfitting 방지
    params['min_child_weight'] = 0.4325  # 높을수록 / 최대 6?
    params['min_child_samples'] = 35  # 100 500 ?
    params['bagging_fraction'] = 0.8  # 낮을수록 overfitting down / 최소 0
    params['subsample_freq'] = 60
    params['lambda_l1'] = 0.1
    params['lambda_l2'] = 0.1
    params['min_gain_to_split'] = 0.1
    params['feature_fraction'] = 0.90288  # 낮을수록 overfitting down / 최소 0

    bst = LGBMRegressor(**params)
    bst.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric='MAE', early_stopping_rounds=25)

    y_pred += bst.predict(test, num_iteration=bst.best_iteration_)

y_pred /= 5

sample_submission = pd.read_csv('../data/sample_submission.csv')
sample_submission['target'] = y_pred
sample_submission.to_csv("../data/submit_stratify_fold.csv", index=False)
