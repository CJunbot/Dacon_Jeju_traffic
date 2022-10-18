import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

train = pd.read_parquet('../data/train_after.parquet')
test = pd.read_parquet('../data/test_after.parquet')

y = train['target']
x = train.drop(columns=['target'])

x2, x_test, y2, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x2, y2, test_size=0.1, random_state=42, shuffle=True)

params = {}
params['objective'] = 'regression'
params["verbose"] = 1
params['metric'] = 'l1'
params['device_type'] = 'gpu'
params['boosting_type'] = 'gbdt'
params['learning_rate'] = 0.1
# 예측력 상승
params['num_iterations'] = 300  # = num round, num_boost_round
params['min_child_samples'] = 120
params['n_estimators'] = 5999
params['subsample'] = 0.8488291
params['num_leaves'] = 1644
params['max_depth'] = 26
# overfitting 방지
params['min_child_weight'] = 1e-2
params['min_child_samples'] = 32
params['subsample_freq'] = 20
params['feature_fraction'] = 0.5
params['min_split_gain'] = 0.1

bst = lgb.LGBMRegressor(**params)
bst.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric='l1', early_stopping_rounds=5)
pred = bst.predict(x_test, num_iteration=bst.best_iteration_)
MAE = mean_absolute_error(y_test, pred)
print('The MAE of prediction is:', MAE)

if MAE < 3:
    ans = np.expm1(bst.predict(test, num_iteration=bst.best_iteration_))
    sample_submission = pd.read_csv('../data/sample_submission.csv')
    sample_submission['target'] = ans
    sample_submission.to_csv("../data/submit.csv", index=False)
