from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

train = pd.read_parquet('../data/train_after.parquet')
test = pd.read_parquet('../data/test_after.parquet')

y = train['target']
x = np.array(train.drop(columns=['target']))

n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

y_for_LR = np.zeros(len(train))
y_pred = np.zeros(len(test))

# StratifiedKFold 사용
for tr_idx, val_idx in kf.split(x):
    x_train, x_val = x[tr_idx], x[val_idx]
    y_train, y_val = y[tr_idx], y[val_idx]

    params = {}
    params['objective'] = 'regression'
    params["verbose"] = 1
    params['metric'] = 'mae'
    params['device_type'] = 'gpu'
    params['boosting_type'] = 'gbdt'
    params['learning_rate'] = 0.013119110575691373  # 0.010 -> 2.884 / 0.028 -> 2.885 / 0.058 -> 2.893
    # 예측력 상승
    params['num_iterations'] = 5000  # = num round, num_boost_round
    params['min_child_samples'] = 118
    params['n_estimators'] = 15918  # 8500
    params['subsample'] = 0.6194512025053622
    params['num_leaves'] = 7868
    params['max_depth'] = 35  # 26?
    # overfitting 방지
    params['min_child_weight'] = 0.7628373492320147  # 높을수록 / 최대 6?
    params['min_child_samples'] = 41  # 100 500 ?
    params['subsample'] = 0.7611163934517731  # 낮을수록 overfitting down / 최소 0  = bagging_fraction
    params['subsample_freq'] = 76
    params['reg_alpha'] = 0.46641059279049957  # = lambda l1
    params['reg_lambda'] = 0.30503746605875  # =dlsw lambda l2
    params['min_gain_to_split'] = 0.05443147365335205  # = min_split_gain
    params['colsample_bytree'] = 0.9009386979948221  # 낮을수록 overfitting down / 최소 0  = feature_fraction

    bst = LGBMRegressor(**params)
    bst.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric='MAE', early_stopping_rounds=25)
    y_for_LR[val_idx] = bst.predict(x_val)
    y_pred += bst.predict(test, num_iteration=bst.best_iteration_)

y_pred /= n_splits

sample_submission = pd.read_csv('../data/sample_submission.csv')
sample_submission['target'] = y_pred
sample_submission.to_csv("../data/submit_fold.csv", index=False)

df = pd.DataFrame(y_for_LR)
df.to_csv('cat_LR.csv', index=False)