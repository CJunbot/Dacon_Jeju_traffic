from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import random
import os


os.environ['PYTHONHASHSEED'] = str(42)

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
    params['learning_rate'] = 0.00901211683  # 0.013119로 고치면 댐
    # 예측력 상승
    params['num_iterations'] = 7000  # = num round, num_boost_round
    params['min_child_samples'] = 137
    params['n_estimators'] = 17260  # 8500
    params['num_leaves'] = 11431
    params['max_depth'] = 35  # 26?
    # overfitting 방지
    params['min_child_weight'] = 1.84548676996724  # 높을수록 / 최대 6?
    params['min_child_samples'] = 37  # 100 500 ?
    params['subsample'] = 0.785612320383  # 낮을수록 overfitting down / 최소 0  = bagging_fraction
    params['subsample_freq'] = 75
    params['reg_alpha'] = 1.37792417179  # = lambda l1
    params['reg_lambda'] = 1.3180813428729763  # = lambda l2
    params['min_gain_to_split'] = 0.0426731612843898  # = min_split_gain
    params['colsample_bytree'] = 0.7485850157993732  # 낮을 수록 overfitting down / 최소 0  = feature_fraction
    params['seed'] = random.seed(42)
    bst = LGBMRegressor(**params)
    bst.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric='MAE', early_stopping_rounds=25)

    # for Ensemble LR
    y_for_LR[val_idx] = bst.predict(x_val, num_iteration=bst.best_iteration_)

    y_pred += bst.predict(test, num_iteration=bst.best_iteration_)

y_pred /= n_splits

sample_submission = pd.read_csv('../data/sample_submission.csv')
sample_submission['target'] = y_pred
sample_submission.to_csv("../data/submit_LGBM_fold2.csv", index=False)

df = pd.DataFrame(y_for_LR)
df.to_parquet('LGBM_LR2.parquet', index=False)
