from xgboost import XGBRegressor
import pandas as pd
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

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
    params['objective'] = 'reg:squaredlogerror'
    params['eval_metric'] = 'rmse'
    params['gpu_id'] = 1
    params['tree_method'] = 'gpu_hist'
    params['learning_rate'] = 0.043119110575691373  # 0.010 -> 2.884 / 0.028 -> 2.885 / 0.058 -> 2.893
    # 예측력 상승
    params['n_estimators'] = 17488  # 8500
    params['max_leaves'] = 10252  # 100~15000
    params['max_depth'] = 40  # 6~40?
    # overfitting 방지
    params['min_child_weight'] = 0.7628373492320147  # 높을수록 / 0~무한대 6?
    params['subsample'] = 0.9028492555488905  # 낮을수록 overfitting down / 0.5~1  = bagging_fraction
    params['reg_alpha'] = 0.2110959615938186  # = lambda l1
    params['reg_lambda'] = 0.43112763032873236  # = lambda l2
    params['min_split_loss'] = 0.011470395498020328  # = gamma
    params['colsample_bytree'] = 0.9997889205965167  # 낮을수록 overfitting down / 0~1  = feature_fraction
    params['colsample_bylevel'] = 0.9304625814875158
    params['colsample_bynode'] = 0.9126534090841069

    model = XGBRegressor(**params)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        early_stopping_rounds=25,
        verbose=100,
    )

    y_for_LR[val_idx] = model.predict(x_val, ntree_limit=model.best_ntree_limit)
    y_pred += model.predict(test, ntree_limit=model.best_ntree_limit)

y_pred /= n_splits

sample_submission = pd.read_csv('../data/sample_submission.csv')
sample_submission['target'] = y_pred
sample_submission.to_csv("../data/submit_xgb_fold.csv", index=False)

df = pd.DataFrame(y_for_LR)
df.to_csv('XGB_LR.csv', index=False)
