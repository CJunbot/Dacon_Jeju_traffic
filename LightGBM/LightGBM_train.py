import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from lightgbm import plot_importance
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

train = pd.read_parquet('../data/train_after.parquet')
test = pd.read_parquet('../data/test_after.parquet')

y = train['target']
x = train.drop(columns=['target'])

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

params = {}
params['objective'] = 'regression'
params["verbose"] = 1
params['metric'] = 'mae'
params['device_type'] = 'gpu'
params['boosting_type'] = 'gbdt'
params['learning_rate'] = 0.013119110575691373  # 0.013119로 고치면 댐
# 예측력 상승
params['num_iterations'] = 5000  # = num round, num_boost_round
params['min_child_samples'] = 118
params['n_estimators'] = 15918  # 8500
params['num_leaves'] = 7868
params['max_depth'] = 35  # 26?
# overfitting 방지
params['min_child_weight'] = 0.7628373492320147  # 높을수록 / 최대 6?
params['min_child_samples'] = 41  # 100 500 ?
params['subsample'] = 0.7611163934517731  # 낮을수록 overfitting down / 최소 0  = bagging_fraction
params['subsample_freq'] = 76
params['reg_alpha'] = 0.46641059279049957  # = lambda l1
params['reg_lambda'] = 0.30503746605875  # = lambda l2
params['min_gain_to_split'] = 0.05443147365335205  # = min_split_gain
params['colsample_bytree'] = 0.9009386979948221  # 낮을 수록 overfitting down / 최소 0  = feature_fraction

bst = lgb.LGBMRegressor(**params)
bst.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric='mae', early_stopping_rounds=25)
fig, ax = plt.subplots(figsize=(12,6))
plot_importance(bst, max_num_features=40, ax=ax)
plt.show()
pred = bst.predict(x_val, num_iteration=bst.best_iteration_)
MAE = mean_absolute_error(y_val, pred)
print('The MAE of prediction is:', MAE)

pred = bst.predict(test, num_iteration=bst.best_iteration_)

sample_submission = pd.read_csv('../data/sample_submission.csv')
sample_submission['target'] = pred
sample_submission.to_csv("../data/submit_lgbm.csv", index=False)

bst.booster_.save_model('model2.txt')
