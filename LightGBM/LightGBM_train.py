import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

train = pd.read_parquet('../data/train_after.parquet')

y = train['target']
x = train.drop(columns=['target'])

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

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

bst = lgb.LGBMRegressor(**params)
bst.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric='mae', early_stopping_rounds=25)
#pred = bst.predict(x_test, num_iteration=bst.best_iteration_)
#MAE = mean_absolute_error(y_test, pred)
#print('The MAE of prediction is:', MAE)
bst.booster_.save_model('model2.txt')
