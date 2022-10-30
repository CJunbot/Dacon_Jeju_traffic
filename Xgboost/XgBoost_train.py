from xgboost import XGBRegressor
import pandas as pd
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import time

train = pd.read_parquet('../data/train_after.parquet')

y = train['target']
x = train.drop(columns=['target'])

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

params = {}
params['objective'] = 'reg:squaredlogerror'
params['eval_metric'] = 'rmse'
params['gpu_id'] = 1
params['tree_method'] = 'gpu_hist'
params['learning_rate'] = 0.053119110575691373  # 0.010 -> 2.884 / 0.028 -> 2.885 / 0.058 -> 2.893
# 예측력 상승
params['n_estimators'] = 15918  # 8500
params['max_leaves'] = 150  # 100~15000
params['max_depth'] = 35  # 6~40?
# overfitting 방지
params['min_child_weight'] = 0.7628373492320147  # 높을수록 / 0~무한대 6?
params['subsample'] = 0.7611163934517731  # 낮을수록 overfitting down / 0.5~1  = bagging_fraction
params['reg_alpha'] = 0.46641059279049957  # = lambda l1
params['reg_lambda'] = 0.30503746605875  # = lambda l2
params['min_split_loss'] = 0.05443147365335205  # = gamma
params['colsample_bytree'] = 0.9009386979948221  # 낮을수록 overfitting down / 0~1  = feature_fraction

start = time.time()
model = XGBRegressor(**params)
model.fit(
    x_train,
    y_train,
    eval_set=[(x_val, y_val)],
    early_stopping_rounds=25,
    verbose=10,
)

pred = model.predict(x_val, ntree_limit=model.best_ntree_limit)
mae = mean_absolute_error(y_val, pred)
print(f'MAE: {mae}')
print(f'걸린시간: {(time.time()-start)//60}분')
fig, ax = plt.subplots(figsize=(12,6))
plot_importance(model, max_num_features=8, ax=ax)
plt.show()
