import pandas as pd
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae

# initialize data
train = pd.read_parquet('../data/train_cat.parquet')
y = train['target']
x = train.drop(columns=['target'])

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

# initialize Pool
train_pool = Pool(x_train, y_train,
                  cat_features=['road_name', 'start_node_name', 'end_node_name',
                                'start_region_2', 'end_region_2'])
# day_of_week
val_pool = Pool(x_val, y_val,
                cat_features=['road_name', 'start_node_name', 'end_node_name',
                              'start_region_2', 'end_region_2'])

test_pool = Pool(x_val,
                     cat_features=['road_name', 'start_node_name', 'end_node_name',
                                   'start_region_2', 'end_region_2'])

# specify the training parameters
cb_model = CatBoostRegressor(
                             learning_rate=0.025,  # 0.025
                             depth=15,
                             n_estimators=10000,  # 10000 -> 1시간 반정도
                             bootstrap_type='Bernoulli',
                             devices='0:1',
                             task_type='GPU',
                             eval_metric='RMSE',
                             random_seed=42,
                             min_data_in_leaf=47,
                             l2_leaf_reg=0.8130860044896614,
                             subsample=0.9540988370165997,
                             metric_period=10)

# train the model
cb_model.fit(train_pool, eval_set=(val_pool), early_stopping_rounds=25, verbose=100, use_best_model=True)
y_pred = cb_model.predict(test_pool)
MAE = mae(y_val, y_pred)
print(MAE)
# make the prediction using the resulting model
cb_model.save_model('model.cbm')
