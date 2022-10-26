import pandas as pd
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import train_test_split

# initialize data
train = pd.read_parquet('../data/train_cat.parquet')
y = train['target']
x = train.drop(columns=['target'])

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

# initialize Pool
train_pool = Pool(x_train, y_train,
                  cat_features=['road_name', 'start_node_name', 'end_node_name',
                                'start_region_2', ' end_region_2'])
# day_of_week
val_pool = Pool(x_val, y_val,
                cat_features=['road_name', 'start_node_name', 'end_node_name',
                              'start_region_2', ' end_region_2'])

# specify the training parameters
cb_model = CatBoostRegressor(learning_rate=0.04,
                             n_estimators=3000,
                             max_depth=8,
                             devices='GPU',
                             eval_metric='RMSE',
                             random_seed=42,
                             metric_period=50)
#param['l2_leaf_reg'] = 2
#param['random_strength'] = 5

# train the model
cb_model.fit(train_pool, eval_set=(val_pool), early_stopping_rounds=30, verbose=100, use_best_model=True)

# make the prediction using the resulting model
cb_model.save_model('model.cbm')
