import pandas as pd
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae

# initialize data
train = pd.read_parquet('../data/train_cat.parquet')
y = train['target']
x = train.drop(columns=['target'])

x2, x_test, y2, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x2, y2, test_size=0.1, random_state=42, shuffle=True)

# initialize Pool
train_pool = Pool(x_train,
                  y_train,
                  cat_features=['day_of_week', 'road_name', 'start_node_name',
                                'start_turn_restricted', 'end_node_name', 'end_turn_restricted',
                                'road_types'])

val_pool = Pool(x_val, y_val,
                cat_features=['day_of_week', 'road_name', 'start_node_name',
                               'start_turn_restricted', 'end_node_name', 'end_turn_restricted',
                               'road_types'])

test_pool = Pool(x_test,
                 cat_features=['day_of_week', 'road_name', 'start_node_name',
                               'start_turn_restricted', 'end_node_name', 'end_turn_restricted',
                               'road_types'])

# specify the training parameters
cb_model = CatBoostRegressor(iterations=50000,
                             devices='GPU',
                             eval_metric='RMSE',
                             random_seed = 42,
                             od_type='Iter',
                             metric_period = 50,
                             od_wait=20)
#param['l2_leaf_reg'] = 2
#param['random_strength'] = 5

# train the model
cb_model.fit(train_pool,
             eval_set=(val_pool),
             use_best_model=True)

# make the prediction using the resulting model
cb_model.save_model('model.cbm')
y_pred = cb_model.predict(test_pool)
MAE = mae(y_pred, y_test)
print(MAE)
