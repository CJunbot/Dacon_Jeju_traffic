import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae

train = pd.read_parquet('data/train_after.parquet')

y = train['target']
x = train.drop(columns=['target'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
d_train = lgb.Dataset(x_train, label=y_train,
                      categorical_feature=['road_rating',
                                           'road_name', 'connect_code', 'road_type',
                                           'start_node_name',
                                           'start_turn_restricted', 'end_node_name', 'end_turn_restricted'])

params = {}
params['objective'] = 'regression'
params["verbose"] = -1
params['metric'] = 'mae'
params['device_type'] = 'gpu'
params['learning_rate'] = 0.007713208719461027
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'mae'
params['sub_feature'] = 0.5
params['num_leaves'] = 23
params['min_data'] = 50
params['max_depth'] = 11
params['min_child_samples'] = 76
params['n_estimators'] = 1546
params['subsample'] = 0.5348779873185086


bst = lgb.train(params, d_train, 100)

y_pred = bst.predict(x_test)

accuracy = mae(y_pred, y_test)
print(accuracy)