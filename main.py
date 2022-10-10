import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train = pd.read_parquet('data/train_after.parquet')

y = train['target']
x = train.drop(columns=['target'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
d_train = lgb.Dataset(x_train, label=y_train,
                      categorical_feature=['road_in_use', 'road_rating',
                                           'road_name', 'connect_code', 'road_type',
                                           'start_node_name',
                                           'start_turn_restricted', 'end_node_name', 'end_turn_restricted'])

params = {}
params['objective'] = 'regression'
params["verbose"] = -1
params['metric'] = 'mse'
params['device_type'] = 'gpu'
params['learning_rate'] = 0.001
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'mse'
params['sub_feature'] = 0.5
params['num_leaves'] = 20
params['min_data'] = 50
params['max_depth'] = 10

bst = lgb.train(params, d_train, 1000)

y_pred = bst.predict(x_test)

accuracy = mean_squared_error(y_pred, y_test)
print(accuracy)
