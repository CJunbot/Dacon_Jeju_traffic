import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

train = pd.read_parquet('data/train_after.parquet')

y = train['target']
x = train.drop(columns=['target'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

d_train = lgb.Dataset(x_train, label=y_train,
                      categorical_feature=['road_rating',
                                           'road_name', 'connect_code', 'road_type',
                                           'start_node_name', 'day_of_week',
                                           'start_turn_restricted', 'end_node_name', 'end_turn_restricted',
                                           'multi_linked', 'road_in_use'])

params = {}
params['objective'] = 'regression'
params["verbose"] = -1
params['metric'] = 'mae'
params['device_type'] = 'gpu'
params['boosting_type'] = 'gbdt'
params['learning_rate'] = 0.00773607
params['sub_feature'] = 0.5
params['num_leaves'] = 1034
params['max_depth'] = 34
params['min_child_samples'] = 127
params['n_estimators'] = 4923
params['subsample'] = 00.8488291

bst = lgb.train(params, d_train)
bst.save_model('model4  .txt')
accuracy = mean_absolute_error(y_test, bst.predict(x_test))
print(accuracy)

