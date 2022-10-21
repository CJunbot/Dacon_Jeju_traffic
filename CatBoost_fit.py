import pandas as pd
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae

# initialize data
train = pd.read_parquet('data/train_cat.parquet')
y = train['target']
x = train.drop(columns=['target'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)


# initialize Pool
train_pool = Pool(x_train,
                  y_train,
                  cat_features=['day_of_week', 'road_name', 'start_node_name',
                                'start_turn_restricted', 'end_node_name', 'end_turn_restricted',
                                'road_types'])

test_pool = Pool(x_test,
                 cat_features=['day_of_week', 'road_name', 'start_node_name',
                               'start_turn_restricted', 'end_node_name', 'end_turn_restricted',
                               'road_types'])

# specify the training parameters
param = {}
param['eval_metric'] = 'MAE'
#param['task_type'] = 'GPU'
param['iterations'] = 20000
#param['learning_rate'] = 0.05
param['od_type'] = 'Iter'
param['od_wait'] = 25
#param['depth'] = 4
#param['l2_leaf_reg'] = 2
#param['random_strength'] = 5
model = CatBoostRegressor(**param)

# train the model
model.fit(train_pool)
# make the prediction using the resulting model
model.save_model('model.cbm')
y_pred = model.predict(test_pool)
MAE = mae(y_pred, y_test)
print(MAE)
