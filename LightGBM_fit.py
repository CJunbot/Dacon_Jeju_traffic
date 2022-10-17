import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

train = pd.read_parquet('data/train_after.parquet')
test = pd.read_parquet('data/test_after.parquet')

y = train['target']
x = train.drop(columns=['target'])

<<<<<<< HEAD
x2, x_test, y2, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x2, y2, test_size=0.1, random_state=42, shuffle=True)
d_train = lgb.Dataset(x_train, label=y_train,
                      categorical_feature=[
                                           'road_rating', 'connect_code', 'road_type',
                                           'day_of_week', 'start_turn_restricted', 'end_turn_restricted',
                                           'multi_linked', 'road_in_use'])

d_val = lgb.Dataset(x_val, label=y_val,
                      categorical_feature=[
                                           'road_rating', 'connect_code', 'road_type',
                                           'day_of_week', 'start_turn_restricted', 'end_turn_restricted',
                                           'multi_linked', 'road_in_use'])

# 'road_name', 'start_node_name', 'end_node_name',
=======
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

d_train = lgb.Dataset(x_train, label=y_train,
                      categorical_feature=['road_rating',
                                           'road_name', 'connect_code', 'road_type',
                                           'start_node_name', 'day_of_week',
                                           'start_turn_restricted', 'end_node_name', 'end_turn_restricted',
                                           'multi_linked', 'road_in_use'])
>>>>>>> 9ee693be1db224917e85fe1efd172bd7330b199f

params = {}
params['objective'] = 'regression'
params["verbose"] = 1
params['metric'] = 'mae'
params['device_type'] = 'gpu'
params['boosting_type'] = 'gbdt'
params['learning_rate'] = 0.00773607
params['sub_feature'] = 0.5
<<<<<<< HEAD
# 예측력 상승
params['num_iterations'] = 300  # = num round, num_boost_round
params['min_child_samples'] = 127
params['n_estimators'] = 4923
params['subsample'] = 0.8488291
params['num_leaves'] = 1034
params['max_depth'] = 34
# overfitting 방지
params['min_sum_hessian_in_leaf'] = 1e-2
params['min_data_in_leaf'] = 32
params['bagging_fraction'] = 1
params['bagging_freq'] = 20
params['feautre_fraction'] = 0.6
params['lambda_l1'] = 0.1
params['lambda_l2'] = 0.1
params['min_gain_to_split'] = 0.1

bst = lgb.train(params, d_train, valid_sets=[d_val], callbacks=[lgb.early_stopping(stopping_rounds=10)])
bst.save_model('model.txt', num_iteration=bst.best_iteration)
pred = bst.predict(x_test, num_iteration=bst.best_iteration)
accuracy = mean_absolute_error(y_test, pred)

print(accuracy)
if accuracy < 3.2:
    sample_submission = pd.read_csv('data/sample_submission.csv')
    sample_submission['target'] = pred
    sample_submission.to_csv("data/submit.csv", index=False)
=======
params['num_leaves'] = 1034
params['max_depth'] = 34
params['min_child_samples'] = 127
params['n_estimators'] = 4923
params['subsample'] = 00.8488291

bst = lgb.train(params, d_train)
bst.save_model('model4  .txt')
accuracy = mean_absolute_error(y_test, bst.predict(x_test))
print(accuracy)

>>>>>>> 9ee693be1db224917e85fe1efd172bd7330b199f
