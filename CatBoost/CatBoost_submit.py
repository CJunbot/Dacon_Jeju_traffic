import pandas as pd
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae

train = pd.read_parquet('../data/train_cat.parquet')
test = pd.read_parquet('../data/test_cat.parquet')

y = train['target']
x = train.drop(columns=['target'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

val_pool = Pool(x_test,
                 cat_features=['day_of_week', 'road_name', 'start_node_name',
                               'start_turn_restricted', 'end_node_name', 'end_turn_restricted',
                               'road_types'])

test_pool = Pool(test,
                 cat_features=['day_of_week', 'road_name', 'start_node_name',
                               'start_turn_restricted', 'end_node_name', 'end_turn_restricted',
                               'road_types'])

cb_model = CatBoostRegressor()      # parameters not required.
cb_model.load_model('model.cbm')
y_pred = cb_model.predict(val_pool)
MAE = mae(y_pred, y_test)
print(MAE)

pred = cb_model.predict(test_pool)
sample_submission = pd.read_csv('../data/sample_submission.csv')
sample_submission['target'] = pred
sample_submission.to_csv("../data/submit2.csv", index=False)
