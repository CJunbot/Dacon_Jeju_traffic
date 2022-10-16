import pandas as pd
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae

# initialize data
train = pd.read_parquet('data/train_cat.parquet')
y = train['target']
x = train.drop(columns=['target'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# initialize Pool
train_pool = Pool(x_train,
                  y_train,
                  cat_features=['day_of_week', 'road_name', 'road_rating', 'connect_code', 'road_type', 'start_node_name',
                                'start_turn_restricted', 'end_node_name', 'end_turn_restricted'])

test_pool = Pool(x_test,
                 cat_features=['day_of_week', 'road_name', 'road_rating', 'connect_code', 'road_type',
                               'start_node_name',
                               'start_turn_restricted', 'end_node_name', 'end_turn_restricted'])

# specify the training parameters
model = CatBoostRegressor(iterations=2,
                          depth=2,
                          learning_rate=1,
                          loss_function='MAE')

#train the model
model.fit(train_pool)
# make the prediction using the resulting model
y_pred = model.predict(test_pool)
accuracy = mae(y_pred, y_test)
print(accuracy)
