import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce

def extract_year(row):
    return int(str(row)[0:4])-2020

def extract_month(row):
    return int(str(row)[4:6])

def extract_day(row):
    return int(str(row)[6:])

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 400)

train = pd.read_parquet('data/train.parquet')
test = pd.read_parquet('data/test.parquet')

# Normalization
scaler = MinMaxScaler()
for col_name in ['start_latitude', 'start_longitude', 'end_latitude', 'end_longitude']:
    x, y = train[col_name].values.reshape(-1, 1), test[col_name].values.reshape(-1, 1)
    x_scaled, y_scaled = scaler.fit_transform(x), scaler.fit_transform(y)
    train[col_name], test[col_name] = x_scaled, y_scaled

# Categorical to Numerical
for index, embark in enumerate(['월', '화', '수', '목', '금', '토', '일']):
  train.loc[(train['day_of_week'] == embark), 'day_of_week'] = index
  test.loc[(test['day_of_week'] == embark), 'day_of_week'] = index

for index, embark in enumerate(['없음','있음']):
  train.loc[(train['start_turn_restricted'] == embark), 'start_turn_restricted'] = index
  test.loc[(test['start_turn_restricted'] == embark), 'start_turn_restricted'] = index
  train.loc[(train['end_turn_restricted'] == embark), 'end_turn_restricted'] = index
  test.loc[(test['end_turn_restricted'] == embark), 'end_turn_restricted'] = index

# Category Encoder
train['road_name'] = train['road_name'].replace('-', None)
test['road_name'] = test['road_name'].replace('-', None)

for name in ['road_name', 'start_node_name', 'end_node_name']:
    glmm_encoder = ce.CatBoostEncoder(cols=[name], handle_missing='return_nan')
    train[name] = glmm_encoder.fit_transform(train[name], train['target'])
    test[name] = glmm_encoder.transform(test[name])

# feature scaling
train['road_rating'] = train['road_rating']-100
test['road_rating'] = test['road_rating']-100
for name, divide in [('maximum_speed_limit', 10), ('weight_restricted', 10000)]:
    train[name] = train[name]/divide
    test[name] = test[name]/divide

# separate base date to year, month, day
train['year'] = train['base_date'].apply(extract_year)
train['month'] = train['base_date'].apply(extract_month)
train['day'] = train['base_date'].apply(extract_day)
test['year'] = test['base_date'].apply(extract_year)
test['month'] = test['base_date'].apply(extract_month)
test['day'] = test['base_date'].apply(extract_day)

# drop cols
train.drop(columns=['id', 'base_date', 'height_restricted', 'year', 'day', 'month', 'multi_linked', 'start_longitude',
                    'end_longitude', 'day_of_week', 'vehicle_restricted'], inplace=True)
test.drop(columns=['id', 'base_date', 'height_restricted', 'year', 'day', 'month', 'multi_linked', 'start_longitude',
                    'end_longitude', 'day_of_week', 'vehicle_restricted'], inplace=True)

print(train.head(50))
print('\n')
print(test.head(50))

# save processed dataset to parquet
train.to_parquet('data/train_after.parquet', index=False)
test.to_parquet('data/test_after.parquet', index=False)

