import pandas as pd
import category_encoders as ce

def extract_year(row):
    return int(str(row)[0:4])-2020

def extract_month(row):
    return int(str(row)[4:6])

def extract_day(row):
    return int(str(row)[6:])

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 600)

train = pd.read_parquet('../data/train.parquet')
test = pd.read_parquet('../data/test.parquet')

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
    cat_encoder = ce.CatBoostEncoder(cols=[name], handle_missing='return_nan')
    train[name] = cat_encoder.fit_transform(train[name], train['target'])
    test[name] = cat_encoder.transform(test[name])


# separate base date to year, month, day
train['year'] = train['base_date'].apply(extract_year)
train['month'] = train['base_date'].apply(extract_month)
train['day'] = train['base_date'].apply(extract_day)
test['year'] = test['base_date'].apply(extract_year)
test['month'] = test['base_date'].apply(extract_month)
test['day'] = test['base_date'].apply(extract_day)

# drop cols
train.drop(columns=['id', 'base_date', 'height_restricted', 'vehicle_restricted', 'road_in_use'], inplace=True)
test.drop(columns=['id', 'base_date', 'height_restricted', 'vehicle_restricted', 'road_in_use'], inplace=True)

print(train.head(50))
print('\n')
print(test.head(50))

# save processed dataset to parquet
train.to_parquet('../data/train_after.parquet', index=False)
test.to_parquet('../data/test_after.parquet', index=False)

