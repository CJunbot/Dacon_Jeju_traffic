import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def extract_year(row):
    return int(str(row)[0:4])

def extract_month(row):
    return int(str(row)[4:6])

def extract_day(row):
    return int(str(row)[6:])

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 400)

train = pd.read_parquet('../data/train.parquet')
test = pd.read_parquet('../data/test.parquet')

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
train.to_parquet('../data/train_cat.parquet', index=False)
test.to_parquet('../data/test_cat.parquet', index=False)
