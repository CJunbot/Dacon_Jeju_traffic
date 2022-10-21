import pandas as pd
from haversine import haversine

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
# add feature
train.loc[(train['maximum_speed_limit'] <= 40), 'road_types'] = '인접도로'  # 인접 도로
train.loc[(train['maximum_speed_limit'] == 50), 'road_types'] = '도심부도로'  # 도심부 도로
train.loc[(train['maximum_speed_limit'] == 60) & (train['lane_count'] == 1), 'road_types'] = '도심부외도로'  # 도심부 외 도로
train.loc[(train['maximum_speed_limit'] == 60) & (train['lane_count'] >= 2), 'road_types'] = '도심부도로'  # 도심부 도로
train.loc[(train['maximum_speed_limit'] > 60), 'road_types'] = '도심부외도로'  # 도심부 외 도로

test.loc[(train['maximum_speed_limit'] <= 40), 'road_types'] = '인접도로'  # 인접 도로
test.loc[(train['maximum_speed_limit'] == 50), 'road_types'] = '도심부도로'  # 도심부 도로
test.loc[(train['maximum_speed_limit'] == 60) & (test['lane_count'] == 1), 'road_types'] = '도심부외도로'  # 도심부 외 도로
test.loc[(train['maximum_speed_limit'] == 60) & (test['lane_count'] >= 2), 'road_types'] = '도심부도로'  # 도심부 도로
test.loc[(train['maximum_speed_limit'] > 60), 'road_types'] = '도심부외도로'  # 도심부 외 도로

# 성수기 비수기 추가(7월 23일~ 8월 5일)
train.loc[(7 == train['month']) & (train['day'] >= 23), 'peak_season'] = '1'
train.loc[(8 == train['month']) & (train['day'] <= 5), 'peak_season'] = '1'
train['peak_season'] = train['peak_season'].fillna(0)
train['peak_season'] = train['peak_season'].astype(dtype='int64')

test.loc[(7 == test['month']) & (test['day'] >= 23), 'peak_season'] = '1'
test.loc[(8 == test['month']) & (test['day'] <= 5), 'peak_season'] = '1'
test['peak_season'] = test['peak_season'].fillna(0)
test['peak_season'] = test['peak_season'].astype(dtype='int64')

# 도로 연장 추가
train['km'] = 0
test['km'] = 0
for haver in range(len(train)):
    start = ((train['start_latitude'][haver]), (train['start_longitude'][haver]))  # (lat, lon)
    goal = ((train['end_latitude'][haver]), (train['end_longitude'][haver]))  # (lat, lon)
    train['km'][haver] = haversine(start, goal)

for haver2 in range(len(test)):
    start = ((test['start_latitude'][haver2]), (test['start_longitude'][haver2]))  # (lat, lon)
    goal = ((test['end_latitude'][haver2]), (test['end_longitude'][haver2]))  # (lat, lon)
    test['km'][haver2] = haversine(start, goal)

# drop cols
train.drop(columns=['id', 'base_date', 'height_restricted', 'vehicle_restricted', 'multi_linked', 'connect_code', 'road_in_use'], inplace=True)
test.drop(columns=['id', 'base_date', 'height_restricted', 'vehicle_restricted', 'multi_linked', 'connect_code', 'road_in_use'], inplace=True)

print(train.head(50))
print('\n')
print(test.head(50))

# save processed dataset to parquet
train.to_parquet('../data/train_cat.parquet', index=False)
test.to_parquet('../data/test_cat.parquet', index=False)
