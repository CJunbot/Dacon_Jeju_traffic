import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

train = pd.read_parquet('../data/train_address.parquet')
test = pd.read_parquet('../data/test_address.parquet')

csv = pd.read_csv('../Bus_station_per_region.csv', encoding='ANSI')
bus_region = csv['region3']

for indexx, region in enumerate(bus_region):
    train.loc[(train['start_region_2'] == region), 'bus_station_start'] = csv['count'][indexx]
    train.loc[(train['start_region_3'] == region), 'bus_station_start'] = csv['count'][indexx]
    train.loc[(train['end_region_2'] == region), 'bus_station_end'] = csv['count'][indexx]
    train.loc[(train['end_region_3'] == region), 'bus_station_end'] = csv['count'][indexx]
train['bus_station_start'] = train['bus_station_start'].fillna(0)
train['bus_station_end'] = train['bus_station_end'].fillna(0)
train['bus_station_start'] = train['bus_station_start'].astype(dtype='int64')
train['bus_station_end'] = train['bus_station_end'].astype(dtype='int64')

for indexx, region in enumerate(bus_region):
    test.loc[(test['start_region_2'] == region), 'bus_station_start'] = csv['count'][indexx]
    test.loc[(test['start_region_3'] == region), 'bus_station_start'] = csv['count'][indexx]
    test.loc[(test['end_region_2'] == region), 'bus_station_end'] = csv['count'][indexx]
    test.loc[(test['end_region_3'] == region), 'bus_station_end'] = csv['count'][indexx]
test['bus_station_start'] = test['bus_station_start'].fillna(0)
test['bus_station_end'] = test['bus_station_end'].fillna(0)
test['bus_station_start'] = test['bus_station_start'].astype(dtype='int64')
test['bus_station_end'] = test['bus_station_end'].astype(dtype='int64')

print(train.head(50))
print('\n')
print(test.head(50))
# save processed dataset to parquet
train.to_parquet('../data/train_address_test.parquet', index=False)
test.to_parquet('../data/test_address_test.parquet', index=False)
