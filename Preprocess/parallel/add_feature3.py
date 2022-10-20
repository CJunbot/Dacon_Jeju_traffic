import pandas as pd
from haversine import haversine

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 400)

train = pd.read_parquet('../../data/train.parquet')
test = pd.read_parquet('../../data/test.parquet')
bus_station = pd.read_csv('../../jeju_bus.csv', encoding='cp949')

train['start_bus_km'] = 0
train['end_bus_km'] = 0
for trains in range(800000, 1200000):
    start = (train['start_latitude'][trains], train['start_longitude'][trains])
    start2 = (train['end_latitude'][trains], train['end_longitude'][trains])
    min1, min2 = 1000, 1000
    print(trains)
    for bus in range(len(bus_station)):
        end = (bus_station['long'][bus], bus_station['lati'][bus])
        cash_start = haversine(start, end)
        cash_end = haversine(start2, end)
        if cash_start < min1:
            min1 = cash_start
        if cash_end < min2:
            min2 = cash_end
    train['start_bus_km'][trains] = min1
    train['end_bus_km'][trains] = min2

train.to_parquet('../../data/train_bus3.parquet', index=False)
test.to_parquet('../../data/test_bus3.parquet', index=False)