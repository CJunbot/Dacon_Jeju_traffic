import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 600)

train = pd.read_parquet('../data/train.parquet')
test = pd.read_parquet('../data/test.parquet')

train.loc[(train['maximum_speed_limit'] <= 40), 'test'] = '인접 도로'
train.loc[(train['maximum_speed_limit'] == 50), 'test'] = '도심부 도로'
train.loc[(train['maximum_speed_limit'] == 60) & (train['lane_count'] == 1), 'test'] = '도심부 외 도로'
train.loc[(train['maximum_speed_limit'] == 60) & (train['lane_count'] == 2), 'test'] = '도심부 도로'
train.loc[(train['maximum_speed_limit'] > 60), 'test'] = '도심부 외 도로'

print(train.head(20))
