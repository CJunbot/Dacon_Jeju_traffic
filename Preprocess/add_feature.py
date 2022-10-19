import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 600)

train = pd.read_parquet('../data/train_after.parquet')
test = pd.read_parquet('../data/test_after.parquet')

train.loc[(1 <= train['month']) & (train['month'] < 6), 'peak_season'] = '0'
train.loc[(6 <= train['month']) & (train['month'] <= 9), 'peak_season'] = '1'
train.loc[(9 < train['month']) & (train['month'] <= 12), 'peak_season'] = '0'

print(train.head(20))
