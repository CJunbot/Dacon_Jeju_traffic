import pandas as pd
from scipy.stats import skew
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

train = pd.read_parquet('../data/train_after.parquet')
test = pd.read_parquet('../data/test_after.parquet')

train.loc[(1 <= train['month']) & (train['month'] <= 3) & (7 <= train['base_hour']) & (train['base_hour'] <= 18), 'sun'] = 1  # 해뜸
train.loc[(4 == train['month']) & (6 <= train['base_hour']) & (train['base_hour'] <= 19), 'sun'] = 1  # 해뜸
train.loc[(5 <= train['month']) & (train['month'] <= 7) & (5 <= train['base_hour']) & (train['base_hour'] <= 19), 'sun'] = 1  # 해뜸
train.loc[(8 <= train['month']) & (train['month'] <= 9) & (6 <= train['base_hour']) & (train['base_hour'] <= 18), 'sun'] = 1  # 해뜸
train.loc[(10 <= train['month']) & (train['month'] <= 11) & (6 <= train['base_hour']) & (train['base_hour'] <= 17), 'sun'] = 1  # 해뜸
train.loc[(12 == train['month']) & (7 <= train['base_hour']) & (train['base_hour'] <= 17), 'sun'] = 1  # 해뜸
train['sun'] = train['sun'].fillna(0)

test.loc[(1 <= test['month']) & (test['month'] <= 3) & (7 <= test['base_hour']) & (test['base_hour'] <= 18), 'sun'] = 1  # 해뜸
test.loc[(4 == test['month']) & (6 <= test['base_hour']) & (test['base_hour'] <= 19), 'sun'] = 1  # 해뜸
test.loc[(5 <= test['month']) & (test['month'] <= 7) & (5 <= test['base_hour']) & (test['base_hour'] <= 19), 'sun'] = 1  # 해뜸
test.loc[(8 <= test['month']) & (test['month'] <= 9) & (6 <= test['base_hour']) & (test['base_hour'] <= 18), 'sun'] = 1  # 해뜸
test.loc[(10 <= test['month']) & (test['month'] <= 11) & (6 <= test['base_hour']) & (test['base_hour'] <= 17), 'sun'] = 1  # 해뜸
test.loc[(12 == test['month']) & (7 <= test['base_hour']) & (test['base_hour'] <= 17), 'sun'] = 1  # 해뜸
test['sun'] = test['sun'].fillna(0)


print(train.head(50))
print('\n')
print(test.head(50))
# save processed dataset to parquet
train.to_parquet('../data/train_after_test.parquet', index=False)
test.to_parquet('../data/test_after_test.parquet', index=False)
