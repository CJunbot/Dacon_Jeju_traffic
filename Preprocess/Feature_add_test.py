import pandas as pd
import category_encoders as ce
from scipy.stats import skew
from haversine import haversine
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

train = pd.read_parquet('../data/train_after.parquet')
test = pd.read_parquet('../data/test_after.parquet')

train.loc[(3 < train['base_hour']) & (train['base_hour'] <= 8), 'base_day_night'] = 0  # 새벽
train.loc[(8 < train['base_hour']) & (train['base_hour'] <= 18), 'base_day_night'] = 1  # 오전
train.loc[(18 < train['base_hour']) | (train['base_hour'] <= 3), 'base_day_night'] = 2  # 오후

test.loc[(test['base_hour'] <= 8) & (3 < test['base_hour']), 'base_day_night'] = 0  # 새벽
test.loc[(8 < test['base_hour']) & (test['base_hour'] <= 18), 'base_day_night'] = 1  # 오전
test.loc[(18 < test['base_hour']) | (test['base_hour'] <= 3), 'base_day_night'] = 2  # 오후


print(train.head(50))
print('\n')
print(test.head(50))
# save processed dataset to parquet
train.to_parquet('../data/train_after_test.parquet', index=False)
test.to_parquet('../data/test_after_test.parquet', index=False)
