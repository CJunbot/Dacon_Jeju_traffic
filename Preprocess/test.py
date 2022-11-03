import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 400)

train = pd.read_parquet('../data/train_after.parquet')
test = pd.read_parquet('../data/test_after.parquet')

train.loc[(train['base_hour'] <= 40), 'road_types'] = 0  # 인접 도로


train.to_parquet('../data/train_after_test.parquet', index=False)
test.to_parquet('../data/test_after_test.parquet', index=False)