import pandas as pd
import category_encoders as ce
from haversine import haversine
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 600)

train = pd.read_parquet('../data/train_after.parquet')
test = pd.read_parquet('../data/test_after.parquet')

train.loc[(train['start_node_name'] == train['end_node_name']), 'road_name'] = train['start_node_name']
test.loc[(test['start_node_name'] == test['end_node_name']), 'road_name'] = test['start_node_name']
print(train.head(20))

train.to_parquet('../data/train_after.parquet', index=False)
test.to_parquet('../data/test_after.parquet', index=False)

