import json
import requests
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 800)

train = pd.read_parquet('../data/train_address.parquet')
train2 = pd.read_parquet('../data/train_address2.parquet')

for sex in range(480000):
    if train['start_region_1'][sex] == '':
        train.loc[sex, 'start_region_1'] = train2['start_region_1'][sex]
        train.loc[sex, 'start_region_2'] = train2['start_region_2'][sex]
        train.loc[sex, 'start_region_3'] = train2['start_region_3'][sex]
        train.loc[sex, 'end_region_1'] = train2['end_region_1'][sex]
        train.loc[sex, 'end_region_1'] = train2['end_region_2'][sex]
        train.loc[sex, 'end_region_1'] = train2['end_region_3'][sex]

train.to_parquet('../data/train_address_after.parquet', index=False)
