import pandas as pd
from scipy.stats import skew
from haversine import haversine
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

train = pd.read_csv('../data/submit_cat_fold.csv')

train.drop(columns=['index'], inplace=True)
train.to_csv('../data/submit_cat_fold2.parquet', index=False)
