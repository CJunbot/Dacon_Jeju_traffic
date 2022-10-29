from scipy.stats import skew
import pandas as pd
import numpy as np

train = pd.read_parquet('../data/train_after.parquet')
test = pd.read_parquet('../data/test_after.parquet')

# population_dong2 ?
features_index = ['km', 'start_bus_km', 'end_bus_km', 'population_dong2']

skew_features = train[features_index].apply(lambda x : skew(x))
skew_features_top = skew_features[skew_features > 1]
print(skew_features_top)
print('----------------------------------------------')

train[skew_features_top.index] = np.log1p(train[skew_features_top.index])
test[skew_features_top.index] = np.log1p(test[skew_features_top.index])

train.to_parquet('../data/train_after.parquet', index=False)
test.to_parquet('../data/test_after.parquet', index=False)
