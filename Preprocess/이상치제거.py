from scipy.stats import skew
import pandas as pd
import numpy as np

train = pd.read_parquet('../data/train_after.parquet')

#features_index = train.dtypes[train != 'object'].index
features_index = ['km', 'start_bus_km', 'end_bus_km']

skew_features = train[features_index].apply(lambda x : skew(x))
skew_features_top = skew_features[skew_features > 1]
print(skew_features_top)

train[skew_features_top.index] = np.log1p(train[skew_features_top.index])

skew_features = train[features_index].apply(lambda x : skew(x))
skew_features_top = skew_features[skew_features > 1]
print(skew_features_top)
