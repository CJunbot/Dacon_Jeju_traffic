from scipy.stats import skew
import pandas as pd
import numpy as np

train = pd.read_parquet('../data/train_address_pop.parquet')

# population_dong2 ?
#features_index = ['km', 'start_bus_km', 'end_bus_km']
features_index = train.dtypes[train.dtypes != 'object'].index

skew_features = train[features_index].apply(lambda x : skew(x))
skew_features_top = skew_features[skew_features > 1]
print(skew_features_top)
print('----------------------------------------------')
train[skew_features_top.index] = np.log1p(train[skew_features_top.index])

print('----------------------------------------------')
skew_features = train[features_index].apply(lambda x : skew(x))
skew_features_top = skew_features[skew_features > 1]
print(skew_features_top)
