import pandas as pd
from lightgbm import LGBMRegressor
from boruta import BorutaPy

train = pd.read_parquet('../data/train_after_test.parquet')
train.drop(columns=['road_name'], inplace=True)

y = train['target']
x = train.drop(columns=['target'])

lgb = LGBMRegressor(num_boost_round=100)
feat_selector = BorutaPy(lgb, n_estimators='auto', verbose=0, random_state=1)
feat_selector.fit(x.values, y.values)

# Check the selected features
print(x.columns[feat_selector.support_])

print(x.columns-x.columns[feat_selector.support_])
