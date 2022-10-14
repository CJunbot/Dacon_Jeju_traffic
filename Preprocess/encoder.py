import pandas as pd
import category_encoders as ce

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 400)

train = pd.read_parquet('sfa/train.parquet')
test = pd.read_parquet('sfa/test.parquet')

train['road_name'] = train['road_name'].replace('-', None)
test['road_name'] = test['road_name'].replace('-', None)

for name in ['road_name']:
    glmm_encoder = ce.JamesSteinEncoder(cols=[name], handle_missing='return_nan')
    train[name] = glmm_encoder.fit_transform(train[name], train['target'])
    test[name] = glmm_encoder.transform(test[name])

print(train['road_name'].head(20))
print(train['road_name'].describe())
