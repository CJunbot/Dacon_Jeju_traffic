# 인구수 / 밀도
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 600)

train = pd.read_parquet('../../data/train_address.parquet')
test = pd.read_parquet('../../data/test_address.parquet')

# 동별 인구수 추가
jeju_2021 = pd.read_excel('../../jeju_population_per_year.xlsx', sheet_name='jeju_2021')
jeju_list = list(jeju_2021['행정동2'])

west_2021 = pd.read_excel('../../jeju_population_per_year.xlsx', sheet_name='seogipo_2021')
west_list = list(west_2021['행정동2'])

# 제주시 인구수
train.loc[(train['year'] == 2021) & (train['start_region_1'] == '제주시'), 'population_city'] = jeju_2021['인구'][0]
train.loc[(train['year'] == 2022) & (train['start_region_1'] == '제주시'), 'population_city'] = jeju_2021['인구2'][0]
for indexx, region in enumerate(jeju_list):
    # 제주시-2021년
    train.loc[(train['year'] == 2021) & (train['start_region_1'] == '제주시') & (train['start_region_3'] == region), 'population_dong'] = jeju_2021['인구'][indexx]
    train.loc[(train['year'] == 2021) & (train['start_region_1'] == '제주시') & (train['start_region_2'] == region), 'population_dong'] = jeju_2021['인구'][indexx]
    train.loc[(train['year'] == 2021) & (train['end_region_1'] == '제주시') & (train['end_region_3'] == region), 'population_dong2'] = jeju_2021['인구'][indexx]
    train.loc[(train['year'] == 2021) & (train['end_region_1'] == '제주시') & (train['end_region_2'] == region), 'population_dong2'] = jeju_2021['인구'][indexx]
    # 제주시-2022년
    train.loc[(train['year'] == 2022) & (train['start_region_1'] == '제주시') & (train['start_region_2'] == region), 'population_dong'] = jeju_2021['인구2'][indexx]
    train.loc[(train['year'] == 2022) & (train['start_region_1'] == '제주시') & (train['start_region_3'] == region), 'population_dong'] = jeju_2021['인구2'][indexx]
    train.loc[(train['year'] == 2022) & (train['end_region_1'] == '제주시') & (train['end_region_3'] == region), 'population_dong2'] = jeju_2021['인구2'][indexx]
    train.loc[(train['year'] == 2022) & (train['end_region_1'] == '제주시') & (train['end_region_2'] == region), 'population_dong2'] = jeju_2021['인구2'][indexx]

test.loc[(test['year'] == 2021) & (test['start_region_1'] == '제주시'), 'population_city'] = jeju_2021['인구'][0]
test.loc[(test['year'] == 2022) & (test['start_region_1'] == '제주시'), 'population_city'] = jeju_2021['인구2'][0]
for indexxx, region in enumerate(jeju_list):
    # 제주시-2021년
    test.loc[(test['year'] == 2021) & (test['start_region_1'] == '제주시') & (test['start_region_3'] == region), 'population_dong'] = jeju_2021['인구'][indexxx]
    test.loc[(test['year'] == 2021) & (test['start_region_1'] == '제주시') & (test['start_region_2'] == region), 'population_dong'] = jeju_2021['인구'][indexxx]
    test.loc[(test['year'] == 2021) & (test['end_region_1'] == '제주시') & (test['end_region_3'] == region), 'population_dong2'] = jeju_2021['인구'][indexxx]
    test.loc[(test['year'] == 2021) & (test['end_region_1'] == '제주시') & (test['end_region_2'] == region), 'population_dong2'] = jeju_2021['인구'][indexxx]
    # 제주시-2022년
    test.loc[(test['year'] == 2022) & (test['start_region_1'] == '제주시') & (test['start_region_2'] == region), 'population_dong'] = jeju_2021['인구2'][indexxx]
    test.loc[(test['year'] == 2022) & (test['start_region_1'] == '제주시') & (test['start_region_3'] == region), 'population_dong'] = jeju_2021['인구2'][indexxx]
    test.loc[(test['year'] == 2022) & (test['end_region_1'] == '제주시') & (test['end_region_3'] == region), 'population_dong2'] = jeju_2021['인구2'][indexxx]
    test.loc[(test['year'] == 2022) & (test['end_region_1'] == '제주시') & (test['end_region_2'] == region), 'population_dong2'] = jeju_2021['인구2'][indexxx]
print('제주시 인구 매핑 끝')

# 서귀포시 인구수
train.loc[(train['year'] == 2021) & (train['start_region_1'] == '서귀포시'), 'population_city'] = west_2021['인구'][0]
train.loc[(train['year'] == 2022) & (train['start_region_1'] == '서귀포시'), 'population_city'] = west_2021['인구2'][0]
for indexx, region in enumerate(west_list):
    # 서귀포시-2021년
    train.loc[(train['year'] == 2021) & (train['start_region_1'] == '서귀포시') & (train['start_region_2'] == region), 'population_dong'] = west_2021['인구'][indexx]
    train.loc[(train['year'] == 2021) & (train['start_region_1'] == '서귀포시') & (train['start_region_3'] == region), 'population_dong'] = west_2021['인구'][indexx]
    train.loc[(train['year'] == 2021) & (train['end_region_1'] == '서귀포시') & (train['end_region_2'] == region), 'population_dong2'] = west_2021['인구'][indexx]
    train.loc[(train['year'] == 2021) & (train['end_region_1'] == '서귀포시') & (train['end_region_3'] == region), 'population_dong2'] = west_2021['인구'][indexx]
    # 제주시-2022년
    train.loc[(train['year'] == 2022) & (train['start_region_1'] == '서귀포시') & (train['start_region_2'] == region), 'population_dong'] = west_2021['인구2'][indexx]
    train.loc[(train['year'] == 2022) & (train['start_region_1'] == '서귀포시') & (train['start_region_3'] == region), 'population_dong'] = west_2021['인구2'][indexx]
    train.loc[(train['year'] == 2022) & (train['end_region_1'] == '서귀포시') & (train['end_region_2'] == region), 'population_dong2'] = west_2021['인구2'][indexx]
    train.loc[(train['year'] == 2022) & (train['end_region_1'] == '서귀포시') & (train['end_region_3'] == region), 'population_dong2'] = west_2021['인구2'][indexx]

test.loc[(test['year'] == 2021) & (test['start_region_1'] == '서귀포시'), 'population_city'] = west_2021['인구'][0]
test.loc[(test['year'] == 2022) & (test['start_region_1'] == '서귀포시'), 'population_city'] = west_2021['인구2'][0]
for indexxxx, region in enumerate(west_list):
    # 서귀포시-2021년
    test.loc[(test['year'] == 2021) & (test['start_region_1'] == '서귀포시') & (test['start_region_2'] == region), 'population_dong'] = west_2021['인구'][indexxxx]
    test.loc[(test['year'] == 2021) & (test['start_region_1'] == '서귀포시') & (test['start_region_3'] == region), 'population_dong'] = west_2021['인구'][indexxxx]
    test.loc[(test['year'] == 2021) & (test['end_region_1'] == '서귀포시') & (test['end_region_2'] == region), 'population_dong2'] = west_2021['인구'][indexxxx]
    test.loc[(test['year'] == 2021) & (test['end_region_1'] == '서귀포시') & (test['end_region_3'] == region), 'population_dong2'] = west_2021['인구'][indexxxx]
    # 제주시-2022년
    test.loc[(test['year'] == 2022) & (test['start_region_1'] == '서귀포시') & (test['start_region_2'] == region), 'population_dong'] = west_2021['인구2'][indexxxx]
    test.loc[(test['year'] == 2022) & (test['start_region_1'] == '서귀포시') & (test['start_region_3'] == region), 'population_dong'] = west_2021['인구2'][indexxxx]
    test.loc[(test['year'] == 2022) & (test['end_region_1'] == '서귀포시') & (test['end_region_2'] == region), 'population_dong2'] = west_2021['인구2'][indexxxx]
    test.loc[(test['year'] == 2022) & (test['end_region_1'] == '서귀포시') & (test['end_region_3'] == region), 'population_dong2'] = west_2021['인구2'][indexxxx]
print('서귀포시 인구 매핑 끝')

train['population_dong2'] = train['population_dong2'].fillna(0)
train['population_city'] = train['population_city'].astype(dtype='int64')
train['population_dong'] = train['population_dong'].astype(dtype='int64')
train['population_dong2'] = train['population_dong2'].astype(dtype='int64')

test['population_dong2'] = test['population_dong2'].fillna(0)
test['population_city'] = test['population_city'].astype(dtype='int64')
test['population_dong'] = test['population_dong'].astype(dtype='int64')
test['population_dong2'] = test['population_dong2'].astype(dtype='int64')

print(train.head(20))
print(test.head(20))

# save processed dataset to parquet
train.to_parquet('../../data/train_address.parquet', index=False)
test.to_parquet('../../data/test_address.parquet', index=False)
