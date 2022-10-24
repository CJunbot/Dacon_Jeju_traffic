# 인구수 / 밀도
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 600)

train = pd.read_parquet('../../data/train_address.parquet')
test = pd.read_parquet('../../data/test_address.parquet')
# 동별 인구수 추가
jeju_2021 = pd.read_excel('../../jeju_population_per_year.xlsx', sheet_name='jeju_2021')
jeju_list = list(jeju_2021['행정동2'][2:])
west_2021 = pd.read_excel('../../jeju_population_per_year.xlsx', sheet_name='seogipo_2021')
west_list = list(west_2021['행정동2'][2:])

# 제주시 인구수
for indexx, region in enumerate(jeju_list):
    # 제주시-2021년
    for monthh in range(1, 8):
        print(f'index: {indexx}')
        train.loc[(monthh == train['month']) & (train['year'] == 2021) & (train['start_region_1'] == 0), 'population_city'] = jeju_2021['인구'][0]
        train.loc[(monthh == train['month']) & (train['year'] == 2021) & (train['start_region_1'] == 0) &
                  (train['start_region_3'] == region), 'population_dong'] = jeju_2021['인구'][indexx]
        train.loc[(monthh == train['month']) & (train['year'] == 2021) & (train['start_region_1'] == 0) &
                  (train['start_region_2'] == region), 'population_dong'] = jeju_2021['인구'][indexx]
    # 제주시-2022년
    for month2 in range(9, 13):
        train.loc[(month2 == train['month']) & (train['year'] == 2022) & (train['start_region_1'] == 0), 'population_city'] = jeju_2021['인구2'][0]
        train.loc[(month2 == train['month']) & (train['year'] == 2022) & (train['start_region_1'] == 0) &
                  (train['start_region_2'] == region), 'population_dong'] = jeju_2021['인구2'][indexx]
        train.loc[(month2 == train['month']) & (train['year'] == 2022) & (train['start_region_1'] == 0) &
                  (train['start_region_3'] == region), 'population_dong'] = jeju_2021['인구2'][indexx]

# 서귀포시 인구수
for indexx, region in enumerate(west_list):
    # 서귀포시-2021년
    for monthh in range(1, 8):
        train.loc[(monthh == train['month']) & (train['year'] == 2021) & (train['start_region_1'] == 1), 'population_city'] = west_2021['인구'][0]
        train.loc[(monthh == train['month']) & (train['year'] == 2021) & (train['start_region_1'] == 1) &
                  (train['start_region_2'] == region), 'population_dong'] = west_2021['인구'][indexx]
        train.loc[(monthh == train['month']) & (train['year'] == 2021) & (train['start_region_1'] == 1) &
                  (train['start_region_3'] == region), 'population_dong'] = west_2021['인구'][indexx]
    # 제주시-2022년
    for month2 in range(9, 13):
        train.loc[(month2 == train['month']) & (train['year'] == 2022) & (train['start_region_1'] == 1), 'population_city'] = west_2021['인구2'][0]
        train.loc[(month2 == train['month']) & (train['year'] == 2022) & (train['start_region_1'] == 1) &
                  (train['start_region_2'] == region), 'population_dong'] = west_2021['인구2'][indexx]
        train.loc[(month2 == train['month']) & (train['year'] == 2022) & (train['start_region_1'] == 1) &
                  (train['start_region_3'] == region), 'population_dong'] = west_2021['인구2'][indexx]

print(train.head(100))