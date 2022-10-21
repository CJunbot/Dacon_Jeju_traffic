# 인구수 / 밀도
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 600)

train = pd.read_parquet('../../data/train_after.parquet')
test = pd.read_parquet('../../data/test_after.parquet')

jeju_2021 = pd.read_excel('../jeju_population_per_year.xlsx', sheet_name='jeju_2021')
jeju_2022 = pd.read_excel('../jeju_population_per_year.xlsx', sheet_name='jeju_2022')
jeju_list = list(jeju_2021['행정동2'])

# 제주시-2021년
for indexx, region in enumerate(jeju_list):
    for monthh in range(1, 8):
        train.loc[(monthh == train['month']) & (train['year'] == 2021) & (train['start_region_1'] == '제주시'), 'population_city'] = jeju_2021['인구'][0]
        train.loc[(monthh == train['month']) & (train['year'] == 2021) & (train['start_region_1'] == '제주시') &
                  (train['start_region_2' == region]), 'population_dong'] = jeju_2021['인구'][indexx]
        train.loc[(monthh == train['month']) & (train['year'] == 2021) & (train['start_region_1'] == '제주시') &
                  (train['start_region_3' == region]), 'population_dong'] = jeju_2021['인구'][indexx]
    # 제주시-2022년
    for month2 in range(9, 13):
        train.loc[(month2 == train['month']) & (train['year'] == 2022) & (train['start_region_1'] == '제주시'), 'population_city'] = jeju_2022['인구'][0]
        train.loc[(month2 == train['month']) & (train['year'] == 2022) & (train['start_region_1'] == '제주시') &
                  (train['start_region_2' == region]), 'population_dong'] = jeju_2022['인구'][jeju_2022['행정동1'].index(jeju_2021['행정동1'][indexx])]*jeju_2021['비율'][indexx]
        train.loc[(month2 == train['month']) & (train['year'] == 2022) & (train['start_region_1'] == '제주시') &
                  (train['start_region_3' == region]), 'population_dong'] = jeju_2022['인구'][jeju_2022['행정동1'].index(jeju_2021['행정동1'][indexx])] * jeju_2021['비율'][indexx]
