import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 600)

train = pd.read_parquet('../../data/train_after.parquet')
test = pd.read_parquet('../../data/test_after.parquet')

# 월 말 차량 등록 대수
car_gwan_jeju = [1630,1627,1639,1634,1638,1633,1612,1608,1611,1620,1639]
car_mine_jeju = [251254,251857,252617,253020,253298,253893,254393,254782,254907,254889,255447]
car_youngup_jeju = [40120,40107,40194,39887,40154,40089,40039,40257,40525,40391,40454]
car_gwan_west = [793,789,787,792,788,784,781,777,777,777,782]
car_mine_west = [104600,104821,105184,105380,10557,105881,106098,106373,106225,106025,106219]
car_youngup_west = [1981,1988,1995,1990,1988,1992,2020,2038,1865,1871,1876]

for twentyone in range(9,13):
    train.loc[(twentyone == train['month']) & (train['year'] == 2021) & (train['start_region_1'] == '제주시'), 'car_gwan'] = car_gwan_jeju[twentyone-9]
    train.loc[(twentyone == train['month']) & (train['year'] == 2021) & (train['start_region_1'] == '제주시'), 'car_mine'] = car_mine_jeju[twentyone - 9]
    train.loc[(twentyone == train['month']) & (train['year'] == 2021) & (train['start_region_1'] == '제주시'), 'car_youngup'] = car_youngup_jeju[twentyone - 9]
    train.loc[(twentyone == train['month']) & (train['year'] == 2021) & (train['start_region_1'] == '서귀포시'), 'car_gwan'] = car_gwan_west[twentyone - 9]
    train.loc[(twentyone == train['month']) & (train['year'] == 2021) & (train['start_region_1'] == '서귀포시'), 'car_mine'] = car_mine_west[twentyone - 9]
    train.loc[(twentyone == train['month']) & (train['year'] == 2021) & (train['start_region_1'] == '서귀포시'), 'car_youngup'] = car_youngup_west[twentyone - 9]

for twentytwo in range(1, 8):
    train.loc[(twentytwo == train['month']) & (train['year'] == 2022) & (train['start_region_1'] == '제주시'), 'car_gwan'] = car_gwan_jeju[twentytwo + 3]
    train.loc[(twentytwo == train['month']) & (train['year'] == 2022) & (train['start_region_1'] == '제주시'), 'car_mine'] = car_mine_jeju[twentytwo + 3]
    train.loc[(twentytwo == train['month']) & (train['year'] == 2022) & (train['start_region_1'] == '제주시'), 'car_youngup'] = car_youngup_jeju[twentytwo + 3]
    train.loc[(twentytwo == train['month']) & (train['year'] == 2022) & (train['start_region_1'] == '서귀포시'), 'car_gwan'] = car_gwan_west[twentytwo + 3]
    train.loc[(twentytwo == train['month']) & (train['year'] == 2022) & (train['start_region_1'] == '서귀포시'), 'car_mine'] = car_mine_west[twentytwo + 3]
    train.loc[(twentytwo == train['month']) & (train['year'] == 2022) & (train['start_region_1'] == '서귀포시'), 'car_youngup'] = car_youngup_west[twentytwo + 3]



