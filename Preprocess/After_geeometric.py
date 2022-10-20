import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 600)

train = pd.read_parquet('../data/train_after.parquet')
test = pd.read_parquet('../data/test_after.parquet')

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

# 월 별 유입 / 유출 인구
def in_jejudo():
    in_jeju_weekdays_2021 = [47445, 56835, 64137, 54879]
    in_jeju_weekend_2021 = [52111, 62606, 75889, 49812]
    in_west_weekdays_2021 = [46021, 54031, 62073, 56879]
    in_west_weekend_2021 = [47473, 54960, 68857, 46193]
    in_jeju_weekdays_2022 = [64930, 59757, 48707, 60278, 65030, 60952, 64688, 61254]
    in_jeju_weekend_2022 = [58055, 62785, 46007, 73909, 77201, 76137, 73186, 74487]
    in_west_weekdays_2022 = [62992, 56429, 48044, 58482, 62410, 61981, 62363, 56817]
    in_west_weekend_2022 = [60876, 56237, 43194, 67144, 68895, 67783, 65585, 65593]
    # 2021년
    for k in range(9, 13):
        # 평일
        train.loc[('제주시' == train['start_region_1']) & (train['month'] == k) &
                  (train['day'] <= 4) & (train['year'] == 2021), 'in_jeju_people'] = in_jeju_weekdays_2021[k-9]
        train.loc[('서귀포시' == train['start_region_1']) & (train['month'] == k) &
                  (train['day'] <= 4) & (train['year'] == 2021), 'in_west_people'] = in_west_weekdays_2021[k-9]
        # 주말
        train.loc[('제주시' == train['start_region_1']) & (train['month'] == k) &
                  (train['day'] > 4) & (train['year'] == 2021), 'in_jeju_people'] = in_jeju_weekend_2021[k-9]
        train.loc[('서귀포시' == train['start_region_1']) & (train['month'] == k) &
                  (train['day'] <= 4) & (train['year'] == 2021), 'in_west_people'] = in_west_weekend_2021[k-9]
    # 2022년
    for k2 in range(1, 9):
        # 평일
        train.loc[('제주시' == train['start_region_1']) & (train['month'] == k2) &
                  (train['day'] <= 4) & (train['year'] == 2022), 'in_jeju_people'] = in_jeju_weekdays_2022[k2 - 9]
        train.loc[('서귀포시' == train['start_region_1']) & (train['month'] == k2) &
                  (train['day'] <= 4) & (train['year'] == 2022), 'in_west_people'] = in_west_weekdays_2022[k2 - 9]
        # 주말
        train.loc[('제주시' == train['start_region_1']) & (train['month'] == k2) &
                  (train['day'] > 4) & (train['year'] == 2022), 'in_jeju_people'] = in_jeju_weekend_2022[k2 - 9]
        train.loc[('서귀포시' == train['start_region_1']) & (train['month'] == k2) &
                  (train['day'] <= 4) & (train['year'] == 2022), 'in_west_people'] = in_west_weekend_2022[k2 - 9]


def out_jejudo():
    out_jeju_weekdays_2021 = [51730, 60401,71218,62645]
    out_jeju_weekend_2021 = [56490,65312,79009,55512]
    out_west_weekdays_2021 = [44231,50254,59668,51095]
    out_west_weekend_2021 = [48527,57725,69804,46532]
    out_jeju_weekdays_2022 = [73431,67646,54145,66076,71756,71546,71219,63680]
    out_jeju_weekend_2022 = [68553,65952,53526,77318,81846,77837,77267,76093]
    out_west_weekdays_2022 = [59782,54032,45122,56657,61234,57317,61490,55516]
    out_west_weekend_2022 = [51558,56236,44477,70904,72832,70748,69365,68734]
    # 2021년
    for k in range(9, 13):
        # 평일
        train.loc[('제주시' == train['start_region_1']) & (train['month'] == k) &
                  (train['day'] <= 4) & (train['year'] == 2021), 'out_jeju_people'] = out_jeju_weekdays_2021[k-9]
        train.loc[('서귀포시' == train['start_region_1']) & (train['month'] == k) &
                  (train['day'] <= 4) & (train['year'] == 2021), 'out_west_people'] = out_west_weekdays_2021[k-9]
        # 주말
        train.loc[('제주시' == train['start_region_1']) & (train['month'] == k) &
                  (train['day'] > 4) & (train['year'] == 2021), 'out_jeju_people'] = out_jeju_weekend_2021[k-9]
        train.loc[('서귀포시' == train['start_region_1']) & (train['month'] == k) &
                  (train['day'] <= 4) & (train['year'] == 2021), 'out_west_people'] = out_west_weekend_2021[k-9]
    # 2022년
    for k2 in range(1, 9):
        # 평일
        train.loc[('제주시' == train['start_region_1']) & (train['month'] == k2) &
                  (train['day'] <= 4) & (train['year'] == 2022), 'out_jeju_people'] = out_jeju_weekdays_2022[k2 - 9]
        train.loc[('서귀포시' == train['start_region_1']) & (train['month'] == k2) &
                  (train['day'] <= 4) & (train['year'] == 2022), 'out_west_people'] = out_west_weekdays_2022[k2 - 9]
        # 주말
        train.loc[('제주시' == train['start_region_1']) & (train['month'] == k2) &
                  (train['day'] > 4) & (train['year'] == 2022), 'out_jeju_people'] = out_jeju_weekend_2022[k2 - 9]
        train.loc[('서귀포시' == train['start_region_1']) & (train['month'] == k2) &
                  (train['day'] <= 4) & (train['year'] == 2022), 'out_west_people'] = out_west_weekend_2022[k2 - 9]

