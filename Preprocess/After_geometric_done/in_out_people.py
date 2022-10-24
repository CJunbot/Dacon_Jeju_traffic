import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 600)

train = pd.read_parquet('../../data/train_after.parquet')
test = pd.read_parquet('../../data/test_after.parquet')


#월 별 유입 / 유출 인구

in_jeju_weekdays_2021 = [47445, 56835, 64137, 54879]
in_jeju_weekend_2021 = [52111, 62606, 75889, 49812]
in_west_weekdays_2021 = [46021, 54031, 62073, 56879]
in_west_weekend_2021 = [47473, 54960, 68857, 46193]
in_jeju_weekdays_2022 = [64930, 59757, 48707, 60278, 65030, 60952, 64688, 61254,57658]
in_jeju_weekend_2022 = [58055, 62785, 46007, 73909, 77201, 76137, 73186, 74487,71815]
in_west_weekdays_2022 = [62992, 56429, 48044, 58482, 62410, 61981, 62363, 56817,57534]
in_west_weekend_2022 = [60876, 56237, 43194, 67144, 68895, 67783, 65585, 65593,65984]
# 2021년
for k in range(9, 13):
    # 평일
    train.loc[(0 == train['start_region_1']) & (train['month'] == k) &
              (train['day_of_week'] <= 4) & (train['year'] == 2021), 'in_people'] = in_jeju_weekdays_2021[k-9]
    train.loc[(1 == train['start_region_1']) & (train['month'] == k) &
              (train['day_of_week'] <= 4) & (train['year'] == 2021), 'in_people'] = in_west_weekdays_2021[k-9]
    # 주말
    train.loc[(0 == train['start_region_1']) & (train['month'] == k) &
              (train['day_of_week'] > 4) & (train['year'] == 2021), 'in_people'] = in_jeju_weekend_2021[k-9]
    train.loc[(1 == train['start_region_1']) & (train['month'] == k) &
              (train['day_of_week'] > 4) & (train['year'] == 2021), 'in_people'] = in_west_weekend_2021[k-9]
# 2022년
for k2 in range(1, 9):
    # 평일
    train.loc[(0 == train['start_region_1']) & (train['month'] == k2) &
              (train['day_of_week'] <= 4) & (train['year'] == 2022), 'in_people'] = in_jeju_weekdays_2022[k2 - 9]
    train.loc[(1 == train['start_region_1']) & (train['month'] == k2) &
              (train['day_of_week'] <= 4) & (train['year'] == 2022), 'in_people'] = in_west_weekdays_2022[k2 - 9]
    # 주말
    train.loc[(0 == train['start_region_1']) & (train['month'] == k2) &
              (train['day_of_week'] > 4) & (train['year'] == 2022), 'in_people'] = in_jeju_weekend_2022[k2 - 9]
    train.loc[(1 == train['start_region_1']) & (train['month'] == k2) &
              (train['day_of_week'] > 4) & (train['year'] == 2022), 'in_people'] = in_west_weekend_2022[k2 - 9]


out_jeju_weekdays_2021 = [51730, 60401,71218,62645]
out_jeju_weekend_2021 = [56490,65312,79009,55512]
out_west_weekdays_2021 = [44231,50254,59668,51095]
out_west_weekend_2021 = [48527,57725,69804,46532]
out_jeju_weekdays_2022 = [73431,67646,54145,66076,71756,71546,71219,63680,66901]
out_jeju_weekend_2022 = [68553,65952,53526,77318,81846,77837,77267,76093,76522]
out_west_weekdays_2022 = [59782,54032,45122,56657,61234,57317,61490,55516,55976]
out_west_weekend_2022 = [51558,56236,44477,70904,72832,70748,69365,68734,66969]
# 2021년
for k in range(9, 13):
    # 평일
    train.loc[(0 == train['start_region_1']) & (train['month'] == k) &
              (train['day_of_week'] <= 4) & (train['year'] == 2021), 'out_people'] = out_jeju_weekdays_2021[k-9]
    train.loc[(1 == train['start_region_1']) & (train['month'] == k) &
              (train['day_of_week'] <= 4) & (train['year'] == 2021), 'out_people'] = out_west_weekdays_2021[k-9]
    # 주말
    train.loc[(0 == train['start_region_1']) & (train['month'] == k) &
              (train['day_of_week'] > 4) & (train['year'] == 2021), 'out_people'] = out_jeju_weekend_2021[k-9]
    train.loc[(1 == train['start_region_1']) & (train['month'] == k) &
              (train['day_of_week'] > 4) & (train['year'] == 2021), 'out_people'] = out_west_weekend_2021[k-9]
# 2022년
for k2 in range(1, 9):
    # 평일
    train.loc[(0 == train['start_region_1']) & (train['month'] == k2) &
              (train['day_of_week'] <= 4) & (train['year'] == 2022), 'out_people'] = out_jeju_weekdays_2022[k2 - 9]
    train.loc[(1 == train['start_region_1']) & (train['month'] == k2) &
              (train['day_of_week'] <= 4) & (train['year'] == 2022), 'out_people'] = out_west_weekdays_2022[k2 - 9]
    # 주말
    train.loc[(0 == train['start_region_1']) & (train['month'] == k2) &
              (train['day_of_week'] > 4) & (train['year'] == 2022), 'out_people'] = out_jeju_weekend_2022[k2 - 9]
    train.loc[(1 == train['start_region_1']) & (train['month'] == k2) &
              (train['day_of_week'] > 4) & (train['year'] == 2022), 'out_people'] = out_west_weekend_2022[k2 - 9]

# 2022년
for k2 in range(1, 9):
    # 평일
    test.loc[(0 == test['start_region_1']) & (test['month'] == k2) &
              (test['day_of_week'] <= 4) & (test['year'] == 2022), 'in_people'] = in_jeju_weekdays_2022[k2 - 9]
    test.loc[(1 == test['start_region_1']) & (test['month'] == k2) &
              (test['day_of_week'] <= 4) & (test['year'] == 2022), 'in_people'] = in_west_weekdays_2022[k2 - 9]
    # 주말
    test.loc[(0 == test['start_region_1']) & (test['month'] == k2) &
              (test['day_of_week'] > 4) & (test['year'] == 2022), 'in_people'] = in_jeju_weekend_2022[k2 - 9]
    test.loc[(1 == test['start_region_1']) & (test['month'] == k2) &
              (test['day_of_week'] > 4) & (test['year'] == 2022), 'in_people'] = in_west_weekend_2022[k2 - 9]
for k2 in range(1, 9):
    # 평일
    test.loc[(0 == test['start_region_1']) & (test['month'] == k2) &
              (test['day_of_week'] <= 4) & (test['year'] == 2022), 'out_people'] = out_jeju_weekdays_2022[k2 - 9]
    test.loc[(1 == test['start_region_1']) & (test['month'] == k2) &
              (test['day_of_week'] <= 4) & (test['year'] == 2022), 'out_people'] = out_west_weekdays_2022[k2 - 9]
    # 주말
    test.loc[(0 == test['start_region_1']) & (test['month'] == k2) &
              (test['day_of_week'] > 4) & (test['year'] == 2022), 'out_people'] = out_jeju_weekend_2022[k2 - 9]
    test.loc[(1 == test['start_region_1']) & (test['month'] == k2) &
              (test['day_of_week'] > 4) & (test['year'] == 2022), 'out_people'] = out_west_weekend_2022[k2 - 9]
print(train.head(20))
print(test.head(20))
test.to_parquet('../../data/test_bus_after.parquet', index=False)
train.to_parquet('../../data/train_bus_after.parquet', index=False)