import requests, json
import pandas as pd

api_key = '610093332b1a7815b94a196b4179ff1d'

train = pd.read_parquet('../data/train_bus.parquet')
test = pd.read_parquet('../data/test_bus.parquet')


def lat_lon_to_addr(longitude, latitude):
    try:
        url = 'https://dapi.kakao.com/v2/local/geo/coord2regioncode.json?x={longitude}&y={latitude}'.format(longitude=longitude, latitude=latitude)
        headers = {"Authorization": "KakaoAK " + api_key}
        result = json.loads(str(requests.get(url, headers=headers).text))
        return result['documents'][0]['region_2depth_name'], result['documents'][0]['region_3depth_name'], result['documents'][0]['region_4depth_name']
    except Exception as e:
        print(e)
        return None, None, None

# unique value들을 매핑
start_list, end_list, index_start, index_end, i = [], [], [], [], 0
while (len(start_list) or len(end_list)) < 586:
    if (train['start_longitude'][i], train['start_latitude'][i]) not in start_list:
        start_list.append((train['start_longitude'][i], train['start_latitude'][i]))
        start_1, start_2, start_3 = lat_lon_to_addr(train['start_longitude'][i], train['start_latitude'][i])
        train.loc[i, 'start_region_1'] = start_1
        train.loc[i, 'start_region_2'] = start_2
        train.loc[i, 'start_region_3'] = start_3
        index_start.append(i)
    if (train['end_longitude'][i], train['end_latitude'][i]) not in end_list:
        end_list.append((train['end_longitude'][i], train['end_latitude'][i]))
        end_1, end_2, end_3 = lat_lon_to_addr(train['end_longitude'][i], train['end_latitude'][i])
        train.loc[i, 'end_region_1'] = end_1
        train.loc[i, 'end_region_2'] = end_2
        train.loc[i, 'end_region_3'] = end_3
        index_end.append(i)
    i += 1
    print(f'i: {i} | start list 개수: {len(start_list)}, end list 개수: {len(end_list)}')

# 할당 해주기
for geo in range(len(start_list)):
    train.loc[(train['start_longitude'] == start_list[geo][0]) & (train['start_latitude'] == start_list[geo][1]),
              'start_region_1'] = train['start_region_1'][index_start[geo]]
    train.loc[(train['start_longitude'] == start_list[geo][0]) & (train['start_latitude'] == start_list[geo][1]),
              'start_region_2'] = train['start_region_2'][index_start[geo]]
    train.loc[(train['start_longitude'] == start_list[geo][0]) & (train['start_latitude'] == start_list[geo][1]),
              'start_region_3'] = train['start_region_3'][index_start[geo]]

for geos in range(len(end_list)):
    train.loc[(train['end_longitude'] == end_list[geos][0]) & (train['end_latitude'] == end_list[geos][1]),
              'end_region_1'] = train['end_region_1'][index_end[geos]]
    train.loc[(train['end_longitude'] == end_list[geos][0]) & (train['end_latitude'] == end_list[geos][1]),
              'end_region_2'] = train['end_region_2'][index_end[geos]]
    train.loc[(train['end_longitude'] == end_list[geos][0]) & (train['end_latitude'] == end_list[geos][1]),
              'end_region_3'] = train['end_region_3'][index_end[geos]]

start_list, end_list, index_start, index_end, i = [], [], [], [], 0
while len(start_list) < 294 or len(end_list) < 296:
    if (test['start_longitude'][i], test['start_latitude'][i]) not in start_list:
        start_list.append((test['start_longitude'][i], test['start_latitude'][i]))
        start_1, start_2, start_3 = lat_lon_to_addr(test['start_longitude'][i], test['start_latitude'][i])
        test.loc[i, 'start_region_1'] = start_1
        test.loc[i, 'start_region_2'] = start_2
        test.loc[i, 'start_region_3'] = start_3
        index_start.append(i)
    if (test['end_longitude'][i], test['end_latitude'][i]) not in end_list:
        end_list.append((test['end_longitude'][i], test['end_latitude'][i]))
        end_1, end_2, end_3 = lat_lon_to_addr(test['end_longitude'][i], test['end_latitude'][i])
        test.loc[i, 'end_region_1'] = end_1
        test.loc[i, 'end_region_2'] = end_2
        test.loc[i, 'end_region_3'] = end_3
        index_end.append(i)
    i += 1
    print(f'i: {i} | start list 개수: {len(start_list)}, end list 개수: {len(end_list)}')

for geo in range(len(start_list)):
    test.loc[(test['start_longitude'] == start_list[geo][0]) & (test['start_latitude'] == start_list[geo][1]),
              'start_region_1'] = test['start_region_1'][index_start[geo]]
    test.loc[(test['start_longitude'] == start_list[geo][0]) & (test['start_latitude'] == start_list[geo][1]),
              'start_region_2'] = test['start_region_2'][index_start[geo]]
    test.loc[(test['start_longitude'] == start_list[geo][0]) & (test['start_latitude'] == start_list[geo][1]),
              'start_region_3'] = test['start_region_3'][index_start[geo]]

for geos in range(len(end_list)):
    test.loc[(test['end_longitude'] == end_list[geos][0]) & (test['end_latitude'] == end_list[geos][1]),
              'end_region_1'] = test['end_region_1'][index_end[geos]]
    test.loc[(test['end_longitude'] == end_list[geos][0]) & (test['end_latitude'] == end_list[geos][1]),
              'end_region_2'] = test['end_region_2'][index_end[geos]]
    test.loc[(test['end_longitude'] == end_list[geos][0]) & (test['end_latitude'] == end_list[geos][1]),
              'end_region_3'] = test['end_region_3'][index_end[geos]]

print(train.isnull().sum())
print(test.isnull().sum())

train.to_parquet('../data/train_address.parquet', index=False)
test.to_parquet('../data/test_address.parquet', index=False)
