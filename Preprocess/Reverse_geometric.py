import json, time
import requests
import pandas as pd

api_key = '610093332b1a7815b94a196b4179ff1d'

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 800)

train = pd.read_parquet('../data/train_address.parquet')
test = pd.read_parquet('../data/test_address.parquet')


def lat_lon_to_addr(longitude, latitude):
    try:
        url = 'https://dapi.kakao.com/v2/local/geo/coord2regioncode.json?x={longitude}&y={latitude}'.format(longitude=longitude, latitude=latitude)
        headers = {"Authorization": "KakaoAK " + api_key}
        result = json.loads(str(requests.get(url, headers=headers).text))
        return result['documents'][0]['region_2depth_name'], result['documents'][0]['region_3depth_name'], result['documents'][0]['region_4depth_name']
    except Exception as e:
        print(e)
        return None, None, None

start = time.time()
for i in range(40000, 120000):
    start_1, start_2, start_3 = lat_lon_to_addr(train['start_longitude'][i], train['start_latitude'][i])
    end_1, end_2, end_3 = lat_lon_to_addr(train['end_longitude'][i], train['end_latitude'][i])
    train.loc[i, 'start_region_1'] = start_1
    train.loc[i, 'start_region_2'] = start_2
    train.loc[i, 'start_region_3'] = start_3
    train.loc[i, 'end_region_1'] = end_1
    train.loc[i, 'end_region_2'] = end_2
    train.loc[i, 'end_region_3'] = end_3
    if i % 1000 == 0:
        print(i)
end = time.time() - start
print(f'실행시간: {end//60}분 {end%60}초')

train.to_parquet('../data/train_address.parquet', index=False)
train.to_csv('../data/확인용.csv', encoding='ANSI')
#test.to_parquet('../data/test_address.parquet', index=False)
