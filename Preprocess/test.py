import json
import requests
import pandas as pd

api_key = '610093332b1a7815b94a196b4179ff1d'

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 800)

train = pd.read_parquet('../data/train_address2.parquet')
test = pd.read_parquet('../data/test_address2.parquet')


def lat_lon_to_addr(longitude, latitude):
    try:
        url = 'https://dapi.kakao.com/v2/local/geo/coord2regioncode.json?x={longitude}&y={latitude}'.format(longitude=longitude, latitude=latitude)
        headers = {"Authorization": "KakaoAK " + api_key}
        result = json.loads(str(requests.get(url, headers=headers).text))
        return result['documents'][0]['region_2depth_name'], result['documents'][0]['region_3depth_name'], result['documents'][0]['region_4depth_name']
    except:
        return '', '', ''


for i in range(100000, 150000):
    start_1, start_2, start_3 = lat_lon_to_addr(train['start_longitude'][i], train['start_latitude'][i])
    end_1, end_2, end_3 = lat_lon_to_addr(train['end_longitude'][i], train['end_latitude'][i])
    train.loc[i, 'start_region_1'] = start_1
    train.loc[i, 'start_region_2'] = start_2
    train.loc[i, 'start_region_3'] = start_3
    train.loc[i, 'end_region_1'] = end_1
    train.loc[i, 'end_region_2'] = end_2
    train.loc[i, 'end_region_3'] = end_3
    print(f'{i}/150000')

train.to_parquet('../data/train_address2.parquet', index=False)
test.to_parquet('../data/test_address2.parquet', index=False)

print(train.head(20))
