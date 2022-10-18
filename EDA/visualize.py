import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

train = pd.read_parquet('../data/train_after.parquet')
test = pd.read_parquet('../data/test_after.parquet')


# continuous feature
fig, axes = plt.subplots(2, 2, figsize=(20,10))
sns.histplot(x=train['road_rating'],kde=True,ax=axes[0][0])
sns.histplot(x=train['weight_restricted'],kde=True,ax=axes[0][1])
sns.histplot(x=train['maximum_speed_limit'],kde=True,ax=axes[1][0])
sns.histplot(x=train['target'],kde=True,ax=axes[1][1])
plt.tight_layout()
plt.show()

# categorical feature
fig, axes = plt.subplots(12, 2, figsize=(17,30))
sns.countplot(x='day_of_week', data=train, ax=axes[0][0])
sns.countplot(x='day_of_week', data=test, ax=axes[0][1])
sns.countplot(x='base_hour', data=train, ax=axes[1][0])
sns.countplot(x='base_hour', data=test, ax=axes[1][1])
sns.countplot(x='road_in_use', data=train, ax=axes[2][0])
sns.countplot(x='road_in_use', data=test, ax=axes[2][1])
sns.countplot(x='lane_count', data=train, ax=axes[3][0])
sns.countplot(x='lane_count', data=test, ax=axes[3][1])
sns.countplot(x='road_rating', data=train, ax=axes[4][0])
sns.countplot(x='road_rating', data=test, ax=axes[4][1])
sns.countplot(x='multi_linked', data=train, ax=axes[5][0])
sns.countplot(x='multi_linked', data=test, ax=axes[5][1])
sns.countplot(x='connect_code', data=train, ax=axes[6][0])
sns.countplot(x='connect_code', data=test, ax=axes[6][1])
sns.countplot(x='maximum_speed_limit', data=train, ax=axes[7][0])
sns.countplot(x='maximum_speed_limit', data=test, ax=axes[7][1])
sns.countplot(x='weight_restricted', data=train, ax=axes[8][0])
sns.countplot(x='weight_restricted', data=test, ax=axes[8][1])
sns.countplot(x='road_type', data=train, ax=axes[9][0])
sns.countplot(x='road_type', data=test, ax=axes[9][1])
sns.countplot(x='start_turn_restricted', data=train, ax=axes[10][0])
sns.countplot(x='start_turn_restricted', data=test, ax=axes[10][1])
sns.countplot(x='end_turn_restricted', data=train, ax=axes[11][0])
sns.countplot(x='end_turn_restricted', data=test, ax=axes[11][1])
plt.tight_layout()
plt.show()
