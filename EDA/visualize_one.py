import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

train = pd.read_parquet('../data/train_after_test.parquet')
test = pd.read_parquet('../data/test_after_test.parquet')


fig, axes = plt.subplots(2, 1, figsize=(20,10))
xticks = [i for i in range(25)]
#axes.set_xticks(xticks)
#axes.tick_params(labelsize=10)

# sns.lineplot(x='base_hour',y='target',data=train).set(title='base_hour vs target') 숫자형 변수
sns.countplot(x='sun', data=train, ax=axes[0])  # 범주형 변수
sns.countplot(x='sun', data=test, ax=axes[1])  # 범주형 변수
plt.tight_layout()
plt.show()

#6~3시까지 증가
#3~8시까지 감소
#8~18시까지 감소