import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

train = pd.read_parquet('../data/train_after.parquet')
test = pd.read_parquet('../data/test_after.parquet')


fig, axes = plt.subplots(figsize=(20,10))
xticks = [i for i in range(25)]
axes.set_xticks(xticks)
axes.tick_params(labelsize=10)

sns.lineplot(x='base_hour',y='target',data=train).set(title='base_hour vs target')

plt.tight_layout()
plt.show()
