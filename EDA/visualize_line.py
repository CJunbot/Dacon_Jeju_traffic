import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

train = pd.read_parquet('../data/train_after.parquet')
test = pd.read_parquet('../data/test_after.parquet')


def plot_colname_vs_target(col_name):
    train_time = train[[col_name, 'target']]
    train_time = train_time.groupby(col_name).mean().reset_index()
    return train_time

# train_day = train[['day_of_week','target']]
# train_day = train_day.groupby('day_of_week').mean()


fig, ax = plt.subplots(figsize=(10,6))
xticks = [i for i in range(24)]
ax.set_xticks(xticks)
ax.tick_params(labelsize=24)

col_name = 'month'
train_time = plot_colname_vs_target(col_name)
sns.lineplot(x=col_name, y='target', data=train_time).set(title=col_name + ' vs target')
# plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
