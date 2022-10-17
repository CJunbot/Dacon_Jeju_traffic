import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

train = pd.read_parquet('../data/train_after.parquet')
test = pd.read_parquet('../data/test_after.parquet')


def table(feature):
    return train[[feature, "target"]].groupby([feature], as_index=False).mean().sort_values(by='target', ascending=False).style.background_gradient(low=0.75,high=1)


def bar_plot(feature):
    plt.figure(figsize = (5,3))
    sns.barplot(data = train , x = feature , y = "target").set_title(f"{feature} Vs Target")
    plt.show()


plt.figure()
<<<<<<< HEAD
#bar_plot('road_name')  # 값
sns.displot(train['road_name'])  # 분포
=======
bar_plot('road_in_use')
#sns.displot(train['lane_count'])
>>>>>>> 9ee693be1db224917e85fe1efd172bd7330b199f
plt.show()
