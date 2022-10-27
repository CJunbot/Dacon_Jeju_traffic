import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

train = pd.read_parquet('../data/train_address_pop.parquet')
test = pd.read_parquet('../data/test_after.parquet')

col_name = 'population_dong2'
# continuous feature
plt.plot(figsize=(5, 4))
sns.histplot(x=train[col_name], kde=True)
plt.tight_layout()
plt.show()

train[col_name] = np.log1p(train[col_name])
plt.plot(figsize=(5, 4))
sns.histplot(x=train[col_name], kde=True)
plt.tight_layout()
plt.show()
