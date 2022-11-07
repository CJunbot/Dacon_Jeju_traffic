import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

cat = pd.read_csv('cat_LR.csv')
LGBM = pd.read_csv('LGBM_LR.csv')
XGB = pd.read_csv('XGB_LR.csv')

cash = pd.read_parquet('../data/train_after.parquet')
y = cash['target'].values.reshape(-1,1)

train = np.concatenate((cat, LGBM, XGB), axis=1)
train = pd.DataFrame(train)
print(train.shape)
print("-----------------------------------")
print(train.corr())
plt.figure(figsize=(50,40))
sns.heatmap(train.corr(), annot=True)
plt.show()