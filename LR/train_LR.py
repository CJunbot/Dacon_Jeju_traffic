from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

cat = pd.read_csv('cat_LR.csv')
LGBM = pd.read_csv('LGBM_LR.csv')
XGB = pd.read_csv('XGB_LR.csv')

cash = pd.read_parquet('../data/train_after.parquet')
y = cash['target'].values.reshape(-1,1)

X = np.concatenate((cat, LGBM, XGB), axis=1)

reg = LinearRegression().fit(X, y)
pred = reg.predict(test)
sample_submission = pd.read_csv('../data/sample_submission.csv')
sample_submission['target'] = pred
sample_submission.to_csv("../data/submit_LR.csv", index=False)

print(reg.score(X, y))
