from xgboost import XGBRegressor
import pandas as pd
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

test = pd.read_parquet('../data/test_after.parquet')

model = XGBRegressor()
model.load_model("model.json")

pred = model.predict(test, ntree_limit=model.best_ntree_limit)

sample_submission = pd.read_csv('../data/sample_submission.csv')
sample_submission['target'] = pred
sample_submission.to_csv("../data/submit2.csv", index=False)