import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

train = pd.read_parquet('../data/train_after.parquet')
y = train['target']
x = train.drop(columns=['target'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
test = pd.read_parquet('../data/test_after.parquet')

bst = lgb.Booster(model_file='model2.txt')
accuracy = mean_absolute_error(y_test, bst.predict(x_test))
print(accuracy)

pred = bst.predict(test)

sample_submission = pd.read_csv('../data/sample_submission.csv')
sample_submission['target'] = pred
sample_submission.to_csv("../data/submit.csv", index=False)
