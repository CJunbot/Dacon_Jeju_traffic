import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

CAT = pd.read_csv('../data/submit_cat_fold.csv')
LGBM = pd.read_csv('../data/submit_lgbm_fold.csv')
XGB = pd.read_csv('../data/submit_xgb_fold.csv')
sample_submission = pd.read_csv('../data/sample_submission.csv')

ensemble = 0.4*CAT['target'] + 0.5*LGBM['target'] + 0.1*XGB['target']
sample_submission['target'] = ensemble

sample_submission.to_csv("../data/submit_emsemble.csv", index=False)
