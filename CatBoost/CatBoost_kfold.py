import pandas as pd
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import KFold
import numpy as np

# initialize data
train = pd.read_parquet('../data/train_cat.parquet')
test = pd.read_parquet('../data/test_cat.parquet')
y = train['target']
x = train.drop(columns=['target'])

n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

y_for_LR = np.zeros(len(train))
y_pred = np.zeros(len(test))

for tr_idx, val_idx in kf.split(x):
    x_train, x_val = x.iloc[tr_idx], x.iloc[val_idx]
    y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    # initialize Pool
    train_pool = Pool(x_train, y_train,
                      cat_features=['road_name', 'start_node_name', 'end_node_name',
                                    'start_region_2', 'end_region_2'])

    val_pool = Pool(x_val, y_val,
                    cat_features=['road_name', 'start_node_name', 'end_node_name',
                                  'start_region_2', 'end_region_2'])

    val_pool_LR = Pool(x_val,
                    cat_features=['road_name', 'start_node_name', 'end_node_name',
                                  'start_region_2', 'end_region_2'])
    test_pool = Pool(test,
                     cat_features=['road_name', 'start_node_name', 'end_node_name',
                                   'start_region_2', 'end_region_2'])

    # specify the training parameters
    cb_model = CatBoostRegressor(
                                 learning_rate=0.045,  # 0.025
                                 depth=15,
                                 n_estimators=10000,  # 10000 -> 1시간 반정도
                                 bootstrap_type='Bernoulli',
                                 devices='0:1',
                                 task_type='GPU',
                                 eval_metric='RMSE',
                                 random_seed=42,
                                 min_data_in_leaf=47,
                                 l2_leaf_reg= 0.8130860044896614,
                                 subsample=0.9540988370165997,
                                 metric_period=10)

    # train the model
    cb_model.fit(train_pool, eval_set=(val_pool), early_stopping_rounds=25, verbose=100, use_best_model=True)
    # for LR (Ensemble)

    y_for_LR[val_idx] = cb_model.predict(val_pool_LR)
    # make the prediction using the resulting model
    y_pred += cb_model.predict(test_pool)

y_pred /= n_splits

sample_submission = pd.read_csv('../data/sample_submission.csv')
sample_submission['target'] = y_pred
sample_submission.to_csv("../data/submit_cat_fold.csv", index=False)

df = pd.DataFrame(y_for_LR)
df.to_parquet('cat_LR.parquet', index=False)
