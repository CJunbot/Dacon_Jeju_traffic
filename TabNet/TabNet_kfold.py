import pandas as pd
import numpy as np
from pytorch_tabnet.augmentations import RegressionSMOTE
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import KFold
import random, os
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


train = pd.read_parquet('../data/train_after.parquet')
test = pd.read_parquet('../data/test_after.parquet')

y = train['target']
x = train.drop(columns=['target'])

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

seed = 42

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(seed)

cat_idxs = [x.columns.get_loc(c) for c in x.select_dtypes(exclude='float32').columns]
device = "cuda" if torch.cuda.is_available() else "cpu"

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

y_pred = np.zeros(len(test))

for tr_idx, val_idx in kf.split(x):
    x_train, x_val = x[tr_idx], x[val_idx]
    y_train, y_val = y[tr_idx], y[val_idx]

    clf = TabNetRegressor(n_d=16,
                          n_a=16,
                          n_steps=4,
                          gamma=1.9,
                          n_independent=4,
                          n_shared=5,
                          seed=seed,
                          device_name=device,
                          optimizer_fn=torch.optim.Adam,
                          cat_idxs=cat_idxs,
                          scheduler_params={"milestones": [150, 250, 300, 350, 400, 450], 'gamma': 0.2},
                          scheduler_fn=torch.optim.lr_scheduler.MultiStepLR)


    aug = RegressionSMOTE(p=0.2)

    clf.fit(
        X_train=x_train, y_train=y_train,
        eval_set=[(x, y_train), (x_val, y_val)],
        eval_name=['train', 'valid'],
        eval_metric=['mae'],
        max_epochs=1000,
        patience=70,
        batch_size=1024, virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        augmentations=aug,  # aug
    )

    y_pred += clf.predict(test)

y_pred /= n_splits

sample_submission = pd.read_csv('../data/sample_submission.csv')
sample_submission['target'] = y_pred
sample_submission.to_csv("../data/submit_TabNet.csv", index=False)
