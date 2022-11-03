import pandas as pd
import numpy as np
from pytorch_tabnet.augmentations import RegressionSMOTE
from pytorch_tabnet.tab_model import TabNetRegressor
import random, os
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

train = pd.read_parquet('../data/train_rf.parquet')
test = pd.read_parquet('../data/test_rf.parquet')

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

clf = TabNetRegressor(n_d=32,
                       n_a=32,
                       n_steps=6,
                       gamma=1.9,
                       n_independent=4,
                       n_shared=5,
                       seed=seed,
                       verbose=1,
                       device_name=device,
                       optimizer_fn = torch.optim.Adam,
                       optimizer_params = dict(lr=8e-2, weight_decay=2e-5),  # 2e-2
                       scheduler_params = dict(mode="min", patience=5, min_lr=4e-2, factor=0.5),  # 1e-2,  #  {"milestones": [150,250,300,350,400,450],'gamma':0.2}
                       scheduler_fn = torch.optim.lr_scheduler.ReduceLROnPlateau)
aug = RegressionSMOTE(p=0.2)

clf.fit(
    X_train=x_train.values, y_train=y_train.values.reshape(-1,1),
    eval_set=[(x_val.values, y_val.values.reshape(-1,1))],
    eval_metric=['mae'],
    max_epochs=200,
    patience=25,  # early stopping
    batch_size=5120, virtual_batch_size=640,
    num_workers=0,
    drop_last=False,
    augmentations=aug,  # aug
)

y_pred = clf.predict(test)
sample_submission = pd.read_csv('../data/sample_submission.csv')
sample_submission['target'] = y_pred
sample_submission.to_csv("../data/submit_TabNet.csv", index=False)


saving_path_name = "./tabnet_model_test_1"
saved_filepath = clf.save_model(saving_path_name)

plt.figure(figsize=(12, 6))
plt.plot(clf.history['train']['loss'])
plt.plot(clf.history['valid']['loss'])

pd.Series(clf.feature_importances_, index=x.columns).plot.bar(title=f'TabNet Global Feature Importances')





