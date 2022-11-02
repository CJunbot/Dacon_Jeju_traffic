import pandas as pd
import numpy as np
from pytorch_tabnet.augmentations import RegressionSMOTE
from pytorch_tabnet.tab_model import TabNetRegressor
import random, os
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import optuna
from optuna.samplers import TPESampler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


train = pd.read_parquet('../data/train_rf.parquet')
test = pd.read_parquet('../data/test_rf.parquet')

seed = 42

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(seed)

sampler = TPESampler(seed=10)


def objective(trial):
    y = train['target']
    x = train.drop(columns=['target'])
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

    # cat_idxs = [x.columns.get_loc(c) for c in x.select_dtypes(exclude='float32').columns]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    params = {}
    params['mask_type'] = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
    params['n_da'] = trial.suggest_int("n_da", 8, 64, step=4)
    params['n_steps'] = trial.suggest_int("n_steps", 1, 3, step=1)
    params['gamma'] = trial.suggest_float("gamma", 1.0, 2.0, step=0.2)
    params['n_independent'] = trial.suggest_int("n_independent", 1, 5)
    params['n_shared'] = trial.suggest_int("n_shared", 1, 5)
    params['lambda_sparse'] = trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True)
    params['seed'] = seed,
    params['verbose'] = 1,
    params['device_name'] = device,
    params['optimizer_fn'] = torch.optim.Adam,
    params['scheduler_params'] = {"milestones": [150, 250, 300, 350, 400, 450], 'gamma': 0.2},  # {"milestones": [150,250,300,350,400,450],'gamma':0.2}
    params['scheduler_fn'] = torch.optim.lr_scheduler.MultiStepLR

    aug = RegressionSMOTE(p=0.2)

    clf = TabNetRegressor(**params)
    clf.fit(
    X_train = x_train.values, y_train = y_train.values.reshape(-1, 1),
    eval_set = [(x_val.values, y_val.values.reshape(-1, 1))],
    eval_metric = ['mae'],
    max_epochs = 200,
    patience = 70,  # early stopping
    batch_size = 1024, virtual_batch_size = 128,
    num_workers = 0,
    drop_last = False,
    augmentations = aug,  # aug
    )

    score = clf.predict(x_val)
    mae = mean_absolute_error(y_val, score)
    return mae

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=10)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
