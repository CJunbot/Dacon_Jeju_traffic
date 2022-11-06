import optuna
from optuna.samplers import TPESampler
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

sampler = TPESampler(seed=10)

def objective(trial):
    train = pd.read_parquet('../data/train_after.parquet')
    y = train['target']
    x = train.drop(columns=['target'])
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

    params = {}
    params['objective'] = 'regression'
    params["verbose"] = -1
    params['metric'] = 'l1'
    params['device_type'] = 'gpu'
    params['boosting_type'] = 'gbdt'
    params['learning_rate'] = trial.suggest_float("learning_rate", 0.009, 0.05)
    # 예측력 상승
    params['num_iterations'] = 5000
    params['min_child_samples'] = trial.suggest_int('min_child_samples', 100, 150)
    params['n_estimators'] = trial.suggest_int('n_estimators', 15000, 25000)
    params['num_leaves'] = trial.suggest_int('num_leaves', 7000, 15024)
    params['max_depth'] = trial.suggest_int('max_depth', 33, 45)
    # overfitting 방지
    params['min_child_weight'] = trial.suggest_float('min_child_weight', 0.5, 2)
    params['min_child_samples'] = trial.suggest_int('min_child_samplesh', 35, 60)
    params['bagging_fraction'] = trial.suggest_float('bagging_fraction', 0.5, 0.8)
    params['subsample_freq'] = trial.suggest_int('subsample_freq', 70, 99)
    params['lambda_l1'] = trial.suggest_float('lambda_l1', 0.2, 2)
    params['lambda_l2'] = trial.suggest_float('lambda_l2', 0.2, 2)
    params['min_gain_to_split'] = trial.suggest_float('min_gain_to_split', 0.01, 3)
    params['feature_fraction'] = trial.suggest_float('feature_fraction', 0.6, 1)

    # Generate model
    bst = lgb.LGBMRegressor(**params)
    bst.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric='l1', early_stopping_rounds=25)
    pred = bst.predict(x_val, num_iteration=bst.best_iteration_)
    MSE = mean_absolute_error(y_val, pred)
    return MSE


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=40)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))