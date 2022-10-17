import optuna
from optuna.samplers import TPESampler
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

sampler = TPESampler(seed=10)


def objective(trial):
    train = pd.read_parquet('data/train_after.parquet')
    y = train['target']
    x = train.drop(columns=['target'])
    x2, x_test, y2, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(x2, y2, test_size=0.1, random_state=42, shuffle=True)

    params = {}
    params['objective'] = 'regression'
    params["verbose"] = -1
    params['metric'] = 'l1'
    params['device_type'] = 'gpu'
    params['boosting_type'] = 'dart'
    params['learning_rate'] = trial.suggest_float("learning_rate", 0.05, 0.09)
    # 예측력 상승
    params['num_iterations'] = trial.suggest_int('num_iterations', 1000, 10000)  # = num round, num_boost_round
    params['min_child_samples'] = trial.suggest_int('min_child_samples', 100, 200)
    params['n_estimators'] = trial.suggest_int('n_estimators', 5000, 10000)
    params['subsample'] = trial.suggest_float('subsample', 0.6, 1)
    params['num_leaves'] = trial.suggest_int('num_leaves', 1500, 5024)
    params['max_depth'] = trial.suggest_int('max_depth', 20, 40)
    # overfitting 방지
    params['min_child_weight'] = trial.suggest_float('min_child_weight', 1e-3, 0.6)
    params['min_child_samples'] = trial.suggest_int('min_child_samplesh', 25, 60)
    params['subsample_freq'] = trial.suggest_int('subsample_freq', 1, 99)
    params['feature_fraction'] = trial.suggest_float('feature_fraction', 0.7, 0.95)

    # Generate model
    bst = lgb.LGBMRegressor(**params)
    bst.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric='l1', early_stopping_rounds=5)
    pred = bst.predict(x_test, num_iteration=bst.best_iteration_)
    MSE = mean_absolute_error(y_test, pred)
    return MSE


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))