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
    x_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    d_train = lgb.Dataset(x_train, label=y_train,
                          categorical_feature=['road_rating',
                                               'road_name', 'connect_code', 'road_type',
                                               'start_node_name', 'day_of_week',
                                               'start_turn_restricted', 'end_node_name', 'end_turn_restricted'])
    params = {
        'objective': 'regression',
        "verbose": -1,
        'metric': 'mse',
        'device_type': 'gpu',
        'learning_rate': trial.suggest_float("learning_rate", 1e-8, 1e-2),
        'num_leaves': trial.suggest_int('num_leaves', 2, 1024),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        'subsample': trial.suggest_float('subsample', 0.4, 1),
    }

    # Generate model
    bst = lgb.train(params, d_train)
    MSE = mean_absolute_error(y_val, bst.predict(X_val))
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
