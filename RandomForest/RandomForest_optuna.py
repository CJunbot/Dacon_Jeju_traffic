import numpy as np
from sklearn.ensemble import RandomForestRegressor
import optuna
from optuna.samplers import TPESampler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

sampler = TPESampler(seed=10)


def objective(trial):
    train = pd.read_parquet('../data/train_rf.parquet')

    y = train['target']
    x = np.array(train.drop(columns=['target']))
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=42, shuffle=True)

    params = {}
    params["n_estimators"] = trial.suggest_int('n_estimators', 100, 10500),
    params["min_samples_split"] = trial.suggest_int('min_samples_split', 1, 150),  # 8~
    params["max_depth"] = trial.suggest_int('max_depth', 6, 50),  # 6~100
    params["min_samples_leaf"] = trial.suggest_int('min_samples_leaf', 6, 60),  # 8~
    params["max_leaf_nodes"] = trial.suggest_int('max_leaf_nodes', 2, 1000),
    params["min_weight_fraction_leaf"] = trial.sugget_float('min_impurity_decrease', 0.001, 0.5),
    params["min_impurity_decrease"] = trial.sugget_float('min_impurity_decrease', 0.001, 0.5),
    params["max_features"] = trial.sugget_categorical('max_features', ['auto', 'sqrt']),  # sqrt
    params["bootstrap"] = True,  # False
    params["oob_score"] = True,  # False
    params["random_state"] = 42,
    params["n_jobs"] = -1

    model = RandomForestRegressor(**params)
    model.fit(x_train, y_train)

    score = model.predict(x_val)
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
