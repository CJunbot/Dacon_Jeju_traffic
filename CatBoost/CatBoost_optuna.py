import pandas as pd
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import optuna
from optuna.samplers import TPESampler


def objective(trial):
    train = pd.read_parquet('../data/train_cat.parquet')
    y = train['target']
    x = train.drop(columns=['target'])

    x_2, x_test, y_2, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_2, y_2, test_size=0.12, random_state=42)
    train_pool = Pool(x_train,
                      y_train,
                      cat_features=['day_of_week', 'road_name', 'road_rating', 'connect_code', 'road_type',
                                    'start_node_name',
                                    'start_turn_restricted', 'end_node_name', 'end_turn_restricted'])
    val_pool = Pool(x_val,
                      y_val,
                      cat_features=['day_of_week', 'road_name', 'road_rating', 'connect_code', 'road_type',
                                    'start_node_name',
                                    'start_turn_restricted', 'end_node_name', 'end_turn_restricted'])
    test_pool = Pool(x_test,
                     cat_features=['day_of_week', 'road_name', 'road_rating', 'connect_code', 'road_type',
                                   'start_node_name',
                                   'start_turn_restricted', 'end_node_name', 'end_turn_restricted'])
    param = {
        "loss_function": trial.suggest_categorical("loss_function", ["RMSE", "MAE"]),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e0),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-2, 1e0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 1, 10),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 20),
        "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 20),
    }
    # Conditional Hyper-Parameters
    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    regressor = CatBoostRegressor(**param)
    regressor.fit(train_pool, eval_set=[train_pool, val_pool], early_stopping_rounds=35, verbose=100)
    loss = mean_absolute_error(y_test, regressor.predict(test_pool))
    return loss


if __name__ == "__main__":
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=10)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    optuna.visualization.plot_param_importances(study)
