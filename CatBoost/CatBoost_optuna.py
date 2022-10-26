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
              "random_state":42,
               'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.05),
               'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
               "n_estimators":trial.suggest_int("n_estimators", 500, 5000),
               "max_depth":trial.suggest_int("max_depth", 4, 16),
              'random_strength' :trial.suggest_int('random_strength', 0, 100),
               "colsample_bylevel":trial.suggest_float("colsample_bylevel", 0.4, 1.0),
               "l2_leaf_reg":trial.suggest_float("l2_leaf_reg",1e-8,3e-5),
               "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
               "max_bin": trial.suggest_int("max_bin", 200, 500),
               'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
           }

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

