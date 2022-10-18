import pandas as pd
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import optuna


def objective(trial):
    train = pd.read_parquet('data/train_cat.parquet')
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
    param = {}
    param['eval_metric'] = 'RMSE'
    #param['task_type'] = 'GPU'
    param['iterations'] = trial.suggest_int("iterations", 1000, 10000)
    param['learning_rate'] = trial.suggest_float("learning_rate", 0.001, 0.01)
    param['depth'] = trial.suggest_int('depth', 4, 10)
    param['l2_leaf_reg'] = trial.suggest_float('l2_leaf_reg', 2, 10)
    param['min_child_samples'] = trial.suggest_int('min_data_in_leaf', 1, 30)
    param['random_strength'] = trial.suggest_float('random_strength', 0, 10)

    regressor = CatBoostRegressor(**param)
    regressor.fit(train_pool, eval_set=[val_pool], early_stopping_rounds=25)
    loss = mean_absolute_error(y_test, regressor.predict(test_pool))
    return loss


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, n_jobs=-1)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

