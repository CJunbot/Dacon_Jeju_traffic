import pandas as pd
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import optuna
from optuna.samplers import TPESampler


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

    test_pool = Pool(x_test,
                     cat_features=['day_of_week', 'road_name', 'road_rating', 'connect_code', 'road_type',
                                   'start_node_name',
                                   'start_turn_restricted', 'end_node_name', 'end_turn_restricted'])
    param = {}
    param['learning_rate'] = trial.suggest_discrete_uniform("learning_rate", 0.001, 0.02, 0.001)
    param['depth'] = trial.suggest_int('depth', 9, 15)
    param['l2_leaf_reg'] = trial.suggest_discrete_uniform('l2_leaf_reg', 1.0, 5.5, 0.5)
    param['min_child_samples'] = trial.suggest_int('min_data_in_leaf', 1, 30)
    param['grow_policy'] = 'Depthwise'
    param['iterations'] = trial.suggest_int("iterations", 4000, 25000)
    param['eval_metric'] = 'RMSE'
    param['task_type'] = 'GPU'
    param['od_type'] = 'iter'
    param['od_wait'] = trial.suggest_int('od_wait', 20, 500)
    param['random_strength'] = trial.suggest_uniform('random_strength', 10, 50)
    param['logging_level'] = 'Silent'
    param['subsample'] = trial.suggest_uniform('subsample', 0, 1)
    param['bagging_temperature'] = trial.suggest_loguniform('bagging_temperature', 0.01, 100.00)

    regressor = CatBoostRegressor(**param)
    regressor.fit(train_pool, eval_set=[x_val, y_val], early_stopping_rounds=100)
    loss = mean_absolute_error(y_test, regressor.predict(test_pool))
    return loss


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10000, n_jobs=-1)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

