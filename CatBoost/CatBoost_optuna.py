import pandas as pd
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import optuna
from optuna.samplers import TPESampler


def objective(trial):
    # initialize data
    train = pd.read_parquet('../data/train_cat.parquet')
    y = train['target']
    x = train.drop(columns=['target'])

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    train_pool = Pool(x_train,
                      y_train,
                      cat_features=['road_name', 'start_node_name', 'end_node_name',
                                    'start_region_2', 'end_region_2'])
    val_pool = Pool(x_val,
                    y_val,
                    cat_features=['road_name', 'start_node_name', 'end_node_name',
                                  'start_region_2', 'end_region_2'])

    test_pool = Pool(x_val,
                     cat_features=['road_name', 'start_node_name', 'end_node_name',
                                   'start_region_2', 'end_region_2'])
    param = {
        "loss_function": "RMSE",
        "learning_rate": 0.027788803687820704,  # 중요도 1
        "l2_leaf_reg": 0.19300591097027178,
        "depth": 16,  # 중요도 3
        "boosting_type": "Plain",
        "bootstrap_type": 'Bernoulli',
        "subsample": 0.960,
        "min_data_in_leaf": 13,
        "devices": '0:1',
        "task_type": 'CPU',
        "eval_metric": 'RMSE',
        "random_seed": 42,
        "metric_period": 10
    }


    regressor = CatBoostRegressor(**param)
    regressor.fit(train_pool, eval_set=[train_pool, val_pool], early_stopping_rounds=15, verbose=100)
    loss = mean_absolute_error(y_val, regressor.predict(test_pool))
    return loss


if __name__ == "__main__":
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=5)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    optuna.visualization.plot_param_importances(study)
