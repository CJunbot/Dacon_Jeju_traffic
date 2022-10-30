import optuna
from optuna.samplers import TPESampler
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

sampler = TPESampler(seed=10)

def objective(trial):
    train = pd.read_parquet('../data/train_after.parquet')
    y = train['target']
    x = train.drop(columns=['target'])
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

    params = {}
    params['objective'] = 'reg:squaredlogerror'
    params['eval_metric'] = 'rmse'
    params['gpu_id'] = 1
    params['tree_method'] = 'gpu_hist'
    params['learning_rate'] = 0.05  # 0.010 -> 2.884 / 0.028 -> 2.885 / 0.058 -> 2.893
    # 예측력 상승
    params['n_estimators'] = trial.suggest_int('n_estimators', 10500, 20000)  # 8500
    params['max_leaves'] = trial.suggest_int('max_leaves', 150, 15024)  # 100~15000
    params['max_depth'] = trial.suggest_int('max_depth', 4, 40)   # 6~40?
    # overfitting 방지
    params['min_child_weight'] = trial.suggest_int('min_child_weight', 1, 15)  # 높을수록 / 0~무한대 6?
    params['subsample'] = trial.suggest_float('subsample', 0.51, 1.0) # 낮을수록 overfitting down / 0.5~1  = bagging_fraction
    params['reg_alpha'] = trial.suggest_float('reg_alpha', 1e-2, 1)
    params['reg_lambda'] = trial.suggest_float('reg_lambda', 1e-2, 1)
    params['min_split_loss'] = trial.suggest_float('min_split_loss', 0.01, 1, log=True)  # = gamma
    params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.01, 1.0)   # 낮을수록 overfitting down / 0~1
    params['colsample_bylevel'] = trial.suggest_float('colsample_bylevel', 0.01, 1.0)  # 낮을수록 overfitting down / 0~1
    params['colsample_bynode'] = trial.suggest_float('colsample_bynode', 0.01, 1.0)  # 낮을수록 overfitting down / 0~1

    model = XGBRegressor(**params)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        early_stopping_rounds=25,
        verbose=10,
    )
    pred = model.predict(x_val, ntree_limit=model.best_ntree_limit)
    mae = mean_absolute_error(y_val, pred)
    return mae


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=30)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))