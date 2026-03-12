import optuna
import pandas as pd
from typing import Dict, Callable, Any
from src.models.model_builders import set_model_params
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_pinball_loss, make_scorer
from sklearn.model_selection import TimeSeriesSplit



def build_objective_function(model: Any,
                             x_train: pd.DataFrame,
                             y_train: pd.Series,
                             model_config: Dict[str, Dict],
                             alpha: float,
                             cv: int = 5) -> Callable:
    def objective(trial):

        pinball_scorer = make_scorer(mean_pinball_loss, alpha=alpha, greater_is_better=False)
        tscv = TimeSeriesSplit(n_splits=cv)
        trial_params = {}

        for param_name, param_info in model_config.items():
            distribution = param_info["distribution"]
            if distribution == "IntUniformDistribution":
                trial_params[param_name] = trial.suggest_int(param_name, low=param_info["low"], high=param_info["high"])
            elif distribution == "LogUniformDistribution":
                trial_params[param_name] = trial.suggest_float(param_name, low=param_info["low"], high=param_info["high"], log=True)
            elif distribution == "UniformDistribution":
                trial_params[param_name] = trial.suggest_float(param_name, low=param_info["low"], high=param_info["high"])
            else:
                trial_params[param_name] = trial.suggest_categorical(param_name, param_info["categories"])

        set_model_params(pipeline=model, param=trial_params)

        scores = cross_val_score(
            estimator=model,
            X=x_train,
            y=y_train,
            scoring=pinball_scorer,
            cv=tscv)

        return - scores.mean()
    return objective

def run_hyperparameter_optimization(
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        model_config: Dict[str, Dict],
        alpha: float,
        n_trials: int,
        cv: int = 5):

    objective_func = build_objective_function(
        model=model,
        x_train=X,
        y_train=y,
        model_config=model_config,
        alpha=alpha,
        cv=cv)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_func, n_trials=n_trials)
    return study
