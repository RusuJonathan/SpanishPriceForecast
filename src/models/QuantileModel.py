from sklearn.base import BaseEstimator,RegressorMixin
from src.models.model_builders import build_pipeline, set_model_params
from src.models.hpo_tuner import run_hyperparameter_optimization
from lightgbm import LGBMRegressor
import pandas as pd

class QuantileModel(BaseEstimator, RegressorMixin):
    def __init__(self, config: dict, n_trials: int = 300, quantiles : tuple[int] = (0.1, 0.25, 0.5, 0.75, 0.9)):
        self.config = config
        self.quantiles = quantiles
        self.build_pipeline_ = build_pipeline
        self.run_hyperparameter_optimization_ = run_hyperparameter_optimization
        self.set_model_params_ = set_model_params
        self.n_trials = n_trials
        self.models = {}
        self._init_models()

    def _init_models(self):
        for quantile in self.quantiles:
            self.models[quantile] = self.build_pipeline_(LGBMRegressor, param={"objective": "quantile",
                                                                               "alpha": quantile,
                                                                               "verbosity": -1})

    def fit(self, X: pd.DataFrame, y: pd.Series):
        for quantile in self.quantiles:
            self.models[quantile].fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({f"Quantile {q}": model.predict(X) for q, model in self.models.items()}, index=X.index)


    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series):
        for quantile, model in self.models.items():
            study = self.run_hyperparameter_optimization_(
                model=model,
                X=X,
                y=y,
                model_config=self.config,
                alpha=quantile,
                n_trials=self.n_trials
            )
            self.set_model_params_(model, study.best_params)
