from src.preprocessing.utils import clean_data, engineer_features, significant_lags_dict, add_significant_lag
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class CleanData(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return clean_data(X)

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return engineer_features(X)

class LaggedFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        self.significant_lags_ = significant_lags_dict(X)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = add_significant_lag(X, self.significant_lags_)
        return X