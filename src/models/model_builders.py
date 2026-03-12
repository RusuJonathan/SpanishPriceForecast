from sklearn.pipeline import Pipeline
from src.preprocessing.pipeline import build_preprocessing_pipeline
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import StackingRegressor

def build_pipeline(
        model_class: Any,
        param: Dict[str, Any] | None) -> Pipeline:
    param = param or {}
    return Pipeline(steps=[
        #("preprocessing", build_preprocessing_pipeline()),
        ("model", model_class(**param))
    ])

def set_model_params(
        pipeline: Pipeline,
        param: Dict[str, Any]) -> None:

    pipeline.named_steps["model"].set_params(**param)
