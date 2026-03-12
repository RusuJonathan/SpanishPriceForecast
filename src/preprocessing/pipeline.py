from sklearn.pipeline import Pipeline
from src.preprocessing.transformers import CleanData, FeatureEngineering, LaggedFeatures
from src.data.loader import load_parquet, energy_data, processed_energy_data

from sklearn import set_config
set_config(transform_output="pandas")

def build_preprocessing_pipeline():
    return Pipeline(
        steps=[
            ("cleaning", CleanData()),
            ("feature_engineering", FeatureEngineering()),
            ("lagged_features", LaggedFeatures())
        ]
    )

if __name__ == "__main__":
    data = load_parquet(energy_data)

    processed_energy_data.parent.mkdir(parents=True, exist_ok=True)

    preprocessor = build_preprocessing_pipeline()
    new_data = preprocessor.fit_transform(data)
    new_data.to_parquet(processed_energy_data)