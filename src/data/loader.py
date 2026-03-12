import pandas as pd
from pathlib import Path
import yaml

base_dir = Path(__file__).resolve().parent.parent.parent
config_dir = base_dir / "config"
raw_data_dir = base_dir / "data" / "raw"
processed_data_dir = base_dir / "data" / "processed"
prediction_long_dir = base_dir / "data" / "prediction_long"
prediction_short_dir = base_dir / "data" / "prediction_short"
assets_dir = base_dir / "assets"


energy_data = raw_data_dir / "energy_data.parquet"
processed_energy_data = processed_data_dir / "processed_energy_data.parquet"
prediction_long = prediction_long_dir / "prediction_long.parquet"
prediction_short = prediction_short_dir / "prediction_short.parquet"

preprocessing_path = config_dir / "preprocessing.yaml"
model_config_path = config_dir / "model_config.yaml"

def load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.columns = df.columns.str.replace(" ", "_")
    return df

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)