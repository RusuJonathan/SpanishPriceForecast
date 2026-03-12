from src.models.QuantileModel import QuantileModel
from src.data.loader import load_parquet, processed_energy_data, load_yaml, model_config_path, prediction_long, prediction_short
import pandas as pd

target_long = "Long_Imbalance_Price"
target_short = "Short_Imbalance_Price"

data = load_parquet(processed_energy_data)

X = data.drop(columns=[target_long, target_short])
y_long = data[target_long]
y_short = data[target_short]

# 3 months training window
training_window = 3 * 30 * 24 * 4

# 1 month test window
test_window = 30 * 24 * 4

model_config = load_yaml(model_config_path)["lgbm"]
results = []


for start in range(0, len(data) - training_window - test_window, test_window):
    train_end = start + training_window
    test_end = train_end + test_window

    X_train = X.iloc[start:train_end]
    y_train_long = y_long.iloc[start:train_end]
    y_train_short = y_short.iloc[start:train_end]

    X_test = X.iloc[train_end:test_end]
    y_test_long = y_long.iloc[train_end:test_end]
    y_test_short = y_short.iloc[train_end:test_end]

    model_long = QuantileModel(config=model_config, n_trials=200)
    model_short = QuantileModel(config=model_config, n_trials=200)

    model_long.optimize_hyperparameters(X_train, y_train_long)
    model_short.optimize_hyperparameters(X_train, y_train_short)

    model_long.fit(X_train, y_train_long)
    model_short.fit(X_train, y_train_short)

    results.append({
        "long": model_long.predict(X_test),
        "short": model_short.predict(X_test),
        "y_long": y_test_long,
        "y_short": y_test_short
    })

all_predictions_long = []
all_predictions_short = []

for result in results:
    pred_long = result["long"]
    pred_long["y_true"] = result["y_long"].values
    pred_long.index = result["y_long"].index
    all_predictions_long.append(pred_long)

    pred_short = result["short"]
    pred_short["y_true"] = result["y_short"].values
    pred_short.index = result["y_short"].index
    all_predictions_short.append(pred_short)

predictions_long = pd.concat(all_predictions_long)
predictions_short = pd.concat(all_predictions_short)

prediction_long.parent.mkdir(parents=True, exist_ok=True)
prediction_short.parent.mkdir(parents=True, exist_ok=True)

predictions_long.to_parquet(prediction_long)
predictions_short.to_parquet(prediction_short)
