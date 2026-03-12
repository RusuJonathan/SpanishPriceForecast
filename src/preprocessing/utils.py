import pandas as pd
import numpy as np
from src.data.loader import load_parquet,load_yaml, energy_data, preprocessing_path
from statsmodels.tsa.stattools import pacf


preprocessing_config = load_yaml(preprocessing_path)

FILL_ZEROS = preprocessing_config["FILL_ZEROS"]
FILL_FORECASTS = preprocessing_config["FILL_FORECASTS"]
FEATURE_ENGINEERING = preprocessing_config["FEATURE_ENGINEERING"]

def fill_zeros(data: pd.DataFrame, features: list[str] = FILL_ZEROS) -> pd.DataFrame:
    """
    Fills NaN values with 0, mostly relevant for energy storage features

    :param data: dataframe where each column is a variable and each row is an observation
    :param features: list of feature or feature you want to fill 0 with
    :return: dataframe with filled rows
    """
    data[features] = data[features].fillna(0)
    return data


def fill_forecast(data: pd.DataFrame, config: dict = FILL_FORECASTS) -> pd.DataFrame:
    """
    Fills forecast columns using cross-forecast imputation and linear interpolation.
    For each forecast pair defined in the config, NaNs in the intraday forecast are filled
    with the day-ahead counterpart and vice versa. Remaining NaNs are filled with linear interpolation.
    :param data: dataframe where each column is a variable and each row is an observation
    :param config: dictionary mapping forecast names to their intraday and day-ahead column names
    :return: dataframe with filled forecast columns
    """

    for fill_type in config.keys():
        feature_config = config[fill_type]
        intraday_forecast = feature_config["intraday_forecast"]
        day_ahead_forecast = feature_config["day_ahead_forecast"]

        if intraday_forecast is not None and day_ahead_forecast is not None:
            data[intraday_forecast] = data[intraday_forecast].fillna(data[day_ahead_forecast])
            data[day_ahead_forecast] = data[day_ahead_forecast].fillna(data[intraday_forecast])

        cols_to_interpolate = [c for c in [intraday_forecast, day_ahead_forecast] if c is not None]
        data[cols_to_interpolate] = data[cols_to_interpolate].interpolate(method="linear", axis=0)
    return data

def forward_fill(data: pd.DataFrame) -> pd.DataFrame:
    """
    Forward fills all the remaining NaNs
    :param data: dataframe where each column is a variable and each row is an observation
    :return: dataframe with filled NaNs
    """
    return data.ffill()


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all cleaning steps sequentially: fills energy storage with zeros,
    fills forecast columns via cross-imputation and interpolation,
    then forward fills remaining NaNs.
    :param data: dataframe where each column is a variable and each row is an observation
    :return: cleaned dataframe
    """
    data = data.copy()
    data = fill_zeros(data)
    data = fill_forecast(data)
    return forward_fill(data)

def calculate_total_production(data: pd.DataFrame, config: dict = FEATURE_ENGINEERING) -> pd.DataFrame:
    """
    Calculates the net actual total production by summing all production types
    and subtracting production-related consumption (e.g. hydraulic pumping).
    :param data: dataframe where each column is a variable and each row is an observation
    :param config: dictionary containing production and consumption column names and output column name
    :return: dataframe with a new Total_Production_Actual column
    """
    total_production_config = config["TOTAL_PRODUCTION"]
    consumption_cols = total_production_config["CONSUMPTION_COLS"]
    production_cols = total_production_config["PRODUCTION_COLS"]
    output_name = total_production_config["output"]

    data[output_name] = data[production_cols].sum(axis=1) - data[consumption_cols].sum(axis=1)
    return data

def calculate_delta(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Calculates the delta between the forecast and the actual, in the case of total production, we have a special case
    because we need to remove the forecasted consumption from the production
    :param data: dataframe where each column is a variable and each row is an observation
    :param config: dictionary containing forecast and actual column names and output column nam
    :return: dataframe with new delta column
    """
    if "forecast" in config:
        data[config["output"]] = data[config["forecast"]] - data[config["actual"]]
    else:
        data[config["output"]] = data[config["forecast_aggregated"]] - data[config["forecast_consumption"]] - data[config["actual"]]

    return data

def calculate_all_deltas(data: pd.DataFrame, config: dict = FEATURE_ENGINEERING) -> pd.DataFrame:
    """
    Calculates all the deltas between forecast and actual
    :param data: dataframe where each column is a variable and each row is an observation
    :param config: dictionnary containing other dictionnary relevant for calculate_delta function
    :return: dataframe with new delta columns
    """
    for _, config_delta in config["DELTA_FEATURES"].items():
        data = calculate_delta(data=data, config=config_delta)

    return data

def calculate_scheduled_exchanges(data: pd.DataFrame, config: dict = FEATURE_ENGINEERING) -> pd.DataFrame:
    """
    Calculates the scheduled exchange for imports and exports, by summing all of the imports and exports from
    neighbor countries
    :param data: dataframe where each column is a variable and each row is an observation
    :param config: dictionnary containing output names and scheduled exchanges columns names
    :return: dataframe with new features
    """
    scheduled_exchange_config = config["SCHEDULED_EXCHANGES"]
    for _, exchange_dict in scheduled_exchange_config.items():
        data[exchange_dict["output"]] = data[exchange_dict["cols"]].sum(axis=1)

    return data

def calculate_remainder_volume(data: pd.DataFrame, config: dict = FEATURE_ENGINEERING) -> pd.DataFrame: #data[remainder_config[]]
    """
    Calculates the scheduled remainder volume as the net position of the system after scheduled imports/exports and production/consumption
    :param data: dataframe where each column is a variable and each row is an observation
    :param config: dictionnary containing info in order to calculate the remainder volume
    :return: dataframe with new features
    """
    remainder_config = config["REMAINDER_VOLUME"]
    data[remainder_config["output"]] = (data[remainder_config["forecasted_load"]] -
                                        (data[remainder_config["forecast_aggregated"]] -
                                         data[remainder_config["forecast_consumption"]] +
                                         data[remainder_config["scheduled_imports"]] -
                                         data[remainder_config["scheduled_exports"]]))
    return data

def calculate_available_intraday_flow(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Calculate the available intraday flow as the net transfer capacity minus the day ahead allocated flow
    :param data: dataframe where each column is a variable and each row is an observation
    :param config: dictionnary containing info in order to calculate the available flow
    :return: dataframe with new feature
    """
    data[config["output"]] = data[config["ntc"]] - data[config["exchange"]]
    return data

def calculate_all_available_intraday_flow(data: pd.DataFrame, config: dict = FEATURE_ENGINEERING) -> pd.DataFrame:
    """
    Calculate the available intraday flow as the net transfer capacity minus the day ahead allocated flow
    :param data: dataframe where each column is a variable and each row is an observation
    :param config: dictionnary containing info in order to calculate the available flow
    :return: dataframe with new feature
    """

    ntc_config = config["NTC_AVAILABLE"]
    data = data.copy()
    for ntc_dict in ntc_config:
        data = calculate_available_intraday_flow(data, ntc_dict)
    return data

def add_datetime_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds cyclical features to capture cyclical behaviors of energy markets
    :param data: dataframe where each column is a variable and each row is an observation
    :return: dataframe with cyclical features
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex")

    data["hour_sin"] = np.sin(2 * np.pi * data.index.hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data.index.hour / 24)

    data["day_of_week_sin"] = np.sin(2 * np.pi * data.index.dayofweek / 7)
    data["day_of_week_cos"] = np.cos(2 * np.pi * data.index.dayofweek / 7)

    data["month_sin"] = np.sin(2 * np.pi * data.index.month / 12)
    data["month_cos"] = np.cos(2 * np.pi * data.index.month / 12)
    return data

def significant_lag(data: pd.DataFrame, feature: str, nlags: int, top_lags: int, min_lag: int) -> list[int]:
    """
    Identifies the most significant lags of a feature using Partial Autocorrelation Function (PACF).
    Lags are ranked by their absolute PACF value and filtered to only keep those above a minimum lag,
    ensuring that only lags available at prediction time are returned.
    :param data: dataframe where each column is a variable and each row is an observation
    :param feature: name of the column to compute PACF on
    :param nlags: maximum number of lags to consider when computing PACF
    :param top_lags: number of top lags to retain, ranked by absolute PACF value
    :param min_lag: minimum lag threshold — lags below this value are excluded to respect the prediction horizon
    :return: list of the most significant lag indices above min_lag
    """
    pacf_feature = pacf(data[feature], nlags=nlags)
    significant_lags = np.argsort(np.abs(pacf_feature))
    significant_lags = significant_lags[significant_lags > min_lag][-top_lags:]
    return significant_lags.tolist()


def add_significant_lagged_features(data: pd.DataFrame, config: dict = FEATURE_ENGINEERING) -> pd.DataFrame:
    """
    Adds lagged versions of actual features to the dataframe, using PACF to identify the most
    informative lags. Each feature is replaced by its lagged versions, ensuring that only lags
    available at prediction time are kept (above min_lag). A lag of 17 periods is always included
    as a baseline to guarantee at least one lag within the prediction horizon.
    Features with zero variance are skipped as PACF cannot be computed on constant series.
    :param data: dataframe where each column is a variable and each row is an observation
    :param config: dictionary containing lagged feature names and lag period configuration
    :return: dataframe with original actual features replaced by their significant lagged versions
    """
    lagged_features = config["LAGGED_FEATURES"]
    lag_config = config["LAG_PERIODS"]

    all_lagged = []

    for feat in lagged_features:
        if data[feat].std() == 0:
            continue
        significant_lags = significant_lag(data=data,
                                           feature=feat,
                                           nlags=lag_config["max_lags"],
                                           top_lags=lag_config["significant_lags"],
                                           min_lag=lag_config["min_lag"])

        all_lagged.append(data[feat].shift(periods=list(set(significant_lags) | {17})))

    if all_lagged:
        data = pd.concat([data] + all_lagged, axis=1)

    return data.drop(columns=lagged_features).dropna()

def significant_lags_dict(data: pd.DataFrame, config: dict = FEATURE_ENGINEERING) -> dict:
    lagged_features = config["LAGGED_FEATURES"]
    lag_config = config["LAG_PERIODS"]

    most_significant_lag = {}

    for feat in lagged_features:
        if data[feat].std() == 0:
            continue

        significant_lags = significant_lag(data=data,
                                           feature=feat,
                                           nlags=lag_config["max_lags"],
                                           top_lags=lag_config["significant_lags"],
                                           min_lag=lag_config["min_lag"])

        most_significant_lag[feat] = significant_lags

    return most_significant_lag

def add_significant_lag(data: pd.DataFrame, significant_lag_dict: dict, config : dict = FEATURE_ENGINEERING) -> pd.DataFrame:
    lagged_features = config["LAGGED_FEATURES"]

    all_lagged = []
    for feat, lags in significant_lag_dict.items():
        all_lagged.append(data[feat].shift(periods=list(set(lags) | {17})))

    if all_lagged:
        data = pd.concat([data] + all_lagged, axis=1)
    cols_to_drop = list(set(lagged_features) - {"Long_Imbalance_Price"} - {"Short_Imbalance_Price"})
    return data.drop(columns=cols_to_drop).dropna()

def lagged_features(data: pd.DataFrame, config : dict = FEATURE_ENGINEERING) -> pd.DataFrame:
    lag_config = config["FIXED_LAGS"]
    actuals_features = config["LAGGED_FEATURES"]
    all_lagged = []
    for feature, list_of_lags in lag_config.items():
        all_lagged.append(data[feature].shift(periods=list_of_lags))

    data = pd.concat([data] + all_lagged, axis=1)
    return data.drop(columns=actuals_features)

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data = calculate_total_production(data)
    data = calculate_all_deltas(data)
    data = calculate_scheduled_exchanges(data)
    data = calculate_remainder_volume(data)
    data = calculate_all_available_intraday_flow(data)
    data = add_datetime_features(data)
    return data.dropna()

def engineer_features_v2(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data = calculate_total_production(data)
    data = calculate_all_deltas(data)
    data = calculate_scheduled_exchanges(data)
    data = calculate_remainder_volume(data)
    data = calculate_all_available_intraday_flow(data)
    data = add_datetime_features(data)
    data = lagged_features(data)
    return data.dropna()

if __name__ == "__main__":
    data = load_parquet(energy_data)
    data = clean_data(data)
    data = calculate_total_production(data)
    data = calculate_all_deltas(data)
    data = calculate_scheduled_exchanges(data)
    data = calculate_remainder_volume(data)
    data = calculate_all_available_intraday_flow(data)
    data = add_datetime_features(data)
    data = add_significant_lagged_features(data)
    print(data.columns)
