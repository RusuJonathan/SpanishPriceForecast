from entsoe import EntsoePandasClient
from requests import HTTPError, ConnectionError, Timeout
from functools import wraps
from entsoe.exceptions import NoMatchingDataError
from pathlib import Path
from src.data.loader import raw_data_dir
import pandas as pd
from dotenv import load_dotenv
import logging
import os
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


class FetchingException(Exception):
    """
    Fetch exception if an error occurs during the fetching
    """
    pass

def handle_fetch_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = args[1] if len(args) > 1 else "?"
        end = args[2] if len(args) > 2 else "?"
        logger.info(f"Starting fetch...\nCalling {func.__name__} | {start} → {end}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Successfully fetched {len(result)} rows")
        except NoMatchingDataError:
            logger.error(f"No matching data error in {func.__name__}")
            raise FetchingException(f"No Data available for this period")
        except HTTPError as e:
            logger.error(f"HTTP error in {func.__name__}: {e}")
            raise FetchingException(f"HTTP error: {e}")
        except (ConnectionError, Timeout) as e:
            logger.error(f"Impossibility to reach entsoe api in {func.__name__}: {e}")
            raise FetchingException(f"Cannot reach ENTSOE API: {e}") from e

        return result
    return wrapper

class EntsoeFetcher:
    def __init__(self, country_code):
        self.api_key = os.getenv("ENTSOE_API_KEY")
        if self.api_key is None:
            raise ValueError("ENTSOE_API_KEY not found in environment variables")
        self.client = EntsoePandasClient(api_key=self.api_key)
        self.country_code = country_code

    @handle_fetch_errors
    def fetch_imbalance_prices(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """
        Fetches imbalance prices and returns a DataFrame
        :param start_date: Timestamp
        :param end_date: Timestamp
        :return: pd.DataFrame with columns ... to complete
        """
        df = self.client.query_imbalance_prices(self.country_code, start=start_date, end=end_date, psr_type=None)
        df.columns = df.columns + "_Imbalance_Price"
        return df

    @handle_fetch_errors
    def fetch_load_forecast_and_actual(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """
        Fetches consumtion forecast and actual
        :param start_date: Timestamp
        :param end_date: Timestamp
        :return: ... to complete
        """
        return self.client.query_load_and_forecast(self.country_code,
                                                   start=start_date,
                                                   end=end_date)

    @handle_fetch_errors
    def fetch_total_production_forecast(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        df = self.client.query_generation_forecast(self.country_code,
                                                     start=start_date,
                                                     end=end_date)
        df.columns = "Forecast_" + df.columns
        return df

    @handle_fetch_errors
    def fetch_wind_solar_day_ahead_forecast(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        df = self.client.query_wind_and_solar_forecast(self.country_code,
                                                         start=start_date,
                                                         end=end_date,
                                                         psr_type=None)
        df.columns = "Day_ahead_Forecast_" + df.columns
        return df

    @handle_fetch_errors
    def fetch_wind_solar_intraday_forecast(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        df = self.client.query_intraday_wind_and_solar_forecast(self.country_code,
                                                                start=start_date,
                                                                end=end_date,
                                                                psr_type=None)
        df.columns = "Intraday_Forecast_" + df.columns
        return df

    @handle_fetch_errors
    def fetch_actual_production(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        df = self.client.query_generation(self.country_code,
                                          start=start_date,
                                          end=end_date,
                                          psr_type=None)

        df.columns = ["_".join(col).replace(" ", "_") if isinstance(col, tuple) else col for col in df.columns]

        return df


    @handle_fetch_errors
    def fetch_scheduled_exchanges(self, start_date: pd.Timestamp, end_date: pd.Timestamp ,exchange_country: str) -> pd.DataFrame:
        exports = self.client.query_scheduled_exchanges(country_code_from=self.country_code,
                                                        country_code_to=exchange_country,
                                                        start=start_date,
                                                        end=end_date,
                                                        dayahead=False)
        exports.name = f"scheduled_exchange_{self.country_code}>{exchange_country}"

        imports = self.client.query_scheduled_exchanges(country_code_from=exchange_country,
                                                        country_code_to=self.country_code,
                                                        start=start_date,
                                                        end=end_date,
                                                        dayahead=False)

        imports.name = f"scheduled_exchange_{exchange_country}>{self.country_code}"

        df = pd.concat([exports, imports], axis=1).resample("15min").ffill()
        df[f"Net_scheduled_exchange_{self.country_code}-{exchange_country}"] = df[imports.name] - df[exports.name]
        return df

    @handle_fetch_errors
    def fetch_ntc_dayahead(self, start_date: pd.Timestamp, end_date: pd.Timestamp, exchange_country: str) -> pd.DataFrame:
        ntc_exports = self.client.query_net_transfer_capacity_dayahead(country_code_from=self.country_code,
                                                                       country_code_to=exchange_country,
                                                                       start=start_date,
                                                                       end=end_date)

        ntc_exports.name = f"ntc_{self.country_code}>{exchange_country}"

        ntc_imports = self.client.query_net_transfer_capacity_dayahead(country_code_from=exchange_country,
                                                                       country_code_to=self.country_code,
                                                                       start=start_date,
                                                                       end=end_date)

        ntc_imports.name = f"ntc_{exchange_country}>{self.country_code}"

        return pd.concat([ntc_exports, ntc_imports], axis=1).resample("15min").ffill()

    @handle_fetch_errors
    def fetch_day_ahead_prices(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        series = self.client.query_day_ahead_prices(self.country_code,
                                                 start=start_date,
                                                 end=end_date).resample("15min").ffill()
        series.name = "Day_Ahead_Price"
        return series.to_frame()

    def fetch_all(self, start_date: pd.Timestamp, end_date: pd.Timestamp, neighbor_country: list[str]) -> pd.DataFrame:
        data = []
        data.append(self.fetch_day_ahead_prices(start_date, end_date))
        data.append(self.fetch_imbalance_prices(start_date, end_date))
        data.append(self.fetch_load_forecast_and_actual(start_date, end_date))
        data.append(self.fetch_total_production_forecast(start_date, end_date))
        data.append(self.fetch_wind_solar_day_ahead_forecast(start_date, end_date))
        data.append(self.fetch_wind_solar_intraday_forecast(start_date, end_date))
        data.append(self.fetch_actual_production(start_date, end_date))

        for country in neighbor_country:
            data.append(self.fetch_scheduled_exchanges(start_date, end_date, country))
            data.append(self.fetch_ntc_dayahead(start_date, end_date, country))

        df = pd.concat(data, axis=1)
        return df

    def fetch_and_save(self, start_date: pd.Timestamp, end_date: pd.Timestamp, neighbor_country: list[str], data_path: Path) -> None:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df = self.fetch_all(start_date, end_date, neighbor_country)
        df.to_parquet(data_path)


if __name__ == "__main__":
    country_code = "ES"
    start_date = pd.Timestamp("2025-01-01", tz="Europe/Madrid")
    end_date = pd.Timestamp("2026-03-01", tz="Europe/Madrid")
    neighbor_country = ["FR", "PT"]

    fetcher = EntsoeFetcher(country_code=country_code)
    fetcher.fetch_and_save(start_date=start_date,
                           end_date=end_date,
                           neighbor_country=neighbor_country,
                           data_path=raw_data_dir / "energy_data.parquet")