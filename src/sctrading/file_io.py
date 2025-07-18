import pandas as pd
import json
from pathlib import Path
import os
import shutil
from datetime import datetime, timezone
from typing import Tuple, List, Dict, Optional
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow as pa
import logging
import time

# Configure logger for FileIO
logger = logging.getLogger(__name__)

class FileIO:
    """
    Handles all file input/output operations for the data manager.
    This includes reading/writing raw market data (Parquet),
    enriched feature data (Parquet), and ticker lists (JSON).
    """

    def __init__(self, data_dir: Path):
        """
        Initializes the FileIO class with the base data directory.

        Args:
            data_dir (Path): The root directory where all data will be stored.
        """
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True) # Ensure data directory exists
        logger.debug(f"FileIO initialized. Data directory: {self.data_dir}")

        # Define standard file paths relative to data_dir
        self.active_json_path = self.data_dir / "active_tickers.json"
        self.delisted_json_path = self.data_dir / "delisted_tickers.json"

    def _get_data_file_path(self, ticker: str, freq: str, trade_or_quote: str, date: datetime) -> Path:
        """
        Constructs the file path for a specific raw data chunk (monthly parquet file).
        """
        ticker_upper = ticker.upper()
        first_letter = ticker_upper[0]
        base_path = self.data_dir / "data" / first_letter / ticker_upper / f"{trade_or_quote}s" / freq
        base_path.mkdir(parents=True, exist_ok=True)
        file_name = f"{date.year}{date.month:02d}.parquet"
        file_path = base_path / file_name
        logger.debug(f"Constructed raw data file path: {file_path}")
        return file_path

    def _get_enriched_data_file_path(self, ticker: str) -> Path:
        """
        Constructs the file path for an enriched data file.
        These are stored separately from the raw data.
        """
        ticker_upper = ticker.upper()
        first_letter = ticker_upper[0]
        base_path = self.data_dir / "data" / first_letter / ticker_upper / "enriched"
        base_path.mkdir(parents=True, exist_ok=True)
        file_name = f"{ticker_upper}_enriched.parquet"
        file_path = base_path / file_name
        logger.debug(f"Constructed enriched data file path: {file_path}")
        return file_path

    def _write_parquet_safely(self, df: pd.DataFrame, file_path: Path):
        """
        Writes a DataFrame to a final path using a unique temp file and an atomic move with retries.
        This is the robust, centralized writing logic for parallel processes.
        """
        if df.empty:
            logger.debug(f"DataFrame is empty. Skipping write to {file_path}")
            return

        pid = os.getpid()
        temp_file_path = file_path.with_suffix(f'.{pid}.tmp')

        # Step 1: Write to the unique temporary file.
        try:
            df.to_parquet(temp_file_path, index=True)
        except Exception as e:
            logger.error(f"Failed to write to temporary file {temp_file_path}: {e}", exc_info=True)
            if temp_file_path.exists():
                os.remove(temp_file_path)
            raise

        # Step 2: Atomically replace the destination file, retrying on filesystem lag.
        retries = 5
        for i in range(retries):
            try:
                os.replace(temp_file_path, file_path)
                # logger.info(f"Successfully wrote data to {file_path}") # Quieten for parallel runs
                return # Success
            except FileNotFoundError:
                logger.warning(f"Attempt {i+1}/{retries}: Temp file {temp_file_path} not found for move (filesystem lag?). Retrying...")
                if i < retries - 1:
                    time.sleep(0.1 * (i + 1)) # Exponential backoff
                else:
                    logger.critical(f"Gave up finding temp file {temp_file_path} after {retries} retries.")
                    raise
            except Exception as e:
                logger.error(f"Failed to move unique temp file {temp_file_path} to {file_path}: {e}", exc_info=True)
                if temp_file_path.exists():
                    os.remove(temp_file_path)
                raise

    def write_raw_data(self, df: pd.DataFrame, ticker: str, freq: str, trade_or_quote: str, overwrite: bool = False):
        """
        Writes raw market data to monthly Parquet files.
        This function now uses the robust _write_parquet_safely helper.
        """
        if df.empty:
            logger.debug(f"No data to write for {ticker} ({freq}, {trade_or_quote}). Skipping.")
            return

        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error(f"DataFrame for {ticker} must have a DatetimeIndex. Found: {type(df.index)}")
            raise ValueError("DataFrame must have a DatetimeIndex.")
        df = df.sort_index()

        for month_start, month_df in df.groupby(pd.Grouper(freq='MS')):
            file_path = self._get_data_file_path(ticker, freq, trade_or_quote, month_start)
            
            try:
                if overwrite or not file_path.exists():
                    final_df = month_df
                else:
                    existing_df = pd.read_parquet(file_path)
                    conflicting_indices = existing_df.index.intersection(month_df.index)
                    existing_df_updated = existing_df.drop(conflicting_indices)
                    final_df = pd.concat([existing_df_updated, month_df]).sort_index()

                self._write_parquet_safely(final_df, file_path)

            except Exception as e:
                logger.error(f"Error processing month {month_start.date()} for {ticker} at {file_path}: {e}", exc_info=True)
                continue

    def write_enriched_data(self, df: pd.DataFrame, ticker: str):
        """
        Writes an enriched DataFrame to a single Parquet file for a given ticker.
        This function is now robust, using the _write_parquet_safely helper.
        """
        file_path = self._get_enriched_data_file_path(ticker)
        try:
            self._write_parquet_safely(df, file_path)
        except Exception as e:
            logger.error(f"The write_enriched_data operation failed for {ticker}. See previous logs for details.")
            raise e

    def read_raw_data(self, ticker: str, freq: str, trade_or_quote: str,
                      start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Reads raw market data for a given ticker, frequency, and type within a date range
        using pyarrow.dataset for efficient reading and filtering.
        """
        ticker_upper = ticker.upper()
        first_letter = ticker_upper[0]
        ticker_data_dir = self.data_dir / "data" / first_letter / ticker_upper / f"{trade_or_quote}s" / freq
        # logger.debug(f"Attempting to read data from {ticker_data_dir}")

        if not ticker_data_dir.exists():
            return pd.DataFrame()

        try:
            parquet_files = list(ticker_data_dir.glob("*.parquet"))
            if not parquet_files:
                return pd.DataFrame()

            dataset = ds.dataset(parquet_files, format="parquet")
            
            filter_expression = None
            if start_date:
                filter_expression = (ds.field("timestamp") >= start_date)
            if end_date:
                end_filter = (ds.field("timestamp") <= end_date)
                filter_expression = filter_expression & end_filter if filter_expression is not None else end_filter

            table = dataset.to_table(filter=filter_expression)
            if table.num_rows == 0:
                return pd.DataFrame()
            
            df = table.to_pandas()

            if '__index_level_0__' in df.columns:
                df = df.set_index('__index_level_0__')
                df.index.name = 'timestamp'
            elif 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            
            df = df.sort_index()

            return df

        except Exception as e:
            logger.error(f"Error reading data from pyarrow dataset for {ticker_data_dir}: {e}", exc_info=True)
            return pd.DataFrame()

    def read_enriched_data(self, ticker: str) -> pd.DataFrame:
        """
        Reads an enriched data file for a given ticker.
        """
        file_path = self._get_enriched_data_file_path(ticker)
        if not file_path.exists():
            return pd.DataFrame()

        try:
            df = pd.read_parquet(file_path)
            if not isinstance(df.index, pd.DatetimeIndex):
                   logger.warning(f"Index for enriched data {ticker} is not a DatetimeIndex!")
            return df
        except Exception as e:
            logger.error(f"Failed to read enriched data for {ticker} from {file_path}: {e}", exc_info=True)
            return pd.DataFrame()

    def get_last_data_timestamp(self, ticker: str, freq: str, trade_or_quote: str) -> Optional[datetime]:
        """
        Gets the timestamp of the last available data point for a given ticker,
        safely ignoring any temporary files.
        """
        ticker_upper = ticker.upper()
        first_letter = ticker_upper[0]
        ticker_data_dir = self.data_dir / "data" / first_letter / ticker_upper / f"{trade_or_quote}s" / freq
        
        if not ticker_data_dir.exists():
            return None

        try:
            parquet_files = list(ticker_data_dir.glob("*.parquet"))
            if not parquet_files:
                return None

            dataset = ds.dataset(parquet_files, format="parquet")
            
            timestamp_col_name = None
            if '__index_level_0__' in dataset.schema.names:
                timestamp_col_name = '__index_level_0__'
            elif 'timestamp' in dataset.schema.names:
                timestamp_col_name = 'timestamp'

            if timestamp_col_name is None:
                return None
            
            max_timestamp_scalar = pc.max(dataset.to_table(columns=[timestamp_col_name]).column(timestamp_col_name))
            
            if max_timestamp_scalar.is_valid:
                return max_timestamp_scalar.as_py()
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting last timestamp for {ticker_data_dir}: {e}.")
            return None
            
    def write_ticker_file(self, ticker_list: list, delisted: bool = False):
        file_path = self.delisted_json_path if delisted else self.active_json_path
        temp_file_path = file_path.with_suffix('.json.tmp')
        data_to_write = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "tickers": ticker_list
        }
        with open(temp_file_path, 'w') as f:
            json.dump(data_to_write, f, indent=4)
        shutil.move(temp_file_path, file_path)

    def read_ticker_file(self, delisted: bool = False) -> Tuple[List[Dict], Optional[datetime]]:
        """
        Reads a list of tickers from the appropriate JSON file, deduplicates them
        based on the most recent 'last_updated_utc', and returns the list
        and the file's last_updated timestamp.
        """
        file_path = self.delisted_json_path if delisted else self.active_json_path
        ticker_type = "delisted" if delisted else "active"

        if not file_path.exists():
            return [], None
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            tickers = data.get("tickers", [])
            file_last_updated = datetime.fromisoformat(data.get("last_updated")) if data.get("last_updated") else None

            if not tickers:
                return [], file_last_updated

            df = pd.DataFrame(tickers)
            
            if 'ticker' not in df.columns:
                return tickers, file_last_updated

            initial_count = len(df)
            
            if 'last_updated_utc' in df.columns:
                df['last_updated_utc_dt'] = pd.to_datetime(df['last_updated_utc'], errors='coerce')
                df = df.sort_values(by=['ticker', 'last_updated_utc_dt'], ascending=[True, False])

            deduplicated_df = df.drop_duplicates(subset='ticker', keep='first')
            
            final_count = len(deduplicated_df)
            if initial_count > final_count:
                logger.info(f"Deduplicated {initial_count - final_count} tickers from {file_path}. Kept the most recent record for each.")

            if 'last_updated_utc_dt' in deduplicated_df.columns:
                deduplicated_df = deduplicated_df.drop(columns=['last_updated_utc_dt'])
            
            deduplicated_tickers = deduplicated_df.to_dict('records')
            
            return deduplicated_tickers, file_last_updated

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding ticker JSON from {file_path}: {e}. Returning empty list and None.")
            return [], None
        except Exception as e:
            logger.error(f"Error reading or processing ticker file {file_path}: {e}. Returning empty list and None.", exc_info=True)
            return [], None
