import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, date, timedelta, time
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from typing import Optional, Dict
from data_manager import DataManager
from tqdm import tqdm # Import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeaturePreprocessor:
    """
    (Offline Pre-calculation Stage)
    Calculates daily features from 5-minute data and creates a self-contained
    "enriched" data file for each individual symbol.
    """
    def __init__(self, data_manager: DataManager):
        self.dm = data_manager
        self.universe_path = self.dm.get_universe_path()
        self.universe_data = self._load_universe()
        self.all_tickers = sorted(list(set(self.universe_data.keys()) | {'SPY', 'VXX'}))
        self.ewma_periods = [8, 16, 32, 64, 128, 256]

    def _load_universe(self) -> dict:
        try:
            with open(self.universe_path, 'r') as f:
                return json.load(f).get("universe", {})
        except FileNotFoundError:
            logger.error(f"Universe file not found at {self.universe_path}. Cannot proceed.")
            return {}

    def _calculate_daily_features_from_intraday(self, five_min_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Calculates ONLY the daily RTH-based EWMA trendlines from 5-minute data.
        The timestamp for a day's features is marked at 4:00 PM ET on that day.
        """
        if five_min_df.empty:
            return pd.DataFrame()

        try:
            if five_min_df.index.tz is None:
                five_min_df.index = five_min_df.index.tz_localize('UTC')

            eastern_df = five_min_df.tz_convert('US/Eastern')
            rth_closes = eastern_df.between_time('09:30', '16:00')['close'].resample('D').last().ffill()

            if rth_closes.empty or len(rth_closes) < 2:
                return pd.DataFrame()

            daily_features = pd.DataFrame(index=rth_closes.index)
            daily_features.index = daily_features.index.normalize() + pd.DateOffset(hours=16)
            daily_features['close'] = rth_closes.values

            # Calculate EWMAs
            for period in self.ewma_periods:
                ewma = daily_features['close'].ewm(span=period, adjust=False).mean()
                daily_features[f'{symbol.upper()}_ewma_{period}'] = ewma

            feature_cols = [f'{symbol.upper()}_ewma_{period}' for period in self.ewma_periods]
            
            final_features = daily_features[feature_cols].dropna()
            final_features.index = final_features.index.tz_convert('UTC')
            return final_features

        except Exception as e:
            logger.error(f"Could not calculate daily features for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()

    def process_ticker(self, ticker: str):
        """
        Creates a single, self-contained enriched file for a given ticker.
        It calculates the ticker's own daily features and merges them.
        """
        try:
            # For SPY/VXX, they might not be in the universe file, so we define a default full range
            periods = self.universe_data.get(ticker)
            if not periods:
                start_date = date(2007, 1, 1) # A reasonable default start for SPY/VXX
                end_date = date.today()
            else:
                start_date = min(datetime.strptime(p['start'], '%Y%m%d').date() for p in periods)
                end_date = max(datetime.strptime(p['end'], '%Y%m%d').date() for p in periods)

            buffer = timedelta(days=max(self.ewma_periods) * 2)
            fetch_start_date = start_date - buffer

            five_min_data = self.dm.get_data(ticker, freq="5minute", start=fetch_start_date, end=end_date)
            if five_min_data.empty: return

            ticker_daily_features = self._calculate_daily_features_from_intraday(five_min_data, ticker)
            if ticker_daily_features.empty: return

            merged = pd.merge_asof(five_min_data.sort_index(), ticker_daily_features, 
                                   left_index=True, right_index=True, direction='backward')

            # Calculate the intraday "distance from ewma" feature
            for period in self.ewma_periods:
                merged[f'{ticker.upper()}_dist_from_ewma_{period}'] = (merged['close'] - merged[f'{ticker.upper()}_ewma_{period}']) / merged[f'{ticker.upper()}_ewma_{period}']

            # --- FIX: Rename base OHLCV columns to include the ticker prefix ---
            # This ensures every column in the enriched file is uniquely identified.
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            rename_dict = {col: f"{ticker.upper()}_{col}" for col in ohlcv_cols if col in merged.columns}
            merged.rename(columns=rename_dict, inplace=True)
            # --- END FIX ---

            merged.ffill(inplace=True)
            merged.dropna(inplace=True)
            
            if not merged.empty:
                self.dm.write_enriched_data(merged, ticker)
                # logger.info(f"Successfully saved self-contained enriched data for {ticker}.")

        except Exception as e:
            logger.error(f"Failed to process ticker {ticker}: {e}", exc_info=True)

    def run_preprocessing(self, start_date: date, end_date: date, num_workers: int, debug_ticker: Optional[str] = None):
        """ Main execution function. Iterates through all tickers and processes them individually. """
        if debug_ticker:
            logger.info(f"--- RUNNING IN DEBUG MODE FOR TICKER: {debug_ticker} ---")
            tickers_to_process = [debug_ticker]
            num_workers = 1
        else:
            tickers_to_process = self.all_tickers

        logger.info(f"Starting parallel processing for {len(tickers_to_process)} tickers using {num_workers} worker(s)...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.process_ticker, ticker): ticker for ticker in tickers_to_process}
            
            # --- TQDM Progress Bar ---
            for future in tqdm(as_completed(futures), total=len(tickers_to_process), desc="Enriching ticker data"):
                try:
                    # It's good practice to get the result to catch exceptions from workers
                    future.result()
                except Exception as e:
                    ticker = futures[future]
                    logger.error(f"An error occurred in the worker for ticker {ticker}: {e}", exc_info=True)

        logger.info("--- Feature enrichment complete! ---")


if __name__ == "__main__":
    # --- Control Panel ---
    DEBUG_MODE = False
    DEBUG_TICKER = "AAIC"

    # --- QUIET DOWN CHATTY LOGGERS ---
    # This prevents logs from DataManager from overwhelming the tqdm progress bar
    logging.getLogger("data_manager").setLevel(logging.WARNING)
    logging.getLogger("file_io").setLevel(logging.WARNING)

    # --- Path and DataManager Setup ---
    DATA_DIR = Path("/home/kyle/data/polygon_cache")
    dm = DataManager(DATA_DIR)
    
    # --- Date & Worker Configuration ---
    RUN_START_DATE = date(2020, 1, 1)
    RUN_END_DATE = date.today() - timedelta(days=1)
    NUM_WORKERS = max(1, (os.cpu_count() or 4) - 2)

    # --- Execution ---
    preprocessor = FeaturePreprocessor(data_manager=dm)
    preprocessor.run_preprocessing(
        start_date=RUN_START_DATE, 
        end_date=RUN_END_DATE, 
        num_workers=NUM_WORKERS,
        debug_ticker=DEBUG_TICKER if DEBUG_MODE else None
    )
