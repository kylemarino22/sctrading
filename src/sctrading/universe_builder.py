import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone, date, time as datetime_time
from typing import List, Dict, Tuple
from zoneinfo import ZoneInfo
import logging
import shutil
import psutil
import os
import time

# Configure logger
logger = logging.getLogger(__name__)

# Define Eastern Time (ET) timezone for trading hours
ET = ZoneInfo("America/New_York")

# Define constants for default date ranges
TODAY = datetime.now(timezone.utc).date()
HISTORIC_START_DATE = datetime(2000, 1, 1, tzinfo=timezone.utc).date()

class UniverseBuilder:
    """
    Builds and updates a universe of tickers and specific trading dates based on
    defined criteria, such as volume, price, and gap conditions.
    """

    def __init__(self, data_manager_instance, active_tickers: List[Dict], delisted_tickers: List[Dict]):
        """
        Initializes the UniverseBuilder.

        Args:
            data_manager_instance: An instance of the DataManager class.
            active_tickers (List[Dict]): A list of active ticker dictionaries.
            delisted_tickers (List[Dict]): A list of delisted ticker dictionaries.
        """
        self.data_manager = data_manager_instance
        self.active_ticker_symbols = sorted([t['ticker'] for t in active_tickers])
        self.delisted_tickers_symbols = sorted([t['ticker'] for t in delisted_tickers])
        self.all_ticker_symbols = sorted(list(set(self.active_ticker_symbols + self.delisted_tickers_symbols)))
        
        self.universe_output_dir = self.data_manager.io.data_dir / "universes"
        self.universe_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.process = psutil.Process(os.getpid())
        logger.info(f"UniverseBuilder initialized with {len(self.active_ticker_symbols)} active and {len(self.all_ticker_symbols)} total tickers.")

    def _load_universe_from_file(self, filename: str) -> Tuple[Dict, datetime | None, Dict | None]:
        """Loads an existing universe file, its metadata, and its build parameters."""
        file_path = self.universe_output_dir / filename
        if not file_path.exists():
            logger.warning(f"Universe file not found at {file_path}. A new one will be created.")
            return {}, None, None
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            universe = data.get("universe", {})
            parameters = data.get("parameters")
            last_updated_str = data.get("last_updated_utc")
            last_updated = datetime.fromisoformat(last_updated_str) if last_updated_str else None
            
            if parameters:
                 logger.info(f"Successfully loaded universe from {file_path} with parameters: {parameters}")
            else:
                 logger.warning(f"Loaded universe file {file_path} that is missing build parameters.")

            return universe, last_updated, parameters
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Could not read or parse universe file {file_path}: {e}. Starting fresh.")
            return {}, None, None

    def _save_universe_to_file(self, universe_data: Dict[str, List[Dict]], 
                                 output_filename: str, parameters: Dict, is_intermediate_save: bool = False):
        """Helper method to save the universe data and metadata to a JSON file atomically."""
        filename = f"{Path(output_filename).stem}_partial.json" if is_intermediate_save else output_filename
        file_path = self.universe_output_dir / filename
        temp_file_path = file_path.with_suffix('.json.tmp')

        data_to_save = {
            "last_updated_utc": datetime.now(timezone.utc).isoformat(),
            "parameters": parameters,
            "universe": universe_data
        }

        # Atomic file write to avoid corruption
        try:
            with open(temp_file_path, 'w') as f:
                json.dump(data_to_save, f, indent=4)
            shutil.move(temp_file_path, file_path)
            logger.info(f"Universe data saved to: {file_path}")
        except Exception as e:
            logger.error(f"Error saving universe to file {file_path}: {e}")

    def _find_valid_periods(self, symbol: str, data_start_date: date, data_end_date: date, 
                            processing_start_date: date, min_close_price: float, max_avg_dollar_volume: float, 
                            min_gap_pct: float) -> List[Dict]:
        """Core logic to find valid periods or gap events for a single ticker."""
        five_min_data = self.data_manager.get_data(
            symbol=symbol, freq="5minute", quote=False,
            start=data_start_date, end=data_end_date
        )
        if five_min_data.empty: return []

        rth_five_min_data = five_min_data.tz_convert(ET).between_time(datetime_time(9, 30), datetime_time(16, 0))
        if rth_five_min_data.empty: return []
        
        daily_agg = rth_five_min_data.resample('D').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        if daily_agg.empty: return []

        daily_agg['dollar_volume'] = daily_agg['volume'] * daily_agg['close']
        daily_agg['avg_dollar_vol_365d'] = daily_agg['dollar_volume'].rolling(window=365, min_periods=30).mean()
        daily_agg['prev_close'] = daily_agg['close'].shift(1)
        
        full_daily_index = daily_agg.index
        
        daily_agg = daily_agg[daily_agg.index.date >= processing_start_date]
        if daily_agg.empty: return []

        # Build conditions dynamically
        conditions = []
        if max_avg_dollar_volume > 0:
            conditions.append(daily_agg['avg_dollar_vol_365d'] < max_avg_dollar_volume)
        # --- FIXED: Use previous day's close to avoid lookahead bias ---
        if min_close_price > 0:
            conditions.append(daily_agg['prev_close'] > min_close_price)
        
        combined_cond = pd.Series(True, index=daily_agg.index)
        for cond in conditions:
            combined_cond &= cond
        
        found_periods = []
        if min_gap_pct > 0:
            gap_cond = daily_agg['open'] > (daily_agg['prev_close'] * (1 + min_gap_pct))
            qualified_days_df = daily_agg[combined_cond & gap_cond]

            for eval_date_ts in qualified_days_df.index:
                try:
                    current_day_pos = full_daily_index.get_loc(eval_date_ts)
                    if current_day_pos == 0: continue
                    prev_close_ts = full_daily_index[current_day_pos - 1]
                    found_periods.append({
                        'start': prev_close_ts.strftime('%Y%m%d'),
                        'end': eval_date_ts.strftime('%Y%m%d')
                    })
                except Exception as e:
                    logger.warning(f"Could not process gap period for {symbol} on {eval_date_ts.date()}: {e}")
        else:
            daily_agg['is_valid'] = combined_cond
            daily_agg['block'] = (daily_agg['is_valid'].diff() != 0).cumsum()
            valid_blocks = daily_agg[daily_agg['is_valid']]
            
            for _, block_df in valid_blocks.groupby('block'):
                if not block_df.empty:
                    found_periods.append({
                        'start': block_df.index[0].strftime('%Y%m%d'),
                        'end': block_df.index[-1].strftime('%Y%m%d')
                    })
        return found_periods

    def _merge_overlapping_periods(self, periods: List[Dict]) -> List[Dict]:
        """Merges a list of period dictionaries that are overlapping or continuous."""
        if not periods:
            return []
        sorted_periods = sorted(periods, key=lambda x: x['start'])
        merged = [sorted_periods[0]]
        for current_period in sorted_periods[1:]:
            last_merged = merged[-1]
            last_end = datetime.strptime(last_merged['end'], '%Y%m%d').date()
            current_start = datetime.strptime(current_period['start'], '%Y%m%d').date()
            is_continuous = len(pd.bdate_range(start=last_end + timedelta(days=1), end=current_start - timedelta(days=1))) == 0
            if current_start <= last_end or is_continuous:
                current_end = datetime.strptime(current_period['end'], '%Y%m%d').date()
                new_end = max(last_end, current_end)
                last_merged['end'] = new_end.strftime('%Y%m%d')
            else:
                merged.append(current_period)
        return merged

    def build_universe(self, output_filename: str, start_date: date | None = None, end_date: date | None = None,
                       save_interval_tickers: int = 100, min_close_price: float = 0.0, 
                       max_avg_dollar_volume: float = 0.0, min_gap_pct: float = 0.0) -> Dict[str, List[Dict]]:
        """Creates a universe from scratch for all tickers over a given period."""
        effective_start_date = start_date if start_date is not None else HISTORIC_START_DATE
        effective_end_date = end_date if end_date is not None else TODAY
        data_start_date = effective_start_date - timedelta(days=370)
        processing_start_date = effective_start_date
        
        parameters = {
            "min_close_price": min_close_price,
            "max_avg_dollar_volume": max_avg_dollar_volume,
            "min_gap_pct": min_gap_pct
        }
        logger.info(f"Starting FULL universe build from {processing_start_date} to {effective_end_date} with params: {parameters}")
        
        valid_periods_by_ticker = {}
        last_time = time.time()
        for i, symbol in enumerate(self.all_ticker_symbols):
            if (i + 1) % save_interval_tickers == 0:
                elapsed = time.time() - last_time
                mem_info = self.process.memory_info()
                logger.info(f"Processed {i + 1}/{len(self.all_ticker_symbols)} tickers ({symbol}). Last {save_interval_tickers} took {elapsed:.2f}s. Mem: {mem_info.rss / (1024*1024):.2f} MB")
                last_time = time.time()

            new_periods = self._find_valid_periods(symbol, data_start_date, effective_end_date, processing_start_date, 
                                                   min_close_price, max_avg_dollar_volume, min_gap_pct)
            if new_periods:
                valid_periods_by_ticker[symbol] = new_periods
            
            if save_interval_tickers > 0 and (i + 1) % save_interval_tickers == 0:
                self._save_universe_to_file(valid_periods_by_ticker, output_filename, parameters, is_intermediate_save=True)

        logger.info(f"Full universe build complete. Found valid periods for {len(valid_periods_by_ticker)} tickers.")
        self._save_universe_to_file(valid_periods_by_ticker, output_filename, parameters)
        return valid_periods_by_ticker

    def update_universe(self, output_filename: str, save_interval_tickers: int = 100) -> Dict[str, List[Dict]]:
        """
        Updates an existing universe file with new data for active tickers.
        It uses the parameters stored within the file to ensure consistency.
        """
        logger.info(f"Starting universe UPDATE for file: {output_filename}")
        existing_universe, last_updated, params = self._load_universe_from_file(output_filename)
        
        if not params or not last_updated:
            logger.error(f"Cannot update {output_filename}. It's missing a last_updated timestamp or build parameters. Please run a full build first.")
            return {}

        logger.info(f"Updating with stored parameters: {params}")
        min_close_price = params.get("min_close_price", 0.0)
        max_avg_dollar_volume = params.get("max_avg_dollar_volume", 0.0)
        min_gap_pct = params.get("min_gap_pct", 0.0)

        processing_start_date = last_updated.date()
        data_start_date = processing_start_date - timedelta(days=370)
        data_end_date = TODAY
        logger.info(f"Updating universe from {processing_start_date} to {data_end_date} for {len(self.active_ticker_symbols)} active tickers.")
        
        last_time = time.time()
        for i, symbol in enumerate(self.active_ticker_symbols):
            if (i + 1) % save_interval_tickers == 0:
                elapsed = time.time() - last_time
                mem_info = self.process.memory_info()
                logger.info(f"Updated {i + 1}/{len(self.active_ticker_symbols)} tickers ({symbol}). Last {save_interval_tickers} took {elapsed:.2f}s. Mem: {mem_info.rss / (1024*1024):.2f} MB")
                last_time = time.time()

            new_periods = self._find_valid_periods(symbol, data_start_date, data_end_date, processing_start_date, 
                                                   min_close_price, max_avg_dollar_volume, min_gap_pct)
            if not new_periods:
                continue

            if symbol not in existing_universe:
                existing_universe[symbol] = []
            
            existing_universe[symbol].extend(new_periods)
            existing_universe[symbol] = self._merge_overlapping_periods(existing_universe[symbol])
            
            if save_interval_tickers > 0 and (i + 1) % save_interval_tickers == 0:
                self._save_universe_to_file(existing_universe, output_filename, params, is_intermediate_save=True)

        logger.info(f"Universe update complete. Universe now contains {len(existing_universe)} tickers.")
        self._save_universe_to_file(existing_universe, output_filename, params)
        return existing_universe

# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        from data_manager import DataManager
        BASE_DATA_DIR = Path("/mnt/nas/price_data/polygon")
        BASE_DATA_DIR.mkdir(exist_ok=True)
        dm = DataManager(BASE_DATA_DIR)
        
        active_t, delisted_t = dm.get_tickers(force_update=False)
        ub = UniverseBuilder(dm, active_t, delisted_t)

        # --- Control Panel ---
        # Set to True to delete the existing file and build a new one with the parameters below.
        FORCE_REBUILD = True 
        PRETRAIN_UNIVERSE_FILENAME = "pretraining_universe.json"
        
        # --- Universe Build/Update Logic ---
        pretrain_universe_path = ub.universe_output_dir / PRETRAIN_UNIVERSE_FILENAME
        
        if FORCE_REBUILD and pretrain_universe_path.exists():
            print(f"FORCE_REBUILD is True. Deleting existing universe file: {pretrain_universe_path}")
            pretrain_universe_path.unlink()

        if not pretrain_universe_path.exists():
            print(f"\n--- Building the clean, pre-training universe: {PRETRAIN_UNIVERSE_FILENAME} ---")
            # As per our discussion, we remove the dollar volume cap but keep the price floor.
            pretrain_universe = ub.build_universe(
                output_filename=PRETRAIN_UNIVERSE_FILENAME,
                min_close_price=1.0, 
                max_avg_dollar_volume=0.0 # Setting to 0 disables this filter
            )
            print(f"Identified {len(pretrain_universe)} tickers in the new pre-training universe.")
        else:
            print(f"\n--- Updating the existing pre-training universe: {PRETRAIN_UNIVERSE_FILENAME} ---")
            updated_universe = ub.update_universe(
                output_filename=PRETRAIN_UNIVERSE_FILENAME
            )
            print(f"Updated universe now contains {len(updated_universe)} tickers.")

    except ImportError:
        logger.error("Could not import DataManager. Please ensure data_manager.py is in the same directory.")
    except Exception as e:
        logger.error(f"An error occurred during example execution: {e}", exc_info=True)
