import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, date, timedelta, time
from typing import Generator, Tuple, Dict, List, Optional
from zoneinfo import ZoneInfo
import logging
from data_manager import DataManager
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

class PretrainingDatasetBuilder:
    """
    (Online Training Stage)
    Assembles training samples on the fly, now including a target for the next RTH close.
    """
    def __init__(self, data_manager: DataManager, universe_file_path: Path,
                 window_size: int = 512, prediction_horizons_bars: Optional[List[int]] = None,
                 force_regenerate_tasks: bool = False):
        
        self.dm = data_manager
        self.universe_file_path = universe_file_path
        self.window_size = window_size
        self.prediction_horizons_bars = sorted(list(set(prediction_horizons_bars or [1])))
        self.max_prediction_horizon = max(self.prediction_horizons_bars)
        self.ewma_periods = [8, 16, 32, 64, 128, 256]
        
        self.feature_map = self._define_feature_map()
        self.num_features = len(self.feature_map)

        self.tasks_by_date, self.start_date, self.end_date = self._prepare_chronological_tasks(force_regenerate_tasks)
        
        self.spy_df = None
        self.vxx_df = None

        logger.info(f"Builder initialized with {self.num_features} features per timestep.")

    def _define_feature_map(self) -> Dict[str, int]:
        feature_map = {}
        idx = 0
        for name in ["secs_from_pm_open", "secs_to_pm_close", "secs_from_rth_open", "secs_to_rth_close", "secs_to_next_hour", "sin_time", "cos_time"]: feature_map[name] = idx; idx += 1
        for name in ["is_mon", "is_tue", "is_wed", "is_thu", "is_fri"]: feature_map[name] = idx; idx += 1
        feature_map["overnight_gap"] = idx; idx += 1
        for asset_prefix in ["main", "spy", "vxx"]:
            for name in ["open", "high", "low", "close", "volume", "sma_20", "rsi_14"]: feature_map[f"{asset_prefix}_{name}"] = idx; idx += 1
            for period in self.ewma_periods: feature_map[f"{asset_prefix}_ewma_{period}"] = idx; idx += 1
            for period in self.ewma_periods: feature_map[f"{asset_prefix}_dist_from_ewma_{period}"] = idx; idx += 1
        return feature_map

    def _prepare_chronological_tasks(self, force_regenerate: bool) -> Tuple[Dict[date, List[str]], date, date]:
        cache_path = self.universe_file_path.parent / f"{self.universe_file_path.stem}_task_cache.pkl"
        if not force_regenerate and cache_path.exists():
            with open(cache_path, 'rb') as f: cached_data = pickle.load(f)
            return cached_data['tasks_by_date'], cached_data['start_date'], cached_data['end_date']
        tasks_by_date: Dict[date, List[str]] = {}
        all_dates = []
        with open(self.universe_file_path, 'r') as f: universe = json.load(f).get("universe", {})
        for symbol, periods in universe.items():
            for period in periods:
                p_start = datetime.strptime(period['start'], '%Y%m%d').date()
                p_end = datetime.strptime(period['end'], '%Y%m%d').date()
                all_dates.extend([p_start, p_end])
                for day in pd.bdate_range(start=p_start, end=p_end):
                    tasks_by_date.setdefault(day.date(), []).append(symbol)
        min_date, max_date = min(all_dates), max(all_dates)
        with open(cache_path, 'wb') as f: pickle.dump({'tasks_by_date': tasks_by_date, 'start_date': min_date, 'end_date': max_date}, f)
        return tasks_by_date, min_date, max_date

    def _load_market_context_data(self):
        if self.spy_df is None or self.vxx_df is None:
            logger.info("Loading market context data (SPY, VXX) for the first time...")
            self.spy_df = self.dm.get_data('SPY', enriched=True)
            self.vxx_df = self.dm.get_data('VXX', enriched=True)
            if self.spy_df.empty or self.vxx_df.empty:
                raise ValueError("Could not load essential market context data (SPY or VXX).")

    def generate_samples_for_day(self, target_day: date, tickers_to_process: Optional[List[str]] = None):
        self._load_market_context_data()
        if tickers_to_process is None:
            tickers_to_process = self.tasks_by_date.get(target_day, [])
        
        for symbol in tickers_to_process:
            main_df = self.dm.get_data(symbol, enriched=True)
            if main_df.empty: continue
            main_df.name = symbol

            rth_close_indices = np.where(main_df.index.tz_convert(ET).time == time(16, 0))[0]
            if len(rth_close_indices) == 0: continue

            day_start_utc = datetime.combine(target_day, time(4, 0), tzinfo=ET).astimezone(ZoneInfo("UTC"))
            day_end_utc = datetime.combine(target_day, time(20, 0), tzinfo=ET).astimezone(ZoneInfo("UTC"))
            day_start_idx = main_df.index.searchsorted(day_start_utc, side='left')
            day_end_idx = main_df.index.searchsorted(day_end_utc, side='right')
            
            for window_end_idx in range(day_start_idx, day_end_idx):
                window_start_idx = window_end_idx - self.window_size + 1
                if window_start_idx < 0 or (window_end_idx + self.max_prediction_horizon) >= len(main_df): continue
                
                sample = self._build_feature_tensor(main_df, self.spy_df, self.vxx_df, window_start_idx, window_end_idx, rth_close_indices)
                if sample:
                    yield sample

    def _build_feature_tensor(self, main_df, spy_df, vxx_df, start_idx, end_idx, rth_close_indices) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:

        # Pandas slice excludes the end index, so we need to include it with +1
        main_window = main_df.iloc[start_idx:end_idx + 1]
        spy_window = spy_df.reindex(main_window.index, method='ffill')
        vxx_window = vxx_df.reindex(main_window.index, method='ffill')

        main_norm_factor = main_window['close'].iloc[-1]
        spy_norm_factor = spy_window['close'].iloc[-1]
        vxx_norm_factor = vxx_window['close'].iloc[-1]
        if main_norm_factor <= 0 or spy_norm_factor <= 0 or vxx_norm_factor <= 0: return None

        features = np.zeros((self.window_size, self.num_features), dtype=np.float32)
        
        ts_et = main_window.index.tz_convert(ET)

        # Get the start of the day (midnight) for EVERY timestamp in the window
        normalized_days = ts_et.normalize()

        # Add a fixed duration to each day's midnight timestamp to get the session times.
        # This is a fully vectorized operation.
        pm_open  = normalized_days + pd.Timedelta(hours=4)
        rth_open = normalized_days + pd.Timedelta(hours=9, minutes=30)
        rth_close = normalized_days + pd.Timedelta(hours=16)
        pm_close = normalized_days + pd.Timedelta(hours=20)

        # Subtract current time from session times to get the seconds until/after each session.
        # This is also vectorized.
        features[:, self.feature_map["secs_from_pm_open"]] = (ts_et - pm_open).total_seconds().to_numpy().clip(0)
        features[:, self.feature_map["secs_to_pm_close"]] = (pm_close - ts_et).total_seconds().to_numpy().clip(0)
        features[:, self.feature_map["secs_from_rth_open"]] = (ts_et - rth_open).total_seconds().to_numpy().clip(0)
        features[:, self.feature_map["secs_to_rth_close"]] = (rth_close - ts_et).total_seconds().to_numpy().clip(0)

        bar_idx = (ts_et.hour * 12 + ts_et.minute / 5) - (4 * 12)

        # Calculate cos and sin of the time of day, normalized to a 192-bar day.
        features[:, self.feature_map["sin_time"]] = np.sin(2 * np.pi * bar_idx / 192)
        features[:, self.feature_map["cos_time"]] = np.cos(2 * np.pi * bar_idx / 192)

        # One hot encode the day of the week (Monday=0, ..., Friday=4)
        weekdays = ts_et.weekday

        # Iterate through Monday (0) to Friday (4)
        for day_idx in range(5):
            # Create a boolean mask to find all rows matching the current day
            mask = (weekdays == day_idx)
            
            # Get the correct feature column for that day (is_mon, is_tue, etc.)
            feature_col = self.feature_map["is_mon"] + day_idx
            
            # Use the mask to set the feature to 1.0 ONLY for the correct rows
            features[mask, feature_col] = 1.0

        # Add ohlcv data from symbol and market context data (SPY, VXX)
        for asset, window, norm_factor in [("main", main_window, main_norm_factor), ("spy", spy_window, spy_norm_factor), ("vxx", vxx_window, vxx_norm_factor)]:

            prefix = "" if asset == "main" else f"{asset}_"
            close_prices = window[f'{prefix}close']
            delta = close_prices.diff()
            
            # Used for rsi calculation
            gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss.replace(0, 1e-9)
            
            features[:, self.feature_map[f"{asset}_rsi_14"]] = (100 - (100 / (1 + rs))).fillna(50).values / 100.0
            features[:, self.feature_map[f"{asset}_sma_20"]] = np.log(close_prices.rolling(20).mean().values / norm_factor)

            for p_col in ["open", "high", "low", "close"]: features[:, self.feature_map[f"{asset}_{p_col}"]] = np.log(window[f"{prefix}{p_col}"].values / norm_factor)
            
            # Use log1p to avoid log(0) issues
            features[:, self.feature_map[f"{asset}_volume"]] = np.log1p(window[f"{prefix}volume"].values)

            for period in self.ewma_periods:
                ewma_col = f"{main_df.name if asset=='main' else asset.upper()}_ewma_{period}"
                dist_col = f"{main_df.name if asset=='main' else asset.upper()}_dist_from_ewma_{period}"
                features[:, self.feature_map[f"{asset}_ewma_{period}"]] = np.log(window[ewma_col].values / norm_factor)
                features[:, self.feature_map[f"{asset}_dist_from_ewma_{period}"]] = window[dist_col].values
        
        # --- TARGET CALCULATION ---
        # 1. Fixed bar-based targets
        target_indices = end_idx + np.array(self.prediction_horizons_bars)
        target_prices = main_df['close'].values[target_indices]
        bar_targets = np.log(target_prices / main_norm_factor)

        # 2. Next RTH close target
        next_close_idx_pos = np.searchsorted(rth_close_indices, end_idx, side='right')
        if next_close_idx_pos >= len(rth_close_indices): return None # No more closes in the data
        next_close_idx = rth_close_indices[next_close_idx_pos]
        next_close_price = main_df['close'].iloc[next_close_idx]
        close_target = np.log(next_close_price / main_norm_factor)
        
        # 3. Concatenate all targets
        targets = np.append(bar_targets, close_target).astype(np.float32)

        context = {'symbol': main_df.name, 'timestamp': main_window.index[-1].isoformat()}
        if np.any(np.isnan(features)) or np.any(np.isinf(features)) or np.any(np.isnan(targets)): return None
        
        return features, targets, context
  
