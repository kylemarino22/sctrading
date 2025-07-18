import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone, date, time as datetime_time
from typing import List, Dict, Tuple, Union
from zoneinfo import ZoneInfo
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pandas.tseries.offsets import BDay

# Configure logger for Backtester
logger = logging.getLogger(__name__)

# Define Eastern Time (ET) timezone for trading hours
ET = ZoneInfo("America/New_York")

class Backtester:
    """
    Performs a backtest of a short-gap trading strategy on a given universe of tickers.
    This version is designed for analysis: it records pre-trade conditions for every
    trade to allow for post-backtest performance analysis.
    """

    def __init__(self, universe_file: Path, data_manager_instance,
                 initial_capital: float = 100_000.0,
                 fixed_capital_per_trade: bool = False,
                 cost_per_trade_pct: float = 0.05,
                 strategy_name: str = "gap_50pct_sl200", # 'gap_50pct_sl200' or 'gap_up_long'
                 volatility_multiplier: float = 1.0,
                 use_volatility_scaling: bool = True,
                 trade_integer_shares: bool = True,
                 use_trailing_stop: bool = False,
                 trailing_stop_type: str = 'fixed_amount', # 'percent' or 'fixed_amount'
                 trailing_stop_pct: float = 0.10,
                 require_entry_confirmation: bool = False,
                 long_momentum_max_threshold: float = 2.5, # For 'gap_up_long' strategy
                 long_path_chord_max_threshold: float = 1.2, # For 'gap_up_long' strategy
                 short_momentum_min_threshold: float = 5.0): # For 'gap_50pct_sl200' strategy
        if not universe_file.exists():
            logger.error(f"Universe file not found: {universe_file}")
            raise FileNotFoundError(f"Universe file not found: {universe_file}")

        self.universe_file = universe_file
        self.data_manager = data_manager_instance
        self.initial_capital = initial_capital
        self.fixed_capital_per_trade = fixed_capital_per_trade
        self.cost_per_trade_pct = cost_per_trade_pct
        self.strategy_name = strategy_name
        self.volatility_multiplier = volatility_multiplier
        self.use_volatility_scaling = use_volatility_scaling
        self.trade_integer_shares = trade_integer_shares
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_type = trailing_stop_type
        self.trailing_stop_pct = trailing_stop_pct
        self.require_entry_confirmation = require_entry_confirmation
        # Renamed for clarity
        self.long_momentum_max_threshold = long_momentum_max_threshold
        self.long_path_chord_max_threshold = long_path_chord_max_threshold
        # New parameter for short strategy filter
        self.short_momentum_min_threshold = short_momentum_min_threshold
        self.universe_data: Dict[str, List[Dict]] = self._load_universe(universe_file)
        logger.info(f"Backtester initialized with initial capital: ${self.initial_capital:,.2f}")
        logger.info(f"Selected strategy: {self.strategy_name}")

    def _load_universe(self, universe_file: Path) -> Dict[str, List[Dict]]:
        try:
            with open(universe_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error loading universe file {universe_file}: {e}")
            raise

    def _calculate_stdev_of_returns(self, df: pd.DataFrame) -> float:
        if df.empty or len(df) < 2: return np.nan
        return df['close'].pct_change().std()

    def _calculate_path_over_chord(self, price_series: pd.Series) -> float:
        if len(price_series) < 2: return np.nan
        price_deltas = price_series.diff().dropna()
        path_length = price_deltas.abs().sum()
        chord_length = abs(price_series.iloc[-1] - price_series.iloc[0])
        if chord_length == 0: return np.inf
        return path_length / chord_length

    def _calculate_cumulative_positive_momentum(self, price_series: pd.Series) -> float:
        if len(price_series) < 2: return 0.0
        price_deltas = price_series.diff().dropna()
        return price_deltas[price_deltas > 0].sum()

    def _calculate_position_size_value(self, current_account_value: float, entry_price: float, stdev_of_returns: float) -> Tuple[float, float]:
        base_capital = self.initial_capital if self.fixed_capital_per_trade else current_account_value
        max_position_value = self.initial_capital * 0.05
        position_value = 0.0
        num_shares = 0.0
        if self.use_volatility_scaling:
            capital_to_risk = base_capital * 0.01
            dollar_risk_per_share = entry_price * stdev_of_returns * self.volatility_multiplier
            if dollar_risk_per_share >= 0.01:
                num_shares = capital_to_risk / dollar_risk_per_share
                position_value = num_shares * entry_price
        else:
            position_value = base_capital * 0.01
        if position_value > max_position_value:
            position_value = max_position_value
        position_value = max(0, position_value)
        if entry_price > 0:
            num_shares = position_value / entry_price
        else:
            num_shares = 0
        if self.trade_integer_shares:
            num_shares = int(num_shares)
        if num_shares <= 0:
            return 0.0, 0
        return num_shares * entry_price, num_shares

    def _strategy_gap_50pct_sl200(self, rth_5min_data_df: pd.DataFrame, prev_day_rth_close: float, current_day_rth_open: float, current_account_value: float, stdev_of_returns: float) -> Tuple[float, float, float]:
        rth_data_et = rth_5min_data_df.tz_convert(ET)
        entry_trigger_price = prev_day_rth_close * 1.50
        entry_price = 0
        entry_time = None
        first_hour_data = rth_data_et.between_time("09:30", "10:30")

        if not self.require_entry_confirmation:
            if current_day_rth_open >= entry_trigger_price:
                entry_price, entry_time = current_day_rth_open, rth_data_et.index[0]
            else:
                trigger_bars = first_hour_data[first_hour_data['high'] >= entry_trigger_price]
                if not trigger_bars.empty:
                    entry_price, entry_time = entry_trigger_price, trigger_bars.index[0]
        else: # With confirmation
            trigger_bars = first_hour_data[first_hour_data['high'] >= entry_trigger_price]
            if not trigger_bars.empty:
                first_trigger_time = trigger_bars.index[0]
                potential_entry_bars = first_hour_data.loc[first_trigger_time:]
                confirmation_bars = potential_entry_bars[potential_entry_bars['close'] < potential_entry_bars['open']]
                if not confirmation_bars.empty:
                    entry_time = confirmation_bars.index[0]
                    entry_price = confirmation_bars.loc[entry_time]['close']
        
        if entry_price == 0: return 0.0, 0.0, 0
        _, num_shares = self._calculate_position_size_value(current_account_value, entry_price, stdev_of_returns)
        if num_shares <= 0: return 0.0, 0.0, 0

        trade_cost = (num_shares * entry_price) * self.cost_per_trade_pct
        hard_stop_price = entry_price * 3.00
        exit_price = rth_data_et['close'].iloc[-1]
        for _, row in rth_data_et.loc[entry_time:].iterrows():
            if row['high'] >= hard_stop_price:
                exit_price = hard_stop_price
                break
        
        pnl = (entry_price - exit_price) * num_shares - trade_cost
        return pnl, trade_cost, num_shares

    def _strategy_gap_up_long(self, rth_5min_data_df: pd.DataFrame, current_account_value: float, stdev_of_returns: float) -> Tuple[float, float, float]:
        rth_data_et = rth_5min_data_df.tz_convert(ET)
        if rth_data_et.empty: return 0.0, 0.0, 0

        first_bar = rth_data_et.iloc[0]
        breakout_level = first_bar['high']
        stop_loss_price = first_bar['low']
        
        entry_price = 0
        entry_time = None
        
        # Look for entry within the first hour, starting from the second bar
        first_hour_data = rth_data_et.between_time("09:30", "10:30")
        for idx, row in first_hour_data.iloc[1:].iterrows():
            # Check if the high of the current bar broke the breakout level
            if row['high'] >= breakout_level:
                # Realistic entry price: either the breakout level or the bar's open if it gapped above.
                entry_price = max(breakout_level, row['open'])
                entry_time = idx
                break
        
        if entry_price == 0: return 0.0, 0.0, 0

        _, num_shares = self._calculate_position_size_value(current_account_value, entry_price, stdev_of_returns)
        if num_shares <= 0: return 0.0, 0.0, 0

        trade_cost = (num_shares * entry_price) * self.cost_per_trade_pct
        exit_price = rth_data_et['close'].iloc[-1] # Default exit is End of Day

        for _, row in rth_data_et.loc[entry_time:].iterrows():
            if row['low'] <= stop_loss_price:
                exit_price = stop_loss_price # Exited at stop loss
                break
        
        # PnL for a LONG trade
        pnl = (exit_price - entry_price) * num_shares - trade_cost
        return pnl, trade_cost, num_shares

    def run_backtest_on_ticker(self, symbol: str, start_date: date | None = None) -> Tuple[str, pd.Series, pd.Series, List[Dict]]:
        current_account_value = self.initial_capital
        daily_pnl, daily_cost = {}, {}
        trade_log = []
        opportunities = sorted([(datetime.strptime(p['end'], '%Y%m%d').date(), p['start']) for p in self.universe_data.get(symbol, []) if start_date is None or datetime.strptime(p['end'], '%Y%m%d').date() >= start_date])

        for trade_date, prev_day_str in opportunities:
            prev_day_date = datetime.strptime(prev_day_str, '%Y%m%d').date()
            fetch_start_date = (pd.to_datetime(trade_date) - BDay(5)).date()
            trade_window_data = self.data_manager.get_data(symbol=symbol, freq="5minute", quote=False, start=fetch_start_date, end=trade_date)
            if trade_window_data.empty: continue

            all_prior_data = trade_window_data[trade_window_data.index.date < trade_date]
            volatility_metric = self._calculate_stdev_of_returns(all_prior_data.tail(300))
            prev_day_data = all_prior_data[all_prior_data.index.date == prev_day_date]
            if prev_day_data.empty: continue
            prev_day_rth_5min_data = prev_day_data.tz_convert(ET).between_time("09:30", "16:00")
            if prev_day_rth_5min_data.empty: continue
            prev_day_rth_close_val = prev_day_rth_5min_data['close'].iloc[-1]

            today_data = trade_window_data[trade_window_data.index.date == trade_date]
            if today_data.empty: continue
            today_data_et = today_data.tz_convert(ET)
            pre_market_data = today_data_et.between_time("04:00", "09:29")
            rth_5min_data_today = today_data_et.between_time("09:30", "16:00").tz_convert(timezone.utc)
            if rth_5min_data_today.empty: continue
            current_day_rth_open_val = rth_5min_data_today['open'].iloc[0]

            # --- Calculate metrics for logging and filtering ---
            data_for_metrics = pd.concat([pre_market_data, rth_5min_data_today.tz_convert(ET).first('5T')])
            cum_pos_mom = self._calculate_cumulative_positive_momentum(data_for_metrics['close'])
            path_chord_ratio = self._calculate_path_over_chord(data_for_metrics['close'])

            # --- Main Strategy Logic ---
            pnl, cost, shares = 0.0, 0.0, 0
            if self.strategy_name == 'gap_50pct_sl200':
                # NEW FILTER: Only short if momentum is high enough to be worth fading
                if cum_pos_mom > self.short_momentum_min_threshold:
                    pnl, cost, shares = self._strategy_gap_50pct_sl200(rth_5min_data_today, prev_day_rth_close_val, current_day_rth_open_val, current_account_value, volatility_metric)
            
            elif self.strategy_name == 'gap_up_long':
                # Filter for the long strategy based on init parameters
                if cum_pos_mom < self.long_momentum_max_threshold and path_chord_ratio < self.long_path_chord_max_threshold:
                    pnl, cost, shares = self._strategy_gap_up_long(rth_5min_data_today, current_account_value, volatility_metric)

            if shares > 0:
                daily_pnl[trade_date] = daily_pnl.get(trade_date, 0.0) + pnl
                daily_cost[trade_date] = daily_cost.get(trade_date, 0.0) + cost
                current_account_value += pnl
                trade_log.append({
                    "symbol": symbol, "trade_date": trade_date, "pnl": pnl,
                    "strategy": self.strategy_name,
                    "path_chord_ratio": path_chord_ratio,
                    "positive_momentum": cum_pos_mom
                })
        
        return symbol, pd.Series(daily_pnl).sort_index(), pd.Series(daily_cost).sort_index(), trade_log

    def run_backtest(self, max_workers: int = 16, start_date: date | None = None) -> Tuple[pd.Series, pd.DataFrame, List[Dict]]:
        all_tickers = list(self.universe_data.keys())
        individual_pnls, individual_costs = {}, {}
        full_trade_log = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(self.run_backtest_on_ticker, symbol, start_date): symbol for symbol in all_tickers}
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    _, daily_pnl, daily_cost, trade_log = future.result()
                    if not daily_pnl.empty:
                        individual_pnls[symbol] = daily_pnl
                        individual_costs[symbol] = daily_cost
                    full_trade_log.extend(trade_log)
                except Exception as exc:
                    logger.error(f"  {symbol} generated an exception: {exc}")
        pnl_df = pd.DataFrame(individual_pnls).fillna(0).sort_index()
        portfolio_daily_pnl = pnl_df.sum(axis=1)
        total_pnl_series = portfolio_daily_pnl.cumsum()
        total_pnl_series.name = 'Cumulative PnL'
        return total_pnl_series, pnl_df, full_trade_log

    def _calculate_statistics(self, pnl_series: pd.Series) -> Dict[str, float]:
        if pnl_series.empty or len(pnl_series) < 2:
            return {"Sharpe Ratio (Annualized)": np.nan, "Annualized Return": np.nan, "Annualized Std Dev": np.nan, "Total Trading Days": 0}
        daily_pnl = pnl_series.diff().fillna(pnl_series.iloc[0])
        daily_returns = []
        cap = self.initial_capital
        for pnl in daily_pnl:
            daily_returns.append(pnl / cap if cap != 0 else 0.0)
            cap += pnl
        daily_returns_series = pd.Series(daily_returns, index=pnl_series.index)
        if daily_returns_series.empty:
            return {"Sharpe Ratio (Annualized)": np.nan, "Annualized Return": np.nan, "Annualized Std Dev": np.nan, "Total Trading Days": 0}
        days = len(daily_returns_series)
        annual_factor = 252
        ann_return = (1 + daily_returns_series).prod()**(annual_factor / days) - 1
        ann_std_dev = daily_returns_series.std() * np.sqrt(annual_factor)
        sharpe = ann_return / ann_std_dev if ann_std_dev != 0 else np.nan
        return {"Sharpe Ratio (Annualized)": sharpe, "Annualized Return": ann_return, "Annualized Std Dev": ann_std_dev, "Total Trading Days": days}
