import os
import json
import time
import requests
import pandas as pd

from pathlib import Path
from datetime import date, timedelta, datetime

# ============================================
# CONFIGURATION
# ============================================

API_KEY = ""
BASE_URL = "https://api.polygon.io"
DATA_DIR = Path("data")
VOL_FILE = DATA_DIR / "daily_volumes.csv"
TICKERS_FILE = DATA_DIR / "all_tickers.json"
FIFTEEN_MIN_DIR = DATA_DIR / "15min"

# Ensure directories exist
for folder in (DATA_DIR, FIFTEEN_MIN_DIR):
    folder.mkdir(parents=True, exist_ok=True)


# ============================================
# STEP 1: Fetch (or load) the full universe of US tickers
# ============================================
def get_all_us_tickers() -> list[str]:
    """
    Returns a list of all active US stock tickers.
    If DATA_DIR/all_tickers.json exists, load from disk. Otherwise, paginate
    through /v3/reference/tickers, save to JSON, and return.
    """
    if TICKERS_FILE.exists():
        with open(TICKERS_FILE, "r") as f:
            return json.load(f)

    all_tickers: list[str] = []
    url = f"{BASE_URL}/v3/reference/tickers"
    params = {
        "market": "stocks",
        "active": "true",
        "limit": 1000,
        "apiKey": API_KEY
    }
    cursor = None

    while True:
        if cursor:
            params["cursor"] = cursor

        resp = requests.get(url, params=params).json()
        results = resp.get("results", [])
        for entry in results:
            symbol = entry.get("ticker")
            if symbol:
                all_tickers.append(symbol)

        cursor = resp.get("next_url")
        if not cursor:
            break

        # Throttle to avoid transient 429s
        time.sleep(0.2)

    # Save to disk for future runs
    with open(TICKERS_FILE, "w") as f:
        json.dump(all_tickers, f)

    return all_tickers


# ============================================
# STEP 2: Incrementally update daily volumes for the full universe
# ============================================
def update_daily_volumes():
    """
    Ensures that DATA_DIR/daily_volumes.csv contains one row per 
    (date, ticker) with that day's volume, for all dates up to yesterday.
    If the file already exists, we append data from (last_date + 1) through yesterday.
    Otherwise, start from one year ago through yesterday.
    Columns: date (YYYY-MM-DD), ticker, volume
    """
    # Determine date range
    today = date.today()
    yesterday = today - timedelta(days=1)

    if VOL_FILE.exists():
        # Load existing and find the maximum date present
        df_existing = pd.read_csv(VOL_FILE, parse_dates=["date"])
        last_date_in_file = df_existing["date"].max().date()
        start_date = last_date_in_file + timedelta(days=1)
    else:
        # No file exists; start from one year ago
        start_date = today - timedelta(days=365)

    # If start_date > yesterday, nothing to do
    if start_date > yesterday:
        print(f"No new daily-volume data needed; last date is already {start_date - timedelta(days=1)}.")
        return

    # Loop from start_date through yesterday
    current_date = start_date
    while current_date <= yesterday:
        # Skip weekends
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue

        date_str = current_date.isoformat()  # "YYYY-MM-DD"
        url = f"{BASE_URL}/v2/aggs/grouped/locale/us/market/stocks/{date_str}"
        params = {"adjusted": "true", "apiKey": API_KEY}

        try:
            resp = requests.get(url, params=params).json()
            day_results = resp.get("results", [])

            if day_results:
                # Build DataFrame with columns: date, ticker, volume
                records = [
                    {"date": date_str, "ticker": bar["T"], "volume": bar["v"]}
                    for bar in day_results
                ]
                df_day = pd.DataFrame.from_records(records)

                # Append to CSV
                header = not VOL_FILE.exists()
                df_day.to_csv(VOL_FILE, mode="a", index=False, header=header)
                print(f"Appended {len(df_day)} rows for {date_str}.")
            else:
                # No trading data returned (unlikely on a weekday)
                print(f"Warning: No results for {date_str} (market closed or error).")

        except Exception as e:
            print(f"⚠️ Error fetching grouped data for {date_str}: {e}")

        # Throttle at ~10 QPS (0.1s sleep) to safely stay under 100 QPS
        time.sleep(0.1)
        current_date += timedelta(days=1)

    print("Daily volumes update complete.")


# ============================================
# STEP 3: Identify tickers with daily volume < 15M for the past year
# ============================================
def get_low_vol_tickers(threshold: int = 15_000_000) -> list[str]:
    """
    Reads DATA_DIR/daily_volumes.csv, filters to the last 365 days, and
    returns a list of symbols whose maximum daily volume over that window
    is strictly less than `threshold`.
    """
    if not VOL_FILE.exists():
        raise FileNotFoundError(f"{VOL_FILE} not found. Please run update_daily_volumes() first.")

    df = pd.read_csv(VOL_FILE, parse_dates=["date"])
    cutoff_date = date.today() - timedelta(days=365)
    df_recent = df[df["date"].dt.date >= cutoff_date]

    # Group by ticker, get the max volume in that window
    max_vol_by_ticker = df_recent.groupby("ticker")["volume"].max().reset_index()

    # Filter those with max < threshold
    low_vol_df = max_vol_by_ticker[max_vol_by_ticker["volume"] < threshold]
    low_vol_tickers = low_vol_df["ticker"].tolist()

    print(f"Found {len(low_vol_tickers)} tickers with max daily volume < {threshold:,} in the past year.")
    return low_vol_tickers


# ============================================
# STEP 4: Fetch 1 year of 15-minute OHLCV for filtered tickers and save as CSV
# ============================================
def fetch_1yr_15min_bars(ticker_list: list[str]):
    """
    If you still want 15-minute bars, switch to timespan="minute":
    /range/15/minute/{from}/{to}. Save as CSV in data/15min_csv/{TICKER}.csv
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=2*365)
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()

    base = f"{BASE_URL}/v2/aggs/ticker"
    total = len(ticker_list)

    if total == 0:
        print("No low-vol tickers to fetch 15-minute bars for.")
        return

    for idx, sym in enumerate(ticker_list, start=1):
        dest_file = FIFTEEN_MIN_DIR / f"{sym}.csv"
        # Skip re-downloading if file exists?
        # if dest_file.exists():
        #     print(f"[{idx}/{total}] Skipping {sym} (already has 15-min CSV).")
        #     continue

        # CORRECT endpoint for 15-minute bars:
        url = f"{base}/{sym}/range/15/minute/{start_str}/{end_str}"
        params = {"adjusted": "true", "limit": 5000, "apiKey": API_KEY}

        try:
            resp = requests.get(url, params=params).json()
            bars = resp.get("results", [])

            if not bars:
                print(f"[{idx}/{total}] {sym}: no 15-min bars returned.")
            else:
                df_bars = pd.DataFrame.from_records(bars)
                df_bars["timestamp"] = df_bars["t"].apply(lambda ms: datetime.fromtimestamp(ms / 1000))
                df_bars = df_bars.rename(columns={
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                    "n": "transactions"
                })
                df_bars = df_bars[["timestamp", "open", "high", "low", "close", "volume", "transactions"]]

                df_bars.to_csv(dest_file, index=False)
                print(f"[{idx}/{total}] {sym}: saved {len(df_bars)} rows of 15-min to {dest_file.name}.")

        except Exception as e:
            print(f"[{idx}/{total}] ⚠️ Error fetching 15-min bars for {sym}: {e}")

        time.sleep(0.02)

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    # 1) Ensure we have the full ticker universe
    print("Step 1: Loading all US stock tickers...")
    all_tickers = get_all_us_tickers()
    print(f" -> Total tickers in universe: {len(all_tickers)}")

    # 2) Update (or create) daily_volumes.csv incrementally
    print("\nStep 2: Updating daily volume data for all tickers...")
    update_daily_volumes()

    # 3) Identify low-volume tickers (vol < 15M for past 365 days)
    print("\nStep 3: Filtering tickers by 1-year daily volume < 15M...")
    low_vol_tickers = get_low_vol_tickers(threshold=15_000_000)

    print("\nStep 4: Fetching 1 year of daily OHLCV bars for each low-vol ticker...")
    fetch_1yr_15min_bars(low_vol_tickers)

    print("\nAll tasks completed.")
