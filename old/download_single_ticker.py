#!/usr/bin/env python3
import time
import json
import sys
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
import threading

import requests
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION: adjust paths and your API key here
# ──────────────────────────────────────────────────────────────────────────────
API_KEY       = "8PB9Ur5yWGX7d7okqxhhvB9i_0bCsOut"
BASE_URL      = "https://api.polygon.io"

DATA_DIR      = Path("/mnt/nas/price_data/polygon")
METADATA_FILE = DATA_DIR / "5min_metadata.json"
FIVE_MIN_DIR  = DATA_DIR / "5min_parquet"

TODAY = date.today()
HISTORIC_START = date(1980, 1, 1)

# Lock to serialize metadata writes
_metadata_lock = threading.Lock()
# ──────────────────────────────────────────────────────────────────────────────

def load_metadata() -> dict:
    """
    Loads { ticker: "YYYY-MM-DD" } → last fetched date.
    """
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {}

def update_metadata_entry(metadata: dict, sym: str, date_str: str):
    """
    Thread-safe update of metadata[sym] = date_str, then write to METADATA_FILE.
    """
    with _metadata_lock:
        metadata[sym] = date_str
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[DEBUG] metadata updated and saved for {sym} → {date_str}")

def fetch_5min_for_symbol(sym: str, start_date: date) -> list[dict]:
    """
    Fetches all 5-minute bars (including pre-/post-market) for `sym`
    from `start_date` through TODAY by following next_url paging.
    Returns a list of dicts: [{"timestamp":datetime, "open":…, …}, …].
    """
    all_bars = []
    base_url = f"{BASE_URL}/v2/aggs/ticker/{sym}/range/5/minute/{start_date.isoformat()}/{TODAY.isoformat()}"
    params = {
        "adjusted": "true",
        "sort":     "asc",
        "limit":    5000,
        "apiKey":   API_KEY
    }

    count = 1
    resp = requests.get(base_url, params=params).json()

    while True:
        bars = resp.get("results", [])
        print(f"    • [{sym}] batch {count} → {len(bars)} bars")
        for b in bars:
            ts = datetime.fromtimestamp(b["t"] / 1000.0, tz=timezone.utc)
            all_bars.append({
                "timestamp":    ts,
                "open":         b["o"],
                "high":         b["h"],
                "low":          b["l"],
                "close":        b["c"],
                "volume":       b["v"],
                "transactions": b.get("n", None)
            })

        next_url = resp.get("next_url")
        if not next_url:
            break

        # Ensure apiKey is present
        if "apiKey=" not in next_url:
            sep = "&" if "?" in next_url else "?"
            next_url = f"{next_url}{sep}apiKey={API_KEY}"

        count += 1
        time.sleep(0.01)
        resp = requests.get(next_url).json()

    return all_bars

def process_single_symbol(sym: str, metadata: dict):
    """
    Download 5-minute history for a single ticker, starting from 1980-01-01:
      – Fetch all bars from HISTORIC_START through TODAY.
      – Partition by month (YYYYMM) and write/appended per-month Parquet.
      – Update metadata[sym] = TODAY.isoformat().
    """
    try:
        print(f"[DEBUG] process_single_symbol starting for {sym} from {HISTORIC_START}")

        # 1) Always start from January 1, 1980
        start_date = HISTORIC_START

        # 2) Fetch all bars from HISTORIC_START to TODAY
        new_bars = fetch_5min_for_symbol(sym, start_date)
        if not new_bars:
            print(f"[NO-NEW] {sym}: no bars from {start_date} to {TODAY}. Marking as done.")
            update_metadata_entry(metadata, sym, TODAY.isoformat())
            return

        # 3) Convert to DataFrame
        df_new = pd.DataFrame(new_bars)
        df_new["timestamp"] = pd.to_datetime(df_new["timestamp"])

        # 4) Ensure ticker folder exists
        ticker_dir = FIVE_MIN_DIR / sym
        ticker_dir.mkdir(exist_ok=True, parents=True)

        # 5) Partition df_new by year_month
        df_new["year_month"] = df_new["timestamp"].dt.strftime("%Y%m")
        for ym, group in df_new.groupby("year_month"):
            month_file = ticker_dir / f"{ym}.parquet"
            group = group.drop(columns=["year_month"])
            if month_file.exists():
                df_existing = pd.read_parquet(month_file)
                df_combined = pd.concat([df_existing, group], ignore_index=True)
                df_combined.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
                df_combined.sort_values("timestamp", inplace=True)
                df_combined.to_parquet(month_file, index=False)
                net_new = len(df_combined) - len(df_existing)
                print(f"[{sym}] Updated {ym}.parquet: {len(df_combined)} rows (+{net_new} new).")
            else:
                group.sort_values("timestamp", inplace=True)
                group.to_parquet(month_file, index=False)
                print(f"[{sym}] Wrote {ym}.parquet: {len(group)} rows.")

        # 6) Update metadata
        update_metadata_entry(metadata, sym, TODAY.isoformat())
        print(f"[DEBUG] process_single_symbol finished for {sym} and metadata updated.")

    except Exception as e:
        print(f"[ERROR] {sym} encountered an exception: {e}")
        update_metadata_entry(metadata, sym, TODAY.isoformat())
        print(f"[DEBUG] process_single_symbol errored for {sym} and metadata updated.")

if __name__ == "__main__":
    # Expect exactly one ticker as argument
    if len(sys.argv) != 2:
        print("Usage: python download_single_ticker.py <TICKER>")
        sys.exit(1)

    ticker = sys.argv[1].strip().upper()

    # 1) Load metadata
    metadata = load_metadata()

    # 2) Check if directory already exists; if so, skip
    if (FIVE_MIN_DIR / ticker).exists():
        print(f"[SKIP] {ticker}: directory already exists. Marking metadata and exiting.")
        update_metadata_entry(metadata, ticker, TODAY.isoformat())
        sys.exit(0)

    # 3) Process that one ticker
    process_single_symbol(ticker, metadata)

    print("Done.")
