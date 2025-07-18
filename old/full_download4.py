import time
import requests
import json
import sys
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
import threading
import concurrent.futures

import pandas as pd

API_KEY       = ""
BASE_URL      = "https://api.polygon.io"

DATA_DIR      = Path("/mnt/nas/price_data/polygon")
ACTIVE_JSON   = DATA_DIR / "active_tickers.json"
DELISTED_JSON = DATA_DIR / "delisted_tickers.json"
METADATA_FILE = DATA_DIR / "5min_metadata.json"

# Each ticker gets its own folder; inside that, one Parquet per YYYYMM
FIVE_MIN_DIR  = DATA_DIR / "5min_parquet"

TODAY = date.today()
HISTORIC_START = date(1980, 1, 1)

# Lock to serialize metadata updates (dict + file write)
_metadata_lock = threading.Lock()


def download_tickers(active: bool, dest_file: Path) -> list[str]:
    """
    Fetches all tickers with active={true|false}&market=stocks.
    Caches to dest_file. Returns list of symbols.
    """
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    if dest_file.exists():
        with open(dest_file, "r") as f:
            return json.load(f)

    symbols = []
    url = f"{BASE_URL}/v3/reference/tickers"
    params = {
        "market": "stocks",
        "active": "true" if active else "false",
        "limit": 1000,
        "apiKey": API_KEY
    }

    page = 1
    while True:
        resp = requests.get(url, params=params).json()
        batch = resp.get("results", [])
        print(f"[{'A' if active else 'D'}:{page}] Retrieved {len(batch)} tickers")
        symbols.extend([r["ticker"] for r in batch if r.get("ticker")])

        nxt = resp.get("next_url")
        if not nxt:
            break
        if "apiKey=" not in nxt:
            sep = "&" if "?" in nxt else "?"
            nxt = f"{nxt}{sep}apiKey={API_KEY}"
        url = nxt
        params = None
        page += 1
        time.sleep(0.1)

    with open(dest_file, "w") as f:
        json.dump(symbols, f, indent=2)
    print(f"Wrote {len(symbols)} {'active' if active else 'delisted'} tickers to {dest_file}")
    return symbols


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
    Thread-safe update of metadata[sym] = date_str and write to METADATA_FILE.
    Both the in-memory dict and file write happen under the same lock.
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


def process_symbol(sym: str, metadata: dict):
    """
    Worker function for one symbol:
      – Always start from HISTORIC_START (01-01-1980).
      – Fetch all 5-min bars from HISTORIC_START to TODAY.
      – Partition by month (YYYYMM), write or append per-month Parquet.
      – Update metadata[sym] = TODAY.isoformat().
    """
    try:
        print(f"[DEBUG] Starting historical fetch for {sym} from {HISTORIC_START}")

        # 1) Use HISTORIC_START for all symbols
        start_date = HISTORIC_START

        # 2) Fetch all bars from HISTORIC_START to TODAY
        new_bars = fetch_5min_for_symbol(sym, start_date)
        if not new_bars:
            print(f"[NO-NEW] {sym}: no bars from {start_date} to {TODAY}.")
            update_metadata_entry(metadata, sym, TODAY.isoformat())
            return

        # 3) Convert to DataFrame
        df_new = pd.DataFrame(new_bars)
        df_new["timestamp"] = pd.to_datetime(df_new["timestamp"])

        # 4) Ensure ticker‐folder exists
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
        print(f"[DEBUG] Finished historical fetch for {sym} and updated metadata.")

    except Exception as e:
        print(f"[ERROR] {sym} encountered an exception: {e}")
        update_metadata_entry(metadata, sym, TODAY.isoformat())
        print(f"[DEBUG] process_symbol errored for {sym} and metadata updated.")


def fetch_historic_for_all(ticker_list: list[str], max_workers: int = 10):
    """
    For each ticker:
      – If FIVE_MIN_DIR/{ticker} exists, skip and mark metadata as done.
      – Otherwise, submit to threadpool to fetch all historical data from 1980-01-01.
      – Update metadata.json for every ticker, whether skipped or processed.
    """
    FIVE_MIN_DIR.mkdir(exist_ok=True, parents=True)
    metadata = load_metadata()
    total = len(ticker_list)
    print(f"Starting historic fetch from {HISTORIC_START} for {total} tickers with {max_workers} workers…")

    # Build list of symbols that need processing
    to_process = []
    for sym in ticker_list:
        ticker_dir = FIVE_MIN_DIR / sym
        if ticker_dir.exists():
            # Already have a folder, so skip but mark metadata
            print(f"[SKIP] {sym}: directory already exists.")
            update_metadata_entry(metadata, sym, TODAY.isoformat())
        else:
            to_process.append(sym)

    print(f"{len(to_process)} remaining tickers to fetch.")

    in_flight = {}
    ticker_iter = iter(to_process)
    completed_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Prime the pump with up to max_workers initial submissions
        for _ in range(min(max_workers, len(to_process))):
            try:
                sym = next(ticker_iter)
            except StopIteration:
                break
            print(f"[DEBUG] Submitting initial historic task for {sym}")
            future = executor.submit(process_symbol, sym, metadata)
            in_flight[future] = sym

        # As each future completes, submit the next symbol until exhausted
        try:
            while in_flight:
                done, _ = concurrent.futures.wait(
                    in_flight.keys(),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                    timeout=1
                )
                if not done:
                    # Timeout: loop again to allow KeyboardInterrupt
                    continue

                for fut in done:
                    sym = in_flight.pop(fut)
                    completed_count += 1
                    try:
                        fut.result()
                        print(f"[DEBUG] Future for {sym} finished successfully.")
                    except Exception as e:
                        print(f"[ERROR] {sym} raised exception in future: {e}")

                    print(f"[{completed_count}/{len(to_process)}] Completed {sym}")

                    # Submit next symbol, if any remain
                    try:
                        nxt_sym = next(ticker_iter)
                        print(f"[DEBUG] Submitting next historic task for {nxt_sym}")
                        nxt_future = executor.submit(process_symbol, nxt_sym, metadata)
                        in_flight[nxt_future] = nxt_sym
                    except StopIteration:
                        print("[DEBUG] No more tickers to submit.")

        except KeyboardInterrupt:
            print("\n[Interrupted] Cancelling in‐flight workers…")
            for fut in in_flight:
                fut.cancel()
            executor.shutdown(wait=False)
            sys.exit(1)

    print("Historic Parquet fetch finished.")


if __name__ == "__main__":
    # 1) Pull active tickers
    active = download_tickers(True, ACTIVE_JSON)
    # 2) Pull delisted tickers
    delisted = download_tickers(False, DELISTED_JSON)
    # 3) Combine
    all_symbols = active + delisted
    print(f"{len(active)} active tickers, {len(delisted)} delisted tickers, {len(all_symbols)} total tickers")

    # 4) Fetch from 1980-01-01 for all tickers (skip existing dirs)
    fetch_historic_for_all(all_symbols, max_workers=30)
