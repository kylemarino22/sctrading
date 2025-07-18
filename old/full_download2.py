import time
import requests
import json
import sys
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
import threading
import concurrent.futures

import pandas as pd

API_KEY       = "8PB9Ur5yWGX7d7okqxhhvB9i_0bCsOut"
BASE_URL      = "https://api.polygon.io"

DATA_DIR      = Path("/mnt/nas/price_data/polygon")
ACTIVE_JSON   = DATA_DIR / "active_tickers.json"
DELISTED_JSON = DATA_DIR / "delisted_tickers.json"
METADATA_FILE = DATA_DIR / "5min_metadata.json"

# Each ticker gets its own folder; inside that, one parquet per YYYYMM
FIVE_MIN_DIR  = DATA_DIR / "5min_parquet"

TODAY = date.today()

# Lock to serialize writes to METADATA_FILE
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


def save_metadata(meta: dict):
    """
    Saves the metadata dict to disk in a thread‐safe manner, with debug prints.
    """
    print(f"[DEBUG] Attempting to acquire metadata_lock to save metadata for {len(meta)} tickers...")
    with _metadata_lock:
        print("[DEBUG] metadata_lock acquired in save_metadata.")
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        with open(METADATA_FILE, "w") as f:
            json.dump(meta, f, indent=2)
        print("[DEBUG] Saved metadata to disk; releasing metadata_lock.")
    # lock automatically released here


def get_list_date(symbol: str) -> date | None:
    """
    Calls /v3/reference/tickers/{symbol} to read list_date.
    Returns a Python date or None if not found.
    """
    try:
        resp = requests.get(
            f"{BASE_URL}/v3/reference/tickers/{symbol}",
            params={"apiKey": API_KEY}
        ).json().get("results", {})
        ld = resp.get("list_date")  # e.g. "2005-03-15"
        if ld:
            return datetime.strptime(ld, "%Y-%m-%d").date()
    except:
        pass
    return None


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
      – Determine start_date using metadata (or list_date).
      – If start_date > TODAY, mark up‐to‐date and return.
      – Otherwise, fetch 5-min bars, convert to DataFrame.
      – Partition new data by month (YYYYMM), and append/update per‐month parquet.
      – Update metadata[sym] = TODAY.isoformat() via save_metadata() directly.
    """
    try:
        print(f"[DEBUG] process_symbol starting for {sym}")

        # 1) Determine start_date
        last_str = metadata.get(sym)
        if last_str:
            try:
                last_date = datetime.strptime(last_str, "%Y-%m-%d").date()
                start_date = last_date + timedelta(days=1)
            except:
                start_date = get_list_date(sym) or TODAY
        else:
            ld = get_list_date(sym)
            if not ld:
                print(f"[SKIP] {sym}: no list_date.")
                # Directly call save_metadata without extra lock
                metadata[sym] = TODAY.isoformat()
                save_metadata(metadata)
                return
            start_date = ld

        if start_date > TODAY:
            print(f"[UP-TO-DATE] {sym}: already up to date.")
            # Directly call save_metadata without extra lock
            metadata[sym] = TODAY.isoformat()
            save_metadata(metadata)
            return

        # 2) Fetch new bars
        new_bars = fetch_5min_for_symbol(sym, start_date)
        if not new_bars:
            print(f"[NO-NEW] {sym}: no new bars ({start_date}→{TODAY}).")
            metadata[sym] = TODAY.isoformat()
            save_metadata(metadata)
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
        metadata[sym] = TODAY.isoformat()
        save_metadata(metadata)
        print(f"[DEBUG] process_symbol finished for {sym} and metadata updated.")

    except Exception as e:
        # Catch any exception so a single ticker error doesn't kill the entire run
        print(f"[ERROR] {sym} encountered an exception: {e}")
        metadata[sym] = TODAY.isoformat()
        save_metadata(metadata)
        print(f"[DEBUG] process_symbol errored for {sym} and metadata updated.")


def fetch_incremental_5min_history(ticker_list: list[str], max_workers: int = 10):
    """
    Runs in “batches of max_workers”:
      – Reads metadata
      – Submits up to max_workers symbols at a time
      – As each future completes, immediately submit the next symbol
      – Handles clean shutdown on KeyboardInterrupt
    """
    FIVE_MIN_DIR.mkdir(exist_ok=True, parents=True)
    metadata = load_metadata()
    total = len(ticker_list)
    print(f"Starting incremental fetch for {total} tickers with up to {max_workers} workers…")

    in_flight = {}
    ticker_iter = iter(ticker_list)
    completed_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 1) Prime with up to max_workers initial submissions
        for _ in range(min(max_workers, total)):
            try:
                sym = next(ticker_iter)
            except StopIteration:
                break
            print(f"[DEBUG] Submitting initial task for {sym}")
            future = executor.submit(process_symbol, sym, metadata)
            in_flight[future] = sym

        # 2) As each future completes, submit the next symbol until exhausted
        try:
            while in_flight:
                done, _ = concurrent.futures.wait(
                    in_flight.keys(),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                    timeout=1
                )
                if not done:
                    # Timeout: allow for KeyboardInterrupt
                    continue

                for fut in done:
                    sym = in_flight.pop(fut)
                    completed_count += 1
                    try:
                        fut.result()
                        print(f"[DEBUG] Future for {sym} returned successfully.")
                    except Exception as e:
                        print(f"[ERROR] {sym} raised exception in future: {e}")

                    print(f"[{completed_count}/{total}] Completed {sym}")

                    # Submit next symbol, if any remain
                    try:
                        nxt_sym = next(ticker_iter)
                        print(f"[DEBUG] Submitting next task for {nxt_sym}")
                        nxt_future = executor.submit(process_symbol, nxt_sym, metadata)
                        in_flight[nxt_future] = nxt_sym
                    except StopIteration:
                        print("[DEBUG] No more symbols to submit.")

        except KeyboardInterrupt:
            print("\n[Interrupted] Cancelling in‐flight workers…")
            for fut in in_flight:
                fut.cancel()
            executor.shutdown(wait=False)
            sys.exit(1)

    print("Incremental Parquet fetch finished.")


if __name__ == "__main__":
    # 1) Pull active tickers
    active = download_tickers(True, ACTIVE_JSON)
    # 2) Pull delisted tickers
    delisted = download_tickers(False, DELISTED_JSON)
    # 3) Combine
    all_symbols = active + delisted
    print(f"{len(active)} active tickers, {len(delisted)} delisted tickers, {len(all_symbols)} total tickers")

    # 4) Fetch incremental history in parallel, 10 at a time
    fetch_incremental_5min_history(all_symbols, max_workers=30)
