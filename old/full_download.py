import time
import requests
import json
from pathlib import Path
from datetime import date, datetime, timedelta

import pandas as pd

API_KEY        = ""
BASE_URL       = "https://api.polygon.io"

# Store everything under the NAS parquet directory
DATA_DIR       = Path("/mnt/nas/price_data/polygon")
ACTIVE_JSON    = DATA_DIR / "active_tickers.json"
DELISTED_JSON  = DATA_DIR / "delisted_tickers.json"
METADATA_FILE  = DATA_DIR / "5min_metadata.json"

# All symbol‐level Parquet files will live here
FIVE_MIN_DIR   = DATA_DIR / "5min_parquet"

TODAY = date.today()


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
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    with open(METADATA_FILE, "w") as f:
        json.dump(meta, f, indent=2)


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
    Returns a list of dicts: [{"timestamp":"YYYY-MM-DD HH:MM:SS","open":…, …}, …].
    """
    all_bars = []
    base_url = f"{BASE_URL}/v2/aggs/ticker/{sym}/range/5/minute/{start_date.isoformat()}/{TODAY.isoformat()}"
    params = {
        "adjusted": "true",
        "sort":     "asc",
        "limit":    5000,
        "apiKey":   API_KEY
    }

    # 1) First request
    resp = requests.get(base_url, params=params).json()
    count = 1

    while True:
        bars = resp.get("results", [])
        print(f"    • {sym}: batch {count} → {len(bars)} bars")

        for b in bars:
            ts = datetime.fromtimestamp(b["t"] / 1000.0)  # UTC → local
            all_bars.append({
                "timestamp":    ts,  # keep as datetime; Parquet will preserve
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


def fetch_incremental_5min_history(ticker_list: list[str]):
    """
    For each symbol:
      – Determine start_date using metadata (or list_date).
      – If start_date > TODAY, skip.
      – Otherwise, fetch 5-min bars, convert to DataFrame.
      – If a Parquet exists already, read it into pandas, concat new rows, and overwrite.
      – If not, write a fresh Parquet file.
      – Update metadata[sym] = TODAY.isoformat().
    """
    FIVE_MIN_DIR.mkdir(exist_ok=True, parents=True)
    metadata = load_metadata()
    total = len(ticker_list)

    for idx, sym in enumerate(ticker_list, start=1):
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
                print(f"[{idx}/{total}] {sym}: no list_date → skip.")
                continue
            start_date = ld

        if start_date > TODAY:
            print(f"[{idx}/{total}] {sym}: already up to date.")
            continue

        # 2) Fetch new bars
        new_bars = fetch_5min_for_symbol(sym, start_date)
        if not new_bars:
            print(f"[{idx}/{total}] {sym}: no new bars ({start_date}→{TODAY}).")
            metadata[sym] = TODAY.isoformat()
            save_metadata(metadata)
            continue

        # 3) Convert to DataFrame
        df_new = pd.DataFrame(new_bars)
        # Ensure timestamp column is parsed as datetime (Parquet will store it correctly)
        df_new["timestamp"] = pd.to_datetime(df_new["timestamp"])

        # 4) Write or append to Parquet
        dest = FIVE_MIN_DIR / f"{sym}.parquet"
        if dest.exists():
            # Read existing Parquet, concatenate, drop duplicates, and overwrite
            df_existing = pd.read_parquet(dest)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            # Optionally sort by timestamp and drop any duplicates
            df_combined.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
            df_combined.sort_values("timestamp", inplace=True)
            df_combined.to_parquet(dest, index=False)
            appended = len(df_combined) - len(df_existing)
            print(f"[{idx}/{total}] {sym}: combined and wrote {len(df_combined)} total rows "
                  f"( +{len(df_new)} new, {appended} net new after dedupe ).")
        else:
            # First time for this symbol: write fresh Parquet
            df_new.sort_values("timestamp", inplace=True)
            df_new.to_parquet(dest, index=False)
            print(f"[{idx}/{total}] {sym}: wrote {len(df_new)} rows to new Parquet.")

        # 5) Update metadata
        metadata[sym] = TODAY.isoformat()
        save_metadata(metadata)

    print("Incremental Parquet fetch finished.")


if __name__ == "__main__":
    # 1) Pull active tickers
    active = download_tickers(True, ACTIVE_JSON)
    # 2) Pull delisted tickers
    delisted = download_tickers(False, DELISTED_JSON)
    # 3) Combine
    all_symbols = active + delisted
    print(f"{len(active)} active tickers, {len(delisted)} delisted tickers, {len(all_symbols)} total tickers")
    # 4) Fetch incremental history into Parquet
    fetch_incremental_5min_history(all_symbols)
