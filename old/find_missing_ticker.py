#!/usr/bin/env python3
import json
from pathlib import Path
from typing import List

# Adjust these paths if your layout differs
DATA_DIR      = Path("/mnt/nas/price_data/polygon")
ACTIVE_JSON   = DATA_DIR / "active_tickers.json"
DELISTED_JSON = DATA_DIR / "delisted_tickers.json"
FIVE_MIN_DIR  = DATA_DIR / "5min_parquet"

def load_symbol_list() -> List[str]:
    """
    Returns the concatenation of active + delisted tickers
    (in the same order as in the JSON files).
    """
    symbols = []
    for file in (ACTIVE_JSON, DELISTED_JSON):
        if file.exists():
            with open(file, "r") as f:
                batch = json.load(f)
                symbols.extend(batch)
        else:
            print(f"[WARN] {file} not found; skipping.")
    return symbols

def find_first_missing():
    """
    Iterates through all symbols and prints the first one
    for which FIVE_MIN_DIR/{symbol} does not exist.
    """
    all_symbols = load_symbol_list()
    for sym in all_symbols:
        ticker_dir = FIVE_MIN_DIR / sym
        if not ticker_dir.exists():
            print(sym)
            return
    print("🎉 All symbols have a directory under 5min_parquet.")

if __name__ == "__main__":
    find_first_missing()
