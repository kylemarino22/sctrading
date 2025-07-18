import os
import requests
import datetime

API_KEY = "_74P3fk7zYzp5VS6SbPgPW_ChyVhF_IA"
BASE_URL = "https://api.polygon.io/v3/quotes"

def fetch_extreme_quote(symbol: str, order: str = "asc") -> dict:
    """
    Fetches either the earliest (order="asc") or latest (order="desc")
    NBBO quote for `symbol`, returning the single quote record.
    """
    params = {
        "limit": 1,
        "sort": "timestamp",    # sort by the quote timestamp
        "order": order,         # "asc" for earliest, "desc" for latest
        "apiKey": API_KEY
    }
    resp = requests.get(f"{BASE_URL}/{symbol}", params=params)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    return results[0] if results else None

def ns_to_datetime(ns: int) -> datetime.datetime:
    """Convert a nanosecond UNIX timestamp to a UTC datetime."""
    return datetime.datetime.fromtimestamp(ns / 1e9, tz=datetime.timezone.utc)

def main():
    symbol = input("Enter ticker symbol: ").strip().upper()

    earliest = fetch_extreme_quote(symbol, order="asc")
    latest   = fetch_extreme_quote(symbol, order="desc")

    if not earliest:
        print(f"No quote data found for {symbol}.")
        return

    dt_earliest = ns_to_datetime(earliest["t"])
    dt_latest   = ns_to_datetime(latest["t"])

    span = dt_latest - dt_earliest

    print(f"\nQuote coverage for {symbol}:")
    print(f"  • Earliest quote: {dt_earliest.isoformat()}")
    print(f"  • Latest   quote: {dt_latest.isoformat()}")
    print(f"  • Total span   : {span.days} days (~{span.days/365:.1f} years)")

if __name__ == "__main__":
    main()
