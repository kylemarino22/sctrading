import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List, Dict, Optional
import logging

# Import the FileIO class from the separate file
from file_io import FileIO


# --- Configuration ---
# Configure logger for the application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Constants ---
API_KEY = "_74P3fk7zYzp5VS6SbPgPW_ChyVhF_IA"  # Placeholder: Use a secure method to manage API keys.
BASE_URL = "https://api.polygon.io"
HISTORIC_START_DATE = datetime(1980, 1, 1, tzinfo=timezone.utc)
TICKER_UPDATE_INTERVAL = timedelta(hours=24) # Using timedelta is more explicit.
TODAY = datetime.now(timezone.utc).date()
MAX_WORKERS = 30 # Concurrency limit for downloads

# Mapping for Polygon API parameters
FREQ_MAP = {
    "minute": {"multiplier": 1, "timespan": "minute"},
    "5minute": {"multiplier": 5, "timespan": "minute"},
    "hour": {"multiplier": 1, "timespan": "hour"},
    "day": {"multiplier": 1, "timespan": "day"},
}

class PolygonDataDownloader:
    """
    Handles data downloading from the Polygon.io API.
    """
    def __init__(self, file_io_instance: FileIO):
        self.api_key = API_KEY
        self.base_url = BASE_URL
        self.file_io = file_io_instance
        self.session = self._create_session()
        logger.debug("PolygonDataDownloader initialized.")

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({'Authorization': f'Bearer {self.api_key}'})
        return session

    def _download_paginated_data(self, url: str, params: Dict, desc: str) -> List[Dict]:
        all_results = []
        page_num = 1
        next_url = url
        while next_url:
            try:
                response = self.session.get(next_url, params=params)
                response.raise_for_status()
                data = response.json()
                batch = data.get("results", [])
                logger.debug(f"[{desc}:{page_num}] Retrieved {len(batch)} items.")
                all_results.extend(batch)
                next_url = data.get("next_url")
                params = None
                page_num += 1
                time.sleep(0.01)
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP Error for {desc}: {e.response.status_code} - {e.response.text}")
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error for {desc}: {e}")
                break
        logger.info(f"Finished downloading {len(all_results)} total {desc} items.")
        return all_results

    def _download_tickers(self, active: bool) -> List[Dict]:
        ticker_type = "active" if active else "delisted"
        url = f"{self.base_url}/v3/reference/tickers"
        params = {"market": "stocks", "active": str(active).lower(), "limit": 1000}
        logger.info(f"Starting download of {ticker_type} tickers...")
        return self._download_paginated_data(url, params, f"{ticker_type} tickers")

    def get_tickers(self, force_update: bool = False) -> Tuple[List[Dict], List[Dict]]:
        now_utc = datetime.now(timezone.utc)
        active_tickers, active_last_updated = self.file_io.read_ticker_file(delisted=False)
        if force_update or not active_last_updated or (now_utc - active_last_updated) > TICKER_UPDATE_INTERVAL:
            logger.info("Fetching latest active tickers.")
            active_tickers = self._download_tickers(active=True)
            if active_tickers: self.file_io.write_ticker_file(active_tickers, delisted=False)
        else:
            logger.info("Active tickers up-to-date.")

        delisted_tickers, delisted_last_updated = self.file_io.read_ticker_file(delisted=True)
        if force_update or not delisted_last_updated or (now_utc - delisted_last_updated) > TICKER_UPDATE_INTERVAL:
            logger.info("Fetching latest delisted tickers.")
            delisted_tickers = self._download_tickers(active=False)
            if delisted_tickers: self.file_io.write_ticker_file(delisted_tickers, delisted=True)
        else:
            logger.info("Delisted tickers up-to-date.")
        return active_tickers, delisted_tickers

    def fetch_data_for_symbol(self, symbol: str, freq: str, start_date: datetime) -> pd.DataFrame:
        if freq not in FREQ_MAP:
            logger.warning(f"Invalid frequency '{freq}'. Skipping.")
            return pd.DataFrame()
        p = FREQ_MAP[freq]
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{p['multiplier']}/{p['timespan']}/{start_date.date().isoformat()}/{TODAY.isoformat()}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000}
        logger.debug(f"Fetching {symbol} ({freq}) from {start_date.strftime('%Y-%m-%d')}.")
        bars = self._download_paginated_data(url, params, f"{symbol} {freq} bars")
        
        if not bars:
            logger.info(f"No data fetched for {symbol} ({freq}).")
            return pd.DataFrame()
        df = pd.DataFrame(bars).rename(columns={'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 'vw': 'vwap', 'n': 'transactions'})
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.set_index('timestamp')
        expected_cols = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']
        for col in expected_cols:
            if col not in df.columns: df[col] = None
        
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').round(0).astype('Int64')
        if 'transactions' in df.columns:
            df['transactions'] = pd.to_numeric(df['transactions'], errors='coerce').round(0).astype('Int64')
        
        logger.info(f"Fetched {len(df)} records for {symbol} ({freq}).")
        return df[expected_cols]

    def download_symbol(self, symbol: str, freq: str, trade_or_quote: str,
                        start_date: Optional[datetime] = None, update: bool = False) -> Tuple[str, bool]:
        if trade_or_quote == "quote":
            logger.warning("Quote data not supported. Skipping.")
            return symbol, False
            
        effective_start_date = start_date
        was_up_to_date = False
        
        # --- FIX: When update is False, we want to redownload and overwrite. ---
        # The overwrite flag is the opposite of the update flag.
        should_overwrite = not update

        if update and not effective_start_date:
            last_ts = self.file_io.get_last_data_timestamp(symbol, freq, trade_or_quote)
            if last_ts:
                effective_start_date = last_ts.replace(hour=0, minute=0, second=0, microsecond=0)
                
                if effective_start_date.date() >= TODAY:
                    was_up_to_date = True
                    logger.info(f"Data for {symbol} ({freq}) is current. Skipping.")
                    return symbol, was_up_to_date
            else:
                effective_start_date = HISTORIC_START_DATE
        elif not effective_start_date:
            effective_start_date = HISTORIC_START_DATE

        try:
            df = self.fetch_data_for_symbol(symbol, freq, effective_start_date)
            if not df.empty:
                # Pass the dynamically determined overwrite flag
                self.file_io.write_raw_data(df, symbol, freq, trade_or_quote, overwrite=should_overwrite)
            else:
                logger.info(f"No new data for {symbol} ({freq}).")
        except Exception as e:
            logger.error(f"Failed to download/process {symbol} ({freq}): {e}", exc_info=True)
        
        return symbol, was_up_to_date

    def download(self, update: bool = True, target_freqs: Optional[List[str]] = None,
                 target_types: Optional[List[str]] = None, only_active_tickers: bool = False):
        logger.info(f"Starting download run: update={update}, only_active={only_active_tickers}")
        active_tickers, delisted_tickers = self.get_tickers()
        
        tickers_to_process_symbols = {t['ticker'] for t in active_tickers if 'ticker' in t}

        if not only_active_tickers:
            delisted_ticker_symbols = {t['ticker'] for t in delisted_tickers if 'ticker' in t}
            tickers_to_process_symbols.update(delisted_ticker_symbols)

        tickers_to_process = sorted(list(tickers_to_process_symbols))
        
        logger.info(f"Processing {len(tickers_to_process)} unique tickers.")

        frequencies = target_freqs or list(FREQ_MAP.keys())
        data_types = target_types or ["trade"]

        tasks = [(symbol, freq, trade_type, None, update) for symbol in tickers_to_process for freq in frequencies for trade_type in data_types]
        
        logger.info(f"Submitting {len(tasks)} tasks to {MAX_WORKERS} workers...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(self.download_symbol, *task): task for task in tasks}
            for i, future in enumerate(as_completed(futures), 1):
                symbol, freq, _, _, _ = futures[future]
                try:
                    _, was_up_to_date = future.result()
                    if not was_up_to_date:
                        logger.info(f"({i}/{len(tasks)}) Completed task for {symbol} ({freq}).")
                except Exception as e:
                    logger.error(f"({i}/{len(tasks)}) Task for {symbol} ({freq}) failed: {e}")

        logger.info("Data download process finished.")

# --- Execution Example ---
if __name__ == "__main__":
    data_directory = Path("/mnt/nas/price_data/polygon")
    file_io = FileIO(data_dir=data_directory)
    downloader = PolygonDataDownloader(file_io_instance=file_io)
    
    logger.info("="*50)
    logger.info("STARTING DOWNLOAD FOR ACTIVE TICKERS (DAILY TRADES)")
    logger.info("="*50)
    
    # downloader.download(
    #     update=True,
    #     target_freqs=["5minute"],
    #     target_types=["trade"],
    #     only_active_tickers=True
    # )

    downloader.download_symbol("ACIW", "5minute", "trade", update=False)
    
    logger.info("\n" + "="*50)
    logger.info("DOWNLOAD FOR ACTIVE TICKERS COMPLETE.")
    logger.info("="*50)
