import pandas as pd
from pathlib import Path
from datetime import datetime, date, timezone
from typing import Tuple, List, Dict, Union, Optional
import logging
from file_io import FileIO
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from data_downloader import PolygonDataDownloader

logger = logging.getLogger(__name__)

class DataManager:
    """
    Manages access to market data. Acts as a clean interface
    to the underlying FileIO and handles all timezone conversions.
    """

    def __init__(self, base_dir: Path):
        """
        Initializes the DataManager with a single base directory.
        """
        self.io = FileIO(base_dir)
        try:
            # Use 'America/Los_Angeles' for PST/PDT
            self.default_tz = ZoneInfo("America/Los_Angeles")
        except ZoneInfoNotFoundError:
            logger.warning("zoneinfo 'America/Los_Angeles' not found. Falling back to UTC-8.")
            self.default_tz = timezone(pd.Timedelta(hours=-8))

        self.downloader = PolygonDataDownloader(self.io)

        
        logger.info(f"DataManager initialized. Base directory: {base_dir}")

    def _normalize_date(self, dt: Union[date, datetime, None], is_end_date: bool = False) -> Optional[datetime]:
        """
        Normalizes a date or datetime object to a timezone-aware UTC datetime.
        - Naive datetimes are assumed to be in the default timezone (PST).
        - Date objects are converted to datetime at the start/end of the day.
        """
        if dt is None:
            return None

        # If it's a date object, convert to datetime at start or end of the day
        if isinstance(dt, date) and not isinstance(dt, datetime):
            if is_end_date:
                dt = datetime.combine(dt, datetime.max.time())
            else:
                dt = datetime.combine(dt, datetime.min.time())

        # If it's naive, attach the default timezone (PST)
        if dt.tzinfo is None:
            # --- FIX: Use .replace() for modern zoneinfo, not .localize() ---
            dt = dt.replace(tzinfo=self.default_tz)
        
        # Convert to UTC for consistent internal handling
        return dt.astimezone(timezone.utc)

    def get_data(self, symbol: str, freq: str = 'day', quote: bool = False,
                 start: Union[date, datetime, None] = None,
                 end: Union[date, datetime, None] = None,
                 enriched: bool = False) -> pd.DataFrame:
        """
        Retrieves market data, handling all timezone conversions before passing
        the request to the FileIO layer.
        """
        if enriched:
            logger.debug(f"Getting ENRICHED data for {symbol}...")
            # Enriched data is not typically filtered by date in this design,
            # as it's a single file per ticker.
            return self.io.read_enriched_data(symbol)
        
        trade_or_quote = "quote" if quote else "trade"
        logger.debug(f"Getting RAW {trade_or_quote} {freq} data for {symbol}...")

        # Normalize start and end dates to timezone-aware UTC datetimes
        utc_start = self._normalize_date(start, is_end_date=False)
        utc_end = self._normalize_date(end, is_end_date=True)

        return self.io.read_raw_data(symbol, freq, trade_or_quote, utc_start, utc_end)

    def write_enriched_data(self, df: pd.DataFrame, ticker: str):
        """Writes an enriched data file for a given ticker."""
        logger.info(f"Writing enriched data for {ticker}...")
        self.io.write_enriched_data(df, ticker)

    def get_universe_path(self) -> Path:
        """Returns the path to the main universe file from FileIO."""
        # This assumes a specific universe file name, which is fine for this project
        return self.io.data_dir / "universes" / "pretraining_universe.json"

    def get_tickers(self, force_update: bool = False) -> Tuple[List[Dict], List[Dict]]:
        """Retrieves active and delisted ticker lists."""
        return self.downloader.get_tickers(force_update=force_update)

    def download(self, update: bool = True, target_freqs: list[str] = None, 
                target_types: list[str] = None, only_active_tickers: bool = False):
        """Initiates the data download process."""
        logger.info("Initiating data download via DataManager...")
        self.downloader.download(update=update, target_freqs=target_freqs, 
                                 target_types=target_types, only_active_tickers=only_active_tickers)
        logger.info("Data download initiated.")