import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional
import json
import pandas as pd
import os
import shutil
import sys
import pyarrow.parquet as pq

from tqdm import tqdm

# Add the project's root directory to the Python path to find other modules
# Assumes the project structure is: project_root/scripts/ and project_root/raw_data/
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / 'raw_data'))

from data_manager import DataManager

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- SCRIPT SETTINGS ---
# Set the base directory for your cached data
# BASE_DATA_DIR = Path("/home/kyle/data/polygon_cache")
BASE_DATA_DIR = Path("/mnt/nas/price_data/polygon")

# Set the number of parallel workers to use for the migration
NUM_WORKERS = max(1, (os.cpu_count()//2 or 4) - 2)

# Set to True to run the script on a single ticker for testing before running on the whole universe.
DEBUG_MODE = False
DEBUG_TICKER = "AMZE"


def migrate_ticker_data(ticker: str, dm: DataManager) -> Optional[str]:
    """
    Worker function to migrate a single ticker. It inspects the file
    schema and uses the robust FileIO write method if a fix is needed.
    Returns the ticker name if an error occurs, otherwise None.
    """
    try:
        logger.debug(f"Starting migration for {ticker}...")
        
        ticker_upper = ticker.upper()
        first_letter = ticker_upper[0]
        data_dir = dm.io.data_dir / "data" / first_letter / ticker_upper / "trades" / "5minute"

        if not data_dir.is_dir():
            return None

        monthly_files = list(data_dir.glob("*.parquet"))
        if not monthly_files:
            return None

        for file_path in monthly_files:
            try:
                schema = pq.read_schema(file_path)
                needs_fix = False
                
                if 'volume' in schema.names and schema.field('volume').type != 'int64':
                    needs_fix = True
                
                if 'transactions' in schema.names and schema.field('transactions').type != 'int64':
                    needs_fix = True

                if not needs_fix:
                    continue

                logger.info(f"Schema mismatch found for {file_path}. Migrating...")
                df = pd.read_parquet(file_path)

                if 'volume' in df.columns:
                    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').round(0).astype('Int64')
                if 'transactions' in df.columns:
                    df['transactions'] = pd.to_numeric(df['transactions'], errors='coerce').round(0).astype('Int64')

                dm.io._write_parquet_safely(df, file_path)
                logger.debug(f"Successfully migrated file: {file_path}")

            except Exception as e:
                logger.error(f"Failed to process file {file_path} for ticker {ticker}: {e}", exc_info=True)
                # If one file fails, we consider the whole ticker failed.
                return ticker
        
        logger.debug(f"Successfully migrated {ticker}.")
        return None # Return None on success
    
    except Exception as e:
        logger.error(f"A critical error occurred during migration for ticker {ticker}: {e}", exc_info=True)
        return ticker # Return the ticker name on failure


def run_migration(tickers_to_process: List[str], dm: DataManager) -> List[str]:
    """
    Orchestrates the migration process using a process pool.
    Returns a list of tickers that failed during migration.
    """
    logger.info(f"Starting data migration for {len(tickers_to_process)} tickers using {NUM_WORKERS} worker(s)...")
    failed_tickers = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(migrate_ticker_data, ticker, dm): ticker for ticker in tickers_to_process}

        for future in tqdm(as_completed(futures), total=len(tickers_to_process), desc="Migrating ticker data"):
            try:
                result = future.result()
                if result is not None:
                    # If the worker returned a ticker name, it means it failed.
                    failed_tickers.append(result)
            except Exception as e:
                ticker = futures[future]
                logger.error(f"A critical error occurred in the executor for ticker {ticker}: {e}", exc_info=True)
                failed_tickers.append(ticker)
    
    logger.info("--- Data Migration Complete ---")
    return failed_tickers


if __name__ == "__main__":
    # Suppress chatty loggers to keep the progress bar clean
    logging.getLogger("file_io").setLevel(logging.WARNING)
    logging.getLogger("data_manager").setLevel(logging.WARNING)
    
    dm = DataManager(BASE_DATA_DIR)
    
    tickers = []
    if DEBUG_MODE:
        logger.warning(f"--- RUNNING IN DEBUG MODE FOR TICKER: {DEBUG_TICKER} ---")
        tickers = [DEBUG_TICKER]
    else:
        universe_path = dm.get_universe_path()
        if not universe_path.exists():
            logger.critical(f"Universe file not found at {universe_path}. Cannot proceed.")
        else:
            logger.info(f"Loading tickers from universe file: {universe_path}")
            with open(universe_path, 'r') as f:
                universe_data = json.load(f).get("universe", {})
                tickers = sorted(list(universe_data.keys()))
            logger.info(f"Found {len(tickers)} tickers to migrate.")

    if tickers:
        failed_list = run_migration(tickers, dm)
        
        if failed_list:
            logger.warning("\n" + "="*50)
            logger.warning("MIGRATION FINISHED WITH ERRORS.")
            logger.warning(f"The following {len(failed_list)} tickers failed to migrate:")
            for ticker in sorted(failed_list):
                print(f" - {ticker}")
            logger.warning("="*50)
        else:
            logger.info("Migration finished successfully with no errors.")
    else:
        logger.error("No tickers found to process. Exiting.")
