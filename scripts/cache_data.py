import json
from pathlib import Path
import logging
import shutil
import os
import tempfile
import subprocess
from typing import List, Optional, Set
from tqdm import tqdm

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_all_required_files(source_dir: Path, tickers_to_scan: List[str]) -> Set[str]:
    """
    Determines the set of all monthly parquet files needed for a given list of tickers
    by scanning the source directory for existing files. This is much more efficient
    than generating file paths from date ranges.
    """
    required_files = set()
    logger.info(f"Scanning source directory for all existing files for {len(tickers_to_scan)} tickers...")

    for symbol in tqdm(tickers_to_scan, desc="Scanning for ticker files"):
        ticker_upper = symbol.upper()
        # Handle potential empty strings or invalid ticker names
        if not ticker_upper:
            continue
        first_letter = ticker_upper[0]
        
        # Path to the directory containing the monthly parquet files for the ticker
        ticker_data_dir = source_dir / "data" / first_letter / ticker_upper / "trades" / "5minute"

        if not ticker_data_dir.is_dir():
            continue
        
        # Use glob to find all .parquet files in the directory
        for file_path in ticker_data_dir.glob("*.parquet"):
            # We need the path relative to the source_dir for rsync's -R flag
            relative_path = file_path.relative_to(source_dir)
            required_files.add(str(relative_path))

    logger.info(f"Found {len(required_files)} existing Parquet files on the source to sync.")
    return required_files

def run_rsync_chunk(chunk: List[str], source_dir: Path, cache_dir: Path):
    """Runs rsync for a single chunk of files."""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
        for f in chunk:
            tmp.write(str(f) + '\n')
        files_from_path = tmp.name

    try:
        # Using flags that provide good output for smaller chunks
        rsync_command = [
            "/usr/bin/rsync",
            "-avR", # Archive, verbose, relative paths
            f"--files-from={files_from_path}",
            str(source_dir) + "/", # Source directory
            str(cache_dir) # Destination directory
        ]
        
        result = subprocess.run(rsync_command, capture_output=True, text=True)
        if result.returncode != 0:
            # Log errors but don't stop the whole process for "file not found"
            logger.warning(f"Rsync chunk finished with errors (some files might have been deleted between scan and sync):\n{result.stderr}")

    finally:
        os.remove(files_from_path)


def cache_full_universe_with_rsync(universe_path: Path, source_dir: Path, cache_dir: Path, full_cache_tickers: Optional[List[str]] = None):
    """
    Main function to orchestrate the caching process for the entire universe using rsync,
    processing files in manageable chunks.
    """
    logger.info("Starting data caching process with rsync...")
    logger.info(f"Source (NAS): {source_dir}")
    logger.info(f"Destination (Cache): {cache_dir}")

    # Ensure the cache directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy the universe file to the cache for reference
    if universe_path.exists():
        universe_dest_path = cache_dir / "universes"
        universe_dest_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(universe_path, universe_dest_path / universe_path.name)
        logger.info(f"Copied universe file to cache: {universe_dest_path / universe_path.name}")

    if not full_cache_tickers:
        logger.error("No tickers provided for full cache. Nothing to do.")
        return

    files_to_copy = list(get_all_required_files(source_dir, tickers_to_scan=full_cache_tickers))
    total_files = len(files_to_copy)

    if not files_to_copy:
        logger.warning("No data files identified to copy for the given universe.")
        return

    # --- Chunking Logic ---
    chunk_size = 1000  # Process 1000 files at a time
    file_chunks = [files_to_copy[i:i + chunk_size] for i in range(0, total_files, chunk_size)]
    
    logger.info(f"Splitting {total_files} files into {len(file_chunks)} chunks of up to {chunk_size} files each.")

    # Process each chunk with a progress bar
    for chunk in tqdm(file_chunks, desc="Syncing file chunks"):
        run_rsync_chunk(chunk, source_dir, cache_dir)

    logger.info("--- Caching Complete ---")


if __name__ == "__main__":
    # --- Configuration ---
    DEBUG = False
    DEBUG_TICKER = "SPY"
    # If FULL_CACHE_TICKERS is not empty and DEBUG is False, it will do a full cache for these tickers.
    FULL_CACHE_TICKERS = []

    NAS_DATA_DIR = Path("/mnt/nas/price_data/polygon")
    LOCAL_CACHE_DIR = Path("/home/kyle/data/polygon_cache")
    
    UNIVERSE_DIR = NAS_DATA_DIR / "universes"
    UNIVERSE_FILENAME = "pretraining_universe.json"
    
    universe_path = UNIVERSE_DIR / UNIVERSE_FILENAME
    
    tickers_for_full_cache = None
    
    if DEBUG:
        logger.warning("--- SCRIPT MODE: DEBUG ---")
        tickers_for_full_cache = [DEBUG_TICKER]
    elif FULL_CACHE_TICKERS:
        logger.warning("--- SCRIPT MODE: FULL CACHE FOR SPECIFIC TICKERS ---")
        tickers_for_full_cache = FULL_CACHE_TICKERS
    else:
        logger.warning("--- SCRIPT MODE: FULL CACHE FOR ALL TICKERS IN UNIVERSE FILE ---")
        if universe_path.exists():
            with open(universe_path, 'r') as f:
                universe_data = json.load(f).get("universe", {})
                tickers_for_full_cache = list(universe_data.keys())
            logger.info(f"Loaded {len(tickers_for_full_cache)} tickers from {UNIVERSE_FILENAME} for a full history cache.")
        else:
            logger.critical(f"Universe file not found at {universe_path}. Cannot proceed.")
            tickers_for_full_cache = [] # Set to empty list to prevent execution

    # Proceed only if we have tickers to process
    if tickers_for_full_cache:
        cache_full_universe_with_rsync(
            universe_path=universe_path,
            source_dir=NAS_DATA_DIR,
            cache_dir=LOCAL_CACHE_DIR,
            full_cache_tickers=tickers_for_full_cache
        )
    else:
        logger.error("No tickers were identified for caching. Exiting.")
