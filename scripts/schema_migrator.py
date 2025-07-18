import pandas as pd
from pathlib import Path
import os
import logging
from tqdm import tqdm
import argparse
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Local Imports ---
# Assuming file_io.py is in the same directory or accessible in the python path
try:
    from file_io import FileIO
except ImportError:
    print("Error: Could not import FileIO. Make sure 'file_io.py' is in the same directory.")
    exit(1)


# --- Configuration ---
# Configure logger for the migration script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def _normalize_transactions_column(df: pd.DataFrame, file_path_for_logging: Path) -> Tuple[pd.DataFrame, bool]:
    """
    Handles schema evolution for the transactions column and returns a flag indicating if changes were made.
    It coalesces data from legacy column names ('number_of_transactions', 'num_transactions')
    into the modern 'transactions' column.
    """
    df_copy = df.copy()
    was_changed = False
    
    # List of possible names, in order of preference
    possible_names = ['transactions', 'number_of_transactions', 'num_transactions']
    
    # Find the first valid column name from the list that exists in the DataFrame
    target_col = None
    for name in possible_names:
        if name in df_copy.columns:
            target_col = name
            break
    
    # If none of the columns exist, there's nothing to do
    if not target_col:
        return df, False
        
    # If the best available column isn't 'transactions', rename it
    if target_col != 'transactions':
        df_copy = df_copy.rename(columns={target_col: 'transactions'})
        logger.info(f"Schema migration: Renamed '{target_col}' to 'transactions' for {file_path_for_logging}.")
        was_changed = True
        
    # Now, iterate through the other legacy names and merge them if they exist
    other_legacy_names = [name for name in possible_names if name in df_copy.columns and name != 'transactions']
    
    for legacy_name in other_legacy_names:
        # Use direct assignment instead of inplace=True on a slice
        df_copy['transactions'] = df_copy['transactions'].fillna(df_copy[legacy_name])
        df_copy = df_copy.drop(columns=[legacy_name])
        logger.info(f"Schema migration: Merged '{legacy_name}' into 'transactions' for {file_path_for_logging}.")
        was_changed = True
        
    return df_copy, was_changed

def process_single_file(file_path: Path, file_io_instance: FileIO) -> bool:
    """
    Processes a single parquet file: checks for duplicates, normalizes schema, and uses FileIO to overwrite if necessary.
    Returns True if the file was modified, False otherwise.
    """
    try:
        # Read the original file
        df = pd.read_parquet(file_path)
        
        # --- 1. Drop Duplicates ---
        original_row_count = len(df)
        if df.index.hasnans:
            logger.warning(f"File has NaN in index, cannot check for duplicates: {file_path}")
            df_cleaned = df
        else:
            df_cleaned = df[~df.index.duplicated(keep='first')]
        
        rows_dropped = original_row_count - len(df_cleaned)
        duplicates_were_dropped = rows_dropped > 0

        # --- 2. Normalize Schema ---
        normalized_df, schema_was_changed = _normalize_transactions_column(df_cleaned, file_path)
        
        # --- 3. Decide whether to write the file ---
        if schema_was_changed or duplicates_were_dropped:
            if duplicates_were_dropped:
                logger.info(f"Dropped {rows_dropped} duplicate rows from {file_path}.")

            # --- 4. Parse info from path to call write_raw_data ---
            # Example path: .../data/S/SPY/trades/5minute/202507.parquet
            # We need to extract: ticker, trade_or_quote, and freq
            parts = file_path.parts
            ticker = parts[-4]
            trade_or_quote = parts[-3].removesuffix('s') # 'trades' -> 'trade'
            freq = parts[-2]

            # --- 5. Use FileIO to write the data ---
            # Using overwrite=True ensures the existing monthly file is completely replaced with the cleaned data.
            file_io_instance.write_raw_data(
                df=normalized_df,
                ticker=ticker,
                freq=freq,
                trade_or_quote=trade_or_quote,
                overwrite=True
            )
            logger.info(f"Successfully updated and saved file via FileIO: {file_path}")
            return True

    except Exception as e:
        logger.error(f"Failed to process file {file_path}: {e}", exc_info=True)
    
    return False

def migrate_data_schema(root_data_dir: Path, test_ticker: Optional[str] = None):
    """
    Scans a directory for parquet files, checks for legacy transaction column names and duplicate rows,
    and overwrites the file with a corrected version if necessary using multiple threads.
    If test_ticker is provided, it only scans files for that specific ticker.
    """
    if test_ticker:
        ticker_upper = test_ticker.upper()
        first_letter = ticker_upper[0]
        search_dir = root_data_dir / "data" / first_letter / ticker_upper
        logger.info(f"Starting TEST migration for ticker '{test_ticker}' in: {search_dir}")
    else:
        search_dir = root_data_dir / "data"
        logger.info(f"Starting FULL schema migration for all .parquet files in: {search_dir}")

    if not search_dir.exists():
        logger.error(f"Search directory does not exist: {search_dir}")
        return

    # Find all parquet files recursively within the chosen directory
    all_files = list(search_dir.rglob("*.parquet"))
    
    if not all_files:
        logger.warning(f"No .parquet files found in {search_dir}. Nothing to do.")
        return

    logger.info(f"Found {len(all_files)} total .parquet files to check.")
    
    # Instantiate the FileIO class to handle all writes
    file_io = FileIO(data_dir=root_data_dir)
    files_modified = 0
    
    max_workers = os.cpu_count() or 4
    logger.info(f"Processing files using up to {max_workers} worker threads.")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all file processing tasks, passing the FileIO instance to each
        future_to_file = {executor.submit(process_single_file, file_path, file_io): file_path for file_path in all_files}
        
        for future in tqdm(as_completed(future_to_file), total=len(all_files), desc="Processing files"):
            try:
                was_modified = future.result()
                if was_modified:
                    files_modified += 1
            except Exception as e:
                file_path = future_to_file[future]
                logger.error(f"A task for file {file_path} generated an exception: {e}")

    logger.info("--- Migration Complete ---")
    logger.info(f"Total files checked: {len(all_files)}")
    logger.info(f"Total files modified (schema updated or duplicates removed): {files_modified}")

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Scans a data directory and migrates Parquet files to a consistent 'transactions' column schema and removes duplicate rows."
    )
    parser.add_argument(
        "data_directory",
        type=str,
        help="The root data directory to scan for Parquet files (e.g., /mnt/nas/price_data/polygon)."
    )
    parser.add_argument(
        "--test-ticker",
        type=str,
        default=None,
        help="Run the migration on only a single ticker for testing purposes (e.g., SPY)."
    )
    
    args = parser.parse_args()
    
    root_dir = Path(args.data_directory)
    
    if not root_dir.is_dir():
        logger.critical(f"Error: The provided path is not a valid directory: {root_dir}")
    else:
        # It's good practice to ask for confirmation before modifying files
        if args.test_ticker:
            prompt = f"This script will modify files for ticker '{args.test_ticker}' in-place within '{root_dir}'.\nAre you sure you want to continue? (yes/no): "
        else:
            prompt = f"This script will modify ALL files in-place within '{root_dir}'.\nHave you backed up your data? (yes/no): "
        
        response = input(prompt)
        if response.lower() == 'yes':
            migrate_data_schema(root_dir, test_ticker=args.test_ticker)
        else:
            logger.info("Migration cancelled by user.")
