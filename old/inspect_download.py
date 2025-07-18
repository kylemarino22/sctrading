import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pandas as pd

from pathlib import Path
from datetime import datetime, timezone

def load_and_filter_ticker_native(
    ticker: str,
    start_ts: datetime,
    end_ts: datetime,
    data_root: Path = Path("/mnt/nas/price_data/polygon/5min_parquet")
):
    """
    1) Treats data_root / ticker as a PyArrow Dataset of multiple monthly Parquet files.
    2) Pushes down a timestamp filter so only row groups overlapping [start_ts, end_ts) are read.
    3) Converts the resulting PyArrow Table to pandas, prints all rows + count.
    4) Computes business‐day count between start_ts.date() and end_ts.date().
    """
    # -----------------------------------------------------------------------------
    # A) Build the dataset
    # -----------------------------------------------------------------------------
    ticker_dir = data_root / ticker
    if not ticker_dir.exists():
        raise FileNotFoundError(f"No folder found for ticker '{ticker}' at {ticker_dir}")

    # This will recursively find all .parquet files under ticker_dir
    dataset = ds.dataset(str(ticker_dir), format="parquet")

    # -----------------------------------------------------------------------------
    # B) Make sure our datetimes are UTC‐aware
    # -----------------------------------------------------------------------------
    if start_ts.tzinfo is None:
        start_ts = start_ts.replace(tzinfo=timezone.utc)
    if end_ts.tzinfo is None:
        end_ts = end_ts.replace(tzinfo=timezone.utc)

    # Convert to Arrow scalar of type timestamp("ns", tz="UTC")
    start_scalar = pa.scalar(int(start_ts.timestamp() * 1e9), type=pa.timestamp("ns", tz="UTC"))
    end_scalar   = pa.scalar(int(end_ts.timestamp()   * 1e9), type=pa.timestamp("ns", tz="UTC"))

    # -----------------------------------------------------------------------------
    # C) Build a boolean filter expression using bitwise &
    # -----------------------------------------------------------------------------
    time_filter = (
        (pc.field("timestamp") >= start_scalar) &
        (pc.field("timestamp") <  end_scalar)
    )

    # -----------------------------------------------------------------------------
    # D) Push that filter into the dataset scan
    # -----------------------------------------------------------------------------
    filtered_table = dataset.to_table(filter=time_filter)

    # -----------------------------------------------------------------------------
    # E) Convert to pandas and print all rows + count
    # -----------------------------------------------------------------------------
    df = filtered_table.to_pandas()

    # Temporarily expand pandas display so we see every row/column
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)
    pd.set_option("display.expand_frame_repr", False)

    print(f"\n=== Rows for {ticker} from {start_ts} to {end_ts} ===")
    if df.empty:
        print("No rows in this range.")
    else:
        print(df)

    print(f"\nTotal rows in range: {len(df)}")

    # -----------------------------------------------------------------------------
    # F) Compute number of business days (Mon–Fri) between start_ts.date() and end_ts.date()
    #    We treat [start_date, end_date) → so we exclude end_date itself:
    # -----------------------------------------------------------------------------
    # Generate a Business‐day range from start_date to (end_date – 1 day)
    first_day = start_ts.date()
    last_day_inclusive = (end_ts.date() - pd.Timedelta(days=1))
    if last_day_inclusive < first_day:
        num_bdays = 0
    else:
        bdr = pd.bdate_range(start=first_day, end=last_day_inclusive, freq="C")
        num_bdays = len(bdr)

    print(f"Number of business days between {first_day} and {end_ts.date()}: {num_bdays}")

    # Reset pandas display options
    pd.reset_option("display.max_rows")
    pd.reset_option("display.max_columns")
    pd.reset_option("display.width")
    pd.reset_option("display.expand_frame_repr")


# ------------------------------------------------------------
# Example usage:
# ------------------------------------------------------------
if __name__ == "__main__":
    ticker     = "AAPB"
    # e.g. “2025-05-01 09:30 UTC” up to “2025-05-05 16:00 UTC”
    start_ts   = datetime.fromisoformat("2025-04-25T09:30:00+00:00")
    end_ts     = datetime.fromisoformat("2025-05-05T16:00:00+00:00")

    load_and_filter_ticker_native(ticker, start_ts, end_ts)
