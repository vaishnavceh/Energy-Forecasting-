"""
Preprocess: CD_INTERVAL_READING_ALL_NO_QUOTES.csv (16.5 GB)
============================================================
This script processes the file in chunks since it's too large to fit in memory.

Columns:
  CUSTOMER_ID, READING_DATETIME, CALENDAR_KEY, EVENT_KEY,
  GENERAL_SUPPLY_KWH, CONTROLLED_LOAD_KWH, GROSS_GENERATION_KWH,
  NET_GENERATION_KWH, OTHER_KWH

Run: python preprocess_large_csv.py
"""

import pandas as pd
import numpy as np
import os
import time

# ======================== CONFIG ========================
INPUT_FILE = r'V:\DATASETS\DATASET 2\CD_INTERVAL_READING_ALL_NO_QUOTES.csv'
OUTPUT_DIR = r'V:\DATASETS\DATASET 2\Preprocessed'
CHUNK_SIZE = 2_000_000
# ========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Columns to use (drop CALENDAR_KEY and EVENT_KEY as they're not useful)
USE_COLS = [
    'CUSTOMER_ID', 'READING_DATETIME',
    'GENERAL_SUPPLY_KWH', 'CONTROLLED_LOAD_KWH',
    'GROSS_GENERATION_KWH', 'NET_GENERATION_KWH', 'OTHER_KWH'
]

# Data types to reduce memory usage
DTYPES = {
    'CUSTOMER_ID': 'int32',
    'GENERAL_SUPPLY_KWH': 'float32',
    'CONTROLLED_LOAD_KWH': 'float32',
    'GROSS_GENERATION_KWH': 'float32',
    'NET_GENERATION_KWH': 'float32',
    'OTHER_KWH': 'float32',
}

KWH_COLS = [
    'GENERAL_SUPPLY_KWH', 'CONTROLLED_LOAD_KWH',
    'GROSS_GENERATION_KWH', 'NET_GENERATION_KWH', 'OTHER_KWH'
]


def preprocess_chunk(chunk):
    """Preprocess a single chunk of data."""

    # 1. Parse datetime
    chunk['READING_DATETIME'] = pd.to_datetime(chunk['READING_DATETIME'], errors='coerce')

    # 2. Drop rows where datetime couldn't be parsed
    chunk = chunk.dropna(subset=['READING_DATETIME'])

    # 3. Remove exact duplicates
    chunk = chunk.drop_duplicates(subset=['CUSTOMER_ID', 'READING_DATETIME'], keep='first')

    # 4. Replace negative kWh values with 0 (meter errors)
    for col in KWH_COLS:
        chunk[col] = chunk[col].clip(lower=0)

    # 5. Cap extreme outliers per kWh column (99.9th percentile)
    for col in KWH_COLS:
        upper = chunk[col].quantile(0.999)
        if upper > 0:
            chunk[col] = chunk[col].clip(upper=upper)

    # 6. Create total consumption column
    chunk['TOTAL_KWH'] = chunk[KWH_COLS].sum(axis=1)

    # 7. Feature engineering from datetime
    dt = chunk['READING_DATETIME']
    chunk['hour'] = dt.dt.hour.astype('int8')
    chunk['dayofweek'] = dt.dt.dayofweek.astype('int8')
    chunk['month'] = dt.dt.month.astype('int8')
    chunk['year'] = dt.dt.year.astype('int16')
    chunk['is_weekend'] = chunk['dayofweek'].isin([5, 6]).astype('int8')

    # Cyclical encoding
    chunk['hour_sin'] = np.sin(2 * np.pi * chunk['hour'] / 24).astype('float32')
    chunk['hour_cos'] = np.cos(2 * np.pi * chunk['hour'] / 24).astype('float32')
    chunk['month_sin'] = np.sin(2 * np.pi * chunk['month'] / 12).astype('float32')
    chunk['month_cos'] = np.cos(2 * np.pi * chunk['month'] / 12).astype('float32')

    return chunk


def main():
    print("=" * 60)
    print("Preprocessing: CD_INTERVAL_READING_ALL_NO_QUOTES.csv")
    print(f"Chunk size: {CHUNK_SIZE:,} rows")
    print("=" * 60)

    output_file = os.path.join(OUTPUT_DIR, 'CD_INTERVAL_preprocessed.csv')
    start_time = time.time()
    total_rows_in = 0
    total_rows_out = 0
    first_chunk = True

    # Read and process in chunks
    reader = pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        usecols=USE_COLS,
        dtype=DTYPES,
        skipinitialspace=True,  # handle spaces after commas in header
    )

    for i, chunk in enumerate(reader):
        chunk_start = time.time()
        total_rows_in += len(chunk)

        # Preprocess
        processed = preprocess_chunk(chunk)
        total_rows_out += len(processed)

        # Write to CSV (header only on first chunk)
        processed.to_csv(
            output_file,
            mode='w' if first_chunk else 'a',
            header=first_chunk,
            index=False
        )
        first_chunk = False

        elapsed = time.time() - chunk_start
        total_elapsed = time.time() - start_time
        print(f"  Chunk {i+1}: {len(chunk):>10,} rows in -> {len(processed):>10,} rows out  |  "
              f"Chunk: {elapsed:.1f}s  |  Total: {total_elapsed:.0f}s  |  "
              f"Rows processed: {total_rows_in:,}")

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("DONE!")
    print(f"  Total rows in:  {total_rows_in:,}")
    print(f"  Total rows out: {total_rows_out:,}")
    print(f"  Rows removed:   {total_rows_in - total_rows_out:,}")
    print(f"  Time taken:     {total_time/60:.1f} minutes")
    print(f"  Output saved:   {output_file}")
    print("=" * 60)


if __name__ == '__main__':
    main()
