"""
preprocess_smartmeter.py
========================
Preprocess the large CD_INTERVAL_READING_ALL_NO_QUOTES.csv (16.5 GB)
smart meter dataset using chunked processing.

The file contains 30-minute interval readings per customer with columns:
  CUSTOMER_ID, READING_DATETIME, CALENDAR_KEY, EVENT_KEY,
  GENERAL_SUPPLY_KWH, CONTROLLED_LOAD_KWH, GROSS_GENERATION_KWH,
  NET_GENERATION_KWH, OTHER_KWH

This script processes in chunks to handle memory constraints.

Usage:
    python scripts/preprocess_smartmeter.py
    python scripts/preprocess_smartmeter.py --chunk-size 500000
    python scripts/preprocess_smartmeter.py --input path/to/file.csv --output path/to/output/

Output:
    data/preprocessed/CD_INTERVAL_preprocessed.csv
"""

import pandas as pd
import numpy as np
import os
import time
import argparse


# ======================== CONFIG ========================
DEFAULT_INPUT = r'V:\DATASETS\DATASET 2\CD_INTERVAL_READING_ALL_NO_QUOTES.csv'
DEFAULT_OUTPUT_DIR = r'V:\DATASETS\DATASET 2\Preprocessed'
DEFAULT_CHUNK_SIZE = 1_000_000
# ========================================================

# Columns to keep (drop CALENDAR_KEY and EVENT_KEY)
USE_COLS = [
    'CUSTOMER_ID', 'READING_DATETIME',
    'GENERAL_SUPPLY_KWH', 'CONTROLLED_LOAD_KWH',
    'GROSS_GENERATION_KWH', 'NET_GENERATION_KWH', 'OTHER_KWH'
]

# Memory-efficient data types
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
    """
    Preprocess a single chunk of smart meter data.

    Steps:
        1. Parse datetime
        2. Drop unparseable rows
        3. Remove duplicates per customer+timestamp
        4. Clip negative kWh values to 0
        5. Cap extreme outliers at 99.9th percentile
        6. Create total consumption column
        7. Feature engineering (time features + cyclical encoding)

    Parameters
    ----------
    chunk : pd.DataFrame
        A chunk of raw smart meter data.

    Returns
    -------
    pd.DataFrame
        Preprocessed chunk.
    """
    # 1. Parse datetime
    chunk['READING_DATETIME'] = pd.to_datetime(
        chunk['READING_DATETIME'], errors='coerce'
    )

    # 2. Drop rows where datetime couldn't be parsed
    chunk = chunk.dropna(subset=['READING_DATETIME'])

    # 3. Remove exact duplicates per customer + timestamp
    chunk = chunk.drop_duplicates(
        subset=['CUSTOMER_ID', 'READING_DATETIME'], keep='first'
    )

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

    # Cyclical encoding for periodic features
    chunk['hour_sin'] = np.sin(2 * np.pi * chunk['hour'] / 24).astype('float32')
    chunk['hour_cos'] = np.cos(2 * np.pi * chunk['hour'] / 24).astype('float32')
    chunk['month_sin'] = np.sin(2 * np.pi * chunk['month'] / 12).astype('float32')
    chunk['month_cos'] = np.cos(2 * np.pi * chunk['month'] / 12).astype('float32')

    return chunk


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess large smart meter CSV using chunked processing'
    )
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT,
                        help='Path to input CSV file')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save preprocessed output')
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE,
                        help='Number of rows per chunk (default: 1,000,000)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, 'CD_INTERVAL_preprocessed.csv')

    print("=" * 60)
    print("Smart Meter Data Preprocessing (Chunked)")
    print(f"Input:      {args.input}")
    print(f"Output:     {output_file}")
    print(f"Chunk size: {args.chunk_size:,} rows")
    print("=" * 60)

    start_time = time.time()
    total_rows_in = 0
    total_rows_out = 0
    first_chunk = True

    # Read and process in chunks
    reader = pd.read_csv(
        args.input,
        chunksize=args.chunk_size,
        usecols=USE_COLS,
        dtype=DTYPES,
        skipinitialspace=True,
    )

    for i, chunk in enumerate(reader):
        chunk_start = time.time()
        total_rows_in += len(chunk)

        # Preprocess the chunk
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
        print(
            f"  Chunk {i+1}: {len(chunk):>10,} in -> {len(processed):>10,} out  |  "
            f"Chunk: {elapsed:.1f}s  |  Total: {total_elapsed:.0f}s  |  "
            f"Processed: {total_rows_in:,}"
        )

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
