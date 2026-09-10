"""
preprocess_smartmeter_unified.py
=================================
Preprocess large smart meter dataset (16.5 GB CSV) into the Unified Common Schema.

Steps:
  1. Chunked Reading & Memory Safety (1M rows / chunk)
  2. Datetime Parsing & Null Dropping
  3. Deduplication per (customer_id, datetime)
  4. Outlier Handling (Clip lower at 0.0, upper cap at 99.9th percentile)
  5. Min-Max Normalization (Scaling load to [0, 1] range)
  6. Unified Feature Engineering (Time slots, Lags 30m/1h/24h/7d, Rolling Mean/Std, Sin/Cos)
  7. Target Creation (Future load shift=-1)
  8. Streaming PyArrow Parquet Compression Output (Snappy)

Usage:
    python scripts/preprocess_smartmeter_unified.py
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Config
DEFAULT_INPUT = r"V:\Dataset original\CD_INTERVAL_READING_ALL_NO_QUOTES.csv"
DEFAULT_OUTPUT_DIR = r"V:\DATASETS\DATASET 2\Preprocessed"
OUTPUT_PARQUET = os.path.join(DEFAULT_OUTPUT_DIR, "CD_INTERVAL_unified_preprocessed.parquet")
SCALER_JSON = os.path.join(DEFAULT_OUTPUT_DIR, "scaler_params.json")
CHUNK_SIZE = 1_000_000

USE_COLS = ['CUSTOMER_ID', 'READING_DATETIME', 'GENERAL_SUPPLY_KWH']
DTYPES = {
    'CUSTOMER_ID': 'int32',
    'GENERAL_SUPPLY_KWH': 'float32'
}

def main():
    print("=" * 70)
    print("UNIFIED SMART METER PREPROCESSING PIPELINE (WITH NORMALIZATION)")
    print(f"Input File:       {DEFAULT_INPUT}")
    print(f"Output Parquet:   {OUTPUT_PARQUET}")
    print(f"Chunk Size:       {CHUNK_SIZE:,} rows")
    print("=" * 70)

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    start_time = time.time()

    # Pass 1: Estimate Global Bounds for Min-Max Normalization (Fast Sampling)
    print("\n[Step 1/4] Estimating Global Min-Max Normalization Bounds...")
    sample_df = pd.read_csv(
        DEFAULT_INPUT, nrows=2_000_000, usecols=USE_COLS, dtype=DTYPES, skipinitialspace=True
    )
    sample_loads = sample_df['GENERAL_SUPPLY_KWH'].clip(lower=0.0)
    min_val = 0.0
    max_val = float(sample_loads.quantile(0.999))
    if max_val <= 0:
        max_val = 10.0  # Fallback

    scaler_params = {"min": min_val, "max": max_val}
    with open(SCALER_JSON, 'w') as f:
        json.dump(scaler_params, f, indent=2)

    print(f"  Min Load: {min_val:.4f} kWh")
    print(f"  Max Load (99.9th %ile Cap): {max_val:.4f} kWh")
    print(f"  Saved Scaler Parameters: {SCALER_JSON}")
    del sample_df, sample_loads

    # Pass 2: Main Processing & Feature Generation Loop
    print("\n[Step 2/4] Processing Chunks & Generating Unified Features...")

    reader = pd.read_csv(
        DEFAULT_INPUT,
        chunksize=CHUNK_SIZE,
        usecols=USE_COLS,
        dtype=DTYPES,
        skipinitialspace=True
    )

    writer = None
    total_rows_in = 0
    total_rows_out = 0

    for i, chunk in enumerate(reader):
        chunk_start = time.time()
        total_rows_in += len(chunk)

        # Rename columns to Common Schema
        chunk = chunk.rename(columns={
            'CUSTOMER_ID': 'customer_id',
            'READING_DATETIME': 'datetime',
            'GENERAL_SUPPLY_KWH': 'load_val'
        })

        # 1. Datetime Parse & Null Drop
        chunk['datetime'] = pd.to_datetime(chunk['datetime'], errors='coerce')
        chunk = chunk.dropna(subset=['datetime'])

        # 2. Deduplicate per (customer_id, datetime)
        chunk = chunk.drop_duplicates(subset=['customer_id', 'datetime'], keep='first')

        # 3. Outlier Handling (Clip negative to 0, cap upper at max_val)
        chunk['load_val'] = chunk['load_val'].clip(lower=0.0, upper=max_val)

        # 4. Min-Max Normalization
        chunk['load_scaled'] = ((chunk['load_val'] - min_val) / (max_val - min_val)).astype('float32')

        # 5. Feature Engineering
        dt = chunk['datetime']
        hour = dt.dt.hour.astype('int8')
        minute = dt.dt.minute.astype('int8')
        step_of_day = (hour * 2 + (minute // 30)).astype('int8')
        day_of_week = dt.dt.dayofweek.astype('int8')
        month = dt.dt.month.astype('int8')
        is_weekend = day_of_week.isin([5, 6]).astype('int8')

        chunk['step_of_day'] = step_of_day
        chunk['day_of_week'] = day_of_week
        chunk['month'] = month
        chunk['is_weekend'] = is_weekend

        # Cyclical Transformations
        chunk['step_sin'] = np.sin(2 * np.pi * step_of_day / 48).astype('float32')
        chunk['step_cos'] = np.cos(2 * np.pi * step_of_day / 48).astype('float32')
        chunk['month_sin'] = np.sin(2 * np.pi * month / 12).astype('float32')
        chunk['month_cos'] = np.cos(2 * np.pi * month / 12).astype('float32')

        # Multi-Resolution Lags
        chunk['lag_1'] = chunk['load_scaled'].shift(1).fillna(0.0).astype('float32')
        chunk['lag_2'] = chunk['load_scaled'].shift(2).fillna(0.0).astype('float32')
        chunk['lag_day'] = chunk['load_scaled'].shift(48).fillna(0.0).astype('float32')
        chunk['lag_week'] = chunk['load_scaled'].shift(336).fillna(0.0).astype('float32')

        # Rolling Statistics (12 steps = 6 hours)
        chunk['rolling_mean'] = chunk['load_scaled'].rolling(12, min_periods=1).mean().fillna(0.0).astype('float32')
        chunk['rolling_std'] = chunk['load_scaled'].rolling(12, min_periods=1).std().fillna(0.0).astype('float32')

        # Target Creation (Next 30-min load)
        chunk['target_load'] = chunk['load_val'].shift(-1).ffill().astype('float32')
        chunk['target_scaled'] = chunk['load_scaled'].shift(-1).ffill().astype('float32')

        # Drop intermediate nulls
        chunk = chunk.dropna()
        total_rows_out += len(chunk)

        # Convert to PyArrow Table & Stream to Parquet Writer
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(OUTPUT_PARQUET, table.schema, compression='snappy')
        writer.write_table(table)

        elapsed = time.time() - chunk_start
        total_elapsed = time.time() - start_time
        print(f"  Chunk {i+1:3d}: {len(chunk):10,} rows written | Chunk: {elapsed:4.1f}s | Total: {total_elapsed/60:4.1f} min | Processed: {total_rows_in:,}")

    if writer:
        writer.close()

    total_time = time.time() - start_time
    input_size_gb = os.path.getsize(DEFAULT_INPUT) / 1e9
    output_size_gb = os.path.getsize(OUTPUT_PARQUET) / 1e9
    compression_ratio = input_size_gb / output_size_gb if output_size_gb > 0 else 0

    print("\n" + "=" * 70)
    print("UNIFIED PREPROCESSING & NORMALIZATION COMPLETE!")
    print(f"  Total Rows Input:      {total_rows_in:,}")
    print(f"  Total Rows Written:    {total_rows_out:,}")
    print(f"  Raw CSV Size:          {input_size_gb:.2f} GB")
    print(f"  Parquet Output Size:   {output_size_gb:.2f} GB")
    print(f"  Compression Ratio:     {compression_ratio:.1f}x smaller")
    print(f"  Total Time Elapsed:    {total_time / 60:.1f} minutes")
    print(f"  Output Parquet:        {OUTPUT_PARQUET}")
    print("=" * 70)

if __name__ == '__main__':
    main()
