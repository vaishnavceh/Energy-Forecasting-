"""
preprocess_pjm_unified.py
==========================
Preprocess all 12 PJM regional hourly CSV files (Dataset 1) into the Unified Common Schema.

Produces a single unified compressed Parquet file (`PJM_unified_preprocessed.parquet`)
and scaler parameters (`pjm_scaler_params.json`) matching the exact schema of Dataset 2.

Usage:
    python scripts/preprocess_pjm_unified.py
"""

import os
import sys
import time
import json
import glob
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_DATA_DIR = r"V:\DATASETS\DATASET 1\Preprocessed"
OUTPUT_PARQUET = os.path.join(DEFAULT_DATA_DIR, "PJM_unified_preprocessed.parquet")
SCALER_JSON = os.path.join(DEFAULT_DATA_DIR, "pjm_scaler_params.json")

PJM_RAW_DIR = r"c:\Users\vaish\OneDrive\Desktop\Dataset"

def get_region_name(filepath):
    filename = os.path.basename(filepath)
    return filename.replace('_hourly.csv', '').replace('_preprocessed.csv', '')

def main():
    print("=" * 70)
    print("UNIFIED PJM REGIONAL DATASET PREPROCESSING (DATASET 1)")
    print(f"Output Parquet: {OUTPUT_PARQUET}")
    print("=" * 70)

    os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
    start_time = time.time()

    # Find raw or preprocessed files
    csv_files = glob.glob(os.path.join(PJM_RAW_DIR, "*_hourly.csv"))
    if not csv_files:
        csv_files = glob.glob(os.path.join(DEFAULT_DATA_DIR, "*_preprocessed.csv"))
        # Exclude output parquet if present
        csv_files = [f for f in csv_files if not f.endswith('PJM_unified_preprocessed.csv')]

    print(f"Found {len(csv_files)} PJM regional CSV files to unify.")

    # Pass 1: Global Min-Max Normalization Bounds Calculation
    print("\n[Step 1/3] Estimating Global Min-Max Normalization Bounds across PJM regions...")
    all_loads = []
    for f in csv_files:
        df_temp = pd.read_csv(f)
        load_col = [c for c in df_temp.columns if 'MW' in c or 'load' in c.lower()][0]
        all_loads.append(df_temp[load_col].dropna().clip(lower=0.0))

    combined_loads = pd.concat(all_loads)
    min_val = 0.0
    max_val = float(combined_loads.quantile(0.999))

    scaler_params = {"min": min_val, "max": max_val}
    with open(SCALER_JSON, 'w') as out_j:
        json.dump(scaler_params, out_j, indent=2)

    print(f"  PJM Global Min Load: {min_val:.2f} MW")
    print(f"  PJM Global Max Load (99.9th %ile Cap): {max_val:.2f} MW")
    print(f"  Saved Scaler Parameters: {SCALER_JSON}")
    del all_loads, combined_loads

    # Pass 2: Process Regions & Build Unified Schema Parquet
    print("\n[Step 2/3] Processing PJM Regions & Building Unified Parquet...")
    writer = None
    total_rows = 0

    for file_idx, fpath in enumerate(csv_files):
        region = get_region_name(fpath)
        print(f"  [{file_idx+1}/{len(csv_files)}] Processing Region: {region:12s}...", end="")
        region_start = time.time()

        df = pd.read_csv(fpath)
        load_col = [c for c in df.columns if 'MW' in c or 'load' in c.lower()][0]
        date_col = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()][0]

        # 1. Clean & Rename
        df['datetime'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=['datetime']).sort_values('datetime')
        df = df.drop_duplicates(subset=['datetime'], keep='first')

        df['load_val'] = df[load_col].clip(lower=0.0, upper=max_val).astype('float32')
        df['customer_id'] = file_idx + 1  # Integer ID for Region

        # 2. Min-Max Normalization
        df['load_scaled'] = ((df['load_val'] - min_val) / (max_val - min_val)).astype('float32')

        # 3. Feature Engineering (1-Hour Resolution)
        dt = df['datetime']
        hour = dt.dt.hour.astype('int8')
        day_of_week = dt.dt.dayofweek.astype('int8')
        month = dt.dt.month.astype('int8')
        is_weekend = day_of_week.isin([5, 6]).astype('int8')

        df['step_of_day'] = hour
        df['day_of_week'] = day_of_week
        df['month'] = month
        df['is_weekend'] = is_weekend

        # Cyclical Transformations (24-hour cycle)
        df['step_sin'] = np.sin(2 * np.pi * hour / 24).astype('float32')
        df['step_cos'] = np.cos(2 * np.pi * hour / 24).astype('float32')
        df['month_sin'] = np.sin(2 * np.pi * month / 12).astype('float32')
        df['month_cos'] = np.cos(2 * np.pi * month / 12).astype('float32')

        # Lags (1h, 2h, 24h=1d, 168h=7d)
        df['lag_1'] = df['load_scaled'].shift(1).fillna(0.0).astype('float32')
        df['lag_2'] = df['load_scaled'].shift(2).fillna(0.0).astype('float32')
        df['lag_day'] = df['load_scaled'].shift(24).fillna(0.0).astype('float32')
        df['lag_week'] = df['load_scaled'].shift(168).fillna(0.0).astype('float32')

        # Rolling Statistics (24 hours)
        df['rolling_mean'] = df['load_scaled'].rolling(24, min_periods=1).mean().fillna(0.0).astype('float32')
        df['rolling_std'] = df['load_scaled'].rolling(24, min_periods=1).std().fillna(0.0).astype('float32')

        # Target Creation (Next hour load)
        df['target_load'] = df['load_val'].shift(-1).ffill().astype('float32')
        df['target_scaled'] = df['load_scaled'].shift(-1).ffill().astype('float32')

        # Select Final Unified Columns (Exact match with Dataset 2)
        unified_cols = [
            'customer_id', 'datetime', 'load_val', 'load_scaled',
            'step_of_day', 'day_of_week', 'month', 'is_weekend',
            'step_sin', 'step_cos', 'month_sin', 'month_cos',
            'lag_1', 'lag_2', 'lag_day', 'lag_week',
            'rolling_mean', 'rolling_std', 'target_load', 'target_scaled'
        ]
        df_out = df[unified_cols].dropna()
        total_rows += len(df_out)

        # Write to Parquet
        table = pa.Table.from_pandas(df_out, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(OUTPUT_PARQUET, table.schema, compression='snappy')
        writer.write_table(table)

        elapsed = time.time() - region_start
        print(f" {len(df_out):>8,} rows written ({elapsed:.2f}s)")

    if writer:
        writer.close()

    total_time = time.time() - start_time
    output_size_mb = os.path.getsize(OUTPUT_PARQUET) / 1e6

    print("\n" + "=" * 70)
    print("DATASET 1 UNIFIED PREPROCESSING COMPLETE!")
    print(f"  Total Regions Unified: {len(csv_files)}")
    print(f"  Total Rows Written:   {total_rows:,}")
    print(f"  Parquet Size:         {output_size_mb:.2f} MB")
    print(f"  Time Elapsed:         {total_time:.2f} seconds")
    print(f"  Output Parquet:       {OUTPUT_PARQUET}")
    print("=" * 70)

if __name__ == '__main__':
    main()
