"""
preprocess_pjm.py
=================
Preprocess all 12 PJM regional hourly energy consumption CSV files.

Each file has 2 columns: Datetime, <REGION>_MW
This script applies a complete preprocessing pipeline to each file:
  1. Parse datetime and set as index
  2. Remove duplicate timestamps (DST transitions)
  3. Sort and reindex to continuous hourly frequency
  4. Interpolate missing values (time-based)
  5. Clip outliers using IQR method
  6. Feature engineering (time features, cyclical encoding, lag features)
  7. Save preprocessed output

Usage:
    python scripts/preprocess_pjm.py

Output:
    data/preprocessed/<REGION>_preprocessed.csv
"""

import pandas as pd
import numpy as np
import os
import time
import argparse


# ======================== CONFIG ========================
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'preprocessed')

PJM_FILES = [
    'AEP_hourly.csv',
    'COMED_hourly.csv',
    'DAYTON_hourly.csv',
    'DEOK_hourly.csv',
    'DOM_hourly.csv',
    'DUQ_hourly.csv',
    'EKPC_hourly.csv',
    'FE_hourly.csv',
    'NI_hourly.csv',
    'PJME_hourly.csv',
    'PJMW_hourly.csv',
    'PJM_Load_hourly.csv',
]
# ========================================================


def preprocess_single_file(filepath, output_dir):
    """
    Preprocess a single PJM regional CSV file.

    Parameters
    ----------
    filepath : str
        Path to the input CSV file.
    output_dir : str
        Directory to save the preprocessed output.

    Returns
    -------
    dict
        Summary statistics of the preprocessing.
    """
    filename = os.path.basename(filepath)
    print(f"\n{'='*50}")
    print(f"Processing: {filename}")

    # 1. Load data
    df = pd.read_csv(filepath, parse_dates=['Datetime'], index_col='Datetime')
    col = df.columns[0]  # e.g., 'AEP_MW'
    rows_initial = len(df)

    # 2. Remove duplicate timestamps
    duplicates = df.index.duplicated().sum()
    df = df[~df.index.duplicated(keep='first')]
    print(f"  Duplicates removed: {duplicates}")

    # 3. Sort and reindex to continuous hourly frequency
    df = df.sort_index()
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
    missing_hours = len(full_idx) - len(df)
    print(f"  Missing hours filled: {missing_hours}")
    df = df.reindex(full_idx)
    df.index.name = 'Datetime'

    # 4. Interpolate missing values
    df = df.interpolate(method='time')

    # 5. Handle outliers using IQR method
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    print(f"  Outliers clipped: {outliers}")

    # 6. Feature engineering
    # Time-based features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['quarter'] = df.index.quarter
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Lag features
    df['lag_1h'] = df[col].shift(1)
    df['lag_24h'] = df[col].shift(24)
    df['lag_168h'] = df[col].shift(168)

    # Rolling statistics
    df['rolling_24h_mean'] = df[col].rolling(24).mean()
    df['rolling_24h_std'] = df[col].rolling(24).std()

    # Drop NaN rows created by lag/rolling features
    df = df.dropna()
    rows_final = len(df)

    # 7. Save preprocessed file
    output_name = filename.replace('_hourly.csv', '_preprocessed.csv')
    output_path = os.path.join(output_dir, output_name)
    df.to_csv(output_path)
    print(f"  Saved: {output_name} ({rows_final:,} rows)")

    return {
        'file': filename,
        'rows_in': rows_initial,
        'rows_out': rows_final,
        'duplicates': duplicates,
        'missing_hours': missing_hours,
        'outliers': outliers,
    }


def train_test_split_timeseries(filepath, split_date='2017-01-01', output_dir=None):
    """
    Split a preprocessed file into train and test sets chronologically.

    Parameters
    ----------
    filepath : str
        Path to the preprocessed CSV file.
    split_date : str
        Date string to split on. Everything before is train, after is test.
    output_dir : str, optional
        Directory to save splits. Defaults to same directory as input.

    Returns
    -------
    tuple
        (train_df, test_df)
    """
    df = pd.read_csv(filepath, parse_dates=['Datetime'], index_col='Datetime')

    train = df.loc[:split_date]
    test = df.loc[split_date:]

    if output_dir is None:
        output_dir = os.path.dirname(filepath)

    basename = os.path.basename(filepath).replace('_preprocessed.csv', '')
    train.to_csv(os.path.join(output_dir, f'{basename}_train.csv'))
    test.to_csv(os.path.join(output_dir, f'{basename}_test.csv'))

    print(f"  Train: {train.shape[0]:,} rows ({train.index.min()} to {train.index.max()})")
    print(f"  Test:  {test.shape[0]:,} rows ({test.index.min()} to {test.index.max()})")

    return train, test


def main():
    parser = argparse.ArgumentParser(description='Preprocess PJM hourly energy consumption data')
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR,
                        help='Directory containing raw CSV files')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save preprocessed files')
    parser.add_argument('--split', action='store_true',
                        help='Also generate train/test splits')
    parser.add_argument('--split-date', type=str, default='2017-01-01',
                        help='Date to split train/test (default: 2017-01-01)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 50)
    print("PJM Energy Data Preprocessing Pipeline")
    print(f"Input:  {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print("=" * 50)

    start_time = time.time()
    results = []

    for filename in PJM_FILES:
        filepath = os.path.join(args.data_dir, filename)
        if os.path.exists(filepath):
            summary = preprocess_single_file(filepath, args.output_dir)
            results.append(summary)

            if args.split:
                output_name = filename.replace('_hourly.csv', '_preprocessed.csv')
                output_path = os.path.join(args.output_dir, output_name)
                print(f"  Splitting {output_name}...")
                train_test_split_timeseries(output_path, args.split_date, args.output_dir)
        else:
            print(f"\n  WARNING: {filepath} not found, skipping.")

    total_time = time.time() - start_time

    # Print summary
    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETE")
    print(f"  Files processed: {len(results)}")
    print(f"  Total time:      {total_time:.1f}s")
    print("=" * 50)

    # Summary table
    print(f"\n{'File':<25} {'Rows In':>10} {'Rows Out':>10} {'Dupes':>6} {'Missing':>8} {'Outliers':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['file']:<25} {r['rows_in']:>10,} {r['rows_out']:>10,} "
              f"{r['duplicates']:>6} {r['missing_hours']:>8} {r['outliers']:>8}")


if __name__ == '__main__':
    main()
