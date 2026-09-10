import argparse
import os
import gc
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for DeepDeFF model")
    parser.add_argument("--input", type=str, default=r"V:\DATASETS\DATASET 2\Preprocessed\CD_INTERVAL_preprocessed.parquet", help="Path to input preprocessed parquet file")
    parser.add_argument("--output-dir", type=str, default=r"V:\DATASETS\DATASET 2\Prepared\DeepDeFF", help="Directory to save prepared .npz files")
    parser.add_argument("--num-customers", type=int, default=69, help="Number of top customers to process")
    parser.add_argument("--k-steps", type=int, default=2, help="Number of past steps to consider (K)")
    parser.add_argument("--start-date", type=str, default="2013-06-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2013-08-31", help="End date (YYYY-MM-DD)")
    return parser.parse_args()

def extract_features(df_cust, k_steps):
    """
    Extract basic and derived features for a single customer.
    
    Basic features: GENERAL_SUPPLY_KWH, time_slot (one-hot), dayofweek (one-hot), is_weekend (binary)
    Derived features:
      - Rolling mean of GENERAL_SUPPLY_KWH over past K time-steps
      - Rolling std of GENERAL_SUPPLY_KWH over past K time-steps
      - Average of GENERAL_SUPPLY_KWH at same time_slot across past K days
      - Std of GENERAL_SUPPLY_KWH at same time_slot across past K days
    """
    # Ensure dataframe is sorted by datetime
    df_cust = df_cust.sort_values("READING_DATETIME").reset_index(drop=True)
    
    # Extract timeslot 0-47
    dt = df_cust["READING_DATETIME"].dt
    df_cust["time_slot"] = dt.hour * 2 + dt.minute // 30
    
    # 1. Basic Features
    # GENERAL_SUPPLY_KWH (1), time_slot (48), dayofweek (7), is_weekend (1)
    kwh = df_cust["GENERAL_SUPPLY_KWH"].values
    
    # One-hot encoding time_slot (0-47)
    time_slot_ohe = np.zeros((len(df_cust), 48), dtype=np.float32)
    time_slot_ohe[np.arange(len(df_cust)), df_cust["time_slot"].values] = 1.0
    
    # One-hot encoding dayofweek (0-6)
    dow_ohe = np.zeros((len(df_cust), 7), dtype=np.float32)
    dow_ohe[np.arange(len(df_cust)), df_cust["dayofweek"].values] = 1.0
    
    is_weekend = df_cust["is_weekend"].values.astype(np.float32).reshape(-1, 1)
    kwh_feat = kwh.reshape(-1, 1)
    
    basic_features = np.hstack([kwh_feat, time_slot_ohe, dow_ohe, is_weekend])
    
    # 2. Derived Features
    # a & b: Rolling mean and std over past k_steps
    # Pandas rolling includes the current step if not shifted. 
    # To use past K steps to predict current step, we shift by 1 and take rolling of size k_steps.
    # However, sequence generation itself uses past K steps as inputs.
    # We define the derived features at time T as computed from past data.
    # A standard way is to just compute rolling mean/std on the series up to current time, 
    # and sequence windowing will align them.
    # We will use closed='left' equivalent: shift(1).rolling(k_steps)
    shifted_kwh = pd.Series(kwh).shift(1)
    roll_mean = shifted_kwh.rolling(window=k_steps, min_periods=1).mean().fillna(0).values
    roll_std = shifted_kwh.rolling(window=k_steps, min_periods=1).std().fillna(0).values
    
    # c & d: Same time_slot across past K days
    # Past K days means shifting by 48 * K steps if the data is continuous 30-min intervals.
    # A safer way without assuming continuity is to groupby time_slot, shift by 1, and roll.
    grouped_kwh = df_cust.groupby("time_slot")["GENERAL_SUPPLY_KWH"]
    
    def roll_past_days(g, stat='mean'):
        # Shift 1 to look at past days for the same time slot
        s = g.shift(1).rolling(window=k_steps, min_periods=1)
        if stat == 'mean':
            return s.mean().fillna(0)
        return s.std().fillna(0)
    
    df_cust["past_days_mean"] = grouped_kwh.apply(lambda g: roll_past_days(g, 'mean')).reset_index(level=0, drop=True)
    df_cust["past_days_std"] = grouped_kwh.apply(lambda g: roll_past_days(g, 'std')).reset_index(level=0, drop=True)
    
    past_days_mean = df_cust["past_days_mean"].values
    past_days_std = df_cust["past_days_std"].values
    
    derived_features = np.column_stack([roll_mean, roll_std, past_days_mean, past_days_std]).astype(np.float32)
    
    return basic_features, derived_features, df_cust["READING_DATETIME"].values, kwh

def create_sequences(basic_feats, derived_feats, datetimes, targets, k_steps):
    """
    Build input sequences: past K records as input, next record as target
    """
    n_samples = len(basic_feats) - k_steps
    
    seq_basic = np.zeros((n_samples, k_steps, basic_feats.shape[1]), dtype=np.float32)
    seq_derived = np.zeros((n_samples, k_steps, derived_feats.shape[1]), dtype=np.float32)
    seq_targets = np.zeros((n_samples, 1), dtype=np.float32)
    seq_datetimes = np.empty((n_samples,), dtype=object)
    
    for i in range(n_samples):
        seq_basic[i] = basic_feats[i : i + k_steps]
        seq_derived[i] = derived_feats[i : i + k_steps]
        # target is at step i + k_steps
        seq_targets[i, 0] = targets[i + k_steps]
        seq_datetimes[i] = datetimes[i + k_steps]
        
    return seq_basic, seq_derived, seq_targets, seq_datetimes

def split_data(seq_basic, seq_derived, seq_targets, seq_datetimes):
    """
    Split per paper:
    - Train: 01-Jun-2013 to 05-Aug-2013
    - Val: 06-Aug-2013 to 22-Aug-2013
    - Test: 23-Aug-2013 to 31-Aug-2013
    """
    # Convert dates for comparison
    dt_pd = pd.to_datetime(seq_datetimes)
    
    train_mask = (dt_pd >= '2013-06-01') & (dt_pd <= '2013-08-05 23:59:59')
    val_mask = (dt_pd >= '2013-08-06') & (dt_pd <= '2013-08-22 23:59:59')
    test_mask = (dt_pd >= '2013-08-23') & (dt_pd <= '2013-08-31 23:59:59')
    
    def filter_mask(arr, mask):
        return arr[mask]
        
    return {
        'train_basic': filter_mask(seq_basic, train_mask),
        'train_derived': filter_mask(seq_derived, train_mask),
        'train_y': filter_mask(seq_targets, train_mask),
        
        'val_basic': filter_mask(seq_basic, val_mask),
        'val_derived': filter_mask(seq_derived, val_mask),
        'val_y': filter_mask(seq_targets, val_mask),
        
        'test_basic': filter_mask(seq_basic, test_mask),
        'test_derived': filter_mask(seq_derived, test_mask),
        'test_y': filter_mask(seq_targets, test_mask),
    }

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data from {args.input}...")
    columns_to_load = ['CUSTOMER_ID', 'READING_DATETIME', 'GENERAL_SUPPLY_KWH', 'dayofweek', 'is_weekend']
    
    df = pd.read_parquet(args.input, columns=columns_to_load)
    
    # Filter by date range first to save memory
    df = df[(df['READING_DATETIME'] >= args.start_date) & (df['READING_DATETIME'] <= f"{args.end_date} 23:59:59")]
    
    print("Finding top customers by data availability...")
    cust_counts = df['CUSTOMER_ID'].value_counts()
    top_customers = cust_counts.head(args.num_customers).index.tolist()
    print(f"Selected {len(top_customers)} customers.")
    
    for cust_id in tqdm(top_customers, desc="Processing Customers"):
        df_cust = df[df['CUSTOMER_ID'] == cust_id].copy()
        
        if len(df_cust) < args.k_steps + 10:
            print(f"Skipping customer {cust_id} due to insufficient data ({len(df_cust)} records)")
            continue
            
        basic_feats, derived_feats, datetimes, targets = extract_features(df_cust, args.k_steps)
        
        seq_basic, seq_derived, seq_y, seq_dt = create_sequences(
            basic_feats, derived_feats, datetimes, targets, args.k_steps
        )
        
        splits = split_data(seq_basic, seq_derived, seq_y, seq_dt)
        
        out_file = os.path.join(args.output_dir, f"customer_{cust_id}_deepdeff.npz")
        np.savez_compressed(out_file, **splits)
        
        # Free memory
        del df_cust
        del basic_feats, derived_feats, datetimes, targets
        del seq_basic, seq_derived, seq_y, seq_dt
        del splits
        gc.collect()

    print("Data preparation complete!")

if __name__ == "__main__":
    main()
