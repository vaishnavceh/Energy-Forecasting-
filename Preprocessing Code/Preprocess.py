import pandas as pd
import numpy as np
import os

output_dir = r'C:\Users\vaish\OneDrive\Desktop\Dataset\Preprocessed'
os.makedirs(output_dir, exist_ok=True)

# All remaining files to process
files = [
    'AEP_hourly.csv',
    'COMED_hourly.csv',
    'DAYTON_hourly.csv',
    'DEOK_hourly.csv',
    'DOM_hourly.csv',
    'DUQ_hourly.csv',
    'EKPC_hourly.csv',
    'FE_hourly.csv',
    'NI_hourly.csv',
    # 'PJME_hourly.csv',  ← already done ✅
    'PJMW_hourly.csv',
    'PJM_Load_hourly.csv',
]

data_dir = r'C:\Users\vaish\OneDrive\Desktop\Dataset'

for file in files:
    print(f"\n{'='*50}")
    print(f"Processing: {file}")
    
    # 1. Load
    df = pd.read_csv(os.path.join(data_dir, file), 
                     parse_dates=['Datetime'], index_col='Datetime')
    col = df.columns[0]  # e.g., 'AEP_MW'
    
    # 2. Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    # 3. Sort & reindex
    df = df.sort_index()
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
    print(f"  Missing hours: {len(full_idx) - len(df)}")
    df = df.reindex(full_idx)
    df.index.name = 'Datetime'
    
    # 4. Interpolate missing values
    df = df.interpolate(method='time')
    
    # 5. Handle outliers (clip using IQR)
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"  Outliers clipped: {outliers}")
    df[col] = df[col].clip(lower=lower, upper=upper)
    
    # 6. Feature engineering
    df['hour']       = df.index.hour
    df['dayofweek']  = df.index.dayofweek
    df['month']      = df.index.month
    df['year']       = df.index.year
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['hour_sin']   = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']   = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin']  = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']  = np.cos(2 * np.pi * df['month'] / 12)
    
    # 7. Lag features
    df['lag_1h']   = df[col].shift(1)
    df['lag_24h']  = df[col].shift(24)
    df['lag_168h'] = df[col].shift(168)
    df['rolling_24h_mean'] = df[col].rolling(24).mean()
    df = df.dropna()
    
    # 8. Save
    name = file.replace('_hourly.csv', '_preprocessed.csv')
    df.to_csv(os.path.join(output_dir, name))
    print(f"  ✅ Saved: {name} ({df.shape[0]} rows)")

print(f"\n{'='*50}")
print("🎉 All files processed!")