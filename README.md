# ⚡ Energy Consumption Forecasting — Data Preprocessing Pipeline

A comprehensive data preprocessing and feature engineering pipeline for **PJM Interconnection hourly energy consumption** data and **smart meter interval reading** data, designed for time series forecasting and machine learning applications.

## 📋 Project Overview

This project preprocesses large-scale energy consumption datasets to prepare them for load forecasting models. It handles:

- **12 PJM regional hourly datasets** (~120K–145K rows each)
- **1 combined PJM dataset** with all regions (~178K rows)
- **1 large smart meter dataset** (16.5 GB, ~322 million rows of 30-min interval readings)

The pipeline produces clean, feature-rich datasets ready for ML model training (ARIMA, XGBoost, LSTM, etc.)

## 📁 Project Structure

```
├── scripts/
│   ├── preprocess_pjm.py          # Preprocess 12 PJM regional CSV files
│   ├── preprocess_smartmeter.py   # Chunked preprocessing for 16.5 GB smart meter data
│   └── compress_to_parquet.py     # Convert large CSV to Parquet (7-10x compression)
│
├── data/
│   ├── raw/                       # Original CSV files (not tracked in git)
│   │   ├── AEP_hourly.csv
│   │   ├── COMED_hourly.csv
│   │   ├── DAYTON_hourly.csv
│   │   ├── DEOK_hourly.csv
│   │   ├── DOM_hourly.csv
│   │   ├── DUQ_hourly.csv
│   │   ├── EKPC_hourly.csv
│   │   ├── FE_hourly.csv
│   │   ├── NI_hourly.csv
│   │   ├── PJME_hourly.csv
│   │   ├── PJMW_hourly.csv
│   │   ├── PJM_Load_hourly.csv
│   │   ├── pjm_hourly_est.csv
│   │   └── est_hourly.parquet
│   │
│   └── preprocessed/              # Preprocessed output files (not tracked in git)
│       ├── AEP_preprocessed.csv
│       ├── PJME_preprocessed.csv
│       └── ...
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install -r requirements.txt
```

## 📊 Datasets

### Dataset 1: PJM Hourly Energy Consumption

| File | Region | Rows | Date Range |
|------|--------|------|------------|
| `AEP_hourly.csv` | American Electric Power | 121,275 | 2004–2018 |
| `COMED_hourly.csv` | Commonwealth Edison | 66,499 | 2011–2018 |
| `DAYTON_hourly.csv` | Dayton Power & Light | 121,277 | 2004–2018 |
| `DEOK_hourly.csv` | Duke Energy Ohio/Kentucky | 57,741 | 2012–2018 |
| `DOM_hourly.csv` | Dominion Virginia | 116,191 | 2005–2018 |
| `DUQ_hourly.csv` | Duquesne Light | 119,070 | 2005–2018 |
| `EKPC_hourly.csv` | East Kentucky Power | 45,336 | 2013–2018 |
| `FE_hourly.csv` | FirstEnergy | 62,876 | 2011–2018 |
| `NI_hourly.csv` | Northern Illinois | 58,452 | 2004–2018 |
| `PJME_hourly.csv` | PJM East | 145,368 | 2002–2018 |
| `PJMW_hourly.csv` | PJM West | 143,208 | 2002–2018 |
| `PJM_Load_hourly.csv` | PJM Total Load | 32,898 | 1998–2002 |

**Columns:** `Datetime`, `<REGION>_MW`

### Dataset 2: Smart Meter Interval Readings

| Property | Value |
|----------|-------|
| File | `CD_INTERVAL_READING_ALL_NO_QUOTES.csv` |
| Size | 16.5 GB |
| Rows | ~322 million |
| Interval | 30 minutes |
| Columns | `CUSTOMER_ID`, `READING_DATETIME`, `GENERAL_SUPPLY_KWH`, `CONTROLLED_LOAD_KWH`, `GROSS_GENERATION_KWH`, `NET_GENERATION_KWH`, `OTHER_KWH` |

## ⚙️ Usage

### 1. Preprocess PJM Regional Data

```bash
# Preprocess all 12 regional files
python scripts/preprocess_pjm.py --data-dir data/raw --output-dir data/preprocessed

# With train/test split
python scripts/preprocess_pjm.py --data-dir data/raw --output-dir data/preprocessed --split --split-date 2017-01-01
```

### 2. Preprocess Smart Meter Data (Large File)

```bash
# Default: 1M rows per chunk
python scripts/preprocess_smartmeter.py

# Custom chunk size (for lower RAM systems)
python scripts/preprocess_smartmeter.py --chunk-size 500000

# Custom input/output paths
python scripts/preprocess_smartmeter.py --input path/to/file.csv --output-dir path/to/output/
```

### 3. Compress to Parquet

```bash
# Basic conversion
python scripts/compress_to_parquet.py --input data/preprocessed/CD_INTERVAL_preprocessed.csv

# With smart meter dtypes for memory efficiency
python scripts/compress_to_parquet.py --input data/preprocessed/CD_INTERVAL_preprocessed.csv --smartmeter

# Different compression algorithm
python scripts/compress_to_parquet.py --input data/preprocessed/file.csv --compression gzip
```

## 🔧 Preprocessing Pipeline

### PJM Regional Data Pipeline

```
Raw CSV → Parse Datetime → Remove Duplicates → Reindex (Hourly) →
Interpolate Missing → Clip Outliers (IQR) → Feature Engineering → Save
```

| Step | Method | Details |
|------|--------|---------|
| Datetime parsing | `pd.to_datetime()` | Set as DatetimeIndex |
| Duplicate removal | `drop_duplicates()` | Handles DST transition duplicates |
| Reindexing | `pd.date_range()` | Fills gaps to continuous hourly |
| Interpolation | `interpolate(method='time')` | Time-weighted for missing values |
| Outlier clipping | IQR × 1.5 | Clips extreme values |
| Time features | Manual extraction | `hour`, `dayofweek`, `month`, `year`, `quarter`, `is_weekend` |
| Cyclical encoding | Sin/Cos transform | `hour_sin/cos`, `month_sin/cos` |
| Lag features | `.shift()` | 1h, 24h, 168h (1 week) lags |
| Rolling stats | `.rolling()` | 24h rolling mean and std |

### Smart Meter Data Pipeline

```
Read Chunk (1M rows) → Parse Datetime → Remove Duplicates →
Clip Negatives → Cap Outliers (99.9%) → Add TOTAL_KWH →
Feature Engineering → Write Chunk → Repeat
```

| Step | Method | Details |
|------|--------|---------|
| Chunked reading | `pd.read_csv(chunksize=1M)` | Memory-efficient processing |
| Efficient dtypes | `int32`, `float32`, `int8` | Reduces RAM per chunk to ~200 MB |
| Negative clipping | `.clip(lower=0)` | Fixes meter errors |
| Outlier capping | 99.9th percentile | Per-column extreme value capping |
| Total KWH | Sum of all kWh columns | Aggregated consumption metric |

### Compression (CSV → Parquet)

| Metric | CSV | Parquet |
|--------|-----|--------|
| File size | ~37 GB | ~3–5 GB |
| Read speed | Slow (text parsing) | ~10x faster (binary) |
| Compression | None | Snappy (default) |
| Column loading | Full file only | Select specific columns |

## 🖥️ System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| Python | 3.8+ | 3.10+ |
| Disk Space | 50 GB | 100 GB (for raw + processed) |

## 📈 Next Steps

- [ ] Exploratory Data Analysis (EDA) & visualization
- [ ] Statistical tests (ADF stationarity, ACF/PACF)
- [ ] Baseline model (ARIMA / Prophet)
- [ ] ML models (XGBoost, Random Forest)
- [ ] Deep Learning model (LSTM / Transformer)
- [ ] Model evaluation & comparison (MAE, RMSE, MAPE)
- [ ] Deployment / Dashboard

## 🛠️ Tech Stack

- **Python** 3.10+
- **pandas** — Data manipulation
- **NumPy** — Numerical operations
- **scikit-learn** — Scaling, preprocessing
- **PyArrow** — Parquet I/O and compression
- **Matplotlib / Seaborn** — Visualization (upcoming)

## 📝 License

This project is for academic purposes (23EER705 Research Based Mini Project).

## 👥 Team

| Name | Role |
|------|------|
| *(Student 1)* | *(Role)* |
| *(Student 2)* | *(Role)* |
| *(Student 3)* | *(Role)* |
| *(Student 4)* | *(Role)* |
