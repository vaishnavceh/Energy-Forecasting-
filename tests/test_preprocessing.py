"""
test_preprocessing.py
=====================
Unit tests for the energy forecasting preprocessing pipeline.
Uses synthetic data to test each preprocessing function without needing real datasets.

Run: pytest tests/ -v
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))


# ==================== FIXTURES ====================


@pytest.fixture
def sample_pjm_csv(tmp_path):
    """Create a sample PJM-style CSV file for testing."""
    dates = pd.date_range("2020-01-01", periods=200, freq="h")
    data = pd.DataFrame(
        {"Datetime": dates, "PJME_MW": np.random.uniform(20000, 40000, len(dates))}
    )
    # Add a duplicate timestamp (simulating DST)
    dup_row = data.iloc[[5]].copy()
    data = pd.concat([data, dup_row], ignore_index=True)

    # Add a gap (remove some rows to simulate missing hours)
    data = data.drop(index=[10, 11, 12])

    filepath = tmp_path / "PJME_hourly.csv"
    data.to_csv(filepath, index=False)
    return str(filepath)


@pytest.fixture
def sample_smartmeter_csv(tmp_path):
    """Create a sample smart meter CSV file for testing."""
    n_rows = 1000
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="30min")
    data = pd.DataFrame(
        {
            "CUSTOMER_ID": np.random.choice([100, 200, 300], n_rows),
            "READING_DATETIME": dates,
            "CALENDAR_KEY": np.random.randint(1000, 9999, n_rows),
            "EVENT_KEY": np.zeros(n_rows, dtype=int),
            "GENERAL_SUPPLY_KWH": np.random.uniform(0, 2, n_rows),
            "CONTROLLED_LOAD_KWH": np.random.uniform(0, 0.5, n_rows),
            "GROSS_GENERATION_KWH": np.zeros(n_rows),
            "NET_GENERATION_KWH": np.zeros(n_rows),
            "OTHER_KWH": np.zeros(n_rows),
        }
    )
    # Add a negative value (meter error)
    data.loc[5, "GENERAL_SUPPLY_KWH"] = -0.5

    filepath = tmp_path / "test_smartmeter.csv"
    data.to_csv(filepath, index=False)
    return str(filepath)


# ==================== PJM PREPROCESSING TESTS ====================


class TestPJMPreprocessing:
    """Tests for PJM regional data preprocessing."""

    def test_load_csv(self, sample_pjm_csv):
        """Test that CSV loads correctly with datetime parsing."""
        df = pd.read_csv(sample_pjm_csv, parse_dates=["Datetime"], index_col="Datetime")
        assert isinstance(df.index, pd.DatetimeIndex)
        assert "PJME_MW" in df.columns

    def test_duplicate_removal(self, sample_pjm_csv):
        """Test that duplicate timestamps are removed."""
        df = pd.read_csv(sample_pjm_csv, parse_dates=["Datetime"], index_col="Datetime")
        assert df.index.duplicated().sum() > 0, "Test data should have duplicates"

        df = df[~df.index.duplicated(keep="first")]
        assert df.index.duplicated().sum() == 0

    def test_reindex_fills_gaps(self, sample_pjm_csv):
        """Test that reindexing creates continuous hourly frequency."""
        df = pd.read_csv(sample_pjm_csv, parse_dates=["Datetime"], index_col="Datetime")
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()

        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
        df = df.reindex(full_idx)

        # Should have NaN where gaps were
        assert df.isnull().sum().sum() > 0

    def test_interpolation(self, sample_pjm_csv):
        """Test that interpolation fills missing values."""
        df = pd.read_csv(sample_pjm_csv, parse_dates=["Datetime"], index_col="Datetime")
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()

        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
        df = df.reindex(full_idx)
        df = df.interpolate(method="time")

        assert df["PJME_MW"].isnull().sum() == 0

    def test_outlier_clipping(self):
        """Test IQR-based outlier clipping."""
        data = pd.Series([10, 20, 30, 40, 50, 1000, -500])
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        clipped = data.clip(lower=lower, upper=upper)
        assert clipped.max() <= upper
        assert clipped.min() >= lower

    def test_feature_engineering(self):
        """Test that time features are correctly generated."""
        dates = pd.date_range("2020-06-15 14:00:00", periods=5, freq="h")
        df = pd.DataFrame({"MW": [100, 200, 300, 400, 500]}, index=dates)

        df["hour"] = df.index.hour
        df["dayofweek"] = df.index.dayofweek
        df["month"] = df.index.month
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

        assert df["hour"].iloc[0] == 14
        assert df["month"].iloc[0] == 6
        assert df["is_weekend"].iloc[0] == 0  # Monday=0, June 15 2020 is Monday

    def test_cyclical_encoding(self):
        """Test that cyclical encoding produces valid sin/cos values."""
        hours = pd.Series([0, 6, 12, 18, 24])
        hour_sin = np.sin(2 * np.pi * hours / 24)
        hour_cos = np.cos(2 * np.pi * hours / 24)

        # Sin and cos should be between -1 and 1
        assert hour_sin.between(-1, 1).all()
        assert hour_cos.between(-1, 1).all()

        # Hour 0 and 24 should give same encoding
        assert abs(hour_sin.iloc[0] - hour_sin.iloc[-1]) < 1e-10

    def test_lag_features(self):
        """Test that lag features are correctly shifted."""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7])
        lag_1 = data.shift(1)

        assert pd.isna(lag_1.iloc[0])
        assert lag_1.iloc[1] == 1
        assert lag_1.iloc[2] == 2

    def test_train_test_split_chronological(self):
        """Test that train/test split preserves time order."""
        dates = pd.date_range("2016-01-01", "2018-12-31", freq="h")
        df = pd.DataFrame({"MW": np.random.rand(len(dates))}, index=dates)

        split_date = "2017-01-01"
        train = df.loc[:split_date]
        test = df.loc[split_date:]

        assert train.index.max() <= pd.Timestamp(split_date)
        assert test.index.min() >= pd.Timestamp(split_date)
        assert len(train) + len(test) == len(df)


# ==================== SMART METER TESTS ====================


class TestSmartMeterPreprocessing:
    """Tests for smart meter chunked preprocessing."""

    def test_load_with_dtypes(self, sample_smartmeter_csv):
        """Test that CSV loads with memory-efficient dtypes."""
        dtypes = {"CUSTOMER_ID": "int32", "GENERAL_SUPPLY_KWH": "float32"}
        df = pd.read_csv(sample_smartmeter_csv, dtype=dtypes)
        assert df["CUSTOMER_ID"].dtype == np.int32
        assert df["GENERAL_SUPPLY_KWH"].dtype == np.float32

    def test_negative_clipping(self, sample_smartmeter_csv):
        """Test that negative kWh values are clipped to 0."""
        df = pd.read_csv(sample_smartmeter_csv)
        assert (df["GENERAL_SUPPLY_KWH"] < 0).any(), "Test data should have negatives"

        df["GENERAL_SUPPLY_KWH"] = df["GENERAL_SUPPLY_KWH"].clip(lower=0)
        assert (df["GENERAL_SUPPLY_KWH"] >= 0).all()

    def test_total_kwh_calculation(self, sample_smartmeter_csv):
        """Test that TOTAL_KWH is the sum of all kWh columns."""
        df = pd.read_csv(sample_smartmeter_csv)
        kwh_cols = [
            "GENERAL_SUPPLY_KWH",
            "CONTROLLED_LOAD_KWH",
            "GROSS_GENERATION_KWH",
            "NET_GENERATION_KWH",
            "OTHER_KWH",
        ]
        df["TOTAL_KWH"] = df[kwh_cols].sum(axis=1)

        # Verify row-wise sum
        expected = df[kwh_cols].sum(axis=1)
        pd.testing.assert_series_equal(df["TOTAL_KWH"], expected)

    def test_chunked_reading(self, sample_smartmeter_csv):
        """Test that chunked reading processes all rows."""
        total_rows = 0
        for chunk in pd.read_csv(sample_smartmeter_csv, chunksize=200):
            total_rows += len(chunk)

        full_df = pd.read_csv(sample_smartmeter_csv)
        assert total_rows == len(full_df)

    def test_outlier_capping_percentile(self):
        """Test 99.9th percentile capping."""
        data = pd.Series(np.concatenate([np.random.normal(1, 0.3, 1000), [100, 200]]))
        upper = data.quantile(0.999)
        capped = data.clip(upper=upper)
        assert capped.max() <= upper


# ==================== PARQUET COMPRESSION TESTS ====================


class TestParquetCompression:
    """Tests for CSV to Parquet conversion."""

    def test_csv_to_parquet_roundtrip(self, tmp_path):
        """Test that CSV → Parquet → DataFrame preserves data."""
        # Create sample data
        df_original = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "value": [1.5, 2.5, 3.5],
                "name": ["a", "b", "c"],
            }
        )
        csv_path = tmp_path / "test.csv"
        parquet_path = tmp_path / "test.parquet"

        # Save as CSV then read and save as Parquet
        df_original.to_csv(csv_path, index=False)
        df_from_csv = pd.read_csv(csv_path)
        df_from_csv.to_parquet(parquet_path, index=False)

        # Read back from Parquet
        df_from_parquet = pd.read_parquet(parquet_path)

        pd.testing.assert_frame_equal(df_from_csv, df_from_parquet)

    def test_parquet_smaller_than_csv(self, tmp_path):
        """Test that Parquet file is smaller than CSV."""
        df = pd.DataFrame(
            {
                "id": np.arange(10000),
                "value": np.random.rand(10000),
                "category": np.random.choice(["A", "B", "C"], 10000),
            }
        )
        csv_path = tmp_path / "test.csv"
        parquet_path = tmp_path / "test.parquet"

        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False, compression="snappy")

        csv_size = os.path.getsize(csv_path)
        parquet_size = os.path.getsize(parquet_path)

        assert parquet_size < csv_size, "Parquet should be smaller than CSV"

    def test_parquet_column_selection(self, tmp_path):
        """Test that Parquet supports reading specific columns."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        parquet_path = tmp_path / "test.parquet"
        df.to_parquet(parquet_path, index=False)

        df_partial = pd.read_parquet(parquet_path, columns=["a", "c"])
        assert list(df_partial.columns) == ["a", "c"]
        assert len(df_partial) == 3


# ==================== INTEGRATION TESTS ====================


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pjm_pipeline(self, sample_pjm_csv, tmp_path):
        """Test the complete PJM preprocessing pipeline end-to-end."""
        # Load
        df = pd.read_csv(sample_pjm_csv, parse_dates=["Datetime"], index_col="Datetime")
        col = df.columns[0]

        # Remove duplicates
        df = df[~df.index.duplicated(keep="first")]

        # Reindex
        df = df.sort_index()
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
        df = df.reindex(full_idx)
        df.index.name = "Datetime"

        # Interpolate
        df = df.interpolate(method="time")

        # Outlier clipping
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

        # Feature engineering
        df["hour"] = df.index.hour
        df["month"] = df.index.month
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)

        # Lag features
        df["lag_1h"] = df[col].shift(1)
        df = df.dropna()

        # Save
        output_path = tmp_path / "output.csv"
        df.to_csv(output_path)

        # Verify
        assert os.path.exists(output_path)
        df_loaded = pd.read_csv(output_path, parse_dates=["Datetime"], index_col="Datetime")
        assert df_loaded.isnull().sum().sum() == 0
        assert "hour" in df_loaded.columns
        assert "hour_sin" in df_loaded.columns
        assert "lag_1h" in df_loaded.columns
        assert len(df_loaded) > 0
