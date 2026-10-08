"""
train_smartmeter_gru.py
=======================
Training and Evaluation Pipeline for PyTorch EnergyGRU on Dataset 2:
Smart Meter 30-Minute Interval Consumption (CD_INTERVAL).

Assigned Algorithm: Gated Recurrent Unit (GRU)
Part of: Smart Grid Energy Demand Forecasting (Group 15, TKMCE)

Dataset Specifications:
- Source: V:\\DATASETS\\DATASET 2\\Preprocessed\\CD_INTERVAL_simplified_preprocessed.parquet
- Target Benchmark: Customer 10006414 (36,057 half-hour readings)
- Interval: 30 minutes (48 time steps per 24-hour diurnal cycle)
- Physical Unit: Kilowatt-Hours (kWh) per half-hour interval
- Bounds: Min = 0.0 kWh, Max = 2.929 kWh
- Output Directory: V:\\DATASETS\\DATASET 2\\Outputs (Zero space used on C: drive)
"""

import argparse
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import json
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------------------------------------------------------
# 1. MODEL ARCHITECTURE: ENERGYGRU
# -----------------------------------------------------------------------------
class EnergyGRU(nn.Module):
    r"""
    Stacked GRU Neural Network for Smart Meter Load Forecasting.

    Mathematical Formulation:
      r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)       [Reset Gate]
      z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)       [Update Gate]
      \tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h) [Candidate]
      h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t         [Hidden State]
    """
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.2,
                 dense_units=32, output_size=1):
        super(EnergyGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, dense_units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_units, output_size)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        gru_out, _ = self.gru(x)
        # Take hidden state at the last time step
        last_step = gru_out[:, -1, :]
        out = self.dropout(last_step)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# -----------------------------------------------------------------------------
# 2. DATASET & METRIC UTILITIES
# -----------------------------------------------------------------------------
class SlidingWindowDataset(Dataset):
    """Memory-efficient sliding window time-series dataset."""
    def __init__(self, features, targets, seq_len=48):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return x, y


def calculate_metrics(y_true, y_pred, epsilon=1e-7):
    """Compute MAE, RMSE, MAPE %, and R^2 on physical scale (kWh)."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denom = np.maximum(np.abs(y_true), epsilon)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    r2 = float(r2_score(y_true, y_pred))

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "R2": round(r2, 4)
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train GRU on Dataset 2 Smart Meter data.")
    parser.add_argument("--data-path", type=str,
                        default=r"V:\DATASETS\DATASET 2\Preprocessed\CD_INTERVAL_simplified_preprocessed.parquet",
                        help="Path to preprocessed parquet dataset.")
    parser.add_argument("--scaler-path", type=str,
                        default=r"V:\DATASETS\DATASET 2\Preprocessed\scaler_params.json",
                        help="Path to scaler_params.json.")
    parser.add_argument("--customer-id", type=int, default=10006414,
                        help="Target benchmark customer ID (default: 10006414).")
    parser.add_argument("--seq-len", type=int, default=48,
                        help="Lookback sequence window length in half-hours (48 = 24 hours).")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Maximum training epochs (default: 15).")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size (default: 128).")
    parser.add_argument("--hidden-size", type=int, default=64,
                        help="GRU hidden units (default: 64).")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of stacked GRU layers (default: 2).")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Initial learning rate (default: 0.001).")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (default: 5).")
    parser.add_argument("--output-base", type=str,
                        default=r"V:\DATASETS\DATASET 2\Outputs",
                        help="Base output directory on V: drive to protect C: drive space.")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# 3. DATA LOADER FOR CUSTOMER 10006414
# -----------------------------------------------------------------------------
def load_smartmeter_data(parquet_path, scaler_path, customer_id):
    """Loads smart meter observations for benchmark customer."""
    print(f"Reading Dataset 2 Parquet from: {parquet_path}")
    t0 = time.time()
    parquet_file = pq.ParquetFile(parquet_path)

    # Read row group 0 (contains customer 10006414 and ~1M rows)
    rg0 = parquet_file.read_row_group(0).to_pandas()
    df_cust = rg0[rg0['customer_id'] == customer_id].sort_values('datetime').reset_index(drop=True)

    if len(df_cust) == 0:
        customer_id = rg0['customer_id'].iloc[0]
        df_cust = rg0[rg0['customer_id'] == customer_id].sort_values('datetime').reset_index(drop=True)
        print(f"Requested customer not found, switched to primary customer: {customer_id}")

    # Load scaler bounds
    raw_min = 0.0
    raw_max = 2.929
    if os.path.exists(scaler_path):
        with open(scaler_path, 'r') as f:
            sparams = json.load(f)
            raw_min = float(sparams.get('min', 0.0))
            raw_max = float(sparams.get('max', 2.929))

    feature_cols = ['target_scaled', 'step_sin', 'step_cos', 'month_sin', 'month_cos', 'is_weekend']
    print(f"Extracted {len(df_cust):,} continuous 30-min readings for Customer {customer_id} in {time.time()-t0:.2f}s.")
    print(f"Target: target_load (Min: {raw_min:.3f} kWh, Max: {raw_max:.3f} kWh)")
    print(f"Date range: {df_cust['datetime'].min()} to {df_cust['datetime'].max()}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    feature_matrix = df_cust[feature_cols].values.astype(np.float32)
    target_vector = df_cust['target_scaled'].values.astype(np.float32)
    timestamps = df_cust['datetime'].values

    return feature_matrix, target_vector, timestamps, feature_cols, raw_min, raw_max, customer_id


# -----------------------------------------------------------------------------
# 4. TRAINING & EVALUATION PIPELINE
# -----------------------------------------------------------------------------
def train_smartmeter_gru():
    args = parse_args()

    # Destination directories on Drive V:
    models_dir = os.path.join(args.output_base, "models")
    results_dir = os.path.join(args.output_base, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 65)
    print("  SMART METER ENERGY DEMAND FORECASTING (DATASET 2)")
    print("  MODEL: GATED RECURRENT UNIT (GRU)")
    print("  GROUP 15 | DEPT OF EEE | TKM COLLEGE OF ENGINEERING")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Execution Device: {device}")

    # 1. Load Data
    features, targets, timestamps, feature_cols, raw_min, raw_max, customer_id = load_smartmeter_data(
        args.data_path, args.scaler_path, args.customer_id
    )

    # 2. Strict Chronological Split (70% Train, 15% Val, 15% Test)
    n_total = len(features)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    n_test = n_total - n_train - n_val

    train_feat, train_targ = features[:n_train], targets[:n_train]
    val_feat, val_targ = features[n_train : n_train + n_val], targets[n_train : n_train + n_val]
    test_feat, test_targ = features[n_train + n_val:], targets[n_train + n_val:]
    test_timestamps = timestamps[n_train + n_val:]

    print(f"\nChronological Split (Strictly zero data leakage):")
    print(f"  Training Set:   {len(train_feat):,} samples ({len(train_feat)/n_total*100:.1f}%)")
    print(f"  Validation Set: {len(val_feat):,} samples ({len(val_feat)/n_total*100:.1f}%)")
    print(f"  Testing Set:    {len(test_feat):,} samples ({len(test_feat)/n_total*100:.1f}%)")

    train_loader = DataLoader(SlidingWindowDataset(train_feat, train_targ, args.seq_len),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SlidingWindowDataset(val_feat, val_targ, args.seq_len),
                            batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(SlidingWindowDataset(test_feat, test_targ, args.seq_len),
                             batch_size=args.batch_size, shuffle=False)

    # 3. Model Initialization
    model = EnergyGRU(
        input_size=len(feature_cols),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=0.2,
        dense_units=32,
        output_size=1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nEnergyGRU Architecture Initialized:")
    print(f"  • Input features:  {len(feature_cols)}")
    print(f"  • Hidden size:     {args.hidden_size}")
    print(f"  • Stacked layers:  {args.num_layers}")
    print(f"  • Total trainable parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    best_model_path = os.path.join(models_dir, "gru_smartmeter_best.pt")
    history = {'train_loss': [], 'val_loss': []}
    epochs_no_improve = 0

    print(f"\nStarting GRU training for up to {args.epochs} epochs...")
    start_train_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss_acc = 0.0

        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(x_b)
            loss = criterion(preds, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_acc += loss.item() * len(x_b)

        train_loss = train_loss_acc / len(train_loader.dataset)

        model.eval()
        val_loss_acc = 0.0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                preds = model(x_b)
                loss = criterion(preds, y_b)
                val_loss_acc += loss.item() * len(x_b)

        val_loss = val_loss_acc / len(val_loader.dataset)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start
        improved = val_loss < best_val_loss
        marker = "(* Best Model Saved)" if improved else ""

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) | "
              f"Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} {marker}")

        if improved:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
            # Save checkpoint directly on Drive V:
            torch.save({
                'model_state_dict': best_model_state,
                'model_type': 'gru',
                'input_dim': len(feature_cols),
                'hidden_size': args.hidden_size,
                'num_layers': args.num_layers,
                'seq_len': args.seq_len,
                'scaler_min': raw_min,
                'scaler_max': raw_max,
                'customer_id': customer_id,
                'features': feature_cols
            }, best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    total_training_time = time.time() - start_train_time
    print(f"\nTraining completed in {total_training_time:.1f}s ({total_training_time/60:.2f} min).")
    print(f"Best Validation Loss: {best_val_loss:.6f} MSE")

    # 5. Evaluate on Unseen Test Set
    print("\nEvaluating best checkpoint on unseen test set...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds_scaled = []
    all_targets_scaled = []
    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b = x_b.to(device)
            preds = model(x_b)
            all_preds_scaled.append(preds.cpu().numpy())
            all_targets_scaled.append(y_b.numpy())

    all_preds_scaled = np.vstack(all_preds_scaled).flatten()
    all_targets_scaled = np.vstack(all_targets_scaled).flatten()

    # Inverse transform to physical scale (kWh)
    actual_kwh = all_targets_scaled * (raw_max - raw_min) + raw_min
    preds_kwh = all_preds_scaled * (raw_max - raw_min) + raw_min
    preds_kwh = np.maximum(0.0, preds_kwh)

    # 6. Comprehensive Metrics
    metrics = calculate_metrics(actual_kwh, preds_kwh)

    print("\n" + "=" * 55)
    print("      DATASET 2 (SMART METER) FINAL GRU TEST METRICS     ")
    print("=" * 55)
    print(f"  Mean Absolute Error (MAE):      {metrics['MAE']:.4f} kWh")
    print(f"  Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f} kWh")
    print(f"  Mean Abs. Percentage Error:     {metrics['MAPE']:.2f} %")
    print(f"  Coefficient of Determination:   R^2 = {metrics['R2']:.4f}")
    print("=" * 55)

    # 7. Save Metrics JSON on Drive V:
    metrics_summary = {
        "model": "EnergyGRU",
        "dataset": "CD_INTERVAL_simplified_preprocessed.parquet",
        "dataset_category": "Dataset 2: 30-Min Interval Smart Meter",
        "customer_id": customer_id,
        "physical_unit": "kWh (Kilowatt-Hours)",
        "sequence_length_half_hours": args.seq_len,
        "hidden_units": args.hidden_size,
        "num_layers": args.num_layers,
        "epochs_trained": len(history['train_loss']),
        "training_time_seconds": round(total_training_time, 2),
        "metrics_physical_scale": metrics
    }
    metrics_path = os.path.join(results_dir, "gru_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Metrics saved to: {metrics_path}")

    # Update checkpoint with evaluated metrics
    checkpoint['metrics'] = metrics
    torch.save(checkpoint, best_model_path)

    # 8. Generate Visualizations on Drive V:
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Figure 1: Convergence Learning Curve
    plt.figure(figsize=(8, 4.5), dpi=300)
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss (MSE)', color='#059669', lw=1.8)
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss (MSE)', color='#DC2626', lw=1.8, linestyle='--')
    plt.title(f'Dataset 2 Smart Meter GRU: Convergence Curve (Customer {customer_id})', fontsize=11, fontweight='bold', pad=8)
    plt.xlabel('Epochs', fontsize=9.5, fontweight='bold')
    plt.ylabel('Mean Squared Error (Scaled)', fontsize=9.5, fontweight='bold')
    plt.legend(frameon=True, fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    loss_curve_path = os.path.join(results_dir, "gru_loss_curve.png")
    plt.savefig(loss_curve_path, dpi=300)
    plt.close()
    print(f"Saved loss curve to: {loss_curve_path}")

    # Figure 2: Test Horizon Actual vs Predicted Load Curve (1 Week = 336 half-hour intervals)
    plt.figure(figsize=(11, 4.8), dpi=300)
    horizon_steps = 336
    time_slice = range(horizon_steps)

    plt.plot(time_slice, actual_kwh[:horizon_steps], label='Actual Household Demand (kWh)', color='#0F172A', lw=1.6)
    plt.plot(time_slice, preds_kwh[:horizon_steps], label='GRU Forecast (Predicted kWh)', color='#10B981', lw=1.6, linestyle='--')
    plt.fill_between(time_slice, actual_kwh[:horizon_steps], preds_kwh[:horizon_steps], color='#34D399', alpha=0.2, label='Forecast Residual Gap')

    plt.title(f'Dataset 2 Smart Meter: 7-Day Forecast Tracking vs. Actual Household Demand\n(Customer {customer_id} GRU Test Set: MAE = {metrics["MAE"]:.3f} kWh, MAPE = {metrics["MAPE"]:.2f}%)',
              fontsize=10.5, fontweight='bold', pad=8)
    plt.xlabel('Time Horizon (30-Minute Interval Steps across 7 Days)', fontsize=9.5, fontweight='bold')
    plt.ylabel('Household Energy Consumption (kWh)', fontsize=9.5, fontweight='bold')
    plt.xlim(0, horizon_steps)
    plt.xticks(range(0, horizon_steps + 1, 48), [f'Day {d+1}' for d in range(8)])
    plt.legend(loc='upper right', frameon=True, fontsize=8.5)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    forecast_plot_path = os.path.join(results_dir, "gru_predictions_vs_actual.png")
    plt.savefig(forecast_plot_path, dpi=300)
    plt.close()
    print(f"Saved forecast comparison plot to: {forecast_plot_path}")

    # Save sample predictions CSV on Drive V:
    sample_df = pd.DataFrame({
        'Datetime': [str(t) for t in test_timestamps[args.seq_len : args.seq_len + horizon_steps]],
        'Actual_kWh': actual_kwh[:horizon_steps],
        'Predicted_kWh': preds_kwh[:horizon_steps],
        'Absolute_Error_kWh': np.abs(actual_kwh[:horizon_steps] - preds_kwh[:horizon_steps]),
        'Percentage_Error_Pct': np.abs((actual_kwh[:horizon_steps] - preds_kwh[:horizon_steps]) / (actual_kwh[:horizon_steps] + 1e-4)) * 100.0
    })
    preds_csv_path = os.path.join(results_dir, "gru_sample_predictions.csv")
    sample_df.to_csv(preds_csv_path, index=False)
    print(f"Saved sample forecast predictions CSV to: {preds_csv_path}")

    print("\n" + "=" * 65)
    print(f"SUCCESS: GRU model training completed and written to Drive V:")
    print(f"  • Model Checkpoint: {best_model_path}")
    print(f"  • Metrics Summary:  {metrics_path}")
    print(f"  • Sample CSV:       {preds_csv_path}")
    print(f"  • Visual Plots:     {loss_curve_path}")
    print("=" * 65)


if __name__ == "__main__":
    train_smartmeter_gru()
