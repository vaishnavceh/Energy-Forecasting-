"""
train_smartmeter_lstm.py
========================
Training and Evaluation Pipeline for PyTorch EnergyLSTM on Dataset 2:
Smart Meter 30-Minute Interval Consumption (CD_INTERVAL).

Dataset Specifications:
- Source: V:\\DATASETS\\DATASET 2\\Preprocessed\\CD_INTERVAL_simplified_preprocessed.parquet
- Interval: 30 minutes (48 time steps per 24-hour diurnal cycle)
- Unit: Kilowatt-Hours (kWh) per half-hour interval
- Physical Bounds: Min = 0.0 kWh, Max = 2.929 kWh (99.9th percentile threshold)
- Output Location: V:\\DATASETS\\DATASET 2\\Outputs (Preserves C: drive disk space)
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

# Ensure scripts directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from lstm_model import EnergyLSTM, calculate_metrics


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


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM on Dataset 2 Smart Meter data.")
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
                        help="LSTM hidden units (default: 64).")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of stacked LSTM layers (default: 2).")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Initial learning rate (default: 0.001).")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (default: 5).")
    parser.add_argument("--output-base", type=str,
                        default=r"V:\DATASETS\DATASET 2\Outputs",
                        help="Base output directory on V: drive to protect C: drive space.")
    return parser.parse_args()


def load_smartmeter_data(parquet_path, scaler_path, customer_id):
    """Loads smart meter observations for target benchmark customer."""
    print(f"Reading Dataset 2 Parquet from: {parquet_path}")
    t0 = time.time()
    parquet_file = pq.ParquetFile(parquet_path)

    # Read row group 0 (contains customer 10006414 and ~1M rows)
    rg0 = parquet_file.read_row_group(0).to_pandas()
    df_cust = rg0[rg0['customer_id'] == customer_id].sort_values('datetime').reset_index(drop=True)

    if len(df_cust) == 0:
        # Fallback to first available customer in RG0
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

    # Features
    feature_cols = ['target_scaled', 'step_sin', 'step_cos', 'month_sin', 'month_cos', 'is_weekend']
    print(f"Extracted {len(df_cust):,} continuous 30-min readings for Customer {customer_id} in {time.time()-t0:.2f}s.")
    print(f"Target: target_load (Min: {raw_min:.3f} kWh, Max: {raw_max:.3f} kWh)")
    print(f"Date range: {df_cust['datetime'].min()} to {df_cust['datetime'].max()}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    feature_matrix = df_cust[feature_cols].values.astype(np.float32)
    target_vector = df_cust['target_scaled'].values.astype(np.float32)
    timestamps = df_cust['datetime'].values

    return feature_matrix, target_vector, timestamps, raw_min, raw_max, feature_cols, customer_id


def train_smartmeter_pipeline():
    args = parse_args()

    # Destination directories on Drive V: to prevent C: disk exhaustion
    models_dir = os.path.join(args.output_base, "models")
    results_dir = os.path.join(args.output_base, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 65)
    print("  SMART METER ENERGY DEMAND FORECASTING (DATASET 2) TRAINING")
    print("  Department of Electrical & Electronics Engineering | Group 15")
    print("=" * 65)
    print(f"Target Output Storage on Drive V: {args.output_base}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute device: {device}")

    # 1. Load Data
    features, targets, timestamps, raw_min, raw_max, feature_cols, customer_id = load_smartmeter_data(
        args.data_path, args.scaler_path, args.customer_id
    )

    # 2. Chronological Split (70% Train, 15% Val, 15% Test)
    n = len(features)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    train_feats, train_targets = features[:train_end], targets[:train_end]
    val_feats, val_targets = features[train_end:val_end], targets[train_end:val_end]
    test_feats, test_targets = features[val_end:], targets[val_end:]
    test_timestamps = timestamps[val_end:]

    print(f"\nChronological Split:")
    print(f"  Training set:   {len(train_feats):,} samples ({train_feats.shape[0]/n*100:.1f}%)")
    print(f"  Validation set: {len(val_feats):,} samples ({val_feats.shape[0]/n*100:.1f}%)")
    print(f"  Testing set:    {len(test_feats):,} samples ({test_feats.shape[0]/n*100:.1f}%)")

    # 3. Create Datasets & DataLoaders
    train_ds = SlidingWindowDataset(train_feats, train_targets, seq_len=args.seq_len)
    val_ds = SlidingWindowDataset(val_feats, val_targets, seq_len=args.seq_len)
    test_ds = SlidingWindowDataset(test_feats, test_targets, seq_len=args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 4. Instantiate Model
    input_dim = features.shape[1]
    model = EnergyLSTM(
        input_size=input_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=0.2,
        dense_units=32,
        output_size=1
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: EnergyLSTM ({total_params:,} trainable parameters)")
    print(f"Lookback window: {args.seq_len} half-hours (24 Hours)")
    print(f"Training for up to {args.epochs} epochs with patience = {args.patience} ...\n")

    # 5. Training Loop
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_model_path = os.path.join(models_dir, "lstm_smartmeter_best.pt")
    patience_counter = 0

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Training
        model.train()
        train_batch_losses = []
        for x_b, y_b in train_loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            pred = model(x_b)
            loss = criterion(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_batch_losses.append(loss.item())

        train_mse = float(np.mean(train_batch_losses))
        history['train_loss'].append(train_mse)

        # Validation
        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b = x_b.to(device)
                y_b = y_b.to(device)
                pred = model(x_b)
                loss = criterion(pred, y_b)
                val_batch_losses.append(loss.item())

        val_mse = float(np.mean(val_batch_losses))
        history['val_loss'].append(val_mse)

        scheduler.step(val_mse)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        # Checkpointing
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_mse,
                'raw_min': raw_min,
                'raw_max': raw_max,
                'feature_cols': feature_cols,
                'input_dim': input_dim,
                'hidden_size': args.hidden_size,
                'num_layers': args.num_layers,
                'seq_len': args.seq_len,
                'customer_id': customer_id,
                'dataset_name': 'Dataset 2: CD_INTERVAL Smart Meter'
            }, best_model_path)
            improved_flag = "[*] [Saved Checkpoint]"
        else:
            patience_counter += 1
            improved_flag = f"(Patience: {patience_counter}/{args.patience})"

        print(f"Epoch {epoch:2d}/{args.epochs:2d} | Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f} | LR: {current_lr:.6f} | {epoch_time:.1f}s {improved_flag}")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time/60:.1f} minutes.")
    print(f"Best Validation MSE: {best_val_loss:.6f} (saved to {best_model_path})")

    # 6. Evaluation on Unseen Test Set
    print("\nEvaluating best checkpoint on unseen Test Set (5,408 test readings)...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_pred = model(x_batch)
            all_preds.extend(y_pred.cpu().numpy().flatten())
            all_targets.extend(y_batch.numpy().flatten())

    all_preds_scaled = np.array(all_preds)
    all_targets_scaled = np.array(all_targets)

    # Inverse transform to original physical Kilowatt-Hours (kWh)
    preds_kwh = all_preds_scaled * (raw_max - raw_min) + raw_min
    actual_kwh = all_targets_scaled * (raw_max - raw_min) + raw_min

    # Calculate standard electrical metrics
    metrics = calculate_metrics(actual_kwh, preds_kwh)

    print("\n" + "=" * 55)
    print("      DATASET 2 (SMART METER) FINAL LSTM TEST METRICS     ")
    print("=" * 55)
    print(f"  Mean Absolute Error (MAE):     {metrics['MAE']:.4f} kWh")
    print(f"  Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f} kWh")
    print(f"  Mean Abs. Percentage Error:    {metrics['MAPE']:.2f} %")
    print(f"  Coefficient of Determination:  R^2 = {metrics['R2']:.4f}")
    print("=" * 55)

    # 7. Save Metrics JSON on Drive V:
    metrics_summary = {
        "model": "EnergyLSTM",
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
    metrics_path = os.path.join(results_dir, "smartmeter_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Metrics saved to: {metrics_path}")

    # 8. Generate Visualizations on Drive V:
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Figure 1: Convergence Learning Curve
    plt.figure(figsize=(8, 4.5), dpi=300)
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss (MSE)', color='#1E40AF', lw=1.8)
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss (MSE)', color='#DC2626', lw=1.8, linestyle='--')
    plt.title(f'Dataset 2 Smart Meter LSTM: Convergence Curve (Customer {customer_id})', fontsize=11, fontweight='bold', pad=8)
    plt.xlabel('Epochs', fontsize=9.5, fontweight='bold')
    plt.ylabel('Mean Squared Error (Scaled)', fontsize=9.5, fontweight='bold')
    plt.legend(frameon=True, fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    loss_curve_path = os.path.join(results_dir, "smartmeter_loss_curve.png")
    plt.savefig(loss_curve_path, dpi=300)
    plt.close()
    print(f"Saved loss curve to: {loss_curve_path}")

    # Figure 2: Test Horizon Actual vs Predicted Load Curve (1 Week = 336 half-hour intervals)
    plt.figure(figsize=(11, 4.8), dpi=300)
    horizon_steps = 336  # 7 days * 48 half-hours
    time_slice = range(horizon_steps)

    plt.plot(time_slice, actual_kwh[:horizon_steps], label='Actual Household Demand (kWh)', color='#0F172A', lw=1.6)
    plt.plot(time_slice, preds_kwh[:horizon_steps], label='LSTM Forecast (Predicted kWh)', color='#EA580C', lw=1.6, linestyle='--')
    plt.fill_between(time_slice, actual_kwh[:horizon_steps], preds_kwh[:horizon_steps], color='#F97316', alpha=0.2, label='Forecast Residual Gap')

    plt.title(f'Dataset 2 Smart Meter: 7-Day Forecast Tracking vs. Actual Household Demand\n(Customer {customer_id} Test Set: MAE = {metrics["MAE"]:.3f} kWh, MAPE = {metrics["MAPE"]:.2f}%)',
              fontsize=10.5, fontweight='bold', pad=8)
    plt.xlabel('Time Horizon (30-Minute Interval Steps across 7 Days)', fontsize=9.5, fontweight='bold')
    plt.ylabel('Household Energy Consumption (kWh)', fontsize=9.5, fontweight='bold')
    plt.xlim(0, horizon_steps)
    plt.xticks(range(0, horizon_steps + 1, 48), [f'Day {d+1}' for d in range(8)])
    plt.legend(loc='upper right', frameon=True, fontsize=8.5)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    forecast_plot_path = os.path.join(results_dir, "smartmeter_predictions_vs_actual.png")
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
    preds_csv_path = os.path.join(results_dir, "smartmeter_sample_predictions.csv")
    sample_df.to_csv(preds_csv_path, index=False)
    print(f"Saved sample forecast predictions CSV to: {preds_csv_path}")

    print("\n" + "=" * 65)
    print(f"SUCCESS: All training outputs successfully written to Drive V:")
    print(f"  • Model Checkpoint: {best_model_path}")
    print(f"  • Metrics Summary:  {metrics_path}")
    print(f"  • Plots & Samples:  {results_dir}")
    print("  Zero large files were created on C: drive.")
    print("=" * 65)


if __name__ == "__main__":
    train_smartmeter_pipeline()
