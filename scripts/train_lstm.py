"""
train_lstm.py
=============
Training and Evaluation Pipeline for Baseline PyTorch EnergyLSTM Model.

Executes:
1. Chronological Train / Validation / Test data ingestion.
2. Sliding-window sequence tensor generation (lookback T=24 hours).
3. Stacked LSTM training with Adam optimization, LR scheduling, and Early Stopping.
4. Evaluation on unseen Test set using MAE, RMSE, MAPE (%), and R2.
5. Saves model checkpoint (models/lstm_best.pt) and generates publication plots in results/.
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
    def __init__(self, features, targets, seq_len=24):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return x, y


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline LSTM energy forecasting model.")
    parser.add_argument("--data-path", type=str,
                        default=r"V:\DATASETS\DATASET 1\Preprocessed\PJME_preprocessed.csv",
                        help="Path to preprocessed CSV dataset.")
    parser.add_argument("--seq-len", type=int, default=24,
                        help="Lookback sequence window length in hours (default: 24).")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Maximum training epochs (default: 25).")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for DataLoader (default: 128).")
    parser.add_argument("--hidden-size", type=int, default=64,
                        help="Hidden dimensions in LSTM layers (default: 64).")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of stacked LSTM layers (default: 2).")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Initial learning rate (default: 0.001).")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience in epochs (default: 5).")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save plots and metrics.")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Directory to save model weights.")
    return parser.parse_args()


def load_and_preprocess_data(data_path):
    """Loads CSV and sets up normalized feature matrices."""
    print(f"Loading dataset from: {data_path} ...")
    df = pd.read_csv(data_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)

    # Calculate actual physical min and max for scaling restoration
    target_col = 'PJME_MW' if 'PJME_MW' in df.columns else df.columns[1]
    raw_min = float(df[target_col].min())
    raw_max = float(df[target_col].max())

    # Ensure MW_scaled exists
    if 'MW_scaled' in df.columns:
        df['target_scaled'] = df['MW_scaled'].astype(np.float32)
    else:
        df['target_scaled'] = ((df[target_col] - raw_min) / (raw_max - raw_min)).astype(np.float32)

    # Core clean feature subset (normalized [0, 1] or cyclical [-1, 1])
    feature_cols = ['target_scaled']
    cyclical_candidates = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'is_weekend']
    for col in cyclical_candidates:
        if col in df.columns:
            feature_cols.append(col)

    print(f"Dataset loaded: {len(df):,} total hourly observations.")
    print(f"Target variable: {target_col} (Min: {raw_min:,.1f} MW, Max: {raw_max:,.1f} MW)")
    print(f"Features used ({len(feature_cols)}): {feature_cols}")

    feature_matrix = df[feature_cols].values.astype(np.float32)
    target_vector = df['target_scaled'].values.astype(np.float32)
    timestamps = df['Datetime'].values

    return feature_matrix, target_vector, timestamps, target_col, raw_min, raw_max, feature_cols


def train_lstm_pipeline():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute device: {device}")

    # 1. Load Data
    features, targets, timestamps, target_col, raw_min, raw_max, feature_cols = load_and_preprocess_data(args.data_path)

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
    print(f"Training for up to {args.epochs} epochs with patience = {args.patience} ...\n")

    # 5. Training Loop
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.model_dir, "lstm_best.pt")
    patience_counter = 0

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Training
        model.train()
        train_loss_total = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_total += loss.item() * len(x_batch)

        train_mse = train_loss_total / len(train_ds)

        # Validation
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                val_loss_total += loss.item() * len(x_batch)

        val_mse = val_loss_total / len(val_ds)
        scheduler.step(val_mse)

        history['train_loss'].append(train_mse)
        history['val_loss'].append(val_mse)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        # Early Stopping check
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
                'seq_len': args.seq_len
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
    print("\nEvaluating best checkpoint on unseen Test Set...")
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

    # Inverse transform to original Megawatts (MW)
    preds_mw = all_preds_scaled * (raw_max - raw_min) + raw_min
    actual_mw = all_targets_scaled * (raw_max - raw_min) + raw_min

    # Calculate standard electrical metrics
    metrics = calculate_metrics(actual_mw, preds_mw)

    print("\n" + "="*50)
    print("      FINAL LSTM MODEL TEST EVALUATION METRICS     ")
    print("="*50)
    print(f"  Mean Absolute Error (MAE):     {metrics['MAE']:,.2f} MW")
    print(f"  Root Mean Squared Error (RMSE): {metrics['RMSE']:,.2f} MW")
    print(f"  Mean Abs. Percentage Error:    {metrics['MAPE']:.2f} %")
    print(f"  Coefficient of Determination:  R^2 = {metrics['R2']:.4f}")
    print("="*50)

    # Save metrics JSON
    metrics_summary = {
        "model": "EnergyLSTM",
        "dataset": os.path.basename(args.data_path),
        "target": target_col,
        "sequence_length_hours": args.seq_len,
        "hidden_units": args.hidden_size,
        "num_layers": args.num_layers,
        "epochs_trained": len(history['train_loss']),
        "training_time_seconds": round(total_training_time, 2),
        "metrics_physical_scale": metrics
    }
    metrics_path = os.path.join(args.output_dir, "lstm_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Metrics saved to: {metrics_path}")

    # 7. Generate Visualizations
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Figure 1: Training & Validation Convergence Curve
    plt.figure(figsize=(8, 4.5), dpi=300)
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss (MSE)', color='#1E40AF', lw=1.8)
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss (MSE)', color='#DC2626', lw=1.8, linestyle='--')
    plt.title('Baseline EnergyLSTM: Convergence Learning Curve', fontsize=11, fontweight='bold', pad=8)
    plt.xlabel('Epochs', fontsize=9.5, fontweight='bold')
    plt.ylabel('Mean Squared Error (Scaled)', fontsize=9.5, fontweight='bold')
    plt.legend(frameon=True, fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    loss_curve_path = os.path.join(args.output_dir, "lstm_loss_curve.png")
    plt.savefig(loss_curve_path, dpi=300)
    plt.close()
    print(f"Saved loss curve to: {loss_curve_path}")

    # Figure 2: Test Horizon Actual vs Predicted Load Curve (1 Week = 168 hours)
    plt.figure(figsize=(10, 4.8), dpi=300)
    horizon_hours = 168  # 1 full week
    time_slice = range(horizon_hours)

    plt.plot(time_slice, actual_mw[:horizon_hours], label='Actual Demand (PJME MW)', color='#0F172A', lw=1.6)
    plt.plot(time_slice, preds_mw[:horizon_hours], label='LSTM Forecast (Predicted MW)', color='#EA580C', lw=1.6, linestyle='--')
    plt.fill_between(time_slice, actual_mw[:horizon_hours], preds_mw[:horizon_hours], color='#F97316', alpha=0.2, label='Forecast Error Gap')

    plt.title(f'Baseline EnergyLSTM: 7-Day Forecast Tracking vs. Actual Demand\n(Test Set Performance: MAPE = {metrics["MAPE"]:.2f}%, R² = {metrics["R2"]:.4f})',
              fontsize=10.5, fontweight='bold', pad=8)
    plt.xlabel('Time Horizon (Hours across 7 Days)', fontsize=9.5, fontweight='bold')
    plt.ylabel('Power Demand (MW)', fontsize=9.5, fontweight='bold')
    plt.xlim(0, horizon_hours)
    plt.xticks(range(0, horizon_hours + 1, 24), [f'Day {d+1}' for d in range(8)])
    plt.legend(loc='upper right', frameon=True, fontsize=8.5)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    forecast_plot_path = os.path.join(args.output_dir, "lstm_predictions_vs_actual.png")
    plt.savefig(forecast_plot_path, dpi=300)
    plt.close()
    print(f"Saved forecast comparison plot to: {forecast_plot_path}")

    # Save sample predictions CSV
    sample_df = pd.DataFrame({
        'Datetime': [str(t) for t in test_timestamps[args.seq_len: args.seq_len + horizon_hours]],
        'Actual_MW': actual_mw[:horizon_hours],
        'Predicted_MW': preds_mw[:horizon_hours],
        'Absolute_Error_MW': np.abs(actual_mw[:horizon_hours] - preds_mw[:horizon_hours]),
        'Percentage_Error_Pct': np.abs((actual_mw[:horizon_hours] - preds_mw[:horizon_hours]) / actual_mw[:horizon_hours]) * 100.0
    })
    preds_csv_path = os.path.join(args.output_dir, "lstm_sample_predictions.csv")
    sample_df.to_csv(preds_csv_path, index=False)
    print(f"Saved sample forecast predictions CSV to: {preds_csv_path}")


if __name__ == "__main__":
    train_lstm_pipeline()
