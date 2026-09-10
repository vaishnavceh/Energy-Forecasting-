"""
train_deepdeff.py
=================
Train the DeepDeFF model on prepared .npz data files using PyTorch.

Loops over per-customer data, trains all model variants (RNN, LSTM, GRU,
BRNN, BLSTM, BGRU), evaluates with MAPE, and saves results.

Usage:
    python scripts/train_deepdeff.py --data-dir V:/DATASETS/.../DeepDeFF --output-dir results/
"""

import argparse
import os
import sys
import glob
import time
import logging
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Add scripts dir to path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from deepdeff_model import DeepDeFF, mape_loss, calculate_mape


def setup_logger():
    """Sets up the logger."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Train DeepDeFF models on prepared data.")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing prepared .npz files.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save results and models.")
    parser.add_argument("--variants", type=str, default="RNN,LSTM,GRU,BRNN,BLSTM,BGRU",
                        help="Comma-separated list of variants to train (default: all 6).")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum training epochs (default: 100).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32).")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (default: 10).")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001).")
    return parser.parse_args()


def train_one_model(model, train_loader, val_loader, epochs, patience, lr, device):
    """
    Train a single DeepDeFF model with early stopping.

    Args:
        model: DeepDeFF model instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        epochs: Max epochs.
        patience: Early stopping patience.
        lr: Learning rate.
        device: torch device.

    Returns:
        dict: Training history {'train_loss': [...], 'val_loss': [...]}
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'train_loss': [], 'val_loss': []}

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        train_losses = []
        for x_basic, x_derived, y in train_loader:
            x_basic = x_basic.to(device)
            x_derived = x_derived.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x_basic, x_derived)
            loss = mape_loss(y_pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)

        # ---- Validation ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_basic, x_derived, y in val_loader:
                x_basic = x_basic.to(device)
                x_derived = x_derived.to(device)
                y = y.to(device)

                y_pred = model(x_basic, x_derived)
                loss = mape_loss(y_pred, y)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        history['val_loss'].append(avg_val_loss)

        # ---- Early Stopping ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set.

    Returns:
        tuple: (mape, y_true_all, y_pred_all)
    """
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for x_basic, x_derived, y in test_loader:
            x_basic = x_basic.to(device)
            x_derived = x_derived.to(device)

            y_pred = model(x_basic, x_derived)
            all_preds.append(y_pred.cpu().numpy())
            all_true.append(y.numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_true, axis=0)
    mape = calculate_mape(y_pred, y_true)

    return mape, y_true, y_pred


def main():
    """Main training loop."""
    setup_logger()
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    models_dir = os.path.join(args.output_dir, "models")
    predictions_dir = os.path.join(args.output_dir, "predictions")
    history_dir = os.path.join(args.output_dir, "history")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)

    variants = [v.strip().upper() for v in args.variants.split(',')]
    logging.info(f"Training variants: {variants}")

    npz_files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if not npz_files:
        logging.error(f"No .npz files found in {args.data_dir}")
        return

    logging.info(f"Found {len(npz_files)} customer data files")
    results = []
    start_time = time.time()

    for file_idx, npz_file in enumerate(npz_files):
        customer_id = os.path.splitext(os.path.basename(npz_file))[0]
        logging.info(f"\n[{file_idx+1}/{len(npz_files)}] Customer: {customer_id}")

        try:
            data = np.load(npz_file)
            X_basic_train = data['X_basic_train'].astype(np.float32)
            X_derived_train = data['X_derived_train'].astype(np.float32)
            y_train = data['y_train'].astype(np.float32).reshape(-1, 1)
            X_basic_val = data['X_basic_val'].astype(np.float32)
            X_derived_val = data['X_derived_val'].astype(np.float32)
            y_val = data['y_val'].astype(np.float32).reshape(-1, 1)
            X_basic_test = data['X_basic_test'].astype(np.float32)
            X_derived_test = data['X_derived_test'].astype(np.float32)
            y_test = data['y_test'].astype(np.float32).reshape(-1, 1)
        except Exception as e:
            logging.error(f"Error loading data for {customer_id}: {e}")
            continue

        # Check minimum data requirements
        if len(X_basic_train) < 10 or len(X_basic_val) < 5 or len(X_basic_test) < 5:
            logging.warning(f"Skipping {customer_id}: insufficient data "
                            f"(train={len(X_basic_train)}, val={len(X_basic_val)}, "
                            f"test={len(X_basic_test)})")
            continue

        basic_features = X_basic_train.shape[2]
        derived_features = X_derived_train.shape[2]

        # Create DataLoaders
        train_ds = TensorDataset(
            torch.tensor(X_basic_train), torch.tensor(X_derived_train), torch.tensor(y_train))
        val_ds = TensorDataset(
            torch.tensor(X_basic_val), torch.tensor(X_derived_val), torch.tensor(y_val))
        test_ds = TensorDataset(
            torch.tensor(X_basic_test), torch.tensor(X_derived_test), torch.tensor(y_test))

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        customer_results = {'customer_id': customer_id}
        best_mape = float('inf')
        best_variant = None

        for variant in variants:
            try:
                model = DeepDeFF(
                    basic_features=basic_features,
                    derived_features=derived_features,
                    cell_type=variant,
                    n_units=20,
                    dropout=0.2,
                    dense_units=20
                )

                history = train_one_model(
                    model, train_loader, val_loader,
                    epochs=args.epochs, patience=args.patience,
                    lr=args.lr, device=device
                )

                # Evaluate
                test_mape, y_true, y_pred = evaluate_model(model, test_loader, device)
                customer_results[f'MAPE_{variant}'] = round(test_mape, 2)

                logging.info(f"  {variant}: MAPE = {test_mape:.2f}%  "
                             f"(epochs: {len(history['train_loss'])})")

                # Save predictions
                np.savez(
                    os.path.join(predictions_dir, f"{customer_id}_{variant}.npz"),
                    y_true=y_true, y_pred=y_pred
                )

                # Save training history
                np.savez(
                    os.path.join(history_dir, f"{customer_id}_{variant}_history.npz"),
                    train_loss=history['train_loss'],
                    val_loss=history['val_loss']
                )

                # Track best model
                if test_mape < best_mape:
                    best_mape = test_mape
                    best_variant = variant
                    torch.save(model.state_dict(),
                               os.path.join(models_dir, f"{customer_id}_best.pt"))

            except Exception as e:
                logging.error(f"  Error training {variant} for {customer_id}: {e}")
                customer_results[f'MAPE_{variant}'] = np.nan

        customer_results['best_variant'] = best_variant
        customer_results['best_mape'] = round(best_mape, 2) if best_mape < float('inf') else np.nan
        results.append(customer_results)

        logging.info(f"  Best: {best_variant} (MAPE={best_mape:.2f}%)")

    if not results:
        logging.error("No successful results to summarize.")
        return

    # ---- Summary ----
    total_time = time.time() - start_time
    results_df = pd.DataFrame(results)

    mape_cols = [c for c in results_df.columns if c.startswith('MAPE_')]

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"  Customers trained:  {len(results)}")
    print(f"  Model variants:     {variants}")
    print(f"  Total time:         {total_time / 60:.1f} minutes")
    print("=" * 70)

    print("\n--- Average MAPE per Variant ---")
    for col in mape_cols:
        variant_name = col.replace('MAPE_', '')
        avg = results_df[col].mean()
        print(f"  DeepDeFF {variant_name:6s}: {avg:.2f}%")

    print("\n--- Best Variant Distribution ---")
    if 'best_variant' in results_df.columns:
        print(results_df['best_variant'].value_counts().to_string())

    # Save results CSV
    csv_path = os.path.join(args.output_dir, "results.csv")
    results_df.to_csv(csv_path, index=False)
    logging.info(f"\nResults saved to {csv_path}")

    print(f"\nOutput files:")
    print(f"  Results CSV:    {csv_path}")
    print(f"  Models:         {models_dir}")
    print(f"  Predictions:    {predictions_dir}")
    print(f"  History:        {history_dir}")


if __name__ == "__main__":
    main()
