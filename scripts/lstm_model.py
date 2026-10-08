"""
lstm_model.py
=============
PyTorch Baseline Long Short-Term Memory (LSTM) Architecture for
Short-Term Energy Demand Forecasting.

Supports single-step and multi-step lookback windows with stacked recurrent
layers, dropout regularization, and dense prediction heads.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class EnergyLSTM(nn.Module):
    """
    Stacked LSTM neural network for time series load forecasting.

    Args:
        input_size (int): Number of input features per time step.
        hidden_size (int): Hidden dimension size in recurrent units (default: 64).
        num_layers (int): Number of stacked LSTM layers (default: 2).
        dropout (float): Dropout probability between LSTM layers (default: 0.2).
        dense_units (int): Units in intermediate dense layer (default: 32).
        output_size (int): Forecast horizon / output dimension (default: 1).
    """

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2,
                 dense_units=32, output_size=1):
        super(EnergyLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Stacked LSTM
        # Note: PyTorch only applies dropout between layers if num_layers > 1
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout
        )

        # Regularization & Dense head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, dense_units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_units, output_size)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Tensor: Predicted energy load of shape (batch_size, output_size)
        """
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Take representation at the last time step
        last_step = lstm_out[:, -1, :]

        # Dense projection head
        out = self.dropout(last_step)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


def mape_loss(y_pred, y_true, epsilon=1e-7):
    """
    Mean Absolute Percentage Error (MAPE) loss with zero-division safeguard.
    """
    diff = torch.abs(y_true - y_pred)
    denom = torch.clamp(torch.abs(y_true), min=epsilon)
    return torch.mean(diff / denom) * 100.0


def calculate_metrics(y_true, y_pred, epsilon=1e-7):
    """
    Compute comprehensive power engineering forecasting metrics.

    Args:
        y_true (np.ndarray): Actual energy demand values.
        y_pred (np.ndarray): Predicted energy demand values.

    Returns:
        dict: Evaluated metrics (MAE, RMSE, MAPE %, R2 score).
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # MAPE (%)
    denom = np.maximum(np.abs(y_true), epsilon)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

    # R2 Score
    r2 = float(r2_score(y_true, y_pred))

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "R2": round(r2, 4)
    }


if __name__ == "__main__":
    # Smoke test model initialization and forward pass
    print("Testing EnergyLSTM architecture...")
    model = EnergyLSTM(input_size=10, hidden_size=64, num_layers=2)
    sample_input = torch.randn(16, 24, 10)  # (batch=16, seq_len=24 hours, features=10)
    output = model(sample_input)
    print(f"Input shape:  {sample_input.shape}")
    print(f"Output shape: {output.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params:,}")
    print("Architecture verified successfully!")
