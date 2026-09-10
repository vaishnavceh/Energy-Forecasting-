"""
deepdeff_model.py
=================
Deep Derived Feature Fusion (DeepDeFF) Y-shaped sequential model.

Architecture (from the paper):
    Basic Features              Derived Features
          │                           │
    ┌─────▼─────────┐      ┌──────────▼──────────┐
    │ Sequential     │      │ Sequential          │
    │ Layer (20)     │      │ Layer (20)          │
    │ Dropout 0.2    │      │ Dropout 0.2         │
    └─────┬─────────┘      └──────────┬──────────┘
          │                           │
          └───────────┬───────────────┘
                      │ Concatenate
                ┌─────▼─────┐
                │Dense (ReLU)│
                └─────┬─────┘
                ┌─────▼─────┐
                │Output (1) │
                │  Linear   │
                └───────────┘

Supports 6 variants: RNN, LSTM, GRU, BRNN, BLSTM, BGRU
Uses PyTorch (compatible with Python 3.14+)
"""

import torch
import torch.nn as nn


class DeepDeFF(nn.Module):
    """
    Deep Derived Feature Fusion (DeepDeFF) Y-shaped sequential model.

    Two parallel recurrent branches process basic and derived features
    independently, then merge for final prediction.

    Args:
        basic_features (int): Number of basic input features per time step.
        derived_features (int): Number of derived input features per time step.
        cell_type (str): 'RNN', 'LSTM', 'GRU', 'BRNN', 'BLSTM', 'BGRU'.
        n_units (int): Number of hidden units in recurrent layers (default: 20).
        dropout (float): Dropout rate (default: 0.2).
        dense_units (int): Number of units in the dense merge layer (default: 20).
    """

    def __init__(self, basic_features, derived_features, cell_type='BLSTM',
                 n_units=20, dropout=0.2, dense_units=20):
        super(DeepDeFF, self).__init__()
        self.cell_type = cell_type

        # Determine if bidirectional
        bidirectional = cell_type.startswith('B')
        base_type = cell_type[1:] if bidirectional else cell_type
        num_directions = 2 if bidirectional else 1

        # Map to PyTorch RNN classes
        rnn_map = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}
        if base_type not in rnn_map:
            raise ValueError(f"Unsupported cell_type: {cell_type}. "
                             f"Choose from: RNN, LSTM, GRU, BRNN, BLSTM, BGRU")

        RNNClass = rnn_map[base_type]

        # Basic features branch
        self.basic_rnn = RNNClass(
            input_size=basic_features,
            hidden_size=n_units,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.basic_dropout = nn.Dropout(dropout)

        # Derived features branch
        self.derived_rnn = RNNClass(
            input_size=derived_features,
            hidden_size=n_units,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.derived_dropout = nn.Dropout(dropout)

        # Merge layer: concatenated output from both branches
        merge_size = n_units * num_directions * 2  # *2 for two branches
        self.dense = nn.Linear(merge_size, dense_units)
        self.relu = nn.ReLU()

        # Output layer
        self.output_layer = nn.Linear(dense_units, 1)

    def forward(self, basic_input, derived_input):
        """
        Forward pass.

        Args:
            basic_input: Tensor of shape (batch, K, basic_features)
            derived_input: Tensor of shape (batch, K, derived_features)

        Returns:
            Tensor of shape (batch, 1) — predicted load
        """
        # Basic branch: take last hidden state
        basic_out, _ = self.basic_rnn(basic_input)
        basic_out = basic_out[:, -1, :]  # last time step
        basic_out = self.basic_dropout(basic_out)

        # Derived branch: take last hidden state
        derived_out, _ = self.derived_rnn(derived_input)
        derived_out = derived_out[:, -1, :]  # last time step
        derived_out = self.derived_dropout(derived_out)

        # Merge
        merged = torch.cat([basic_out, derived_out], dim=1)
        dense_out = self.relu(self.dense(merged))

        # Output
        output = self.output_layer(dense_out)
        return output


def mape_loss(y_pred, y_true):
    """
    Custom Mean Absolute Percentage Error (MAPE) loss function.

    MAPE = mean(|y_true - y_pred| / |y_true|) * 100

    Args:
        y_pred (Tensor): Predicted values.
        y_true (Tensor): True values.

    Returns:
        Tensor: MAPE loss.
    """
    epsilon = 1e-7
    diff = torch.abs(y_true - y_pred)
    denom = torch.clamp(torch.abs(y_true), min=epsilon)
    return torch.mean(diff / denom) * 100


def calculate_mape(y_pred, y_true):
    """
    Calculate MAPE metric (numpy arrays).

    Args:
        y_pred (ndarray): Predicted values.
        y_true (ndarray): True values.

    Returns:
        float: MAPE percentage.
    """
    import numpy as np
    epsilon = 1e-7
    diff = np.abs(y_true - y_pred)
    denom = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(diff / denom) * 100)


def build_deepdeff_model(basic_features, derived_features, cell_type='BLSTM',
                         n_units=20, dropout=0.2, dense_units=20):
    """
    Build a DeepDeFF model instance.

    Args:
        basic_features (int): Number of basic input features.
        derived_features (int): Number of derived input features.
        cell_type (str): One of 'RNN', 'LSTM', 'GRU', 'BRNN', 'BLSTM', 'BGRU'.
        n_units (int): Hidden units (default: 20).
        dropout (float): Dropout rate (default: 0.2).
        dense_units (int): Dense layer units (default: 20).

    Returns:
        DeepDeFF: Model instance.
    """
    return DeepDeFF(
        basic_features=basic_features,
        derived_features=derived_features,
        cell_type=cell_type,
        n_units=n_units,
        dropout=dropout,
        dense_units=dense_units
    )


def get_all_model_variants(basic_features, derived_features):
    """
    Create all 6 model variants.

    Returns:
        dict: {variant_name: DeepDeFF model instance}
    """
    cell_types = ['RNN', 'LSTM', 'GRU', 'BRNN', 'BLSTM', 'BGRU']
    variants = {}
    for ct in cell_types:
        variants[ct] = build_deepdeff_model(basic_features, derived_features, cell_type=ct)
    return variants


if __name__ == "__main__":
    # Test with example shapes
    basic_feats = 57   # 1 (energy) + 48 (time_slot one-hot) + 7 (day one-hot) + 1 (is_weekend)
    derived_feats = 4  # rolling_mean, rolling_std, slot_mean, slot_std

    print(f"Basic features: {basic_feats}, Derived features: {derived_feats}")
    print(f"{'='*60}")

    models = get_all_model_variants(basic_feats, derived_feats)
    for name, model in models.items():
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--- DeepDeFF {name} ---")
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable:    {trainable_params:,}")

        # Test forward pass
        batch_size = 8
        K = 2
        x_basic = torch.randn(batch_size, K, basic_feats)
        x_derived = torch.randn(batch_size, K, derived_feats)
        out = model(x_basic, x_derived)
        print(f"  Output shape: {out.shape}")

    print(f"\n{'='*60}")
    print("All 6 model variants created successfully!")
