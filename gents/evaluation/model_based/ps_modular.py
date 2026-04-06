from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# ---------------------------------------------------------------------------
# Model hyper-parameter configs
# ---------------------------------------------------------------------------

@dataclass
class RNNConfig:
    hidden_dim: int = 64
    num_layers: int = 1
    cell_type: str = "GRU"  # "GRU", "LSTM", "RNN"
    dropout: float = 0.0


@dataclass
class CNNConfig:
    hidden_channels: int = 64
    num_layers: int = 3
    kernel_size: int = 3
    dropout: float = 0.0


@dataclass
class TransformerConfig:
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1


# ---------------------------------------------------------------------------
# Predictor modules
# ---------------------------------------------------------------------------

class RNNPredictor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, cfg: RNNConfig):
        super().__init__()
        cell_cls = {"GRU": nn.GRU, "LSTM": nn.LSTM, "RNN": nn.RNN}
        if cfg.cell_type not in cell_cls:
            raise ValueError(f"Unknown RNN cell type: {cfg.cell_type}")
        self.rnn = cell_cls[cfg.cell_type](
            input_dim, cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(cfg.hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out)


class CNNPredictor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, cfg: CNNConfig):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i in range(cfg.num_layers):
            out_ch = cfg.hidden_channels
            padding = (cfg.kernel_size - 1)  # causal padding size
            layers.append(nn.Conv1d(in_ch, out_ch, cfg.kernel_size, padding=padding))
            layers.append(nn.ReLU())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(cfg.hidden_channels, output_dim)
        self.kernel_size = cfg.kernel_size
        self.num_layers = cfg.num_layers

    def forward(self, x):
        # x: (B, T, C) -> conv expects (B, C, T)
        out = x.transpose(1, 2)
        out = self.conv(out)
        # trim to causal length: each layer adds (kernel_size - 1) extra steps
        trim = (self.kernel_size - 1) * self.num_layers
        if trim > 0:
            out = out[:, :, :-trim]
        out = out.transpose(1, 2)  # (B, T, C)
        return self.fc(out)


class TransformerPredictor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, seq_len: int, cfg: TransformerConfig):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, cfg.d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.fc = nn.Linear(cfg.d_model, output_dim)
        self.seq_len = seq_len

    def _causal_mask(self, size: int, device: torch.device):
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return mask

    def forward(self, x):
        T = x.size(1)
        out = self.input_proj(x) + self.pos_emb[:, :T, :]
        mask = self._causal_mask(T, x.device)
        out = self.encoder(out, mask=mask)
        return self.fc(out)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _build_predictor(
    model_type: str,
    input_dim: int,
    output_dim: int,
    seq_len: int,
    model_params: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    model_params = model_params or {}
    model_type = model_type.lower()

    if model_type == "rnn":
        cfg = RNNConfig(**model_params)
        return RNNPredictor(input_dim, output_dim, cfg)
    elif model_type == "cnn":
        cfg = CNNConfig(**model_params)
        return CNNPredictor(input_dim, output_dim, cfg)
    elif model_type == "transformer":
        cfg = TransformerConfig(**model_params)
        return TransformerPredictor(input_dim, output_dim, seq_len, cfg)
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose from 'rnn', 'cnn', 'transformer'."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predictive_score(
    ori_data: np.ndarray,
    generated_data: np.ndarray,
    device: str,
    model_type: str = "rnn",
    model_params: Optional[Dict[str, Any]] = None,
    iterations: int = 5000,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
) -> float:
    """Predictive score with configurable predictor model.

    Train a forecasting model on *generated_data* (next-step prediction),
    then evaluate MAE on *ori_data*.

    Args:
        ori_data: Real time series, shape ``(N, T, D)``.
        generated_data: Generated time series, shape ``(N', T, D)``.
        device: PyTorch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
        model_type: One of ``"rnn"``, ``"cnn"``, ``"transformer"``.
        model_params: Dict of model hyper-parameters forwarded to the
            corresponding config dataclass (``RNNConfig`` / ``CNNConfig`` /
            ``TransformerConfig``).  Pass ``None`` to use defaults.
        iterations: Number of training iterations.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.

    Returns:
        Predictive score (MAE on the original data).
    """
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)
    no, seq_len, dim = ori_data.shape

    # Build model
    predictor = _build_predictor(
        model_type=model_type,
        input_dim=dim,
        output_dim=dim,
        seq_len=seq_len - 1,
        model_params=model_params,
    ).to(device)

    optimizer = optim.Adam(predictor.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()

    # --- Training on synthetic data ---
    predictor.train()
    n_gen = len(generated_data)
    for _ in range(iterations):
        idx = np.random.permutation(n_gen)[:batch_size]
        X_mb = torch.from_numpy(generated_data[idx, :-1, :]).float().to(device)
        Y_mb = torch.from_numpy(generated_data[idx, 1:, :]).float().to(device)

        optimizer.zero_grad()
        loss = criterion(predictor(X_mb), Y_mb)
        loss.backward()
        optimizer.step()

    # --- Evaluation on real data ---
    predictor.eval()
    X_test = torch.from_numpy(ori_data[:, :-1, :]).float().to(device)
    Y_test = torch.from_numpy(ori_data[:, 1:, :]).float().to(device)

    with torch.no_grad():
        pred = predictor(X_test).cpu()

    score = torch.abs(Y_test.cpu() - pred).mean().item()
    return score
