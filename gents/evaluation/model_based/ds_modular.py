from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score


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
# Discriminator modules
# ---------------------------------------------------------------------------

class RNNDiscriminator(nn.Module):
    def __init__(self, input_dim: int, cfg: RNNConfig):
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
        self.fc = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, x):
        _, h = self.rnn(x)
        if isinstance(h, tuple):  # LSTM returns (h_n, c_n)
            h = h[0]
        logit = self.fc(h)  # (num_layers, B, 1)
        return logit[-1]     # last layer -> (B, 1)


class CNNDiscriminator(nn.Module):
    def __init__(self, input_dim: int, cfg: CNNConfig):
        super().__init__()
        layers = []
        in_ch = input_dim
        for _ in range(cfg.num_layers):
            out_ch = cfg.hidden_channels
            layers.append(nn.Conv1d(in_ch, out_ch, cfg.kernel_size, padding=cfg.kernel_size // 2))
            layers.append(nn.ReLU())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(cfg.hidden_channels, 1)

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        out = self.conv(x.transpose(1, 2))
        out = self.pool(out).squeeze(-1)  # (B, C)
        return self.fc(out)               # (B, 1)


class TransformerDiscriminator(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, cfg: TransformerConfig):
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
        self.fc = nn.Linear(cfg.d_model, 1)

    def forward(self, x):
        T = x.size(1)
        out = self.input_proj(x) + self.pos_emb[:, :T, :]
        out = self.encoder(out)
        out = out.mean(dim=1)  # global average pooling over time
        return self.fc(out)    # (B, 1)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _build_discriminator(
    model_type: str,
    input_dim: int,
    seq_len: int,
    model_params: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    model_params = model_params or {}
    model_type = model_type.lower()

    if model_type == "rnn":
        cfg = RNNConfig(**model_params)
        return RNNDiscriminator(input_dim, cfg)
    elif model_type == "cnn":
        cfg = CNNConfig(**model_params)
        return CNNDiscriminator(input_dim, cfg)
    elif model_type == "transformer":
        cfg = TransformerConfig(**model_params)
        return TransformerDiscriminator(input_dim, seq_len, cfg)
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose from 'rnn', 'cnn', 'transformer'."
        )


# ---------------------------------------------------------------------------
# Utilities (from ds.py)
# ---------------------------------------------------------------------------

def _train_test_divide(data_x, data_x_hat, train_rate=0.8):
    no = len(data_x)
    idx = np.random.permutation(no)
    train_x = data_x[idx[: int(no * train_rate)]]
    test_x = data_x[idx[int(no * train_rate) :]]

    no_hat = len(data_x_hat)
    idx_hat = np.random.permutation(no_hat)
    train_x_hat = data_x_hat[idx_hat[: int(no_hat * train_rate)]]
    test_x_hat = data_x_hat[idx_hat[int(no_hat * train_rate) :]]

    return train_x, train_x_hat, test_x, test_x_hat


def _batch_indices(n: int, batch_size: int):
    idx = np.random.permutation(n)[:batch_size]
    return idx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discriminative_score(
    ori_data: np.ndarray,
    generated_data: np.ndarray,
    device: str,
    model_type: str = "rnn",
    model_params: Optional[Dict[str, Any]] = None,
    iterations: int = 2000,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    train_rate: float = 0.8,
) -> float:
    """Discriminative score with configurable discriminator model.

    Train a binary classifier to distinguish real from generated time series.
    Report ``|0.5 - accuracy|`` on the held-out test set.

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
        train_rate: Fraction of data used for training.

    Returns:
        Discriminative score  (``|0.5 - accuracy|``).
    """
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)
    _, seq_len, dim = ori_data.shape

    # Train / test split
    train_x, train_x_hat, test_x, test_x_hat = _train_test_divide(
        ori_data, generated_data, train_rate
    )

    # Build model
    model = _build_discriminator(
        model_type=model_type,
        input_dim=dim,
        seq_len=seq_len,
        model_params=model_params,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Training ---
    model.train()
    n_train_real = len(train_x)
    n_train_fake = len(train_x_hat)

    for _ in range(iterations):
        idx_real = _batch_indices(n_train_real, batch_size)
        idx_fake = _batch_indices(n_train_fake, batch_size)

        X_real = torch.from_numpy(train_x[idx_real]).float().to(device)
        X_fake = torch.from_numpy(train_x_hat[idx_fake]).float().to(device)

        logit_real = model(X_real)
        logit_fake = model(X_fake)

        loss = (
            nn.functional.binary_cross_entropy_with_logits(
                logit_real, torch.ones_like(logit_real)
            )
            + nn.functional.binary_cross_entropy_with_logits(
                logit_fake, torch.zeros_like(logit_fake)
            )
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # --- Evaluation ---
    model.eval()
    with torch.no_grad():
        X_test_real = torch.from_numpy(test_x).float().to(device)
        X_test_fake = torch.from_numpy(test_x_hat).float().to(device)

        prob_real = torch.sigmoid(model(X_test_real)).cpu().numpy().reshape(-1)
        prob_fake = torch.sigmoid(model(X_test_fake)).cpu().numpy().reshape(-1)

    y_pred = np.concatenate([(prob_real > 0.5).astype(int),
                             (prob_fake > 0.5).astype(int)])
    y_true = np.concatenate([np.ones(len(prob_real)),
                             np.zeros(len(prob_fake))])

    acc = accuracy_score(y_true, y_pred)
    return float(np.abs(0.5 - acc))
