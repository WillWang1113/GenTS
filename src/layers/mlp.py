import copy
import torch
from torch import nn
from torchvision.ops import MLP


class MLPEncoder(nn.Module):
    def __init__(
        self, seq_len, seq_dim, latent_dim, hidden_size_list=[64, 128, 256], **kwargs
    ):
        super().__init__()
        # Build Encoder
        self.encoder = MLP(seq_len * seq_dim, hidden_size_list + [latent_dim])

    def forward(self, x):
        x = self.encoder(x.flatten(1))
        return x


class MLPDecoder(nn.Module):
    def __init__(
        self,
        seq_len,
        seq_dim,
        latent_dim,
        hidden_size_list=[256, 128, 64],
        **kwargs,
    ):
        super().__init__()
        self.seq_len=seq_len
        self.seq_dim=seq_dim
        # Build Decoder
        self.decoder = MLP(latent_dim, hidden_size_list + [seq_len * seq_dim])

    def forward(self, x):
        x = self.decoder(x)
        return x.reshape(-1, self.seq_len, self.seq_dim)