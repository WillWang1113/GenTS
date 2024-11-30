from typing import List

import lightning as L
import torch
from torch.nn import functional as F
from torchvision.ops import MLP

from .layers import ConvDecoder, ConvEncoder
from .base import BaseVAE


class VanillaVAE(BaseVAE):
    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        latent_dim: int,
        hidden_size_list: list = [64, 128, 256],
        beta: float = 1e-3,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        print(self.hparams)
        self.encoder = ConvEncoder(**self.hparams)
        self.hparams.hidden_size_list.reverse()
        self.decoder = ConvDecoder(**self.hparams)

        self.fc_mu = MLP(latent_dim, [latent_dim])
        self.fc_logvar = MLP(latent_dim, [latent_dim])

    def encode(self, x, c=None):
        x = x.permute(0, 2, 1)
        latents = self.encoder(x)
        mu = self.fc_mu(latents)
        logvar = self.fc_logvar(latents)
        return latents, mu, logvar

    def decode(self, z, c=None):
        return self.decoder(z).permute(0, 2, 1)


    def sample(self, n_sample):
        z = torch.randn((n_sample, self.hparams_initial.latent_dim))
        x_hat = self.decode(z)
        return x_hat


    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            self.hparams_initial.lr,
            weight_decay=self.hparams_initial.weight_decay,
        )
