from typing import List

import lightning as L
import torch
from torch.nn import functional as F
from torchvision.ops import MLP

from .layers import ConvDecoder, ConvEncoder


class VanillaVAE(L.LightningModule):
    def __init__(
        self,
        seq_len: int,
        seq_dim: int,
        latent_dim: int,
        beta: float = 1e-3,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        hidden_size_list: list = [64, 128, 256],
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

    def training_step(self, batch, batch_idx):
        x = batch["seq"]
        x = x.permute(0, 2, 1)

        # encode
        latents = self.encoder(x)
        mu = self.fc_mu(latents)
        logvar = self.fc_logvar(latents)

        # reparameterize
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu

        # decode
        x_hat = self.decoder(z).permute(0, 2, 1)
        x = x.permute(0, 2, 1)

        # loss
        loss = self._get_loss(x, x_hat, mu, logvar)

        self.log_dict({"train_loss": loss})

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["seq"]
        x = x.permute(0, 2, 1)

        # encode
        latents = self.encoder(x)
        mu = self.fc_mu(latents)
        logvar = self.fc_logvar(latents)

        # reparameterize
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu

        # decode
        x_hat = self.decoder(z).permute(0, 2, 1)
        x = x.permute(0, 2, 1)

        loss = self._get_loss(x, x_hat, mu, logvar)
        self.log_dict({"val_loss": loss})
        return loss

    def sample(self, n_sample):
        z = torch.randn((n_sample, self.hparams_initial.latent_dim))
        x_hat = self.decoder(z).permute(0, 2, 1)
        return x_hat

    def _get_loss(self, x, x_hat, mu, logvar):
        recons_loss = F.mse_loss(x_hat, x)
        kld_loss = torch.mean(
            0.5
            * torch.sum(
                -self.hparams_initial.latent_dim - logvar + mu**2 + logvar.exp(), dim=1
            ),
            dim=0,
        )
        loss = recons_loss + self.hparams_initial.beta * kld_loss
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            self.hparams_initial.lr,
            weight_decay=self.hparams_initial.weight_decay,
        )
